# train_hubert_pool_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
from tqdm.auto import tqdm

from disco_gp.data import setup_audio_task

# ----------------- config -----------------
HUBERT_MODEL = "facebook/hubert-base-ls960"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
TARGET_SR = 16000
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 1e-4
LABEL_MAP = {"bonafide": 0, "spoof": 1}
SAVE_HEAD_PATH = "pool_head.pth"
DROPOUT = 0.1
# ------------------------------------------
dls = setup_audio_task()
train_loader = dls.train
dev_loader   = dls.eval
fe = AutoFeatureExtractor.from_pretrained(HUBERT_MODEL)

# 3) load & freeze hubert
hubert = HubertModel.from_pretrained(HUBERT_MODEL)
hubert.eval()
for p in hubert.parameters():
    p.requires_grad = False
hubert.to(DEVICE)

# 4) define head (receives pooled (B, H) and returns logit (B,))
class PoolHeadNoPoolingInside(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(hidden_dim)   # trainable
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, pooled: torch.Tensor):
        # pooled: (B, H)
        x = self.ln(pooled)
        x = self.drop(x)
        logit = self.fc(x).squeeze(-1)   # (B,)
        return logit

# get hidden dim (H) by running one dummy forward to avoid assumptions
with torch.no_grad():
    # create one zero example of 1 second (or small) to get shape
    dummy_wav = torch.zeros(TARGET_SR, dtype=torch.float32)  # [T]
    enc = fe(dummy_wav, sampling_rate=TARGET_SR, return_tensors="pt")
    enc_vals = enc["input_values"].to(DEVICE)
    out = hubert(enc_vals).last_hidden_state
    H = out.size(-1)
    print(f"Hidden size is: {H}")

head = PoolHeadNoPoolingInside(hidden_dim=H, dropout=DROPOUT).to(DEVICE)

# 5) training loop: pooling done outside head (masked average pooling)
optimizer = torch.optim.Adam(head.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

def masked_mean(hidden, attn_mask):
    """
    hidden: (B, T, H)
    attn_mask: (B, T) where 1 indicates real tokens, 0 padded.
    returns: (B, H)
    """
    mask = attn_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
    sum_hidden = (hidden * mask).sum(dim=1)           # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
    return sum_hidden / lengths                       # (B, H)

for epoch in range(NUM_EPOCHS):
    head.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    for batch in pbar:
        input_values = batch["tokens"].to(DEVICE)         # (B, L, D) or (B, L)
        # attention_mask = batch["attention_mask"].to(DEVICE)  # (B,)
        attention_mask = None
        labels = batch["labels"].to(DEVICE) 

        with torch.no_grad():
            # pass attention_mask if present
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
                hubert_out = hubert(input_values, attention_mask=attention_mask).last_hidden_state  # (B, T, H)
                pooled = masked_mean(hubert_out, attention_mask)  # (B, H)
            else:
                hubert_out = hubert(input_values).last_hidden_state
                pooled = F.adaptive_avg_pool1d(hubert_out.transpose(1,2), 1).squeeze(-1)

        logits = head(pooled)  # (B,)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(train_loss=f"{running_loss/ (pbar.n+1):.4f}")

    # eval
    head.eval()
    total = 0
    correct = 0
    dev_loss = 0.0
    with torch.no_grad():
        for batch in pbar:
            input_values = batch["tokens"].to(DEVICE)         # (B, L, D) or (B, L)
            # attention_mask = batch["attention_mask"].to(DEVICE)  # (B,)
            attention_mask = None
            labels = batch["labels"].to(DEVICE) 

            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
                hubert_out = hubert(input_values, attention_mask=attention_mask).last_hidden_state
                pooled = masked_mean(hubert_out, attention_mask)
            else:
                hubert_out = hubert(input_values).last_hidden_state
                pooled = F.adaptive_avg_pool1d(hubert_out.transpose(1,2), 1).squeeze(-1)
                
            logits = head(pooled)
            dev_loss += criterion(logits, labels).item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
    dev_loss /= total
    acc = correct / total
    print(f"Epoch {epoch+1} dev_loss={dev_loss:.4f} acc={acc:.4f}")

# 6) save head (no hubert inside)
def save_head(head_module: nn.Module, path: str):
    torch.save({
        "state_dict": head_module.state_dict(),
        "hidden_dim": head_module.hidden_dim,
        "dropout": head_module.drop.p if isinstance(head_module.drop, nn.Dropout) else 0.0,
    }, path)

def load_head(path: str, device: torch.device = None) -> nn.Module:
    data = torch.load(path, map_location=device)
    hidden_dim = data["hidden_dim"]
    dropout = data.get("dropout", 0.0)
    h = PoolHeadNoPoolingInside(hidden_dim, dropout)
    h.load_state_dict(data["state_dict"])
    if device is not None:
        h.to(device)
    return h

save_head(head, SAVE_HEAD_PATH)
print("Saved head to", SAVE_HEAD_PATH)
