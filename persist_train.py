# --- add imports near the top of train_hubert_pool_head.py ---
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
from tqdm.auto import tqdm

from disco_gp.data import setup_audio_task

# import your persistence helpers (adjust module name as needed)
from persist import neural_persistence_of_module_linears, PoolHeadNoPoolingInside  # or disco_gp.neural_persistence

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
p_num = 2.0
# ------------------------------------------

def compute_head_persistence_details(head_module, p_num=2.0):
    """
    Returns dict:
      {
        "per_layer": {layer_name: (total, normalized), ...},
        "overall_total": float,
        "overall_normalized": float
      }
    """
    raw = neural_persistence_of_module_linears(head_module, p_num=p_num)
    per_layer = {}
    overall_total = 0.0
    overall_norm = 0.0
    for lname, info in raw.items():
        t = float(info.get("total_persistence", 0.0))
        tn = float(info.get("total_persistence_normalized", 0.0))
        per_layer[lname or "<linear>"] = (t, tn)
        overall_total += t
        overall_norm += tn
    return {"per_layer": per_layer, "overall_total": overall_total, "overall_normalized": overall_norm}

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

# --------------------------
# helpers for tracking persistence
# --------------------------
def summarize_persistence_dict(pdict):
    """
    pdict: output of neural_persistence_of_module_linears, e.g.
      { "": {"total_persistence": X, "total_persistence_normalized": Y}, ... }
    Returns:
      per_layer_summary: {layer_name: (total, normalized)}
      overall_total: sum of total_persistence over layers
      overall_norm: sum of normalized persistence over layers (or other aggregator you prefer)
    """
    per_layer = {}
    overall_total = 0.0
    overall_norm = 0.0
    for layer_name, info in pdict.items():
        t = float(info.get("total_persistence", 0.0))
        tn = float(info.get("total_persistence_normalized", 0.0))
        per_layer[layer_name] = (t, tn)
        overall_total += t
        overall_norm += tn
    return per_layer, overall_total, overall_norm

def get_head_persistence(head_module, p_num=2.0):
    """
    Compute persistence for all Linear modules in head_module.
    Returns a dict:
      {
        "per_layer": {layer_name: (total, normalized), ...},
        "overall_total": float,
        "overall_normalized": float
      }
    """
    raw = neural_persistence_of_module_linears(head_module, p_num=p_num)
    per_layer, overall_total, overall_norm = summarize_persistence_dict(raw)
    return {"per_layer": per_layer, "overall_total": overall_total, "overall_normalized": overall_norm}

def evaluate_and_snapshot(hubert, head, dataloader, device, p_num=2.0):
    """
    Runs one eval pass on `dataloader` to compute accuracy, then computes persistence
    from the current head weights. Returns (accuracy, persistence_dict).
    """
    head.eval()
    hubert.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            # handle optional attention mask if present in the batch
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                hubert_out = hubert(input_values, attention_mask=attention_mask).last_hidden_state
                pooled = masked_mean(hubert_out, attention_mask)   # uses your masked_mean helper
            else:
                hubert_out = hubert(input_values).last_hidden_state
                pooled = F.adaptive_avg_pool1d(hubert_out.transpose(1, 2), 1).squeeze(-1)

            logits = head(pooled)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    persist_dict = compute_head_persistence_details(head, p_num=p_num)
    return acc, persist_dict


# --------------------------
# plotting helper
# --------------------------
def save_persistence_accuracy_plot(metrics_path="metrics_history.json",
                                   out_path="persistence_plots/acc_vs_persistence.png",
                                   figsize=(7,5), dpi=150, show_plot=False):
    """
    Load metrics_history.json and save a plot of normalized persistence vs accuracy.
    Does NOT call plt.show() by default. Saves to out_path (PNG).
    
    Args:
        metrics_path (str): path to JSON file with keys "epochs", "accuracy", "persistence_norm".
        out_path (str): path where PNG will be written.
        figsize (tuple): figure size (width, height).
        dpi (int): output DPI for saved image.
        show_plot (bool): if True, call plt.show() (default False).
    """
    # load file
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    epochs = metrics.get("epochs", [])
    persistence = metrics.get("persistence_norm", metrics.get("overall_norm", []))
    accuracy = metrics.get("accuracy", None)

    # Basic validation / alignment: trim/pad to same snapshot length
    n = len(epochs)
    if n == 0:
        raise ValueError(f"No epochs found in {metrics_path} (epochs list is empty).")

    # ensure persistence list length matches
    if len(persistence) < n:
        # pad with NaN to align
        persistence = list(persistence) + [math.nan] * (n - len(persistence))
    elif len(persistence) > n:
        persistence = persistence[:n]

    if accuracy is None:
        acc_vals = [math.nan] * n
    else:
        if len(accuracy) < n:
            acc_vals = list(accuracy) + [math.nan] * (n - len(accuracy))
        elif len(accuracy) > n:
            acc_vals = accuracy[:n]
        else:
            acc_vals = accuracy

    # create output folder
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # build plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, persistence, marker="o", linestyle="-", label="Normalized Persistence")
    ax.plot(epochs, acc_vals, marker="s", linestyle="-", label="Accuracy")

    ax.set_xlabel("Epoch (0 = before training)")
    ax.set_ylabel("Score (0 - 1)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(epochs)
    ax.set_title("Normalized Persistence vs Accuracy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    # save and close
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return out_path


if __name__ == "__main__":
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
    head = PoolHeadNoPoolingInside(hidden_dim=768, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    # --------------------------
    # in your main training code: create a history container BEFORE training
    # --------------------------
    init_acc, init_persist = evaluate_and_snapshot(hubert, head, dev_loader, DEVICE, p_num=p_num)
    metrics_history = {"epochs": [0], "accuracy": [init_acc], "persistence_norm": [init_persist["overall_normalized"]]}
    print(f"[Init] acc={init_acc:.4f}, norm_persist={init_persist['overall_normalized']:.4f}")


    # --------------------------
    # inside your epoch loop (after validation), append a snapshot
    # Replace your existing epoch loop's eval block with this pattern or just
    # add the persistence snapshot code after you've finished the epoch's eval.
    # --------------------------

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
            # after training epoch (and after evaluation if you like), snapshot persistence:

        epoch_idx = epoch + 1
        curr_acc, curr_persist = evaluate_and_snapshot(hubert, head, dev_loader, DEVICE, p_num=p_num)

        metrics_history["epochs"].append(epoch_idx)
        metrics_history["accuracy"].append(curr_acc)
        metrics_history["persistence_norm"].append(curr_persist["overall_normalized"])

        print(f"Epoch {epoch_idx}: acc={curr_acc:.4f}, norm_persist={curr_persist['overall_normalized']:.4f}")

        # optional: save intermediate
        with open("metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    # Example: after training finishes
    saved = save_persistence_accuracy_plot(
        metrics_path="metrics_history.json",
        out_path="persistence_plots/acc_vs_persistence.png",
        figsize=(8,5),
        dpi=200,
        show_plot=False
    )
    print("Saved plot to:", saved)
