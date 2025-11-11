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
from neural_persistence import neural_persistence_of_module_linears, PoolHeadNoPoolingInside  # or disco_gp.neural_persistence

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
p = 2.0
# ------------------------------------------

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

def get_head_persistence(head_module, p=2.0):
    """
    Compute persistence for all Linear modules in head_module.
    Returns a dict:
      {
        "per_layer": {layer_name: (total, normalized), ...},
        "overall_total": float,
        "overall_normalized": float
      }
    """
    raw = neural_persistence_of_module_linears(head_module, p=p)
    per_layer, overall_total, overall_norm = summarize_persistence_dict(raw)
    return {"per_layer": per_layer, "overall_total": overall_total, "overall_normalized": overall_norm}

# --------------------------
# plotting helper
# --------------------------
def plot_persistence_history(history, save_path=None, show=True):
    """
    history: the persistence_history dict used above.
    Produces two plots:
      - overall normalized persistence vs snapshot (initial + per-epoch)
      - per-layer normalized persistence vs snapshot lines
    """
    epochs = history["epochs"]
    # overall
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, history["overall_norm"], marker="o")
    ax1.set_xlabel("snapshot (0=initial, 1..N=after epoch)")
    ax1.set_ylabel("overall normalized persistence (sum over layers)")
    ax1.set_title("Overall normalized persistence vs epoch")
    ax1.grid(True)

    # per-layer normalized
    fig2, ax2 = plt.subplots()
    for layer_name, vals in history["per_layer"].items():
        # vals is list of (total, norm)
        norms = [v[1] for v in vals]
        # If some layers don't have every snapshot, pad with NaN for plotting clarity
        if len(norms) < len(epochs):
            # simple padding at end (shouldn't usually happen)
            norms = norms + [float("nan")] * (len(epochs) - len(norms))
        ax2.plot(epochs, norms, marker="o", label=layer_name)
    ax2.set_xlabel("snapshot (0=initial, 1..N=after epoch)")
    ax2.set_ylabel("layer normalized persistence")
    ax2.set_title("Per-layer normalized persistence vs epoch")
    ax2.legend(loc="best", fontsize="small")
    ax2.grid(True)

    if save_path:
        # save both plots in one figure folder
        os.makedirs(save_path, exist_ok=True)
        fig1.savefig(os.path.join(save_path, "overall_normalized_persistence.png"), bbox_inches="tight")
        fig2.savefig(os.path.join(save_path, "per_layer_normalized_persistence.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


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
    head = PoolHeadNoPoolingInside(hidden_dim=H, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    # --------------------------
    # in your main training code: create a history container BEFORE training
    # --------------------------
    persistence_history = {
        "p": p,
        "epochs": [],          # list of epoch indices (0 for before training)
        "overall_total": [],   # overall_total at each snapshot
        "overall_norm": [],    # overall_normalized at each snapshot
        "per_layer": {}        # layer_name -> list of (total, normalized) per snapshot
    }

    # compute persistence BEFORE training (snapshot 0)
    init_p = get_head_persistence(head, p=p)
    persistence_history["epochs"].append(0)
    persistence_history["overall_total"].append(init_p["overall_total"])
    persistence_history["overall_norm"].append(init_p["overall_normalized"])
    for lname, (t, tn) in init_p["per_layer"].items():
        persistence_history["per_layer"].setdefault(lname, []).append((t, tn))

    print("Initial head persistence:", init_p)

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

        head.eval()
        # (run eval here as in your code) ...
        # then snapshot:
        epoch_idx = epoch + 1   # choose 1..NUM_EPOCHS for readability
        curr_p = get_head_persistence(head, p=p)
        persistence_history["epochs"].append(epoch_idx)
        persistence_history["overall_total"].append(curr_p["overall_total"])
        persistence_history["overall_norm"].append(curr_p["overall_normalized"])
        for lname, (t, tn) in curr_p["per_layer"].items():
            persistence_history["per_layer"].setdefault(lname, []).append((t, tn))

        # optionally save history to disk each epoch
        with open("persistence_history.json", "w") as f:
            # convert tuples to lists for JSON
            serial = {
                "p": persistence_history["p"],
                "epochs": persistence_history["epochs"],
                "overall_total": persistence_history["overall_total"],
                "overall_norm": persistence_history["overall_norm"],
                "per_layer": {k: [list(x) for x in v] for k, v in persistence_history["per_layer"].items()}
            }
            json.dump(serial, f, indent=2)

        print(f"Epoch {epoch_idx} persistence overall_norm={curr_p['overall_normalized']:.6f}, overall_total={curr_p['overall_total']:.6f}")

    # Example: after training finishes
    plot_persistence_history(persistence_history, save_path="persistence_plots", show=True)
