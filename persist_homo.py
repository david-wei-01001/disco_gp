# requirements: aleph, numpy, torch
# pip install aleph numpy torch   # if you don't already have them

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
from tqdm.auto import tqdm

from disco_gp.data import setup_audio_task

import aleph as al

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

def linear_to_bipartite_adjacency(weight_np: np.ndarray) -> np.ndarray:
    """
    Given a weight matrix W of shape (out_features, in_features),
    return bipartite adjacency matrix M of shape (out+in, out+in):
      [ 0   | |W| ]
      [ |W|^T| 0  ]
    """
    W_abs = np.abs(weight_np)
    out, inp = W_abs.shape
    M = np.zeros((out + inp, out + inp), dtype=float)
    M[:out, out:] = W_abs
    M[out:, :out] = W_abs.T
    return M


def compute_neural_persistence_from_adjacency(M: np.ndarray, aleph_module=al):
    """
    Given square adjacency matrix M (vertices = m+n), compute the zero-dimensional
    persistence diagram via aleph and return total persistence and normalized.
    Normalization follows the original neural persistence code: tp / sqrt(n_vertices - 1)
    """
    if M.size == 0:
        return {"total_persistence": 0.0, "total_persistence_normalized": 0.0}

    # normalize by maximum absolute weight to map weights into [0,1]
    W = np.max(M)
    if W == 0:
        M_norm = M.copy()
    else:
        M_norm = M / float(W)

    # Aleph call — reverseFiltration=True so edges with larger weight appear earlier
    D = aleph_module.calculateZeroDimensionalPersistenceDiagramOfMatrix(
        M_norm,
        reverseFiltration=True,
        vertexWeight=1.0,
        unpairedData=0.0
    )

    tp = aleph_module.norms.pNorm(D)  # scalar total persistence
    n_vertices = M.shape[0]           # this is out+in
    tp_norm = tp / math.sqrt(max(1, n_vertices - 1))

    return {"total_persistence": float(tp), "total_persistence_normalized": float(tp_norm)}


def neural_persistence_of_linear_layer(linear: torch.nn.Linear, aleph_module=al):
    """
    Compute neural persistence for a torch.nn.Linear module (ignores bias).
    Returns the same dict as compute_neural_persistence_from_adjacency.
    """
    # get weight as numpy (cpu)
    W = linear.weight.detach().cpu().numpy()  # shape (out, in)
    M = linear_to_bipartite_adjacency(W)
    return compute_neural_persistence_from_adjacency(M, aleph_module=aleph_module)


def neural_persistence_of_module_linears(module: torch.nn.Module, aleph_module=al):
    """
    Compute neural persistence for every nn.Linear inside `module`.
    Returns dict mapping layer name -> persistence dict.
    """
    results = {}
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.Linear):
            results[name or "<linear>"] = neural_persistence_of_linear_layer(mod, aleph_module=aleph_module)
    return results


# -------------------------
# Example usage with your head
# -------------------------
hubert = HubertModel.from_pretrained(HUBERT_MODEL)
hubert.eval()
for p in hubert.parameters():
    p.requires_grad = False
hubert.to(DEVICE)
head = PoolHeadNoPoolingInside(hidden_dim=768, dropout=DROPOUT).to(DEVICE)
# head is your PoolHeadNoPoolingInside instance
# compute persistence for the fc linear only:
score = neural_persistence_of_linear_layer(head.fc)
print("fc persistence:", score)

# OR compute persistence for all Linear layers inside the head (if you later add others):
all_scores = neural_persistence_of_module_linears(head)
print("all linear persistences:", all_scores)
