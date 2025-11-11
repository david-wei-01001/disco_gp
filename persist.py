# requirements: aleph, numpy, torch
# pip install aleph numpy torch   # if you don't already have them

import math
import numpy as np
import torch
import torch.nn as nn


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

# -----------------------
# helper: build bipartite adjacency (same as before)
# -----------------------
def linear_to_bipartite_adjacency(weight_np: np.ndarray) -> np.ndarray:
    """
    Given weight matrix W (out, in), return bipartite adjacency
      [ 0    | |W| ]
      [ |W|^T|  0  ]
    returns shape (out+in, out+in)
    """
    W_abs = np.abs(weight_np)
    out, inp = W_abs.shape
    M = np.zeros((out + inp, out + inp), dtype=float)
    M[:out, out:] = W_abs
    M[out:, :out] = W_abs.T
    return M

# -----------------------
# union-find for merges
# -----------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, a):
        p = self.parent
        # path compression
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True
    def components(self):
        roots = set(self.find(i) for i in range(len(self.parent)))
        return roots

# -----------------------
# compute 0-dim persistence diagram (as list of persistences)
# matching Aleph call: reverseFiltration=True, vertexWeight=1.0, unpairedData=0.0
# -----------------------
def zero_dim_persistences_from_matrix(M_norm: np.ndarray):
    """
    M_norm: square symmetric adjacency matrix with non-negative weights in [0,1]
    Returns a list of persistence values (birth-death) for 0-dim features.
    - We treat vertex birth = 1.0, edges instanced in descending order (reverse filtration).
    - For each union (when an edge connects two previously separate components)
      we record a persistence = 1.0 - edge_weight.
    - After processing all edges, remaining unpaired components die at unpairedData=0.0,
      so each contributes persistence = 1.0.
    This yields exactly N persistence values (one per vertex) as Aleph would (with the options above).
    """
    if M_norm.size == 0:
        return []

    # ensure symmetry (we build bipartite matrices symmetric anyway)
    # consider only upper-triangular edges (i < j)
    n = M_norm.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = float(M_norm[i, j])
            # ignore zero-weight edges (they never connect before unpairedData)
            if w > 0.0:
                edges.append((w, i, j))
    # sort edges by descending weight (reverseFiltration=True)
    edges.sort(key=lambda x: -x[0])

    uf = UnionFind(n)
    persistences = []

    # perform unions; each successful union corresponds to a death at edge weight w
    for w, i, j in edges:
        merged = uf.union(i, j)
        if merged:
            # birth = 1.0, death = w -> persistence = 1 - w
            pers = 1.0 - w
            persistences.append(pers)

    # after all edges, count remaining components
    roots = uf.components()
    c = len(roots)
    # Aleph's unpairedData=0.0 means unpaired intervals die at 0 => persistence = 1 - 0 = 1.0
    # There will be exactly c unpaired features. (Total entries = (N - merges) + merges = N.)
    persistences.extend([1.0] * c)

    # persistences length should be n (vertices)
    # if something odd happened, truncate/pad conservatively
    if len(persistences) > n:
        persistences = persistences[:n]
    elif len(persistences) < n:
        persistences.extend([1.0] * (n - len(persistences)))

    return persistences

# -----------------------
# p-norm of persistence diagram (default p=2)
# -----------------------
def p_norm_of_persistences(persistences, p=2.0):
    if len(persistences) == 0:
        return 0.0

    # Convert all values to floats (handles torch tensors or Parameters)
    arr = []
    for v in persistences:
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().astype(float)
        arr.append(v)
    arr = np.asarray(arr, dtype=float)

    return float((np.sum(np.abs(arr) ** p)) ** (1.0 / p))

# -----------------------
# top-level: compute neural persistence from adjacency matrix (M)
# mirrors earlier code: normalize by global max weight W then use Aleph replacement
# -----------------------
def compute_neural_persistence_from_adjacency(M: np.ndarray, p=2.0):
    """
    M: square adjacency matrix (nonnegative); if not normalized, we'll normalize by its max value.
    Returns dict with total_persistence (p-norm) and normalized version tp / sqrt(n-1)
    """
    if M.size == 0:
        return {"total_persistence": 0.0, "total_persistence_normalized": 0.0}
    W = np.max(M)
    if W == 0:
        M_norm = M.copy()
    else:
        M_norm = M / float(W)

    persistences = zero_dim_persistences_from_matrix(M_norm)
    tp = p_norm_of_persistences(persistences, p=p)
    n_vertices = M.shape[0]
    tp_norm = tp / math.sqrt(max(1, n_vertices - 1))
    return {"total_persistence": float(tp), "total_persistence_normalized": float(tp_norm)}

# -----------------------
# convenience wrappers for torch.nn.Linear
# -----------------------
def neural_persistence_of_linear_layer(linear: nn.Linear, p=2.0):
    W = linear.weight.detach().cpu().numpy()  # (out, in)
    M = linear_to_bipartite_adjacency(W)
    return compute_neural_persistence_from_adjacency(M, p=p)

def neural_persistence_of_module_linears(module: nn.Module, p=2.0):
    results = {}
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            results[name or "<linear>"] = neural_persistence_of_linear_layer(mod, p=p)
    return results
