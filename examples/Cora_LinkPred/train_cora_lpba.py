# examples/LinkPred/train_cora_lpba.py
import os
import sys
import pickle
from typing import Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.data import EdgeAttr, EdgeLayout

from nebula_pyg.nebula_pyg import NebulaPyG

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache


# ================== Configuration ==================
SPACE = "cora"
ETYPE = ("Paper", "Cites", "Paper")

USER = "root"
PASSWORD = "nebula"

GRAPH_HOSTS = [("graphd", 9669)]

META_HOSTS = [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)]

SNAPSHOT_PATH = "snapshot_vid_to_idx_cora.pkl"

# The default mode is x, which means that users do not need to splice feat themselves,
# and the data read from nebula directly meets the requirements of data.x
EXPOSE = "x"


# ================== Factory Function ==================
def make_pool() -> ConnectionPool:
    cfg = Config()
    cfg.max_retry_connect = 1
    pool = ConnectionPool()
    assert pool.init(GRAPH_HOSTS, cfg), "Init ConnectionPool failed"
    return pool

def make_sclient() -> GraphStorageClient:
    meta_cache = MetaCache(META_HOSTS, 50000)
    return GraphStorageClient(meta_cache=meta_cache)


# ================== model ==================
class DotProductLP(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)

    def forward(self, n_id: torch.Tensor, edge_label_index: torch.Tensor):
        z = self.emb(n_id)  # [N_batch_nodes, D]
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)  # [B_edges] logits


# ================== Tool ==================
def split_undirected_pairs(full_pos: torch.Tensor,
                           train_ratio=0.85, val_ratio=0.05
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove duplicate bidirectional edges into undirected pairs, split the data proportionally, and then generate bidirectional edges for each set.
    This prevents information leakage caused by (u,v) and (v,u) falling into different sets.
    """
    assert full_pos.dim() == 2 and full_pos.size(0) == 2
    row, col = full_pos[0].tolist(), full_pos[1].tolist()

    undirected = set()
    for u, v in zip(row, col):
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        undirected.add((a, b))

    pairs = torch.tensor(list(undirected), dtype=torch.long)  # [M, 2]
    M = pairs.size(0)
    perm = torch.randperm(M)
    tr = int(M * train_ratio)
    vr = int(M * val_ratio)
    tr_idx = perm[:tr]
    va_idx = perm[tr:tr + vr]
    te_idx = perm[tr + vr:]

    def both_dir(sub_pairs: torch.Tensor) -> torch.Tensor:
        # sub_pairs: [m, 2]  ->  [2, 2m]
        if sub_pairs.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long)
        u = sub_pairs[:, 0]
        v = sub_pairs[:, 1]
        bi = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
        return bi

    train_pos = both_dir(pairs[tr_idx])
    val_pos   = both_dir(pairs[va_idx])
    test_pos  = both_dir(pairs[te_idx])
    return train_pos, val_pos, test_pos


def make_loader(data_backend,
                pos_edges: torch.Tensor,
                batch_size=2048,
                num_neighbors: List[int] = [10, 10],
                etype=ETYPE,
                num_nodes_dict=None):
    neg_cfg = NegativeSampling(mode="binary", amount=1)
    E = pos_edges.size(1)

    loader = LinkNeighborLoader(
        data=data_backend,                   # (feature_store, graph_store)
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=(etype, pos_edges),
        edge_label=torch.ones(E),
        neg_sampling=neg_cfg,
        shuffle=True,

        # Single process first, so we can inject num_nodes after construction
        num_workers=0,
        persistent_workers=False,
        filter_per_worker=True,
        directed=True,
    )

    # Originally intended to solve the problem of training DDI, it now seems unnecessary
    
    # if num_nodes_dict is not None:
    #     loader.num_nodes = num_nodes_dict
    #     if hasattr(loader, "link_sampler"):
    #         loader.link_sampler.num_nodes = num_nodes_dict

    return loader


def eval_auc(model, device, loader, ntype="Paper", etype=ETYPE):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            # Heterogeneous: node ids are in batch[ntype].n_id, edge labels are in batch[etype]
            n_id = batch[ntype].n_id.to(device)
            edge_label_index = batch[etype].edge_label_index.to(device)
            y = batch[etype].edge_label.float().to(device)

            logits = model(n_id, edge_label_index)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu())
            ps.append(prob.cpu())

    if len(ys) == 0:
        return float("nan")
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    try:
        return roc_auc_score(y, p)
    except Exception:
        return float("nan")


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nebula_pyg = NebulaPyG(make_pool, make_sclient, SPACE, USER, PASSWORD, EXPOSE)

    if not os.path.exists(SNAPSHOT_PATH):
        snapshot = nebula_pyg.create_snapshot(make_pool, make_sclient, SPACE)
        with open(SNAPSHOT_PATH, "wb") as f:
            pickle.dump(snapshot, f)
    else:
        with open(SNAPSHOT_PATH, "rb") as f:
            snapshot = pickle.load(f)

    feature_store, graph_store = nebula_pyg.get_torch_geometric_remote_backend()

    # Read the entire graph's positive edges from Nebula (this example uses Cites full edges)
    full_pos = graph_store.get_edge_index(
        EdgeAttr(edge_type=ETYPE, layout=EdgeLayout.COO)
    ).to(torch.long)  # [2, E]

    # Split train/val/test (undirected deduplication and then restore bidirection, avoid leakage)
    train_pos, val_pos, test_pos = split_undirected_pairs(full_pos)

    num_nodes = len(snapshot["vid_to_idx"]["Paper"])
    num_nodes_dict = {"Paper": num_nodes}

    # Construct loader (heterogeneous + remote)
    train_loader = make_loader(
        data_backend=(feature_store, graph_store),
        pos_edges=train_pos,
        batch_size=4096,
        num_nodes_dict=num_nodes_dict,
    )
    val_loader = make_loader(
        data_backend=(feature_store, graph_store),
        pos_edges=val_pos,
        batch_size=8192,
        num_nodes_dict=num_nodes_dict,
    )
    test_loader = make_loader(
        data_backend=(feature_store, graph_store),
        pos_edges=test_pos,
        batch_size=8192,
        num_nodes_dict=num_nodes_dict,
    )

    # print("train_loader.num_nodes =", getattr(train_loader, "num_nodes", None))

    model = DotProductLP(num_nodes=num_nodes, emb_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, 6):
        model.train()
        total_loss, total_cnt = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", ncols=90):
            n_id = batch["Paper"].n_id.to(device)
            edge_label_index = batch[ETYPE].edge_label_index.to(device)
            y = batch[ETYPE].edge_label.float().to(device)

            opt.zero_grad()
            logits = model(n_id, edge_label_index)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.numel()
            total_cnt += y.numel()

        train_bce = total_loss / max(total_cnt, 1)
        val_auc = eval_auc(model, device, val_loader)
        print(f"Epoch {epoch}: train_bce={train_bce:.4f}  val_auc={val_auc:.4f}")

    test_auc = eval_auc(model, device, test_loader)
    print(f"[Test] AUC = {test_auc:.4f}")


if __name__ == "__main__":
    main()
