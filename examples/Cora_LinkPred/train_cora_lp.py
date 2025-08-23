# This file is the training process without nebula
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling


# ============== dot-product LP model ==============
class DotProductLP(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)

    def forward(self, n_id: torch.Tensor, edge_label_index: torch.Tensor):
        z = self.emb(n_id)  # [N_batch_nodes, D]
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)  # [B_edges] logits


def make_edge_splits(edge_index: torch.Tensor, train_ratio=0.85, val_ratio=0.05):
    E = edge_index.size(1)
    perm = torch.randperm(E)
    train_e = int(E * train_ratio)
    val_e = int(E * val_ratio)
    train_pos = edge_index[:, perm[:train_e]]
    val_pos = edge_index[:, perm[train_e : train_e + val_e]]
    test_pos = edge_index[:, perm[train_e + val_e :]]
    return train_pos, val_pos, test_pos


def make_loader(data, pos_edges, batch_size=1024, num_neighbors=[10, 10], neg_amount=1):
    neg_cfg = NegativeSampling(mode="binary", amount=neg_amount)
    E = pos_edges.size(1)
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=pos_edges,
        edge_label=torch.ones(E),
        neg_sampling=neg_cfg,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        filter_per_worker=True,
        directed=True,
    )
    return loader


def eval_auc(model, device, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            n_id = batch.n_id.to(device)
            edge_label_index = batch.edge_label_index.to(device)
            y = batch.edge_label.float().to(device)
            logits = model(n_id, edge_label_index)
            prob = torch.sigmoid(logits)
            ys.append(y.cpu())
            ps.append(prob.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    try:
        return roc_auc_score(y, p)
    except Exception:
        return float("nan")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Cora (Planetoid)
    root = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    data = data.to("cpu")

    print(f"Graph: num_nodes={data.num_nodes}, num_edges={data.edge_index.size(1)}")
    train_pos, val_pos, test_pos = make_edge_splits(data.edge_index)

    # Construct loaders separately (note that etype is not passed here, Tensor is passed directly)
    train_loader = make_loader(data, train_pos, batch_size=2048)
    val_loader = make_loader(data, val_pos, batch_size=4096)
    test_loader = make_loader(data, test_pos, batch_size=4096)

    print("train_loader.num_nodes =", getattr(train_loader, "num_nodes", None))

    # Model and Optimizer
    model = DotProductLP(num_nodes=data.num_nodes, emb_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, 6):
        model.train()
        total_loss, total_count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", ncols=90):
            n_id = batch.n_id.to(device)
            edge_label_index = batch.edge_label_index.to(device)
            y = batch.edge_label.float().to(device)

            opt.zero_grad()
            logits = model(n_id, edge_label_index)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.numel()
            total_count += y.numel()

        train_loss = total_loss / max(total_count, 1)
        val_auc = eval_auc(model, device, val_loader)
        print(f"Epoch {epoch}: train_bce={train_loss:.4f}  val_auc={val_auc:.4f}")

    test_auc = eval_auc(model, device, test_loader)
    print(f"[Test] AUC = {test_auc:.4f}")


if __name__ == "__main__":
    main()
