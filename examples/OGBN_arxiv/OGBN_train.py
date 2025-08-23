import os
import pickle
import torch
from nebula_pyg.nebula_pyg import NebulaPyG
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from tqdm import tqdm

# NebulaGraph Connection
SPACE = "arxiv"
USER = "root"
PASSWORD = "nebula"
SNAPSHOT_PATH = "../../snapshot_vid_to_idx_arxiv.pkl"
EXPOSE = "x"

# Change to your actual address
NEBULA_HOSTS = [("host.docker.internal", 9669)]
META_HOSTS = [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)]

# config = Config()
# connection_pool = ConnectionPool()
# connection_pool.init(NEBULA_HOSTS, config)
# gclient = connection_pool.get_session(USER, PASSWORD)
#
# meta_cache = MetaCache(META_HOSTS, 50000)
# sclient = GraphStorageClient(meta_cache)


# Factory function (temporary)
def make_pool():
    cfg = Config()
    pool = ConnectionPool()
    ok = pool.init([("graphd", 9669)], cfg)
    assert ok, "Init ConnectionPool failed"
    return pool


def make_sclient():
    meta_cache = MetaCache(META_HOSTS, 50000)
    sclient = GraphStorageClient(meta_cache=meta_cache)
    return sclient


# Create/load snapshot (VID mapping)
if not os.path.exists(SNAPSHOT_PATH):
    snapshot = NebulaPyG.create_snapshot(
        make_pool(), make_sclient(), SPACE, username=USER, password=PASSWORD
    )
    with open(SNAPSHOT_PATH, "wb") as f:
        pickle.dump(snapshot, f)
else:
    with open(SNAPSHOT_PATH, "rb") as f:
        snapshot = pickle.load(f)

# Initialize NebulaPyG backend
nebula_pyg = NebulaPyG(
    make_pool, make_sclient, SPACE, USER, PASSWORD, EXPOSE, snapshot
)  # Pay attention to the order of passing parameters
feature_store, graph_store = nebula_pyg.get_torch_geometric_remote_backend()

from torch_geometric.loader import NeighborLoader

num_nodes = 169343
input_nodes = ("Paper", list(range(num_nodes)))

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors=[10, 10],  # Two-hop sampling of 10 neighbors
    batch_size=32,
    input_nodes=input_nodes,
    num_workers=4,
    filter_per_worker=True,
)

from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 128  # ogbn-arxiv prop dimension
hidden_channels = 128  # hidden layer dimension
out_channels = 40  # ogbn-arxiv has 40 categories

model = GNN(in_channels, hidden_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(1, 6):
    model.train()
    total_loss = 0.0
    total_seeds = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}", ncols=80):
        x = batch["Paper"].x.to(device)
        edge_index = batch["Paper", "Cites", "Paper"].edge_index.to(device)
        y = batch["Paper"].y.squeeze().long().to(device)

        seed_size = batch["Paper"].batch_size
        optimizer.zero_grad()
        out = model(x, edge_index)

        loss = F.cross_entropy(
            out[:seed_size], y[:seed_size].view(-1), reduction="mean"
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seed_size
        total_seeds += seed_size

    epoch_avg_loss = total_loss / max(total_seeds, 1)
    print(f"Epoch {epoch}: avg loss = {epoch_avg_loss:.4f}")

print("Training completed!")
