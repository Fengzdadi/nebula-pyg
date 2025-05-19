import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from nebula_pyg import NebulaConnection, NebulaFeatureStore, NebulaGraphStore

# 1) 连接 NebulaGraph
NebulaConnection.init(
    hosts=[("127.0.0.1", 9669)],
    user="root", password="nebula",
    space="ogbn_products"
)
feat_store = NebulaFeatureStore(NebulaConnection)
graph_store = NebulaGraphStore(NebulaConnection)

# 2) 构造 PyG Data
data = graph_store.as_data(
    num_nodes=100_000,  # mock
    node_attrs={"x": ("product", None, feat_store)},
    edge_attrs={("product", "buys", "product"): graph_store}
)

train_loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],
    batch_size=1024,
    input_nodes=("product", torch.arange(50_000))   # mock
)

# 3) 模型
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GNN(128, 256, 47).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4) 训练循环
for epoch in range(3):
    for batch in train_loader:
        batch = batch.to("cuda")
        out = model(batch.x, batch.edge_index)
        loss = torch.nn.functional.cross_entropy(out, batch.y)
        loss.backward(); optim.step(); optim.zero_grad()
    print(f"Epoch {epoch}: {loss.item():.4f}")
