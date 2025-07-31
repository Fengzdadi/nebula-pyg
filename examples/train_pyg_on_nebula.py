import pickle
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch_geometric.data import TensorAttr

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 假定你已经有这两个类
from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

# ============ 1. 初始化 Nebula 连接与 snapshot ============
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


def fetch_all_features(fs):
    feature_dict = {}
    # player 用 '31'，team 无需特征
    try:
        player_feat = fs.get_tensor(TensorAttr('player', 'age'))
        player_feat = player_feat.view(-1, 1) if player_feat.dim() == 1 else player_feat
        feature_dict['player'] = player_feat
        print(f"[Info] 选用 player.age 作为特征, shape={player_feat.shape}")
    except Exception as e:
        print(f"[Warn] player 特征拉取失败: {e}")

    # team 没有数值特征，可跳过
    return feature_dict


def fetch_all_edge_indices(gs):
    """返回 {(src_tag, edge_name, dst_tag): edge_index_tensor}"""
    edge_index_dict = {}
    # 获取所有边类型
    edge_attrs = gs.get_all_edge_attrs()  # List[EdgeAttr]
    for edge_attr in edge_attrs:
        # edge_attr.edge_type 必然是 (src_tag, edge_name, dst_tag)
        src_tag, edge_name, dst_tag = edge_attr.edge_type
        edge_index = gs.get_edge_index(edge_attr)
        edge_index_dict[(src_tag, edge_name, dst_tag)] = edge_index
    return edge_index_dict



# 参数
SPACE = 'basketballplayer'
USER = 'root'
PASSWORD = 'nebula'
SNAPSHOT_PATH = 'snapshot_vid_to_idx.pkl'
tag = "player"
prop = "age"  

# Nebula 连接（按你的项目实际调整）
config = Config()
connection_pool = ConnectionPool()
connection_pool.init([("host.docker.internal", 9669)], config)
gclient = connection_pool.get_session(USER, PASSWORD)

meta_cache = MetaCache([("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000)
sclient = GraphStorageClient(meta_cache)

with open(SNAPSHOT_PATH, "rb") as f:
    snapshot = pickle.load(f)

# ============ 2. 初始化 FeatureStore & GraphStore ============
fs = NebulaFeatureStore(gclient, sclient, SPACE, snapshot)
gs = NebulaGraphStore(gclient, sclient, SPACE, snapshot)

# ============ 3. 拉取全量特征和边结构 ============
# 注意：这里你需要实现 FeatureStore/GraphStore 的 get_all_features/get_all_edge_index 方法
attr = TensorAttr(group_name=tag, attr_name=prop)
feature_dict = fetch_all_features(fs)      # {ntype: torch.Tensor}
edge_index_dict = fetch_all_edge_indices(gs) # {(src, rel, dst): torch.LongTensor}

# 假如有 label，也可以这样挂（你可以自定义获取方式）
label_dict = fs.get_all_tensor_attrs() if hasattr(fs, 'get_all_tensor_attrs') else {}

# ============ 4. 组装 HeteroData ============
data = HeteroData(drop_last=True)
for ntype, feats in feature_dict.items():
    data[ntype].x = feats
for (src, rel, dst), edge_index in edge_index_dict.items():
    data[(src, rel, dst)].edge_index = edge_index

print("edge_index_dict:", edge_index_dict)
print('data:', data)

# ============ 5. 简单 GNN 训练（以 player 类型为例） =============

# 选用一种边关系和节点类型
ntype = 'player'
rel = 'follow'
print("data.edge_types:", data.edge_types)
if (ntype, rel, ntype) not in data.edge_types:
    raise ValueError(f'Edge ({ntype}, {rel}, {ntype}) 不存在！请检查实际边类型。')

x = data[ntype].x.float()
y = data[ntype].y if 'y' in data[ntype] else torch.randint(0, 2, (x.size(0),)) # 若无label临时生成
edge_index = data[(ntype, rel, ntype)].edge_index.long()
edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)  # 只有一种关系时
print("edge_type unique:", edge_type.unique())
print("x.shape:", x.shape)
print("edge_index.shape:", edge_index.shape)
print("edge_index.max():", edge_index.max().item(), "x.size(0):", x.size(0))
print("labels.size():", y.size())

# ============ Pass ============
# print("==== RGCN 最小测试 ====")
# _x = torch.randn(51, 1)
# _edge_index = torch.randint(0, 51, (2, 81))
# _edge_type = torch.zeros(81, dtype=torch.long)
# _conv = RGCNConv(1, 32, num_relations=1)
# try:
#     _out = _conv(_x, _edge_index, _edge_type)
#     print("最小单元测试通过，out.shape:", _out.shape)
# except Exception as e:
#     print("[ERROR] RGCN 单元测试失败：", e)

#定义模型
class SimpleRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

model = SimpleRGCN(in_channels=x.size(1), hidden_channels=16, out_channels=2, num_relations=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_type)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# from torch_geometric.nn import GCNConv

# class SimpleGCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# model = SimpleGCN(in_channels=x.size(1), hidden_channels=32, out_channels=2)
# out = model(x, edge_index)


# 推理
model.eval()
with torch.no_grad():
    # RGCN
    pred = model(x, edge_index, edge_type).argmax(dim=1)
    # GCN
    # pred = model(x, edge_index).argmax(dim=1)
    acc = (pred == y).float().mean()
    print(f"Final Accuracy: {acc:.4f}")

print("GNN pipeline 完整跑通！")
