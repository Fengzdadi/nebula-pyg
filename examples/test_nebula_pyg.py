import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
from nebula_pyg.nebula_pyg import NebulaPyg  # 你的 NebulaGraph 类所在路径

from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from torch_geometric.data import TensorAttr

# ====== init snapshot ======
SPACE = 'basketballplayer'
USER = 'root'
PASSWORD = 'nebula'
SNAPSHOT_PATH = 'snapshot_vid_to_idx.pkl'

config = Config()
connection_pool = ConnectionPool()
connection_pool.init([("host.docker.internal", 9669)], config)
gclient = connection_pool.get_session(USER, PASSWORD)

meta_cache = MetaCache([("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000)
sclient = GraphStorageClient(meta_cache)

with open(SNAPSHOT_PATH, "rb") as f:
    snapshot = pickle.load(f)

# ====== get remote backend ======
db = NebulaPyg(gclient, sclient, SPACE, snapshot)
feature_store, graph_store = db.get_torch_geometric_remote_backend()

from torch_geometric.loader import NeighborLoader
import multiprocessing

input_nodes = list(range(len(snapshot['idx_to_vid']['player'])))
input_nodes = [0]

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={
        ('player', 'follow', 'player'): [10, 10],
        # ('player', 'serve', 'team'): [10, 10],  # 新加
    },
    batch_size=32,
    input_nodes=('player', input_nodes),
    num_workers=0,
    filter_per_worker=True,
)

print("idx_to_vid['player']:", snapshot['idx_to_vid']['player'])
print("vid_to_idx['player']:", snapshot['vid_to_idx']['player'])


# print("len(snapshot['idx_to_vid']['player']) =", len(snapshot['idx_to_vid']['player']))
# print("feature_store.get_tensor(TensorAttr('player', 'age')).shape =", feature_store.get_tensor(TensorAttr('player', 'age')).shape)


# 确保 graph_store 里有 get_all_edge_attrs 和 get_edge_index 方法
# edge_attrs = graph_store.get_all_edge_attrs()
# for edge_attr in edge_attrs:
#     etype = edge_attr.edge_type  # ('player', 'follow', 'player') ...
#     edge_index = graph_store.get_edge_index(edge_attr)
#     print(f"edge_type: {etype}, edge_index.shape: {edge_index.shape}")
#     print(f"  max: {edge_index.max().item() if edge_index.numel() else 'NA'}"
#           f", min: {edge_index.min().item() if edge_index.numel() else 'NA'}")
#     # 可加 assert 检查越界
#     if edge_index.numel():
#         assert edge_index.max() < len(input_nodes)
#         assert edge_index.min() >= 0




# for i, batch in enumerate(loader):
#     print(f"[Batch {i}] batch['player'].x.shape:", batch['player'].x.shape)
#     # 或
#     print(batch)
#     if i > 0:
#         break  # 只采样2个batch看看


# 遍历采样批次，输入到 GNN 模型
for batch in loader:
    # batch['player'].x, batch['player', 'follow', 'player'].edge_index ...
    print(batch)
    # 可以直接放进你的 GNN forward 函数

gclient.release()
