import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pickle
import torch
from nebula_pyg.nebula_pyg import NebulaPyG

from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from torch_geometric.data import TensorAttr, EdgeAttr

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

# Update only when initialized or needed
snapshot = NebulaPyG.create_snapshot(gclient, sclient, SPACE)
with open(SNAPSHOT_PATH, "wb") as f:
    pickle.dump(snapshot, f)

with open(SNAPSHOT_PATH, "rb") as f:
    snapshot = pickle.load(f)

# If snapshot is not passed in, NebulaPyG will automatically generate one based on Spacename.
nebula_pyg = NebulaPyG(gclient, sclient, SPACE, snapshot)
feature_store, graph_store = nebula_pyg.get_torch_geometric_remote_backend()

from torch_geometric.loader import NeighborLoader
import multiprocessing

input_nodes = list(range(len(snapshot['idx_to_vid']['player'])))

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={
        ('player', 'follow', 'player'): [10, 10],
        ('player', 'serve', 'team'): [10, 10],
    },
    batch_size=32,
    input_nodes=('player', input_nodes),
    num_workers=0,
    filter_per_worker=True,
)

for batch in loader:
    # batch['player'].x, batch['player', 'follow', 'player'].edge_index ...
    print("batch:", batch)
    # 可以直接放进你的 GNN forward 函数

gclient.release()
