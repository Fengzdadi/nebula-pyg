import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

# monkey patch torch.load for OGB compatibility
# The problem is the conflict between torch 2.6 and OGB
# This part can be deleted after ogb solves this problem
# ——————————
real_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    return real_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from numpy.core.multiarray import _reconstruct

torch.serialization.add_safe_globals([_reconstruct, DataEdgeAttr, DataTensorAttr, GlobalStorage])
# ——————————

from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from tqdm import tqdm
import numpy as np

dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0]

NEBULA_HOST = 'host.docker.internal'
NEBULA_PORT = 9669
NEBULA_USER = 'root'
NEBULA_PASS = 'nebula'
SPACE_NAME = 'arxiv'

def get_feat_field_names():
    return ', '.join([f'feat{i}' for i in range(data.x.shape[1])])  # Automatically adapt feature dimension

def get_feat_values(feat):
    return ', '.join([str(float(x)) for x in feat])

if __name__ == '__main__':
    node_feats = data.x.numpy()        # [num_nodes, 100]
    edge_index = data.edge_index.numpy()  # [2, num_edges]
    labels = data.y.numpy().squeeze()  # [num_nodes]

    # Conn to nebula
    config = Config()
    pool = ConnectionPool()
    pool.init([(NEBULA_HOST, NEBULA_PORT)], config)
    client = pool.get_session(NEBULA_USER, NEBULA_PASS)
    client.execute(f'USE {SPACE_NAME};')

    # Batch insert nodes
    print("Inserting nodes...")
    batch_size = 1000
    num_nodes = node_feats.shape[0]
    feat_fields = get_feat_field_names()
    for i in tqdm(range(0, num_nodes, batch_size)):
        stmts = []
        for j in range(i, min(i + batch_size, num_nodes)):
            # String VID, user can choose int or fix_string(),
            # the recommended vid method can be found at 
            # https://docs.nebula-graph.com.cn/3.8.0/1.introduction/3.vid/
            vid = str(j)
            feats = get_feat_values(node_feats[j])
            label = int(labels[j])
            value = f"({feats}, {label})"
            stmts.append(f'"{vid}": {value}')
        ngql = f"INSERT VERTEX Paper({feat_fields}, label) VALUES " + ', '.join(stmts) + ';'

        try:
            resp = client.execute(ngql)
            if not resp.is_succeeded():
                print(f"[Vertex Fail] Batch {i} failed: {resp.error_msg()}")
        except Exception as e:
            print(f"[Vertex Exception] Batch {i} crashed: {e}")
            print("Failing ngql (truncated):", ngql[:300])
            break

    # Batch insert edges
    print("Inserting edges...")
    num_edges = edge_index.shape[1]
    for i in tqdm(range(0, num_edges, batch_size)):
        stmts = []
        for j in range(i, min(i + batch_size, num_edges)):
            src = str(edge_index[0, j])
            dst = str(edge_index[1, j])
            stmts.append(f'"{src}"->"{dst}": ()')
        ngql = f"INSERT EDGE Cites() VALUES " + ', '.join(stmts) + ';'

        try:
            resp = client.execute(ngql)
            if not resp.is_succeeded():
                print(f"[Edge Fail] Batch {i} failed: {resp.error_msg()}")
        except Exception as e:
            print(f"[Edge Exception] Batch {i} crashed: {e}")
            print("Failing ngql (truncated):", ngql[:300])
            break

    print("All done!")
    pool.close()
