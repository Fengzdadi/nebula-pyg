# nebula-pyg
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![python-version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12-blue)](https://www.python.org/)

nebula-pyg is a Python library that connects NebulaGraph and PyTorch Geometric (PyG). It aims to simplify the process of reading and processing graph data from NebulaGraph for graph neural network (GNN) tasks. By encapsulating the underlying graph storage access and data conversion, it helps users seamlessly access node and edge information from distributed graph databases, eliminating tedious data preprocessing and making GNN training and inference more efficient and convenient.

## Features
- 🚀 Directly access NebulaGraph distributed storage data
- 🔄 Automatically handles mapping VIDs to continuous indices
- 📊 Supports large-scale heterogeneous graph data
- ⚙️ Conveniently integrates PyG for GNN training and inference

## Installation

### Install directly from the GitHub repository
```bash
pip install git+https://github.com/Fengzdadi/nebula-pyg.git
```

## Quick Start
A quick example showing how to connect to NebulaGraph, load graph data, and perform sampling using PyG's NeighborLoader:
```python
from nebula_pyg.nebula_pyg import NebulaPyG
from nebula3.gclient.net import ConnectionPool
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from torch_geometric.loader import NeighborLoader
import pickle

SPACE = 'basketballplayer'
USER = 'root'
PASSWORD = 'nebula'
SNAPSHOT_PATH = 'snapshot.pkl'

# Connecting to NebulaGraph
config = Config()
pool = ConnectionPool()
pool.init([("host.docker.internal", 9669)], config)
gclient = pool.get_session(USER, PASSWORD)

meta_cache = MetaCache([("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000)
sclient = GraphStorageClient(meta_cache)

# Generate a snapshot mapping and save it
snapshot = NebulaPyG.create_snapshot(gclient, sclient, SPACE)
with open(SNAPSHOT_PATH, 'wb') as f:
    pickle.dump(snapshot, f)

# Loading a snapshot
with open(SNAPSHOT_PATH, 'rb') as f:
    snapshot = pickle.load(f)

# Initialize nebula-pyg and get PyG data
nebula_pyg = NebulaPyG(gclient, sclient, SPACE, snapshot)
feature_store, graph_store = nebula_pyg.get_torch_geometric_remote_backend()

# Batch Sampling with NeighborLoader
input_nodes = list(range(len(snapshot['idx_to_vid']['player'])))
loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={('player', 'follow', 'player'): [10, 10],
                   ('player', 'serve', 'team'): [10, 10]},
    batch_size=32,
    input_nodes=('player', input_nodes),
)

for batch in loader:
    print(batch)

gclient.release()

```

## Usage
For more usage examples and detailed instructions, see [examples/get_started.ipynb](examples/get_started.ipynb).

