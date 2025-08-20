# nebula-pyg
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![python-version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12-blue)](https://www.python.org/)

nebula-pyg is a Python library that connects NebulaGraph and PyTorch Geometric (PyG). It aims to simplify the process of reading and processing graph data from NebulaGraph for graph neural network (GNN) tasks. By encapsulating the underlying graph storage access and data conversion, it helps users seamlessly access node and edge information from distributed graph databases, eliminating tedious data preprocessing and making GNN training and inference more efficient and convenient.

## Features
- 🚀 **Optimized read-only access to NebulaGraph** — Designed for read-heavy GNN workloads. Metadata snapshots and queries go through `graphd`, while full-graph scans directly hit `storaged` for maximum throughput, eliminating the need for external export/import steps.
- 🔄 **Automatic VID ↔ continuous index mapping** — Transparent mapping layer for PyG-compatible integer indices.
- 📊 **Large-scale heterogeneous graph support** — Handles multiple vertex/edge types efficiently for industrial-scale graphs.
- ⚙️ **Seamless PyG integration** — Out-of-the-box `FeatureStore` and `GraphStore` implementations for training/inference with GNNs.
- 🧵 **Multi-process DataLoader safety** — Connection factories + lazy initialization to avoid socket FD conflicts when using PyTorch `DataLoader` with `num_workers>0`.

## Installation

### Install directly from the GitHub repository
```bash
pip install git+https://github.com/Fengzdadi/nebula-pyg.git
```

## Quick Start
A quick example showing how to connect to NebulaGraph, load graph data, and perform sampling using PyG's NeighborLoader:
```python
import os
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

NEBULA_HOSTS = [("host.docker.internal", 9669)]
# or
# NEBULA_HOSTS = [("graphd", 9669)]
META_HOSTS = [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)]

# Connecting to NebulaGraph
def make_pool():
    cfg = Config()
    pool = ConnectionPool()
    ok = pool.init(NEBULA_HOSTS, cfg)
    assert ok, "Init ConnectionPool failed"
    return pool

def make_sclient():
    meta_cache = MetaCache(META_HOSTS, 50000)
    sclient = GraphStorageClient(meta_cache=meta_cache)
    return sclient

# Generate a snapshot mapping and save it
if not os.path.exists(SNAPSHOT_PATH):
    snapshot = NebulaPyG.create_snapshot(make_pool(), make_sclient(), SPACE, username=USER, password=PASSWORD)
    with open(SNAPSHOT_PATH, "wb") as f:
        pickle.dump(snapshot, f)
else:
    with open(SNAPSHOT_PATH, "rb") as f:
        snapshot = pickle.load(f)

# Initialize nebula-pyg and get PyG data
nebula_pyg = NebulaPyG(make_pool, make_sclient, SPACE, USER, PASSWORD, snapshot)
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

```

## Usage
For more usage examples and detailed instructions, see [examples/get_started.ipynb](examples/get_started.ipynb).

For specific data import and data training examples, see [OBGN.py](OGBN.py) and [OBGN_train.py](OGBN_train.py).

## Acknowledgements

This project originated from the **NebulaGraph PyG Integration** task under [OSPP 2025 (Open Source Promotion Plan)](https://summer-ospp.ac.cn/), with strong support from the **NebulaGraph community**.  

Special thanks to **[wey-gu](https://github.com/wey-gu)**, my Project Mentor, for his invaluable guidance and support throughout the development process.

We also appreciate the **[KUZU](https://blog.kuzudb.com/post/kuzu-pyg-remote-backend/)** team for providing an excellent reference implementation for PyG remote backend design, which inspired parts of this project’s architecture.  

Although the initial implementation was completed during OSPP, this project will continue to be actively maintained and improved.

---
## TODO
- [ ] Improve documentation with more usage examples
  - [x] ~~OGBL~~ Cora for Link Property Prediction
  - [ ] OGBG for Graph Property Prediction
- [ ] Directly provide factory functions without users having to generate them manually
- [x] Provides general snapshots for users to understand the data processing structure
- [x] Currently all vids are based on the fixstring type. Is it necessary to add the int type? In my opinion, users only need to use the fixstring when importing.