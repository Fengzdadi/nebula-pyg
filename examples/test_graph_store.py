import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pickle
from nebula_pyg.graph_store import NebulaGraphStore
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.common.ttypes import HostAddr

from torch_geometric.data import EdgeAttr, EdgeLayout

@pytest.fixture(scope="module")
def graph_store():
    # TODO: Put the parameters in environment variables, pytest.ini or conftest.py to facilitate unified switching
    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")
    meta_cache = MetaCache([("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000)
    sclient = GraphStorageClient(meta_cache)
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)
    gs = NebulaGraphStore(gclient, sclient, "basketballplayer", snapshot)
    yield gs
    gclient.release()

def test_get_edge_index(graph_store):
    # "follow" Edge
    edge_attr = EdgeAttr(edge_type=('player', 'follow', 'player'), layout=EdgeLayout.COO)
    edge_index = graph_store.get_edge_index(edge_attr)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    # "serve" Edge
    edge_attr2 = EdgeAttr(edge_type=('player', 'serve', 'team'), layout=EdgeLayout.COO)
    edge_index2 = graph_store.get_edge_index(edge_attr2)
    assert edge_index2.shape[0] == 2
    assert edge_index2.shape[1] > 0

    # Assert index range
    N_player = len(graph_store.vid_to_idx["player"])
    N_team = len(graph_store.vid_to_idx["team"])
    assert edge_index.max() < N_player
    assert edge_index2[0].max() < N_player and edge_index2[1].max() < N_team

def test_get_all_edge_attrs(graph_store):
    edge_attrs = graph_store.get_all_edge_attrs()
    assert isinstance(edge_attrs, list)
    assert all(hasattr(attr, "edge_type") for attr in edge_attrs)
    assert len(edge_attrs) > 0

