import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import pytest
import torch

from nebula_pyg.feature_store import NebulaFeatureStore
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from torch_geometric.data import TensorAttr

# NebulaGraph Connection
SPACE = 'basketballplayer'
USER = 'root'
PASSWORD = 'nebula'
SNAPSHOT_PATH = 'snapshot_vid_to_idx.pkl'

# Change to your actual address
NEBULA_HOSTS = [("host.docker.internal", 9669)]
META_HOSTS = [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)]

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

@pytest.fixture(scope="module")
def feature_store():
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)
    fs = NebulaFeatureStore(make_pool(), make_sclient(), "basketballplayer", SPACE, snapshot=snapshot, username=USER, password=PASSWORD, expose="x")
    yield fs

def test_get_tensor(feature_store):
    attr = TensorAttr(group_name="player", attr_name="age")
    x = feature_store.get_tensor(attr)
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] > 0

def test_get_tensor_index(feature_store):
    attr = TensorAttr(group_name="player", attr_name="age")
    idxs = [0, 1, 2, 3]
    x = feature_store.get_tensor(attr, index=idxs)
    assert x.shape[0] == len(idxs)

def test_get_tensor_size(feature_store):
    attr = TensorAttr(group_name="player", attr_name="age")
    size = feature_store.get_tensor_size(attr)
    assert isinstance(size, tuple)
    assert size[0] > 0 and size[1] > 0

def test_get_all_tensor_attrs(feature_store):
    attrs = feature_store.get_all_tensor_attrs()
    assert all(isinstance(attr, TensorAttr) for attr in attrs)
