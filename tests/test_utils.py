import pickle
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nebula_pyg.utils.utils import get_edge_type_groups, build_edge_index_dict


def test_edge_type_group_and_index_dict(gclient, sclient, space, snapshot):
    groups = get_edge_type_groups(space, gclient, sclient, snapshot)
    print("All (src_tag, edge_type, dst_tag) groups that actually exist:")
    for g in sorted(groups):
        print(g)
    print(f"A total of {len(groups)} heterogeneous edge groups! \n")

    edge_index_dict = build_edge_index_dict(gclient, sclient, space, snapshot)
    print("PyG edge_index_dict 的 keys：")
    for k in sorted(edge_index_dict):
        print(f"{k}  -> shape: {edge_index_dict[k].shape}")
        assert k in groups, f"Inconsistent grouping: {k}"
        assert edge_index_dict[k].shape[0] == 2
    print(f"There are a total of {len(edge_index_dict)} edge_index groups!\n")

    # Spot check: reverse index
    for k, edge_index in edge_index_dict.items():
        src_tag, _, dst_tag = k
        idx_to_vid_src = snapshot["idx_to_vid"][src_tag]
        idx_to_vid_dst = snapshot["idx_to_vid"][dst_tag]
        src_idx, dst_idx = edge_index[:, 0].tolist()
        src_vid = idx_to_vid_src[src_idx]
        dst_vid = idx_to_vid_dst[dst_idx]
        assert snapshot["vid_to_tag"][src_vid] == src_tag
        assert snapshot["vid_to_tag"][dst_vid] == dst_tag
        print(f"Sample edge ({src_tag},{dst_tag}): {src_vid} -> {dst_vid}")
        break
    print("All edge_index_dict construction and shape checks pass")


def main():
    space = "basketballplayer"
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)

    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")
    meta_cache = MetaCache(
        [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000
    )
    sclient = GraphStorageClient(meta_cache)

    test_edge_type_group_and_index_dict(space, gclient, sclient, snapshot)

    gclient.release()
