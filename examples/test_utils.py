import pickle
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nebula_pyg.utils import get_edge_type_groups, build_edge_index_dict

def test_edge_type_group_and_index_dict(gclient, sclient, space, snapshot):
    groups = get_edge_type_groups(space, sclient, gclient, snapshot)
    print("实际存在的所有 (src_tag, edge_type, dst_tag) 分组：")
    for g in groups:
        print(g)
    assert isinstance(groups, set)
    assert all(len(g) == 3 for g in groups)
    print(f"共计 {len(groups)} 种异构边分组！\n")

    edge_index_dict = build_edge_index_dict(space, sclient, gclient, snapshot)
    print("PyG edge_index_dict 的 keys：")
    for k in edge_index_dict:
        print(f"{k}  -> shape: {edge_index_dict[k].shape}")
        assert k in groups, "分组不一致"
        assert edge_index_dict[k].shape[0] == 2  # [2, num_edges]
    print(f"总共 {len(edge_index_dict)} 个 edge_index 分组！\n")
    print("全部 edge_index_dict 构建和 shape 检查通过！")

    # TODO: spot check：抽一个 key，把 src/dst 下标反查回 vid，和 snapshot 比较

if __name__ == "__main__":
    space = "basketballplayer"
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)

    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")

    meta_cache = MetaCache(
            [("metad0", 9559),("metad1", 9559),("metad2", 9559)], 50000
        )
    sclient = GraphStorageClient(meta_cache)
    test_edge_type_group_and_index_dict(gclient, sclient, space, snapshot)

    gclient.release()
