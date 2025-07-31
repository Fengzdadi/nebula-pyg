import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nebula_pyg.graph_store import NebulaGraphStore
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.common.ttypes import HostAddr
from nebula3.gclient.net import Connection

import pickle

def test_get_edge_index(gs):
    # 测试 "follow" 边，player→player
    edge_index = gs.get_edge_index("follow", src_tag="player", dst_tag="player")
    print("follow edge_index.shape:", edge_index.shape)
    print("follow edge_index (前10条):", edge_index[:, :10])

    # 测试 "serve" 边，player→team
    edge_index2 = gs.get_edge_index("serve", src_tag="player", dst_tag="team")
    print("serve edge_index.shape:", edge_index2.shape)
    print("serve edge_index (前10条):", edge_index2[:, :10])

    # 校验所有 index 都在合理范围内（不越界）
    N_player = len(gs.vid_to_idx["player"])
    N_team = len(gs.vid_to_idx["team"])
    print(f"player count: {N_player}, team count: {N_team}")

    assert edge_index.max() < N_player, "player index越界"
    assert edge_index2[0].max() < N_player and edge_index2[1].max() < N_team, "src/dst越界"

    print("所有边 index 编号都在正确范围内！")

def test_get_all_edge_attrs(gs):
    edge_attrs = gs.get_all_edge_attrs()
    print("EdgeAttrs 列表：")
    for attr in edge_attrs:
        print(attr)

    # 简单断言
    assert isinstance(edge_attrs, list)
    assert all(hasattr(attr, "edge_type") for attr in edge_attrs)
    print(f"总共 {len(edge_attrs)} 个边类型，全部返回成功！")



def main():
    meta_cache = MetaCache(
            [("metad0", 9559),("metad1", 9559),("metad2", 9559)], 50000
        )
    storage_addrs = [HostAddr("host.docker.internal", 45033),
                    HostAddr("host.docker.internal", 46229),
                    HostAddr("host.docker.internal", 44987)
                    ]
    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")
    sclient = GraphStorageClient(meta_cache)

    space = "basketballplayer"

    
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)

    gs = NebulaGraphStore(gclient, sclient, space, snapshot)
    # test_get_edge_index(gs)
    test_get_all_edge_attrs(gs)

    # edge_name = "follow"
    # edge_index = gs.get_edge_index(edge_name)
    # print(f"edge_index: {edge_index}")
    # print(f"edge_index shape: {edge_index.shape}")
    # print("前十条边：", edge_index[:, :10])

    gclient.release()

if __name__ == "__main__":
    main()
