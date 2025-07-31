import time

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nebula_pyg.feature_store import NebulaFeatureStore
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.common.ttypes import HostAddr
from nebula3.gclient.net import Connection
from torch_geometric.data import TensorAttr

import pickle




def main():
    meta_addrs = [("127.0.0.1", 9559)]       # Meta 服务端口（默认9559）
    storage_addrs = [HostAddr("host.docker.internal", 45033),
                     HostAddr("host.docker.internal", 46229),
                     HostAddr("host.docker.internal", 44987)
                     ]
    # Storage 服务端口（默认9779）
    # storage_addrs = [
    #     HostAddr.write("127.0.0.1:49470"),
    #     HostAddr("127.0.0.1", 49470),
    #     HostAddr("127.0.0.1", 49472),
    # ]
    space = "basketballplayer"
    tag = "player"
    prop = "age"                             # 属性名

    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")

    # conn = Connection()
    # conn.open("127.0.0.1", 9669, 1000)
    # conn.open("host.docker.internal", 9669, 1000)
    # auth_result = conn.authenticate("root", "nebula")
    # session_id = auth_result.get_session_id()
    # assert session_id != 0
    # resp = conn.execute(
    #     session_id,
    #     "CREATE SPACE IF NOT EXISTS test_meta_cache1(REPLICA_FACTOR=3, vid_type=FIXED_STRING(8));"
    #     "USE test_meta_cache1;"
    #     "CREATE TAG IF NOT EXISTS tag11(name string);"
    #     "CREATE EDGE IF NOT EXISTS edge11(name string);"
    #     "CREATE SPACE IF NOT EXISTS test_meta_cache2(vid_type=FIXED_STRING(8));"
    #     "USE test_meta_cache2;"
    #     "CREATE TAG IF NOT EXISTS tag22(name string);"
    #     "CREATE EDGE IF NOT EXISTS edge22(name string);",
    # )
    # assert resp.error_code == 0
    # conn.close()
    # time.sleep(2)

    # MetaCache
    meta_cache = MetaCache(
                [("metad0", 9559),("metad1", 9559),("metad2", 9559)], 50000
            )
    # GraphStorageClient
    # GSCconn = GraphStorageClient(meta_cache, storage_addrs)
    sclient = GraphStorageClient(meta_cache)

    # FeatureStore
    with open("snapshot_vid_to_idx.pkl", "rb") as f:
        snapshot = pickle.load(f)

    fs = NebulaFeatureStore(gclient, sclient, space, snapshot)
    # 拉取点的 age 属性
    attr = TensorAttr(group_name=tag, attr_name=prop)
    x = fs.get_tensor(attr)
    print(f"拉取的特征 shape: {x.shape}")
    # print("前10个特征：", x[:10])
    print(x)

    # 拉指定 10 个点的 age 属性
    # attr = TensorAttr(group_name="player", attr_name="age")
    # ids = ['player105', 'player109', 'player111', 'player118', 'player143', 'player104', 'player107', 'player116']
    # idxs = [vid_to_idx[vid] for vid in ids]
    # x = fs.get_tensor(attr, index=idxs)
    # print(x)

    # get tensor size
    # attr = TensorAttr(group_name="player", attr_name="age")
    # size = fs.get_tensor_size(attr)
    # print(size)

    # get all tensor attrs
    # attrs = fs.get_all_tensor_attrs()
    # print(attrs)

    gclient.release()

if __name__ == "__main__":
    main()
