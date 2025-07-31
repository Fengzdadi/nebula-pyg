import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
from nebula_pyg.utils import scan_all_tag_vids
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache

def main():

    space = "basketballplayer"
    output = "snapshot_vid_to_idx.pkl"

    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")

    meta_cache = MetaCache(
            [("metad0", 9559),("metad1", 9559),("metad2", 9559)], 50000
        )
    sclient = GraphStorageClient(meta_cache)

    tag_vids = scan_all_tag_vids(space, gclient, sclient)
    print(f"Total vids found: {len(tag_vids)}")
    for tag in tag_vids:
        print(f"  {tag}: {len(tag_vids[tag])} nodes")
    vid_to_idx = {tag: {vid: idx for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
    idx_to_vid = {tag: {idx: vid for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
    
    vid_to_tag = {}
    for tag, vid_list in tag_vids.items():
        for vid in vid_list:
            vid_to_tag[vid] = tag

    with open(output, "wb") as f:
        pickle.dump({"vid_to_idx": vid_to_idx, "idx_to_vid": idx_to_vid, "vid_to_tag": vid_to_tag}, f)
    print(f"Snapshot saved to {output}")

if __name__ == "__main__":
    main()
