import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import json

from nebula_pyg.utils.utils import scan_all_tag_vids, get_edge_type_groups
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.mclient import MetaCache

# TODO: 1.Wrap gen_snapshot into a function of class NebulaPyG, perhaps as a static function so that users don't have to worry about the pickle logic
#       2.JSON is temporarily used to understand the structure


def main():

    space = "basketballplayer"
    output = "snapshot_vid_to_idx.pkl"
    output_json = "snapshot_vid_to_idx.json"

    config = Config()
    connection_pool = ConnectionPool()
    connection_pool.init([("host.docker.internal", 9669)], config)
    gclient = connection_pool.get_session("root", "nebula")

    meta_cache = MetaCache(
        [("metad0", 9559), ("metad1", 9559), ("metad2", 9559)], 50000
    )
    sclient = GraphStorageClient(meta_cache)

    tag_vids = scan_all_tag_vids(space, gclient, sclient)
    print(f"Total vids found: {len(tag_vids)}")
    for tag in tag_vids:
        print(f"  {tag}: {len(tag_vids[tag])} nodes")
    vid_to_idx = {
        tag: {vid: idx for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids
    }
    idx_to_vid = {
        tag: {idx: vid for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids
    }

    vid_to_tag = {}
    for tag, vid_list in tag_vids.items():
        for vid in vid_list:
            vid_to_tag[vid] = tag

    edge_type_groups = list(
        get_edge_type_groups(gclient, sclient, space, {"vid_to_tag": vid_to_tag})
    )

    with open(output, "wb") as f:
        pickle.dump(
            {
                "vid_to_idx": vid_to_idx,
                "idx_to_vid": idx_to_vid,
                "vid_to_tag": vid_to_tag,
                "edge_type_groups": edge_type_groups,
            },
            f,
        )
    print(f"Snapshot saved to {output}")

    snapshot_json = {
        "vid_to_idx": vid_to_idx,
        "idx_to_vid": idx_to_vid,
        "vid_to_tag": vid_to_tag,
        "edge_type_groups": [list(t) for t in edge_type_groups],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(snapshot_json, f, indent=2, ensure_ascii=False)
    print(f"Snapshot JSON saved to {output_json}")


if __name__ == "__main__":
    main()
