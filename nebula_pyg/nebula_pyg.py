from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

from nebula_pyg.utils import scan_all_tag_vids, get_edge_type_groups

class NebulaPyG:
    def __init__(self, gclient, sclient, space, snapshot = None):
        self.gclient = gclient
        self.sclient = sclient
        self.space = space
        if snapshot is None:
            self.snapshot = self.create_snapshot(gclient, sclient, space)
        else:
            self.snapshot = snapshot

    # TODO: Consider the design logic of snapshot again
    @classmethod
    def create_snapshot(cls, gclient, sclient, space):
        tag_vids = scan_all_tag_vids(space, gclient, sclient)
        vid_to_idx = {tag: {vid: idx for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
        idx_to_vid = {tag: {idx: vid for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
        vid_to_tag = {}
        for tag, vid_list in tag_vids.items():
            for vid in vid_list:
                vid_to_tag[vid] = tag
        edge_type_groups = list(get_edge_type_groups(space, sclient, gclient, {"vid_to_tag": vid_to_tag}))
        return {
            "vid_to_idx": vid_to_idx,
            "idx_to_vid": idx_to_vid,
            "vid_to_tag": vid_to_tag,
            "edge_type_groups": edge_type_groups,
        }

    def get_torch_geometric_remote_backend(self, num_workers=0):
        return (
            NebulaFeatureStore(self.gclient, self.sclient, self.space, self.snapshot),
            NebulaGraphStore(self.gclient, self.sclient, self.space, self.snapshot)
        )
