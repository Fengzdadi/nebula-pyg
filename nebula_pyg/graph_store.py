from torch_geometric.data import GraphStore, EdgeAttr
from torch_geometric.data import EdgeLayout
from abc import ABC
import torch
from nebula3.data.DataObject import ValueWrapper

class NebulaGraphStore(GraphStore, ABC):
    def __init__(self, gclient, sclient, space, snapshot):
        super().__init__()
        self.gclient = gclient      
        self.sclient = sclient      
        self.space = space
        self.idx_to_vid = snapshot["idx_to_vid"]
        self.vid_to_idx = snapshot["vid_to_idx"]
        self.edge_type_groups = snapshot.get("edge_type_groups", [])

    def get_edge_index(self, edge_attr, **kwargs):
        # print(f"get_edge_index called with edge_attr: {edge_attr}")
        if hasattr(edge_attr, "edge_type"):
            etype = edge_attr.edge_type
            if isinstance(etype, (tuple, list)) and len(etype) == 3:
                edge_name = etype[1]
                src_tag = etype[0]
                dst_tag = etype[2]
                return self._get_edge_index(edge_name, src_tag, dst_tag, **kwargs)
            else:
                raise ValueError(
                    f"edge_attr.edge_type must be a 3-tuple (src, rel, dst), but got: {etype}"
                )
        else:
            raise ValueError(
                f"get_edge_index expects EdgeAttr with edge_type 3-tuple, got: {edge_attr}"
            )

    def _get_edge_index(self, edge_name, src_tag, dst_tag, batch_size=4096):
        src_vid_to_idx = self.vid_to_idx[src_tag]
        dst_vid_to_idx = self.vid_to_idx[dst_tag]
        src_list, dst_list = [], []
        for part_id, batch in self.sclient.scan_edge_async(self.space, edge_name, prop_names=[], batch_size=batch_size):
            for rel in batch.as_relationships():
                src_vid = rel.start_vertex_id().cast()
                dst_vid = rel.end_vertex_id().cast()
                if src_vid not in src_vid_to_idx or dst_vid not in dst_vid_to_idx:
                    continue
                src_idx = src_vid_to_idx[src_vid]
                dst_idx = dst_vid_to_idx[dst_vid]
                src_list.append(src_idx)
                dst_list.append(dst_idx)
        return torch.tensor([src_list, dst_list], dtype=torch.long)


    def _put_edge_index(self, edge_name, edge_index, **kwargs):
        raise NotImplementedError

    def _remove_edge_index(self, edge_name, **kwargs):
        raise NotImplementedError

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        edge_info = self.edge_type_groups
        return [EdgeAttr(edge_type=e, layout=EdgeLayout.COO) for e in edge_info]

