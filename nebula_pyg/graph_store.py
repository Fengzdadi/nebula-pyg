from torch_geometric.data import GraphStore, EdgeAttr
from torch_geometric.data import EdgeLayout
from abc import ABC
import torch
from nebula3.data.DataObject import ValueWrapper

class NebulaGraphStore(GraphStore, ABC):
    def __init__(self, gclient, sclient, space, idx_to_vid, vid_to_idx):
        super().__init__()
        self.gclient = gclient      
        self.sclient = sclient      
        self.space = space
        self.idx_to_vid = idx_to_vid
        self.vid_to_idx = vid_to_idx

    def get_edge_index(self, edge_name, **kwargs):
        return self._get_edge_index(edge_name, **kwargs)

    # def _get_edge_index(self, edge_name, batch_size=4096):
    #     src_list = []
    #     dst_list = []

    #     for part_id, batch in self.sclient.scan_edge_async(self.space, edge_name, prop_names=[], batch_size=batch_size):
    #         for rel in batch.as_relationships():
    #             src = rel.start_vertex_id().cast()
    #             dst = rel.end_vertex_id().cast()
    #             src_list.append(src)
    #             dst_list.append(dst)
        
    #     print(src_list)
    #     edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    #     return edge_index
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
        result = self.gclient.execute(f"USE {self.space}; SHOW EDGES;")
        edge_names = [ValueWrapper(row.values[0]).cast() for row in result.rows()]
        return [EdgeAttr(edge_type=e, layout=EdgeLayout.COO) for e in edge_names]

