from abc import ABC

from torch_geometric.data import FeatureStore, TensorAttr
import torch

from nebula_pyg.utils import split_batches
from nebula_pyg.type_helper import get_feature_dim

from nebula3.data.DataObject import ValueWrapper

class NebulaFeatureStore(FeatureStore, ABC):
    def __init__(self, gcilent, sclient, space, idx_to_vid, vid_to_idx):
        super().__init__()
        self.gcilent = gcilent
        self.sclient = sclient
        self.space = space
        self.idx_to_vid = idx_to_vid
        self.vid_to_idx = vid_to_idx
        

    def get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        return self._get_tensor(attr, index=index, **kwargs)

    # TODO: COO edge index
    # def _get_tensor(self, attr: TensorAttr, index=None, **kwargs):
    #     tag = attr.group_name
    #     prop = attr.attr_name
    #     values = []

    #     # TODO: Optimize index sampling from the underlying logic instead of filtering here
    #     N = len(self.vid_to_idx)
    #     if index is not None:
    #         if hasattr(index, "tolist"):
    #             index = index.tolist()
    #         else:
    #             index = list(index)
    #         for batch_vids in split_batches(index, batch_size=4096):
    #             for part_id, batch in self.sclient.scan_vertex_async(
    #                     self.space,
    #                     tag,
    #                     [prop],
    #                     batch_size=len(batch_vids),
    #             ):
    #                 for node in batch.as_nodes():
    #                     vid = node.get_id().cast()
    #                     if vid in batch_vids:
    #                         props = node.properties(tag)
    #                         if prop in props:
    #                             val = props[prop].cast()
    #                             values.append(val)

    #     else:
    #         for part_id, batch in self.sclient.scan_vertex_async(
    #             self.space,
    #             tag,
    #             [prop],
    #             batch_size=4096,
    #         ):
    #             for node in batch.as_nodes():
    #                 props = node.properties(tag)
    #                 if prop in props:
    #                     val = props[prop].cast()
    #                     values.append(val)
    #     return torch.tensor(values)

    def _get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        tag = attr.group_name
        print(tag)
        prop = attr.attr_name

        tag_indices = [idx for idx, (t, v) in self.idx_to_vid.items() if t == tag]
        tag_indices = sorted(tag_indices)

        N = len(tag_indices)
        result = [None] * N
        
        idx_map = {self.idx_to_vid[idx]: i for i, idx in enumerate(tag_indices)} 

        for part_id, batch in self.sclient.scan_vertex_async(
            self.space, tag, [prop], batch_size=4096
        ):
            for node in batch.as_nodes():
                vid = node.get_id().cast()
                tag_vid = (tag, vid)
                if tag_vid in idx_map:
                    idx = idx_map[tag_vid]
                    props = node.properties(tag)
                    if prop in props:
                        val = props[prop].cast()
                        result[idx] = val

        if index is not None:
            out = [result[i] for i in index]
        else:
            out = result
        out = [v if v is not None else 0 for v in out]
        return torch.tensor(out)


    def _put_tensor(self, tensor, attr: TensorAttr):
        raise NotImplementedError

    def _remove_tensor(self, attr: TensorAttr):
        raise NotImplementedError

    def get_tensor_size(self, attr: TensorAttr):
        return self._get_tensor_size(attr)

    def _get_tensor_size(self, attr: TensorAttr):
        tag = attr.group_name
        prop = attr.attr_name

        stats_result = self.gcilent.execute(
            f"USE {self.space};"
            "SHOW STATS;"
        )
        print(stats_result)
        num_nodes = None
        # TODO: optimize the logic of getting the number of nodes, maybe only need to get the number of nodes of the tag
        for row in stats_result.rows():
            print(row)
            row_values = row.values
            row_type = ValueWrapper(row_values[0]).cast()  # "Tag"、"Edge"、"Space"
            row_name = ValueWrapper(row_values[1]).cast()  # Sp. tag/edge/space
            count = ValueWrapper(row_values[2]).cast()     # int 
            if row_type == "Tag" and row_name == tag:
                num_nodes = int(count)
                break
        if num_nodes is None:
            raise ValueError(f"Tag {tag} not found in SHOW STATS")
        # TODO: _meta_cache is not a public attribute, need to find a better way to get the schema
        schema = self.sclient._meta_cache.get_tag_schema(self.space, tag)
        # print(schema)
        feature_dim = None
        for col in schema.columns:
            col_name = col.name
            if isinstance(col_name, bytes):
                col_name = col_name.decode()
            if col_name == prop:
                feature_dim = get_feature_dim(col)
                break
        if feature_dim is None:
            raise ValueError(f"Property {prop} not found in tag {tag} schema")

        return (num_nodes, feature_dim)

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        tags_result = self.gcilent.execute(f"USE {self.space}; SHOW TAGS;")
        tags = []
        for row in tags_result.rows():
            tag_name = ValueWrapper(row.values[0]).cast()
            tags.append(tag_name)
        
        attrs = []
        for tag in tags:
            schema = self.sclient._meta_cache.get_tag_schema(self.space, tag)
            for col in schema.columns:
                col_name = col.name.decode() if isinstance(col.name, bytes) else col.name
                attrs.append(TensorAttr(tag, col_name))
        return attrs

    def _multi_get_tensor(self, attrs: list[TensorAttr], index=None, **kwargs):
        return [self._get_tensor(attr, index=index, **kwargs) for attr in attrs]
