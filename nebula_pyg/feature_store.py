from abc import ABC

from torch_geometric.data import FeatureStore, TensorAttr
import torch
from nebula_pyg.utils import split_batches


class NebulaFeatureStore(FeatureStore, ABC):
    def __init__(self, conn, space):
        super().__init__()
        self.conn = conn
        self.space = space

    def get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        return self._get_tensor(attr, index=index, **kwargs)

    def _get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        tag = attr.group_name
        prop = attr.attr_name
        values = []

        # TODO: Optimize index sampling from the underlying logic instead of filtering here
        if index is not None:
            if hasattr(index, "tolist"):
                index = index.tolist()
            else:
                index = list(index)
            for batch_vids in split_batches(index, batch_size=4096):
                for part_id, batch in self.conn.scan_vertex_async(
                        self.space,
                        tag,
                        [prop],
                        batch_size=len(batch_vids),
                ):
                    for node in batch.as_nodes():
                        vid = node.get_id().cast()
                        if vid in batch_vids:
                            props = node.properties(tag)
                            if prop in props:
                                val = props[prop].cast()
                                values.append(val)

        else:
            for part_id, batch in self.conn.scan_vertex_async(
                self.space,
                tag,
                [prop],
                batch_size=4096,
            ):
                for node in batch.as_nodes():
                    props = node.properties(tag)
                    if prop in props:
                        val = props[prop].cast()
                        values.append(val)
        return torch.tensor(values)

    def _put_tensor(self, tensor, attr: TensorAttr):
        raise NotImplementedError

    def _remove_tensor(self, attr: TensorAttr):
        raise NotImplementedError

    def _get_tensor_size(self, attr: TensorAttr):
        raise NotImplementedError

    def get_all_tensor_attrs(self):
        raise NotImplementedError
