from torch_geometric.data import FeatureStore, TensorAttr
from abc import ABC
from nebula3.common.ttypes import PropertyType
from nebula3.data.DataObject import ValueWrapper
import torch

from nebula_pyg.utils import get_feature_dim

class NebulaFeatureStore(FeatureStore, ABC):
    """
    A PyG-compatible FeatureStore backed by NebulaGraph.

    This class fetches vertex features from NebulaGraph using the storage client.
    It supports lazy scanning and mapping of vertex IDs to PyG indices based on a snapshot.

    Attributes:
        gcilent: NebulaGraph graph client.
        sclient: NebulaGraph storage client.
        space (str): The name of the graph space.
        idx_to_vid (dict): Mapping from tag name to {index: vid}.
        vid_to_idx (dict): Mapping from tag name to {vid: index}.
    """
    def __init__(self, gcilent, sclient, space, snapshot):
        """
        Initializes the FeatureStore with necessary clients and metadata.

        Args:
            gcilent: NebulaGraph graph client.
            sclient: NebulaGraph storage client.
            space (str): Graph space name.
            snapshot (dict): Pre-scanned snapshot including ID mappings.
        """
        super().__init__()
        self.gcilent = gcilent
        self.sclient = sclient
        self.space = space
        self.idx_to_vid = snapshot["idx_to_vid"]
        self.vid_to_idx = snapshot["vid_to_idx"]
        

    def get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        """
        Retrieves feature tensor for a given vertex property.

        Args:
            attr (TensorAttr): The target feature location (tag and property).
            index (Optional[list[int]]): Indices to select from result. If None, return all.

        Returns:
            torch.Tensor: A tensor of features in the order of PyG node indices.
        """
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
        prop = attr.attr_name

        vid_to_idx = self.vid_to_idx[tag]    # {vid: idx}
        N = len(vid_to_idx)
        result = [None] * N

        for part_id, batch in self.sclient.scan_vertex_async(
            self.space, tag, [prop], batch_size=4096
        ):
            for node in batch.as_nodes():
                vid = node.get_id().cast()
                if vid not in vid_to_idx:
                    continue
                idx = vid_to_idx[vid]
                props = node.properties(tag)
                if prop in props:
                    val = props[prop].cast()
                    result[idx] = val

        if index is not None:
            out = [result[i] for i in index]
        else:
            out = result
        out = [v if v is not None else 0 for v in out]
        print("out:", out)
        return torch.tensor(out)


    def _put_tensor(self, tensor, attr: TensorAttr):
        raise NotImplementedError

    def _remove_tensor(self, attr: TensorAttr):
        raise NotImplementedError

    def get_tensor_size(self, attr: TensorAttr):
        """
        Returns the shape (N, D) of a specific tensor in the store.

        Args:
            attr (TensorAttr): The tensor's tag and property.

        Returns:
            Tuple[int, int]: (number of nodes, feature dimension)
        """
        return self._get_tensor_size(attr)

    def _get_tensor_size(self, attr: TensorAttr):
        tag = attr.group_name
        prop = attr.attr_name

        stats_result = self.gcilent.execute(
            f"USE {self.space};"
            "SHOW STATS;"
        )
        print("stats_result:", stats_result)
        num_nodes = None
        # TODO: optimize the logic of getting the number of nodes, maybe only need to get the number of nodes of the tag
        for row in stats_result.rows():
            print("row:", row)
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
                feature_dim = int(get_feature_dim(col))
                break
        if feature_dim is None:
            raise ValueError(f"Property {prop} not found in tag {tag} schema")

        print(f"Feature size for {tag}.{prop}: (num_nodes={num_nodes}, feature_dim={feature_dim})")

        return (num_nodes, feature_dim)

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """
        Returns all valid (numeric) tensor attributes found in the graph space.

        Only numeric scalar types are retained to ensure PyG compatibility.

        Returns:
            List[TensorAttr]: List of all (tag, property) pairs usable as tensors.
        """
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
                # TODO: limit type(reconsider after vecotr available)
                col_type = col.type.type if hasattr(col.type, 'type') else col.type
                # Only keep numeric values
                if col_type in (PropertyType.INT64, PropertyType.INT8, PropertyType.INT16,PropertyType.INT32, PropertyType.FLOAT, PropertyType.DOUBLE, PropertyType.BOOL):
                    attrs.append(TensorAttr(tag, col_name))
                else:
                    print(f"[Skip] {tag}.{col_name} is of type {col_type}, which is not numeric")
        return attrs
