from torch_geometric.data import FeatureStore, TensorAttr
from abc import ABC
from nebula3.common.ttypes import PropertyType
from nebula3.data.DataObject import ValueWrapper
import torch
import numpy as np
import time

from nebula_pyg.type_helper import get_feature_dim

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
    def __init__(self, gcilent, connection_pool, sclient, space, snapshot, expose: str = "x"):
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
        self.connection_pool = connection_pool
        self.sclient = sclient
        self.space = space
        self.idx_to_vid = snapshot["idx_to_vid"]
        self.vid_to_idx = snapshot["vid_to_idx"]

        assert expose in ("x", "feats")
        self.expose = expose

        self._numeric_cols_by_tag: dict[str, list[str]] = {}
        self._x_cols: dict[str, list[str]] = {}
        self._reserved_cols = {"x", "y", "label", "category", "target"}
        

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

    def _get_tensor(self, attr: TensorAttr, **kwargs):
        tag = attr.group_name
        prop = attr.attr_name
        index = attr.index
        vid_to_idx = self.vid_to_idx[tag]
        N = len(vid_to_idx)
        
        # Determine whether it is a full request
        if index is None or (isinstance(index, (list, tuple, np.ndarray)) and len(index) == N):
            # print("use scan")
            return self._get_tensor_by_scan(attr)
        else:
            # print("use query")
            return self._get_tensor_by_query(attr)


    def _get_tensor_by_scan(self, attr: TensorAttr, **kwargs):
        tag = attr.group_name
        prop = attr.attr_name

        vid_to_idx = self.vid_to_idx[tag]    # {vid: idx}
        N = len(vid_to_idx)

        if prop == "x":
            feat_names = self._x_cols.get(tag, [])
            if not feat_names:
                raise ValueError(f"No numeric columns available to build x for tag {tag}")
            D = len(feat_names)
            result = [[0] * D for _ in range(N)]

            for _, batch in self.sclient.scan_vertex_async(
                self.space, tag, feat_names, batch_size=4096
            ):
                for node in batch.as_nodes():
                    vid = node.get_id().cast()
                    if vid not in vid_to_idx:
                        continue
                    i = vid_to_idx[vid]
                    props = node.properties(tag)
                    for j, col in enumerate(feat_names):
                        if col in props:
                            v = props[col].cast()
                            if isinstance(v, bool):
                                v = int(v)
                            result[i][j] = v

            if attr.index is not None:
                idxs = attr.index.tolist() if hasattr(attr.index, "tolist") else list(attr.index)
                result = [result[i] for i in idxs]

            return torch.as_tensor(result) # as_tensor() is a shallow copy

        if prop == "y":
            return self._get_tensor_by_scan(TensorAttr(tag, "label", index=attr.index))

        result = [None] * N

        for _, batch in self.sclient.scan_vertex_async(
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

        if attr.index is not None:
            idxs = attr.index.tolist() if hasattr(attr.index, "tolist") else list(attr.index)
            out = [0 if result[i] is None else result[i] for i in idxs]
        else:
            out = [0 if v is None else v for v in result]
        # print("out:", out)
        return torch.as_tensor(out)
        

    def _get_tensor_by_query(self, attr: TensorAttr):
        tag = attr.group_name
        prop = attr.attr_name
        index = attr.index

        # Map index to actual vid
        idx_to_vid = self.idx_to_vid[tag]
        idxs = index.tolist() if hasattr(index, "tolist") else list(index)
        vids = [idx_to_vid[int(i)] for i in idxs]

        if prop == "x":
            feat_names = self._x_cols.get(tag, [])
            if not feat_names:
                raise ValueError(f"No numeric columns available to build x for tag {tag}")
            cols_expr = ", ".join([f"{tag}.{c}" for c in feat_names])
            vid_list_str = ", ".join(f'"{v}"' for v in vids)
            ngql = f'FETCH PROP ON {tag} {vid_list_str} YIELD {cols_expr}, id(vertex)'

            session = self.connection_pool.get_session("root", "nebula")
            try:
                result = session.execute(f'USE {self.space}; {ngql}')
            finally:
                session.release()

            D = len(feat_names)
            m = {}
            for row in result.rows():
                vals = row.values
                if len(vals) != D + 1:
                    raise RuntimeError(f"Unexpected row width={len(vals)}, expect {D+1}")
                row_vec = []
                for k in range(D):
                    v = ValueWrapper(vals[k]).cast()
                    if isinstance(v, bool):
                        v = int(v)
                    row_vec.append(0 if v is None else v)
                vid = ValueWrapper(vals[D]).cast()
                m[vid] = row_vec

            out = [m.get(v, [0]*D) for v in vids]
            return torch.as_tensor(out)

        # TODO
        if prop == "y":
            return self._get_tensor_by_query(TensorAttr(tag, "label", index=index))


        vid_list_str = ", ".join(f'"{v}"' for v in vids) 
        ngql = f'FETCH PROP ON {tag} {vid_list_str} YIELD {tag}.{prop}, id(vertex)'
        session = self.connection_pool.get_session("root", "nebula")
        try:
            result = session.execute(f'USE {self.space}; {ngql}')
        finally:
            session.release()

        # TODO: 
        # session = self.connection_pool.get_session("root", "nebula")
        # result = session.execute(f'USE {self.space}; {ngql}')
        # session.release()
        # # result = self.gcilent.execute(f'USE {self.space}; {ngql}')
        # rows = result.rows()

        m = {}
        for row in result.rows():
            vals = row.values
            if len(vals) != 2:
                raise RuntimeError(f"Unexpected row width={len(vals)}, expect 2")
            v = ValueWrapper(vals[0]).cast()
            if isinstance(v, bool):
                v = int(v)
            vid = ValueWrapper(vals[1]).cast()
            m[vid] = 0 if v is None else v

        out = [m.get(v, 0) for v in vids]
        return torch.as_tensor(out)


        # if len(out) != len(index):
        #     # print("The vids for this query:", vids)
        #     # print("Query results:", result)
        #     # print("is_succeeded:", result.is_succeeded())
        #     # print("error_msg:", result.error_msg())
        #     raise RuntimeError(f"Expected {len(index)} items, actually returned {len(out)} items, some items were missed.")
        # return torch.tensor(out)


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

    # def _get_tensor_size(self, attr: TensorAttr):
    #     tag = attr.group_name
    #     prop = attr.attr_name

    #     stats_result = self.gcilent.execute(
    #         f"USE {self.space};"
    #         "SHOW STATS;"
    #     )
    #     print("stats_result:", stats_result)
    #     num_nodes = None
    #     # TODO: optimize the logic of getting the number of nodes, maybe only need to get the number of nodes of the tag
    #     for row in stats_result.rows():
    #         print("row:", row)
    #         row_values = row.values
    #         row_type = ValueWrapper(row_values[0]).cast()  # "Tag"、"Edge"、"Space"
    #         row_name = ValueWrapper(row_values[1]).cast()  # Sp. tag/edge/space
    #         count = ValueWrapper(row_values[2]).cast()     # int 
    #         if row_type == "Tag" and row_name == tag:
    #             num_nodes = int(count)
    #             break
    #     if num_nodes is None:
    #         raise ValueError(f"Tag {tag} not found in SHOW STATS")
    #     # TODO: _meta_cache is not a public attribute, need to find a better way to get the schema
    #     schema = self.sclient._meta_cache.get_tag_schema(self.space, tag)
    #     # print(schema)
    #     feature_dim = None
    #     for col in schema.columns:
    #         col_name = col.name
    #         if isinstance(col_name, bytes):
    #             col_name = col_name.decode()
    #         if col_name == prop:
    #             feature_dim = int(get_feature_dim(col))
    #             break
    #     if feature_dim is None:
    #         raise ValueError(f"Property {prop} not found in tag {tag} schema")

    #     print(f"Feature size for {tag}.{prop}: (num_nodes={num_nodes}, feature_dim={feature_dim})")

    #     return (num_nodes, feature_dim)

    def _get_tensor_size(self, attr: TensorAttr):
        tag = attr.group_name
        prop = "label" if attr.attr_name == "y" else attr.attr_name

        num_nodes = len(self.vid_to_idx[tag])

        if prop == "x":
            feature_dim = len(self._x_cols.get(tag, []))
            if feature_dim == 0:
                raise ValueError(f"No numeric columns available to build x for tag {tag}")
        else:
            schema = self.sclient._meta_cache.get_tag_schema(self.space, tag)
            feature_dim = None
            for col in schema.columns:
                col_name = col.name.decode() if isinstance(col.name, bytes) else col.name
                if col_name == prop:
                    feature_dim = int(get_feature_dim(col))
                    break
            if feature_dim is None:
                raise ValueError(f"Property {prop} not found in tag {tag} schema")

        return (num_nodes, feature_dim)

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """
        Returns all valid (numeric) tensor attributes found in the graph space.

        Only numeric scalar types are retained to ensure PyG compatibility.

        Returns:
            List[TensorAttr]: List of all (tag, property) pairs usable as tensors.
        """
        tags_result = self.gcilent.execute(f"USE {self.space}; SHOW TAGS;")
        tags = [ValueWrapper(r.values[0]).cast() for r in tags_result.rows()]
        attrs: list[TensorAttr] = []

        for tag in tags:
            numeric_cols = self._collect_numeric_cols(tag)
            self._numeric_cols_by_tag[tag] = numeric_cols

            x_cols = sorted([c for c in numeric_cols if c not in self._reserved_cols])
            self._x_cols[tag] = x_cols

            if self.expose == "feats":
                for col in x_cols:
                    attrs.append(TensorAttr(tag, col))
            else:  
                if x_cols:  
                    attrs.append(TensorAttr(tag, "x"))

            if "label" in numeric_cols:
                attrs.append(TensorAttr(tag, "y"))

        return attrs


    def _collect_numeric_cols(self, tag):
        schema = self.sclient._meta_cache.get_tag_schema(self.space, tag)
        cols = []
        # TODO: limit type(reconsider after vector available)
        # Only keep numeric values
        numeric_types = (PropertyType.INT64, PropertyType.INT8, PropertyType.INT16,
                        PropertyType.INT32, PropertyType.FLOAT, PropertyType.DOUBLE, PropertyType.BOOL)
        for col in schema.columns:
            name = col.name.decode() if isinstance(col.name, bytes) else col.name
            ctype = col.type.type if hasattr(col.type, 'type') else col.type
            if ctype in numeric_types:
                cols.append(name)
        cols.sort()
        return cols
