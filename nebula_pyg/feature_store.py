from torch_geometric.data import FeatureStore, TensorAttr
from .base_store import NebulaStoreBase
from nebula3.common.ttypes import PropertyType
from nebula3.data.DataObject import ValueWrapper
import torch
import numpy as np
import time

from nebula_pyg.type_helper import get_feature_dim

Y_CANDIDATES = ("label", "y", "target", "category")

class NebulaFeatureStore(NebulaStoreBase, FeatureStore):
    """
    PyG-compatible FeatureStore backed by NebulaGraph with lazy, process-aware
    connection/session management provided by `NebulaStoreBase`.

    Key behaviors:
      - Uses `pool_factory` and `sclient_factory` for lazy initialization of the
        graph query pool and storage client. Sessions are thread-local and rebuilt
        after process changes.
      - Maps Nebula VIDs to PyG indices via a pre-built snapshot and reads vertex
        properties either by full storage scan or targeted queries.

    Exposure policy:
      - expose="x": only expose the synthetic "x" per tag, which is assembled by
        concatenating numeric columns (no individual "feats" are exposed).
      - expose="feats": only expose raw individual numeric properties ("feats"),
        without assembling "x".

    """
    def __init__(self, pool_factory, sclient_factory, space, snapshot,
                 username: str = "root", password: str = "nebula",
                 expose: str = "x"):
        """
        Initialize the FeatureStore with factories and metadata snapshot.

        Args:
          pool_factory: Callable that returns a `ConnectionPool`.
          sclient_factory: Callable that returns a `GraphStorageClient`.
          space: Nebula space name.
          snapshot: Dict containing at least 'vid_to_idx' and 'idx_to_vid' per tag.
          username: Nebula username for session creation.
          password: Nebula password for session creation.
          expose (str): Exposure mode, either "x" or "feats".
            - "x": return a single synthetic feature per tag named "x" (concatenation).
            - "feats": return individual numeric properties as features.
        """
        FeatureStore.__init__(self)
        NebulaStoreBase.__init__(self, pool_factory, sclient_factory, space, username, password)

        if expose not in ("x", "feats"):      # ★ 比 assert 更好：明确报错
            raise ValueError(f"Invalid expose={expose!r}, must be 'x' or 'feats'")
        self.expose = expose
        self.idx_to_vid = snapshot["idx_to_vid"]  # dict[tag] -> {idx: vid}
        self.vid_to_idx = snapshot["vid_to_idx"] 
        
    def get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        """
        Retrieve a tensor for the specified attribute.

        Behavior:
          - If attr.attr_name == "x": assemble and return the synthetic feature
            matrix for the tag (concatenate numeric columns in a fixed order).
          - If attr.attr_name == "y": alias to the tag's "label" property.
          - Otherwise: return the raw property ("feats") as a 1-D tensor.

        The method dispatches to either a full storage scan or a targeted query:
          - Full request (no index or index covers all nodes): `_get_tensor_by_scan`.
          - Partial request (subset of indices): `_get_tensor_by_query`.

        Args:
          attr: TensorAttr with (group_name=tag, attr_name in {"x","y",prop}).
          index: Optional sequence of PyG node indices for sub-selection.

        Returns:
          torch.Tensor:
            - shape (N, D) for "x"
            - shape (N,) for "y" or a raw property
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
        """
        Internal dispatcher selecting between scan or query path.

        - Full request (index is None or covers all nodes): `_get_tensor_by_scan`.
        - Partial request (subset of indices): `_get_tensor_by_query`.
        """
        tag = attr.group_name
        # prop = attr.attr_name
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
        """
        Full-range path using storage scan.

        For "x": scan all numeric columns (`self._x_cols[tag]`), assemble a dense
        (N, D) matrix ordered by PyG indices. Bool values are cast to int.
        For "y": redirect to the "label" property.
        For a raw property: scan that single column and return a length-N vector.

        Missing values are filled with 0 to keep tensors dense.
        """
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
            numeric_cols = self._numeric_cols_by_tag.get(tag) or self._collect_numeric_cols(tag)
            y_prop = self._y_prop(tag, numeric_cols)
            if y_prop is None:
                raise ValueError(f"No usable label column for tag {tag} among {Y_CANDIDATES}")
            return self._get_tensor_by_scan(TensorAttr(tag, y_prop, index=attr.index))

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
        """
        Partial-range path using targeted FETCH queries.

        Steps:
          1) Map requested PyG indices to VIDs via `self.idx_to_vid[tag]`.
          2) For "x": FETCH all x-cols for the selected VIDs and assemble (M, D).
          3) For "y": alias to "label".
          4) For a raw property: FETCH that single column, return length-M vector.

        Missing values and booleans are normalized (None -> 0, bool -> int).
        """
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
            cols_expr = ", ".join(f"{tag}.{c}" for c in feat_names)
            vid_list_str = ", ".join(self._vid_literal(v) for v in vids)
            ngql = f'FETCH PROP ON {tag} {vid_list_str} YIELD {cols_expr}, id(vertex)'
            
            result = self._execute(ngql)

            # print("result:", result)

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

        # TODO：In case ngql takes too long, do a batch process
        # It seems that it is not necessary. Fetching 100,000 points at a time is OK.
        # It is sufficient for the time being under neighborload.
        # If there is an industrial-level demand, you can raise an issue or submit a PR later.

        if prop == "y":
            numeric_cols = self._numeric_cols_by_tag.get(tag) or self._collect_numeric_cols(tag)
            y_prop = self._y_prop(tag, numeric_cols)
            if y_prop is None:
                raise ValueError(f"No usable label column for tag {tag} among {Y_CANDIDATES}")
            return self._get_tensor_by_query(TensorAttr(tag, y_prop, index=index))

        vid_list_str = ", ".join(self._vid_literal(v) for v in vids)
        ngql = f'FETCH PROP ON {tag} {vid_list_str} YIELD {tag}.{prop}, id(vertex)'
        
        # print("ngql:",ngql)

        result = self._execute(ngql)

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
        Return the shape (N, D) for a given tensor.

        - "x": D = number of numeric columns for the tag (excluding reserved ones).
        - "y": D = 1, uses "label" property.
        - other: D = feature dim from tag schema.

        Args:
            attr: TensorAttr with tag and property name ("x", "y", or raw property).
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
        # TODO:
        if attr.attr_name == "y":
            numeric_cols = self._numeric_cols_by_tag.get(tag) or self._collect_numeric_cols(tag)
            prop = self._y_prop(tag, numeric_cols)
            if prop is None:
                raise ValueError(f"No usable label column for tag {tag} among {Y_CANDIDATES}")
        else:
            prop = attr.attr_name

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
        Enumerate all valid tensor attributes in this space based on the exposure policy.

        - If expose="feats": add a TensorAttr for each numeric column.
        - If expose="x": add only one synthetic "x" attr per tag.
        - Always add "y" if "label" exists.

        Returns:
            List[TensorAttr]: All available (tag, property) pairs for features.
        """
        tags_result = self._execute(f"USE {self.space}; SHOW TAGS;")
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

            y_prop = self._y_prop(tag, numeric_cols)
            if y_prop is not None:
                attrs.append(TensorAttr(tag, "y"))

        return attrs


    def _collect_numeric_cols(self, tag):
        """Return sorted numeric property names from tag schema."""
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

    def _y_prop(self, tag: str, numeric_cols: list[str]) -> str | None:
        """Return the first existing y-like column for this tag."""
        return next((c for c in Y_CANDIDATES if c in numeric_cols), None)

    # Reserved for long ngql
    def yield_batched_fetches(tag, cols, vids, max_chars: int):
        """
        Bundle a set of vids into multiple FETCH statements, ensuring each vid does not exceed max_chars characters.
        cols: For example, ["label"] or ["f0","f1",...]
        vids: For example, ["0","1","2",...] (FIXED_STRING -> requires quotes; you can pre-quote them.)
        """
        cols_expr = ", ".join([f"{tag}.{c}" for c in cols] + ["id(vertex)"])
        head = f"FETCH PROP ON {tag} "
        tail = f" YIELD {cols_expr};"

        cur = []
        cur_len = len(head) + len(tail)
        def lit_len(v): return len(v) + 2
        for i, v in enumerate(vids):
            add = lit_len(v)
            if i == 0 and not cur:
                add -= 2
            if cur_len + add > max_chars and cur:
                yield head + ", ".join(cur) + tail
                cur = [v]
                cur_len = len(head) + len(tail) + (len(v))
            else:
                cur.append(v)
                cur_len += add
        if cur:
            yield head + ", ".join(cur) + tail


        """
        quoted_vids = [f'"{v}"' for v in vids]   # FIXED_STRING
        cols = feat_names if prop == "x" else [prop]
        lits = [self._vid_literal(v) for v in vids]
        for ngql in yield_batched_fetches(tag, cols, lits, max_chars=400_000):
            result = self._execute(ngql)
        """


