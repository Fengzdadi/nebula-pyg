from __future__ import annotations
from torch_geometric.data import GraphStore, EdgeAttr, EdgeLayout
import torch
from typing import Iterable, Tuple, Dict
from .base_store import NebulaStoreBase


class NebulaGraphStore(NebulaStoreBase, GraphStore):
    """
    PyG-compatible GraphStore that materializes edge indices from NebulaGraph.

    This adapter reads edges from Nebula storaged via batch scans and converts
    them into index-based COO tensors using tag-scoped VID→index mappings
    supplied by the snapshot.

    Highlights:
      - Snapshot-driven indexing: `vid_to_idx` / `idx_to_vid` are provided per tag.
      - COO-only: returns `torch.LongTensor` of shape [2, num_edges].
      - Lazy & multi-process safe: connections are created on first use by the
        base class (`NebulaStoreBase`), avoiding FD sharing across forked workers.
      - Property-agnostic: edge properties are ignored; only endpoints are read.

    Attributes:
        pool_factory: Factory returning a ConnectionPool (graphd).
        sclient_factory: Factory returning a GraphStorageClient (storaged).
        space (str): Target NebulaGraph space.
        idx_to_vid (dict[str, dict[int, str]]): Per-tag index → VID.
        vid_to_idx (dict[str, dict[str, int]]): Per-tag VID → index.
        edge_type_groups (Iterable[tuple[str, str, str]]): All (src_tag, edge_name, dst_tag).
        default_batch_size (int): Default batch size for storage scans.
    """

    def __init__(
            self,
            pool_factory,
            sclient_factory,
            space: str,
            snapshot: Dict,
            username: str = "root",
            password: str = "nebula",
            default_batch_size: int = 4096,
    ):
        """
        Initialize the GraphStore with connection factories and a metadata snapshot.

        The snapshot is expected to contain:
          - "idx_to_vid": {tag: {idx: vid}}
          - "vid_to_idx": {tag: {vid: idx}}
          - optionally "edge_type_groups": [(src_tag, edge_name, dst_tag), ...]

        Args:
            pool_factory: Factory returning a ConnectionPool (graphd).
            sclient_factory: Factory returning a GraphStorageClient (storaged).
            space: NebulaGraph space name.
            snapshot: Pre-scanned metadata used for VID/index translation.
            username: Nebula username.
            password: Nebula password.
            default_batch_size: Fallback batch size for edge scans.
        """
        GraphStore.__init__(self)
        NebulaStoreBase.__init__(self, pool_factory, sclient_factory, space, username, password)

        self.idx_to_vid: Dict[str, Dict[int, str]] = snapshot["idx_to_vid"]
        self.vid_to_idx: Dict[str, Dict[str, int]] = snapshot["vid_to_idx"]

        self.edge_type_groups: Iterable[Tuple[str, str, str]] = snapshot.get("edge_type_groups", [])
        self.default_batch_size = int(default_batch_size)

    def get_edge_index(self, edge_attr: EdgeAttr, **kwargs) -> torch.Tensor:
        """
        Return the edge index for a specific edge type in COO layout.

        Requirements:
          - `edge_attr.layout` must be `EdgeLayout.COO`.
          - `edge_attr.edge_type` must be a 3-tuple: (src_tag, edge_name, dst_tag).

        Keyword Args:
            batch_size (int): Optional override for scan batch size.

        Returns:
            torch.LongTensor: Shape [2, num_edges] (COO). Empty if mappings are missing.

        Raises:
            NotImplementedError: If a non-COO layout is requested.
            ValueError: If `edge_type` is not a (src, rel, dst) triple.
        """
        # print(f"get_edge_index called with edge_attr: {edge_attr}")
        if edge_attr.layout != EdgeLayout.COO:
            raise NotImplementedError("Only COO layout is supported for now.")

        etype = edge_attr.edge_type
        if not (isinstance(etype, (tuple, list)) and len(etype) == 3):
            raise ValueError(f"edge_attr.edge_type must be (src_tag, edge_name, dst_tag), got: {etype}")
        src_tag, edge_name, dst_tag = etype[0], etype[1], etype[2]

        batch_size = int(kwargs.get("batch_size", self.default_batch_size))
        return self._get_edge_index(edge_name, src_tag, dst_tag, batch_size=batch_size)

    def _get_edge_index(self, edge_name, src_tag, dst_tag, batch_size=4096):
        """
        Scan storaged for all edges of `edge_name` and build a COO edge index.

        Notes:
          - Uses per-tag `vid_to_idx` to translate VIDs into contiguous indices.
          - Silently skips edges whose endpoints are not present in the mappings.
          - Returns an empty [2, 0] tensor if no valid edges are found.

        Args:
            edge_name (str): Edge type name in Nebula.
            src_tag (str): Source vertex tag.
            dst_tag (str): Destination vertex tag.
            batch_size (int): Number of edges per storage scan.

        Returns:
            torch.LongTensor: Edge index [2, num_edges] in COO format.
        """
        src_vid_to_idx = self.vid_to_idx.get(src_tag, {})
        dst_vid_to_idx = self.vid_to_idx.get(dst_tag, {})
        if not src_vid_to_idx or not dst_vid_to_idx:
            return torch.empty((2, 0), dtype=torch.long)

        sclient = self.sclient
        src_list, dst_list = [], []
        for _part_id, batch in sclient.scan_edge_async(
                self.space,
                edge_name,
                prop_names=[],
                batch_size=batch_size,
        ):
            for rel in batch.as_relationships():
                svid = rel.start_vertex_id().cast()
                dvid = rel.end_vertex_id().cast()
                si = src_vid_to_idx.get(svid)
                di = dst_vid_to_idx.get(dvid)
                if si is None or di is None:
                    continue
                src_list.append(si)
                dst_list.append(di)

        if not src_list:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _put_edge_index(self, edge_name, edge_index, **kwargs):
        raise NotImplementedError

    def _remove_edge_index(self, edge_name, **kwargs):
        raise NotImplementedError

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        """
        Enumerate all known edge types as PyG EdgeAttr in COO layout.

        Source of truth is `self.edge_type_groups`, typically populated by the
        snapshot builder. If empty, returns an empty list.

        Returns:
            list[EdgeAttr]: One EdgeAttr per (src_tag, edge_name, dst_tag), COO only.
        """
        return [EdgeAttr(edge_type=e, layout=EdgeLayout.COO) for e in self.edge_type_groups]
