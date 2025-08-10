from __future__ import annotations
from torch_geometric.data import GraphStore, EdgeAttr, EdgeLayout
import torch
from typing import Iterable, Tuple, Dict
from .base_store import NebulaStoreBase


class NebulaGraphStore(NebulaStoreBase, GraphStore):
    """
    A PyG-compatible GraphStore that fetches edge indices from NebulaGraph.

    This class enables integration of NebulaGraph's edge structure with
    PyTorch Geometric, by mapping edge types and vertex IDs into index-based COO format.

    Attributes:
        gclient: NebulaGraph graph client.
        sclient: NebulaGraph storage client.
        space (str): NebulaGraph space to operate in.
        idx_to_vid (dict): Mapping from tag -> {index -> vid}.
        vid_to_idx (dict): Mapping from tag -> {vid -> index}.
        edge_type_groups (list): List of (src_tag, edge_type, dst_tag) triples.
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
        Initializes the GraphStore with NebulaGraph clients and metadata snapshot.

        Args:
            gclient: NebulaGraph graph client.
            sclient: NebulaGraph storage client.
            space (str): The name of the Nebula space.
            snapshot (dict): Contains edge types and vid-index mappings.
        """
        GraphStore.__init__(self)
        NebulaStoreBase.__init__(self, pool_factory, sclient_factory, space, username, password)

        self.idx_to_vid: Dict[str, Dict[int, str]] = snapshot["idx_to_vid"]
        self.vid_to_idx: Dict[str, Dict[str, int]] = snapshot["vid_to_idx"]

        self.edge_type_groups: Iterable[Tuple[str, str, str]] = snapshot.get("edge_type_groups", [])
        self.default_batch_size = int(default_batch_size)

    def get_edge_index(self, edge_attr: EdgeAttr, **kwargs) -> torch.Tensor:
        """
        Returns the edge index tensor for a given edge type in COO format.

        Args:
            edge_attr (EdgeAttr): Must contain a 3-tuple edge_type: (src_tag, rel, dst_tag).

        Returns:
            torch.Tensor: Tensor of shape [2, num_edges] in COO format.

        Raises:
            ValueError: If edge_type is missing or improperly formatted.
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
        Scans NebulaGraph for all edges of a given edge type, and builds an edge index tensor.

        Args:
            edge_name (str): The name of the edge type in Nebula.
            src_tag (str): Source vertex tag.
            dst_tag (str): Destination vertex tag.
            batch_size (int): Number of edges fetched per storage scan.

        Returns:
            torch.Tensor: Edge index in COO format with shape [2, num_edges].
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
        Returns all known edge types in the space as PyG-compatible EdgeAttr objects.

        Returns:
            List[EdgeAttr]: List of edge attributes in COO layout.
        """
        return [EdgeAttr(edge_type=e, layout=EdgeLayout.COO) for e in self.edge_type_groups]
