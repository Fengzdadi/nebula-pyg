from torch_geometric.data import GraphStore, EdgeAttr, EdgeLayout
from abc import ABC
import torch
from nebula3.data.DataObject import ValueWrapper

class NebulaGraphStore(GraphStore, ABC):
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
    def __init__(self, gclient, sclient, space, snapshot):
        """
        Initializes the GraphStore with NebulaGraph clients and metadata snapshot.

        Args:
            gclient: NebulaGraph graph client.
            sclient: NebulaGraph storage client.
            space (str): The name of the Nebula space.
            snapshot (dict): Contains edge types and vid-index mappings.
        """
        super().__init__()
        self.gclient = gclient      
        self.sclient = sclient      
        self.space = space
        self.idx_to_vid = snapshot["idx_to_vid"]
        self.vid_to_idx = snapshot["vid_to_idx"]
        self.edge_type_groups = snapshot.get("edge_type_groups", [])

    def get_edge_index(self, edge_attr, **kwargs):
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
        """
        Returns all known edge types in the space as PyG-compatible EdgeAttr objects.

        Returns:
            List[EdgeAttr]: List of edge attributes in COO layout.
        """
        edge_info = self.edge_type_groups
        return [EdgeAttr(edge_type=e, layout=EdgeLayout.COO) for e in edge_info]

