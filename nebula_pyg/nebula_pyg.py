from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

from nebula_pyg.utils import scan_all_tag_vids, get_edge_type_groups

class NebulaPyG:
    """
    An interface class that integrates NebulaGraph with PyTorch Geometric (PyG).

    This class manages connections to both the graph query client (`gclient`) and
    the graph storage client (`sclient`). It automatically constructs or receives a
    snapshot of the graph structure and indexing, which is later used to initialize
    PyG-compatible feature and graph stores.

    Attributes:
        gclient: The NebulaGraph graph client.
        sclient: The NebulaGraph storage client.
        space (str): The name of the NebulaGraph space to operate on.
        snapshot (dict): A dictionary holding pre-scanned graph metadata
                         such as vid mappings and edge type groups.
    """
    def __init__(self, gclient, sclient, space, snapshot = None):
        """
        Initializes the NebulaPyG interface with NebulaGraph clients and graph space.

        Args:
            gclient: The NebulaGraph graph client instance.
            sclient: The NebulaGraph storage client instance.
            space (str): The name of the target NebulaGraph space.
            snapshot (dict, optional): A precomputed snapshot of the graph structure.
                                       If not provided, one will be generated.
        """
        self.gclient = gclient
        self.sclient = sclient
        self.space = space
        if snapshot is None:
            self.snapshot = self.create_snapshot(gclient, sclient, space)
        else:
            self.snapshot = snapshot

    # TODO: Consider the design logic of snapshot again
    @classmethod
    def create_snapshot(cls, gclient, sclient, space):
        """
        Scans the graph space to build a snapshot containing index and structure metadata.

        This includes mappings between vertex IDs and integer indices required by PyG,
        and all heterogeneous edge type groups in the graph.

        Args:
            gclient: The NebulaGraph query client.
            sclient: The NebulaGraph storage client.
            space (str): The name of the NebulaGraph space to scan.

        Returns:
            dict: A snapshot dictionary with the following keys:
                - 'vid_to_idx': tag-wise mapping from vid to PyG index.
                - 'idx_to_vid': inverse mapping from index to vid.
                - 'vid_to_tag': mapping from vid to its tag (vertex type).
                - 'edge_type_groups': list of (src_tag, edge_type, dst_tag) triples.
        """
        tag_vids = scan_all_tag_vids(space, gclient, sclient)
        vid_to_idx = {tag: {vid: idx for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
        idx_to_vid = {tag: {idx: vid for idx, vid in enumerate(tag_vids[tag])} for tag in tag_vids}
        vid_to_tag = {}
        for tag, vid_list in tag_vids.items():
            for vid in vid_list:
                vid_to_tag[vid] = tag
        edge_type_groups = list(get_edge_type_groups(gclient, sclient, space, {"vid_to_tag": vid_to_tag}))
        return {
            "vid_to_idx": vid_to_idx,
            "idx_to_vid": idx_to_vid,
            "vid_to_tag": vid_to_tag,
            "edge_type_groups": edge_type_groups,
        }

    def get_torch_geometric_remote_backend(self, num_workers=0):
        """
        Returns PyG-compatible remote backends for feature and graph storage.

        Args:
            num_workers (int): Number of workers for data loading (currently unused).

        Returns:
            Tuple[NebulaFeatureStore, NebulaGraphStore]:
                A tuple of remote PyG FeatureStore and GraphStore.
        """
        return (
            NebulaFeatureStore(self.gclient, self.sclient, self.space, self.snapshot),
            NebulaGraphStore(self.gclient, self.sclient, self.space, self.snapshot)
        )
