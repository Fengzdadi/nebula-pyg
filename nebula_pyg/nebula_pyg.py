from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

from nebula_pyg.utils import scan_all_tag_vids, get_edge_type_groups

from nebula3.gclient.net import ConnectionPool
from nebula3.sclient.GraphStorageClient import GraphStorageClient

from typing import Dict, List, Tuple, Callable, Optional, Iterable

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
    def __init__(self, pool_factory, sclient_factory, space: str, username: str = "root", password: str = "nebula", snapshot: dict | None = None):
        """
        Initializes the NebulaPyG interface with NebulaGraph clients and graph space.

        Args:
            gclient: The NebulaGraph graph client instance.
            sclient: The NebulaGraph storage client instance.
            space (str): The name of the target NebulaGraph space.
            snapshot (dict, optional): A precomputed snapshot of the graph structure.
                                       If not provided, one will be generated.
        """
        self.pool_factory = pool_factory
        self.sclient_factory = sclient_factory
        self.space = space
        self.username = username
        self.password = password

        if snapshot is None:
            self.snapshot = self.create_snapshot(
                self.pool_factory, self.sclient_factory, self.space,
                self.username, self.password,  batch_size = 4096
            )
        else:
            self.snapshot = snapshot

    # TODO: Consider the design logic of snapshot again
    @classmethod
    def create_snapshot(
            self,
            pool_factory: Callable[[], "ConnectionPool"],
            sclient_factory: Callable[[], "GraphStorageClient"],
            space: str,
            username: str = "root",
            password: str = "nebula",
            batch_size: int = 4096,
    ) -> dict:
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
        pool = pool_factory()
        sess = pool.get_session(username, password)
        sclient = sclient_factory()

        vid_to_idx, idx_to_vid, vid_to_tag = scan_all_tag_vids(
            space, sess, sclient, batch_size=batch_size
        )
        edge_type_groups = get_edge_type_groups(
            sess, sclient, space, vid_to_tag, batch_size=batch_size
        )

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
            NebulaFeatureStore(self.pool_factory, self.sclient_factory, self.space, self.snapshot, self.username, self.password),
            NebulaGraphStore(self.pool_factory, self.sclient_factory, self.space, self.snapshot, self.username, self.password)
        )
