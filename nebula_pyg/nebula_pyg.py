from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

from nebula_pyg.utils import scan_all_tag_vids, get_edge_type_groups

from nebula3.gclient.net import ConnectionPool
from nebula3.sclient.GraphStorageClient import GraphStorageClient

from typing import Dict, List, Tuple, Callable, Optional, Iterable

class NebulaPyG:
    """
    High-level integration between NebulaGraph and PyTorch Geometric (PyG).

    The main purpose of this class is to provide a single entry point for obtaining
    both a PyG-compatible `NebulaFeatureStore` and `NebulaGraphStore` in one call via
    `get_torch_geometric_remote_backend()`.

    It also handles optional metadata preparation:
        - If a precomputed `snapshot` is provided, it will be reused.
        - If not, `create_snapshot()` scans the NebulaGraph space to build vertex ID
        mappings and heterogeneous edge type groups required by PyG.

    Key features:
        - One-step construction of both FeatureStore and GraphStore backends.
        - Snapshot generation or adaptation for PyG indexing.
        - Supports factory-based lazy connection creation for multi-process safety.

    Attributes:
        pool_factory (Callable[[], ConnectionPool]): Factory for creating a graphd connection pool.
        sclient_factory (Callable[[], GraphStorageClient]): Factory for creating a storaged client.
        space (str): Target NebulaGraph space name.
        username (str): Username for authentication.
        password (str): Password for authentication.
        snapshot (dict): Graph metadata including vid mappings and edge type groups.
    """
    def __init__(self, pool_factory, sclient_factory, space: str, username: str = "root", password: str = "nebula", snapshot: dict | None = None):
        """
        Initialize the NebulaPyG integration.

        If `snapshot` is not provided, `create_snapshot()` will be called to scan
        the target space and build the necessary metadata for PyG backends.

        Args:
            pool_factory (Callable): Factory function returning a ConnectionPool.
            sclient_factory (Callable): Factory function returning a GraphStorageClient.
            space (str): Target NebulaGraph space.
            username (str): Login username (default: "root").
            password (str): Login password (default: "nebula").
            snapshot (dict, optional): Precomputed metadata; skips scanning if provided.
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
            cls,
            pool_factory: Callable[[], "ConnectionPool"],
            sclient_factory: Callable[[], "GraphStorageClient"],
            space: str,
            username: str = "root",
            password: str = "nebula",
            batch_size: int = 4096,
    ) -> dict:
        """
        Scan the target space and return a snapshot of its structure.

        The snapshot contains:
            - vid_to_idx: {tag: {vid: int_idx}}
            - idx_to_vid: {tag: {int_idx: vid}}
            - vid_to_tag: {vid: tag}
            - edge_type_groups: [(src_tag, edge_type, dst_tag), ...]

        Args:
            pool_factory: Factory for a graphd connection pool.
            sclient_factory: Factory for a storaged client.
            space (str): Name of the NebulaGraph space.
            username (str): Login username.
            password (str): Login password.
            batch_size (int): Number of vertices/edges to scan per request.

        Returns:
            dict: The metadata snapshot.
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
        Create PyG-compatible remote FeatureStore and GraphStore.

        Args:
            num_workers (int): Number of DataLoader workers (currently unused here).

        Returns:
            tuple:
                - NebulaFeatureStore
                - NebulaGraphStore
        """
        return (
            NebulaFeatureStore(self.pool_factory, self.sclient_factory, self.space, self.snapshot, self.username, self.password),
            NebulaGraphStore(self.pool_factory, self.sclient_factory, self.space, self.snapshot, self.username, self.password)
        )
