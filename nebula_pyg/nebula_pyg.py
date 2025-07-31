from nebula_pyg.graph_store import NebulaGraphStore
from nebula_pyg.feature_store import NebulaFeatureStore

class NebulaPyg:
    def __init__(self, gclient, sclient, space, snapshot):
        self.gclient = gclient
        self.sclient = sclient
        self.space = space
        self.snapshot = snapshot

    def get_torch_geometric_remote_backend(self, num_workers=0):
        return (
            NebulaFeatureStore(self.gclient, self.sclient, self.space, self.snapshot),
            NebulaGraphStore(self.gclient, self.sclient, self.space, self.snapshot)
        )
