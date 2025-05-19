from torch_geometric.loader import NeighborSampler


class NebulaNeighborSampler(NeighborSampler):
    """
    Reuse PyG NeighborSampler, the only difference:
    edge_index / num_nodes provided by NebulaGraphStore.
    """
    pass
