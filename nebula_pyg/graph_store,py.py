from torch_geometric.data import GraphStore, EdgeAttr


class NebulaGraphStore(GraphStore):
    def __init__(self, conn: "NebulaConnection"):
        self.conn = conn

    def put_edge_index(self, edge_index, *, edge_attr: EdgeAttr):
        pass  # 暂不支持写

    def get_edge_index(self, *, edge_attr: EdgeAttr, layout, is_sorted=False):
        # 可先随机生成小图，验证 API 流程
        import torch
        row = pandas.randint(0, 100_000, (500_000,))
        col = torch.randint(0, 100_000, (500_000,))
        return torch.stack([row, col])
