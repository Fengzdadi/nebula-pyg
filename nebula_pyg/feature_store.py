from torch_geometric.data import FeatureStore, TensorAttr
import torch


class NebulaFeatureStore(FeatureStore):
    def __init__(self, conn: "NebulaConnection"):
        self.conn = conn

    # ---- Interface Implementation ----
    def put_tensor(self, tensor, attr: TensorAttr, **kwargs):
        raise NotImplementedError

    def get_tensor(self, attr: TensorAttr, index=None, **kwargs):
        """
        id List → NebulaGraph `LOOKUP` / Storage scan
        return torch.Tensor
        """
        # demo: 随机值
        from torch import randn
        if index is None:
            # 全量扫描
            return randn(100_000, 128)
        else:
            return randn(len(index), 128)

    def get_tensor_size(self, attr: TensorAttr) -> int:
        return 100_000  # mock
