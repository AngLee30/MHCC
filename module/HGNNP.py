import torch
import torch.nn as nn

class HGNNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate, is_last=False) 
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, drop_rate=0.5, is_last=True)
        )

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, H)
        return X
    

class HGNNPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.0,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)


    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # X: [N, C_t]
        # H: [N, M]

        # [N, C_t] @ [C_t, C_{t+1}] -> [N, C_{t+1}]
        X = self.theta(X)

        # v2e aggregation
        H_T = H.transpose(0, 1) # [M, N]
        X = torch.mm(H_T, X) # [M, N] @ [N, C_{t+1}] -> [M, C_{t+1}]

        D_e = torch.sum(H_T, dim=1).view(-1, 1) # [M, 1]
        D_e[D_e == 0] = float('inf')

        D_e_neg_1 = 1.0 / D_e # [M, 1]
        D_e_neg_1[torch.isinf(D_e_neg_1)] = 0

        X = D_e_neg_1 * X # [M, 1] * [M, C_{t+1}] -> [M, C_{t+1}] (Hadamard product)

        # v2e update: X = W_e @ X(W_e: Identity matrix)

        # e2v aggregation
        X = torch.mm(H, X) # [N, M] @ [M, C_{t+1}] -> [N, C_{t+1}]

        D_v = torch.sum(H, dim=1).view(-1, 1) # [N, 1]
        D_v[D_v == 0] = float('inf')

        D_v_neg_1 = 1.0 / D_v # [N, 1]
        D_v_neg_1[torch.isinf(D_v_neg_1)] = 0

        X = D_v_neg_1 * X # [N, 1] * [N, C_{t+1}] -> [N, C_{t+1}] (hadamard product)

        if not self.is_last:
            X = self.act(X) # activation
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X) # dropout
        return X

