import torch
import torch.nn as nn

class Generator1(nn.Module):
    def __init__(self, H: torch.Tensor):
        super(Generator1, self).__init__()
        print("Initialize Generator version 1 ([N, N])")
        self.N = H.shape[0]
        self.M = H.shape[1]
        self.W = nn.Parameter(torch.eye(self.N)) # diag[N, N]
        self.b = nn.Parameter(torch.zeros((self.N, self.M))) # diag[N, M]
        
    def forward(self, H):
        H_a = torch.mm(self.W, H) # [N, N] @ [N, M] -> [N, M] <=> X = X @ W
        H_a = H_a + self.b # X = X @ W + b
        H_a = torch.sigmoid(H_a)
        H_a = H_a * H
        return H_a


class Generator2(nn.Module):
    def __init__(self, H: torch.Tensor):
        super(Generator2, self).__init__()
        print("Initialize Generator version 2 ([M, M])")
        self.N = H.shape[0]
        self.M = H.shape[1]
        self.W = nn.Parameter(torch.eye(self.M)) # diag[M, M]
        self.b = nn.Parameter(torch.zeros((self.N, self.M))) # diag[N, M]
        
    def forward(self, H):
        H_a = torch.mm(H, self.W) # [N, M] @ [M, M] -> [N, M] <=> X = X @ W
        H_a = H_a + self.b # X = X @ W + b
        H_a = torch.sigmoid(H_a)
        H_a = H_a * H
        return H_a


class Generator3(nn.Module):
    def __init__(self, H: torch.Tensor):
        super(Generator3, self).__init__()
        print("Initialize Generator version 3 (Scaling)")
        self.N = H.shape[0]
        self.M = H.shape[1]
        self.W = nn.Parameter(torch.ones((self.N, self.M))) # [N, M]
        
    def forward(self, H):
        H_a = torch.sigmoid(self.W) * H
        return H_a
    