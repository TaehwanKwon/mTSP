import torch
import torch.nn as nn

class RationalActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 5
        self.n = self.m - 1
        self.param_nominators = nn.Parameter(1e-3 * torch.randn(self.m + 1))
        self.param_denominators = nn.Parameter(1e-3 * torch.randn(self.n + 1))

    def forward(self, x):
        nominator = 0
        for i in range(self.m + 1):
            nominator += self.param_nominators[i] * (x ** i)

        denominator = 1
        for i in range(self.n + 1):
            denominator += self.param_denominators[i] * (x ** i)

        return nominator / denominator

if __name__=='__main__':
    rational_activation = RationalActivation()
