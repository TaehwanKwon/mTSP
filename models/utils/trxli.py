import torch
import torch.nn as nn

class TrXLI(nn.Module):
    def __init__(self, cfg, activation=nn.ReLU()):
        super().__init__()
        self.cfg = cfg
        self.activation = activation
        self.n_head = self.cfg['learning']['n_head']
        self.hidden_size = self.cfg['learning']['base_hidden_size']
        head_hidden_size = self.hidden_size // self.n_head

        self.ln_pre = nn.LayerNorm(self.hidden_size)
        self.fc_qs = nn.ModuleList(
            [ nn.Linear(self.hidden_size, head_hidden_size) for _ in range(self.n_head) ]
        )
        self.fc_ks = nn.ModuleList(
            [ nn.Linear(self.hidden_size, head_hidden_size) for _ in range(self.n_head) ]
        )
        self.fc_vs = nn.ModuleList(
            [ nn.Linear(self.hidden_size, head_hidden_size) for _ in range(self.n_head) ]
        )
        self.fc_mha = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_post = nn.LayerNorm(self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        h_mha = self.ln_pre(x)
        qs = [fc_q(h_mha) for fc_q in self.fc_qs]
        ks = [fc_k(h_mha) for fc_k in self.fc_ks]
        vs = [fc_v(h_mha) for fc_v in self.fc_vs]

        outs = list()
        for i in range(self.n_head):
            attention = torch.softmax(
                torch.matmul(qs[i], ks[i].transpose(-1, -2)) / qs[i].shape[-1] ** 0.5, dim=-1
            )
            out = torch.matmul(attention, vs[i])
            outs.append(out)
        out_mha = torch.cat(outs, dim=-1)
        h_ln_post = self.fc_mha(out_mha) + x

        h_fc_out = self.ln_post(h_ln_post)
        out = self.activation(self.fc_out(h_fc_out)) + h_ln_post

        return out


if __name__=='__main__':
    cfg = dict(n_head=4)
    d_hidden = 64
    trxli_test = TrXLI(cfg, d_hidden)

    x = torch.randn(5, 64)
    with torch.no_grad():
        y = trxli_test(x)