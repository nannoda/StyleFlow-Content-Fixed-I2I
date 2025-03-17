import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Set device
torch_device = xm.xla_device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3).to(torch_device)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        input = input.to(torch_device)
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        output = output.to(torch_device)
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        # Convert to torch tensors
        w_p = torch.from_numpy(w_p.copy()).to(torch_device)
        w_l = torch.from_numpy(w_l.copy()).to(torch_device)
        w_s = torch.from_numpy(w_s.copy()).to(torch_device)
        w_u = torch.from_numpy(w_u.copy()).to(torch_device)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask.copy()).to(torch_device))
        self.register_buffer('l_mask', torch.from_numpy(l_mask.copy()).to(torch_device))
        self.register_buffer('s_sign', torch.sign(w_s).to(torch_device))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]).to(torch_device))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        input = input.to(torch_device)
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s) + 1e-5))
        )
        return weight.unsqueeze(2).unsqueeze(3).to(torch_device)

    def reverse(self, output):
        output = output.to(torch_device)
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0).to(torch_device)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1).to(torch_device))

    def forward(self, input):
        input = input.to(torch_device)
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out
