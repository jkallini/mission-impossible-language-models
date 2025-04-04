# geometric_attention.py
# Author: Róbert Csordás

import torch
import torch.nn.functional as F

class GeometricAttentionFunction(torch.autograd.Function):
    # Numerically stable implementation from https://openreview.net/pdf?id=r8J3DSD5kF
    @staticmethod
    def forward(ctx, att):
        logits = att
        att = att.float()

        prev = F.softplus(att, threshold=15)

        # cumsum from right to left
        prevs = prev.cumsum(dim=-1)
        prevs.neg_()
        prevs += -prevs[..., -1:]
        prevs += prev

        del prev
        prev = None

        res = att - prevs
        res = res.exp().type_as(logits)
        ctx.save_for_backward(res, logits)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        att_out, logits = ctx.saved_tensors
        grad_output = (grad_output * att_out).float()

        sigma = F.sigmoid(logits.float())
        cumgrad = grad_output.cumsum(dim=-1)

        grad_output -= sigma * cumgrad
        return grad_output.type_as(logits)

gatt = GeometricAttentionFunction.apply
