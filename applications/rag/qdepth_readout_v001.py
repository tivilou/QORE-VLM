"""Frozen native depth information diagnostic; no quantum advantage claim."""
from __future__ import annotations

import copy
import torch
from torch import nn

ARMS = ('HEAD', 'BLOCK10', 'BLOCK11', 'CONCAT', 'GLOBALMIX')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def extract(model, row):
    """Return CLS after blocks 10/11/12, using zero-based cached layer10 input."""
    hidden = row['hidden'].cuda()
    mask = None if row['mask'] is None else row['mask'].cuda().expand(row['mask_shape']).contiguous()
    layers = model.span_predictor.encoder.bert_model.encoder.layer
    with torch.no_grad():
        middle = layers[-2](hidden, attention_mask=mask, output_attentions=False)[0]
        final = layers[-1](middle, attention_mask=mask, output_attentions=False)[0][:, 0]
        expected_cls = row['cls'].cuda()
        score = model.span_predictor.qa_classifier(final).squeeze(-1)
        need(torch.allclose(final, expected_cls, atol=1e-5, rtol=1e-6), 'native final CLS replay')
        need(torch.allclose(score, row['native_scores'].cuda(), atol=1e-5, rtol=1e-6), 'native final score replay')
        features = torch.stack((hidden[:, 0], middle[:, 0], expected_cls), dim=1).contiguous()
        need(features.shape == (len(hidden), 3, 768) and features.dtype == torch.float32 and bool(torch.isfinite(features).all()), 'three finite frozen CLS representations')
    return features, float((score - row['native_scores'].cuda()).abs().max()), float((final - expected_cls).abs().max())


class DepthReadout(nn.Module):
    def __init__(self, original_head, kind):
        super().__init__()
        need(kind in ARMS, 'fixed diagnostic arm')
        self.kind = kind
        self.head = copy.deepcopy(original_head)
        self.head.weight.requires_grad_(True)
        self.head.bias.requires_grad_(False)
        if kind == 'CONCAT':
            # Same final-head initial function, two additional zero-initialized readouts.
            self.earlier_weights = nn.Parameter(torch.zeros((1, 2, 768), device=self.head.weight.device))
        elif kind == 'GLOBALMIX':
            self.mix = nn.Parameter(torch.zeros(2, device=self.head.weight.device))

    def forward(self, h):
        need(h.ndim == 3 and h.shape[1:] == (3, 768) and not h.requires_grad, 'frozen shared representations')
        if self.kind == 'BLOCK10':
            x = h[:, 0]
        elif self.kind == 'BLOCK11':
            x = h[:, 1]
        elif self.kind == 'GLOBALMIX':
            x = h[:, 2] + ((h[:, :2] - h[:, 2:3]) * self.mix.reshape(1, 2, 1)).sum(dim=1)
        else:
            x = h[:, 2]
        score = self.head(x).squeeze(-1)
        if self.kind == 'CONCAT':
            score = score + (h[:, :2] * self.earlier_weights).sum(dim=(1, 2))
        return score

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]
