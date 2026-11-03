"""Precision-separated derivative evidence on the same fixed training probe.

Training remains FP32. Record the old FP32 finite differences without treating
their cancellation/truncation error as an automatic gradient failure. Require
the actual FP32 analytic directional derivative to agree with both FP64
autograd and an independently evaluated FP64 central difference. The previous
absolute/relative acceptance bound is retained, not enlarged.
"""
from __future__ import annotations
import copy
import math
import torch


def numeric_probe(model,h,tiers,packed,loss_for):
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    double=copy.deepcopy(model).double()
    double.zero_grad(set_to_none=True)
    h64,packed64=h.double(),packed.double()
    loss64,_=loss_for(double,h64,tiers,packed64)
    loss64.backward()
    rows={};passed=True
    for name,parameter in model.named_parameters():
        direction=gradients[name]/gradients[name].norm().clamp_min(1e-12)
        analytic=float((gradients[name]*direction).sum())
        d64=direction.double();p64=dict(double.named_parameters())[name]
        analytic64=float((p64.grad*d64).sum())
        saved64=p64.detach().clone();curves={}
        try:
            with torch.no_grad():
                for epsilon in (.001,.0001):
                    values=[]
                    for sign in (1,-1):
                        parameter.copy_(original[name]+sign*epsilon*direction)
                        values.append(float(loss_for(model,h,tiers,packed)[0]))
                    curves[str(epsilon)]=(values[0]-values[1])/(2*epsilon)
                values=[]
                for sign in (1,-1):
                    p64.copy_(saved64+sign*.00001*d64)
                    values.append(float(loss_for(double,h64,tiers,packed64)[0]))
                numeric64=(values[0]-values[1])/.00002
        finally:
            with torch.no_grad():
                parameter.copy_(original[name]);p64.copy_(saved64)
        tolerance=2e-4+.05*abs(analytic)
        finite=all(math.isfinite(v) for v in (analytic,analytic64,numeric64,*curves.values()))
        ok=finite and abs(analytic64-analytic)<=tolerance and abs(numeric64-analytic)<=tolerance
        passed=passed and ok
        rows[name]=dict(analytic_fp32=analytic,analytic_fp64=analytic64,
            central_fp64=numeric64,central_fp32=curves,tolerance=tolerance,
            old_fp32_check_would_pass=abs(curves['0.001']-analytic)<=tolerance,passed=ok)
    return dict(schema='qregion.numeric-precision.v1',training_precision='FP32',
                diagnostic_precision='FP64/complex128',epsilon_fp64=.00001,
                same_fixed_two_training_questions=True,rows=rows,passed=passed,
                scientific_utility_claim=False)
