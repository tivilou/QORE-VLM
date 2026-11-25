import copy
import unittest
import torch
from torch.nn import functional as F
from applications.rag.toy100_local_born_v001 import ARMS,input_dephased_intensity,LocalActivation,factory
from applications.rag.toy100_global_ffn_v001 import circuit
from applications.rag.toy100_output_frame_v001 import output_frames
from applications.rag.toy100_ffn_export_v001 import frozen_tail


def dense_local(angles):
    result=angles.new_ones(1,1)
    for q in reversed(range(angles.shape[1])):
        angle=angles[:,q].sum()/2;c,s=angle.cos(),angle.sin()
        result=torch.kron(result,torch.stack((torch.stack((c,-s)),torch.stack((s,c)))))
    return result


class LocalBornTests(unittest.TestCase):
    def test_exact_pure_mixed_density_oracles_gradients_and_boundaries(self):
        torch.manual_seed(147);x=torch.randn(5,8,dtype=torch.float64)
        angles=torch.randn(2,3,dtype=torch.float64,requires_grad=True);u=dense_local(angles)
        pure=circuit(x,angles,False).square();pure_oracle=(x@u.T).square()
        mixed=input_dephased_intensity(x,angles);mixed_oracle=x.square()@u.square().T
        torch.testing.assert_close(pure,pure_oracle,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(mixed,mixed_oracle,atol=1e-12,rtol=1e-12)
        a=torch.autograd.grad(mixed.square().sum(),angles,retain_graph=True)[0]
        b=torch.autograd.grad(mixed_oracle.square().sum(),angles,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
        self.assertGreater(float((pure-mixed).abs().max()),1e-4)
        self.assertLess(float(torch.autograd.grad(mixed.sum(),angles,retain_graph=True)[0].abs().max()),1e-11)
        sign=x.new_tensor([1,-1,1,1,-1,1,-1,1])
        self.assertTrue(torch.equal(input_dephased_intensity(x*sign,angles),mixed))
        onehot=x.new_zeros(2,8);onehot[0,1]=1;onehot[1,6]=-2
        torch.testing.assert_close(circuit(onehot,angles,False).square(),input_dephased_intensity(onehot,angles),atol=1e-12,rtol=1e-12)
        actual=torch.randn(2,1024,dtype=torch.float64);theta=torch.randn(2,10,dtype=torch.float64)
        torch.testing.assert_close(input_dephased_intensity(actual,theta).sum(-1),actual.square().sum(-1),atol=1e-9,rtol=1e-12)
        m=LocalActivation(dict(dpr=6,bge=8),'DEPHASED',20261222,wires=3,chunk=1)
        left=m.activation(x[:,:6].float(),'dpr');m.chunk=5
        torch.testing.assert_close(left,m.activation(x[:,:6].float(),'dpr'),atol=2e-6,rtol=2e-6)

    def test_every_arm_null_private_gradient_and_complete_state_continuation(self):
        torch.manual_seed(147);packed={};coords={};features={};native={}
        for k,d in [('dpr',6),('bge',8)]:
            h=4;z=torch.randn(2,5,d);res=torch.randn(2,5,h)
            dense=dict(weight=torch.randn(h,d)*.2,bias=torch.randn(h)*.1);norm=dict(weight=torch.ones(h),bias=torch.zeros(h))
            bridge=None if k=='dpr' else dict(weight=torch.randn(h,h)*.2,bias=torch.randn(h)*.1)
            f=frozen_tail(z,res,dense,norm,bridge,1e-5);weight=torch.randn(1,h)*.1;bias=torch.zeros(1);n=F.linear(f,weight,bias).squeeze(-1)
            packed[k]=dict(original_head=dict(weight=weight,bias=bias),features=dict(train=f),native=dict(train=n),
                ffn_z=dict(train=z),ffn_residual=dict(train=res),ffn_dense=dense,ffn_normalization=norm,ffn_bridge=bridge,ffn_eps=1e-5,
                ffn_frames=output_frames(dense['weight']))
            coords[k]=torch.cat((z,res),-1);features[k]=f;native[k]=n
        for arm in ARMS:
            make=lambda:factory(packed,arm,20261122,wires=2,chunk=3)
            m=make();out=m(features,native,native,coords,m.references(coords,native,features))
            self.assertTrue(all(torch.equal(out[k],native[k]) for k in out))
            if arm!='HEAD':self.assertFalse(any(n.startswith(('frame_','tails.','probe_')) for n in m.state_dict()))
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=0,foreach=False)
            def step(model,optimizer):
                optimizer.zero_grad(set_to_none=True);out=model(features,native,native,coords,model.references(coords,native,features))
                loss={k:F.softplus(-(v[:,0]-v[:,1])).mean() for k,v in out.items()}
                if arm!='HEAD':
                    target=model.shared.weights
                    for k in ('dpr','bge'):
                        g=torch.autograd.grad(loss[k],target,retain_graph=True)[0];start,size=model.shared.layout[k]
                        self.assertGreater(float(g[start:start+size].abs().max()),0)
                        self.assertEqual(float(torch.cat((g[:start],g[start+size:])).abs().max()),0)
                (sum(loss.values())/2).backward();params=[p for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params))
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            step(m,opt);restored=make();restored.load_state_dict(copy.deepcopy(m.state_dict()))
            other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.001,weight_decay=0,foreach=False)
            other.load_state_dict(copy.deepcopy(opt.state_dict()));step(m,opt);step(restored,other)
            for k in ('dpr','bge'):
                self.assertTrue(torch.equal(m.forward_one(k,features[k],native[k],native[k],coords[k]),
                                            restored.forward_one(k,features[k],native[k],native[k],coords[k])))


if __name__=='__main__':unittest.main()
