"""Run on the collaborator runtime; does not load data/models or access a network."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from applications.rag.top8_methods_v001 import MixedScorer,build,objective
from applications.rag.qclass_fusion_v001 import Scorer,ARMS,loss_for


class BinaryNumerics(unittest.TestCase):
    def test_class_fusion_binary_and_legacy_three_class_simplex_and_gradients(self):
        torch.manual_seed(17)
        h=torch.randn(1,5,768);scores=torch.randn(1,5,2)
        for classes in (2,3):
            y=torch.tensor([[i%classes for i in range(5)]])
            for arm in ARMS:
                with self.subTest(classes=classes,arm=arm):
                    m=Scorer(arm,torch.randn(768),torch.tensor(0.),torch.zeros(2),torch.ones(2),17,1.,
                             priors=torch.ones(classes)/classes)
                    if arm!='HEAD':
                        p=m.probabilities(h,scores)
                        self.assertEqual(p.shape,(1,5,classes))
                        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(1,5),atol=1e-6))
                    loss,_=loss_for(m,h,y,scores);loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                    self.assertTrue(any(p.grad is not None and (p.grad.abs()>0).any() for p in m.parameters()))

    def test_ordinal_binary_likelihood_coarsens_useful_latent_outcomes(self):
        head=nn.Linear(16,1);head.requires_grad_(False)
        m=build('ORDINAL','QUANTUM',17,head,None,None,(),label_classes=2)
        row=dict(readout=torch.randn(5,16));y=torch.tensor([0,1,0,1,1])
        score=m(row);mask=y[:,None]>y[None,:]
        pair=(F.softplus(-(score[:,None]-score[None,:]))*mask).sum()/mask.sum()
        _,p=m.inner.effect(row['readout'])
        binary=torch.stack((p[:,0],p[:,1:].sum(-1)),-1)
        expected=pair-binary.gather(-1,y[:,None]).clamp_min(1e-7).log().mean()
        actual,_=objective(m,'ORDINAL',row,y)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))
        actual.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_mixed_coefficients_match_actual_supervised_classes(self):
        for classes in (2,3):
            head=nn.Linear(16,1);head.requires_grad_(False)
            m=MixedScorer(head,'QUANTUM',17,classes)
            row=dict(depths=torch.randn(5,3,16),mixed={'QUANTUM':torch.randn(5,classes)})
            self.assertEqual(m.coefficients.numel(),classes)
            loss=m(row).sum();loss.backward()
            self.assertTrue(torch.isfinite(m.coefficients.grad).all())


if __name__=='__main__':unittest.main()
