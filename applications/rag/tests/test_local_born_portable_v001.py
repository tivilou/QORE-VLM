"""Numerical adapter/recovery delta tests. Execute on allocated GPU only."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from applications.rag.toy100_local_born_v001 import factory,ARMS
from applications.rag.toy100_output_frame_v001 import output_frames
from applications.rag.toy100_ffn_export_v001 import frozen_tail
from applications.rag.local_born_cache_v001 import QuestionStore,new_tensor
from applications.rag.local_born_io_v001 import immutable_json,RecoveryJournal
from applications.rag.local_born_smallset_v001 import policy,run_cell,score,schedule


def fixture(root):
    torch.manual_seed(14701);packed={};roots={};manifests={};rows=[]
    for i in range(16):
        ps=[dict(id=f'p{j}',text='synthetic') for j in range(50)]
        row=dict(id=f'q{i}',component_id=f'c{i}',question='synthetic '+str(i),paragraphs=ps)
        if i<8:row['labels']={p['id']:int(j%3==0) for j,p in enumerate(ps)}
        rows.append(row)
    for k,d in [('dpr',6),('bge',8)]:
        h=4;z=torch.randn(16,50,d);res=torch.randn(16,50,h)
        dense=dict(weight=torch.randn(h,d)*.2,bias=torch.randn(h)*.1);norm=dict(weight=torch.ones(h),bias=torch.zeros(h))
        bridge=None if k=='dpr' else dict(weight=torch.randn(h,h)*.2,bias=torch.randn(h)*.1)
        f=frozen_tail(z,res,dense,norm,bridge,1e-5);w=torch.randn(1,h)*.1;b=torch.zeros(1);n=F.linear(f,w,b).squeeze(-1)
        packed[k]=dict(original_head=dict(weight=w,bias=b),features=dict(train=f[:8]),native=dict(train=n[:8]),ffn_z=dict(train=z[:8]),
            ffn_dense=dense,ffn_normalization=norm,ffn_bridge=bridge,ffn_eps=1e-5,ffn_frames=output_frames(dense['weight']))
        roots[k]=root/k;roots[k].mkdir();metadata=copy.deepcopy(packed[k])
        for name in ('features','native','ffn_z'):metadata[name]['train']=metadata[name]['train'][:1]
        qs=[]
        for i in range(16):qs.append(dict(tensor=new_tensor(roots[k],f'q{i}',dict(z=z[i],residual=res[i],features=f[i],native=n[i]))))
        manifests[k]=dict(metadata=new_tensor(roots[k],'metadata',metadata),questions=qs)
    return packed,QuestionStore(roots,manifests,4096),rows


def equal_state(left,right):
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):policy()

    def test_all_arm_bookkeeping_adapter_forward_gradient_update(self):
        with tempfile.TemporaryDirectory() as d:
            packed,store,_=fixture(Path(d))
            for arm in ARMS:
                full=factory(packed,arm,20261122).cuda().eval();shard=factory(store.packed,arm,20261122).cuda().eval()
                a,na=score(full,store,list(range(8)));b,nb=score(shard,store,list(range(8)))
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
                (sum(v[:,0].sum()-v[:,1].sum() for v in a.values())/2).backward()
                (sum(v[:,0].sum()-v[:,1].sum() for v in b.values())/2).backward()
                for (n,p),(m,q) in zip(full.named_parameters(),shard.named_parameters()):
                    self.assertEqual(n,m)
                    if p.requires_grad:self.assertTrue(torch.equal(p.grad,q.grad))
                opt=lambda model:torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0,foreach=False)
                opt(full).step();opt(shard).step()
                a,_=score(full,store,list(range(8,16)));b,_=score(shard,store,list(range(8,16)))
                self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_interrupted_full_state_continuation_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,store,rows=fixture(root);bound=dict(test='fixed16synthetic')
            complete=run_cell(root/'complete',bound,store,rows,8,'QUANTUM',20261122,2,1)
            real_save=RecoveryJournal.save
            def save_then_stop(journal,value):
                real_save(journal,value)
                if value['step']==1:raise InterruptedError('injected after durable step1')
            with patch.object(RecoveryJournal,'save',save_then_stop):
                with self.assertRaises(InterruptedError):run_cell(root/'resumed',bound,store,rows,8,'QUANTUM',20261122,2,1)
            resumed=run_cell(root/'resumed',bound,store,rows,8,'QUANTUM',20261122,2,1)
            a=RecoveryJournal(root/'complete/journal',bound).load();b=RecoveryJournal(root/'resumed/journal',bound).load()
            self.assertTrue(equal_state(a,b));self.assertEqual(resumed['resumed_step'],1)
            for k in ('dpr','bge'):
                self.assertEqual(json.loads((root/'complete'/complete['predictions'][k]['name']).read_bytes()),json.loads((root/'resumed'/resumed['predictions'][k]['name']).read_bytes()))
            with patch('applications.rag.local_born_smallset_v001.factory',side_effect=AssertionError('completed cells must not reconstruct or train')):
                self.assertEqual(run_cell(root/'resumed',bound,store,rows,8,'QUANTUM',20261122,2,1),resumed)
            with self.assertRaises(RuntimeError):RecoveryJournal(root/'resumed/journal',dict(changed=True))
            bad=root/'unbound';bad.mkdir();(bad/'step-00000001.json').write_text('{}')
            with self.assertRaises(RuntimeError):RecoveryJournal(bad,bound)


if __name__=='__main__':unittest.main()
