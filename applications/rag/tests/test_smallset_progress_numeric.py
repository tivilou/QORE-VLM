"""32 real tiny CPU optimizer steps with display events on/off; no assets."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn


class ProgressNumerics(unittest.TestCase):
    def test_step32_cache_log_and_events_preserve_optimizer_and_predictions(self):
        from applications.rag.top8_portable_v001 import train_cell
        class Journal:
            def load(self):return None
            def save(self,value):self.last=copy.deepcopy(value)
        torch.manual_seed(17);head=nn.Linear(8,1).eval().requires_grad_(False)
        features={str(i):dict(readout=torch.randn(50,8)) for i in range(4)}
        rows=[]
        for i in range(4):
            rows.append(dict(id=str(i),role='training' if i<2 else 'development',
                candidate_ids=list(range(50)),retained_indices=list(range(50)),targets=[j%2 for j in range(50)],
                native_scores=head(features[str(i)]['readout']).detach().squeeze(-1).tolist(),
                sequences=[dict(positions=list(range(j,min(j+8,50)))) for j in range(0,50,8)]))
        manifest=dict(rows=rows,label_classes=2)
        real_tensor=torch.tensor
        def tensor(*a,**kw):
            if kw.get('device')=='cuda':kw=dict(kw,device='cpu')
            return real_tensor(*a,**kw)
        outputs=[];saved=[]
        with tempfile.TemporaryDirectory() as d,\
             patch('applications.rag.top8_backbone_v001.load_modules',return_value=(None,nn.Identity(),head)),\
             patch.object(nn.Module,'cuda',lambda self,*a,**kw:self),\
             patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self),\
             patch('torch.tensor',side_effect=tensor),patch('torch.cuda.get_rng_state_all',return_value=[]):
            for enabled in ('0','1'):
                out=Path(d)/enabled;out.mkdir();journal=Journal();stream=io.StringIO()
                with patch.dict(os.environ,{'QDUET_PROGRESS_EVENTS':enabled}),contextlib.redirect_stdout(stream):
                    result=train_cell('FIDELITY','HEAD',17,manifest,features,Path(d),out,32,None,lambda:None,journal)
                self.assertEqual(result['steps'],32)
                lines=[json.loads(s) for s in stream.getvalue().splitlines()]
                log=next(e for e in lines if e.get('stage')=='training')
                self.assertEqual(log['step'],32);self.assertIsNone(log['sequence_cache'])
                if enabled=='1':
                    self.assertTrue(any(e.get('checkpoint_step')==32 for e in lines))
                    self.assertTrue(any(e.get('unit')=='questions_checked' for e in lines))
                pred=next(n for n in result['files'] if n['name'].startswith('predictions'))
                outputs.append((out/pred['name']).read_bytes());saved.append(journal.last)
            self.assertEqual(outputs[0],outputs[1])
            self.assertTrue(torch.equal(saved[0]['parameters']['head.weight'],saved[1]['parameters']['head.weight']))
            for k,value in saved[0]['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(value,saved[1]['optimizer']['state'][0][k]))


if __name__=='__main__':unittest.main()
