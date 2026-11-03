"""Native CLS-to-token fidelity adapter; import only inside admitted worker.

Three qubits, q0 least significant, complex data encoding. All padding-valid
keys are eligible; cached prefixes do not contain original segment identities.
No quantum hardware or classical complexity advantage is claimed.
"""
from __future__ import annotations
import copy
import inspect
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

ARMS = ('HEAD', 'MLP', 'GIVENS', 'PRODUCT', 'QUANTUM')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def rotate(state, angle, pairs):
    # Right multiplication of row vectors by the transpose of the gate.
    out = state.clone()
    i = torch.tensor([p[0] for p in pairs], device=state.device)
    j = torch.tensor([p[1] for p in pairs], device=state.device)
    c, s = torch.cos(angle/2), torch.sin(angle/2)
    out[..., i] = c*state[..., i]-s*state[..., j]
    out[..., j] = s*state[..., i]+c*state[..., j]
    return out


def critic(state, theta, kind):
    planes = ((0,3),(1,6),(2,5),(0,7),(1,4),(3,6),
              (0,5),(1,7),(2,6),(3,4),(0,6),(2,7))
    for layer in range(2):
        for q in range(3):
            pairs = tuple((i,i+(1<<q)) for i in range(8) if not i&(1<<q))
            if kind == 'GIVENS': pairs = (planes[6*layer+q],)
            state = rotate(state, theta[6*layer+q], pairs)
        for n,(control,target) in enumerate(((0,1),(1,2),(2,0))):
            pairs = tuple((i,i+(1<<target)) for i in range(8)
                          if not i&(1<<target) and (kind=='PRODUCT' or i&(1<<control)))
            if kind == 'GIVENS': pairs = (planes[6*layer+3+n],)
            state = rotate(state, theta[6*layer+3+n], pairs)
    return state


def encoding(state, z):
    """E2 tensor E1 tensor E0; each Eq=RZ(zq) RY(zq)."""
    for q in range(3):
        pairs = tuple((i,i+(1<<q)) for i in range(8) if not i&(1<<q))
        i = torch.tensor([p[0] for p in pairs],device=state.device)
        j = torch.tensor([p[1] for p in pairs],device=state.device)
        c,s = torch.cos(z[...,q:q+1]/2),torch.sin(z[...,q:q+1]/2)
        a,b = state[...,i],state[...,j]
        out = state.clone()
        out[...,i] = (c*a-s*b)*torch.exp(-.5j*z[...,q:q+1])
        out[...,j] = (s*a+c*b)*torch.exp(.5j*z[...,q:q+1])
        state = out
    return state


class Fidelity(nn.Module):
    def __init__(self, kind, seed, device):
        super().__init__(); self.kind=kind
        g=torch.Generator(device='cpu').manual_seed(seed)
        self.projection=nn.Parameter((torch.randn(3,64,generator=g)/8).to(device))
        self.bias=nn.Parameter((torch.randn(3,generator=g)*.05).to(device))
        if kind=='MLP':
            self.w1=nn.Parameter((torch.randn(4,3,generator=g)/math.sqrt(3)).to(device))
            self.b1=nn.Parameter((torch.randn(4,generator=g)*.05).to(device))
            self.w2=nn.Parameter((torch.randn(8,4,generator=g)/2).to(device))
            self.b2=nn.Parameter((torch.randn(8,generator=g)*.05).to(device))
        else:
            self.theta=nn.Parameter((torch.randn(12,generator=g)*.05).to(device))
        for name,p in list(self.named_parameters()):
            self.register_buffer('reference_'+name,p.detach().clone())

    def parameter_value(self,name,reference):
        return getattr(self,'reference_'+name) if reference else getattr(self,name)

    def states(self, x, reference=False):
        p=lambda name:self.parameter_value(name,reference)
        z=math.pi*torch.tanh(F.linear(F.layer_norm(x,(64,)),p('projection'),p('bias')))
        if self.kind=='MLP':
            raw=F.linear(torch.tanh(F.linear(z,p('w1'),p('b1'))),p('w2'),p('b2'))
            norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
            need(bool((norm>1e-8).all()),'nonzero real classical feature states')
            return raw/norm
        state=torch.zeros(z.shape[:-1]+(8,),device=z.device,dtype=torch.complex64 if z.dtype==torch.float32 else torch.complex128)
        state[...,0]=1
        state=encoding(state,z)
        state=critic(state,p('theta'),self.kind)
        return encoding(state,z)

    def kernel(self,q,k,reference=False):
        b,h,l,_=k.shape
        both=torch.cat((q.unsqueeze(2),k),2)
        s=self.states(both,reference)
        overlap=(s[:,:,0:1].conj()*s[:,:,1:]).sum(-1)
        result=overlap.abs().square()
        need(bool(torch.isfinite(result).all()) and bool((result<=1.00002).all()),'bounded finite state fidelity')
        return result

    def forward(self,q,k):
        with torch.no_grad(): reference=self.kernel(q,k,True)
        return self.kernel(q,k)-reference


def prepare_row(model,row):
    """Original penultimate layer is fixed and shared across all arms."""
    hidden=row['hidden'].cuda()
    mask=None if row['mask'] is None else row['mask'].cuda().expand(row['mask_shape']).contiguous()
    layers=model.span_predictor.encoder.bert_model.encoder.layer
    with torch.no_grad():
        hidden=layers[-2](hidden,attention_mask=mask,output_attentions=False)[0]
        att=layers[-1].attention.self
        q=att.query(hidden).reshape(len(hidden),-1,12,64).transpose(1,2)[:,:,0].contiguous()
        k=att.key(hidden).reshape(len(hidden),-1,12,64).transpose(1,2).contiguous()
        length=hidden.shape[1]
        zeros=torch.zeros((len(hidden),12,length,length),device=hidden.device,dtype=hidden.dtype)
        if mask is None: zero_mask=zeros
        elif mask.dtype==torch.bool: zero_mask=zeros.masked_fill(~mask,torch.finfo(hidden.dtype).min)
        else: zero_mask=zeros+mask
        with sdpa_kernel(SDPBackend.MATH):
            math_cls=layers[-1](hidden,attention_mask=zero_mask,output_attentions=False)[0][:,0]
            math_scores=model.span_predictor.qa_classifier(math_cls).squeeze(-1)
        expected=row['native_scores'].cuda()
        need(torch.allclose(math_scores,expected,atol=1e-5,rtol=1e-6),'every-chunk original native vs zero-bias explicitmath parity')
        math_error=float((math_scores-expected).abs().max())

    need(not hidden.requires_grad and not q.requires_grad and not k.requires_grad,'frozen actual native Q/K')
    return dict(hidden=hidden,mask=mask,q=q,k=k,cls=row['cls'].cuda(),indices=row['indices'],native_scores=row['native_scores'],math_zero_scores=math_scores,native_math_max_error=math_error)


class Scorer(nn.Module):
    def __init__(self,kind,head,seed):
        super().__init__();self.kind=kind
        self.head=copy.deepcopy(head);self.head.weight.requires_grad_(True);self.head.bias.requires_grad_(False)
        self.adapter=None if kind=='HEAD' else Fidelity(kind,seed,head.weight.device)

    @contextmanager
    def install(self,model,row,zero=False,capture=None):
        att=model.span_predictor.encoder.bert_model.encoder.layer[-1].attention.self
        need(type(att).__name__=='BertSdpaSelfAttention' and not att._forward_hooks and not att._forward_pre_hooks,'untouched native SDPA module')
        signature=inspect.signature(att.forward)
        delta=torch.zeros(row['k'].shape[:-1],device=row['k'].device) if zero else self.adapter(row['q'],row['k'])
        length=row['hidden'].shape[1]
        selector=F.one_hot(torch.tensor(0,device=delta.device),num_classes=length).to(delta.dtype).reshape(1,1,length,1)
        def hook(module,args,kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            need(bound.arguments['output_attentions'] is False,'no eager attention fallback')
            original=bound.arguments.get('attention_mask')
            if original is None: base=0.
            elif original.dtype==torch.bool:
                base=torch.zeros_like(original,dtype=delta.dtype).masked_fill(~original,torch.finfo(delta.dtype).min)
            else: base=original
            bias=delta.unsqueeze(2)*selector
            if original is not None:
                valid=original if original.dtype==torch.bool else original==0
                bias=bias*valid.to(bias.dtype)
            mask=base+bias
            need(bool(torch.isfinite(delta).all()) and bool((delta.abs()<=1.00002).all()),'fixed scale range')
            if capture is not None:
                capture.update(delta=delta,mask=mask,original=original)
                if delta.requires_grad: delta.retain_grad();mask.retain_grad()
            bound.arguments['attention_mask']=mask
            return bound.args,bound.kwargs
        handle=att.register_forward_pre_hook(hook,with_kwargs=True)
        try: yield
        finally: handle.remove()

    def hidden(self,model,row,zero=False,capture=None):
        if self.kind=='HEAD':return row['cls']
        with sdpa_kernel(SDPBackend.MATH), self.install(model,row,zero,capture):
            return model.span_predictor.encoder.bert_model.encoder.layer[-1](row['hidden'],attention_mask=row['mask'],output_attentions=False)[0][:,0]

    def forward(self,model,row):return self.head(self.hidden(model,row)).squeeze(-1)


def question_scores(scorer,model,rows):
    return torch.cat([scorer(model,row) for row in rows])


def probe(scorer,model,cases,rows_by_question,targets,ordinal_loss,profile=False):
    """Fixed first two complete train questions, head held fixed for FD."""
    def objective():
        total=0.
        for case in cases:
            rows=rows_by_question[case.question_sha256]
            ys=dict(targets.rows_by_question[case.question_sha256])
            tiers=torch.tensor([ys[i] for row in rows for i in row['indices']],device='cuda')
            total=total+ordinal_loss(question_scores(scorer,model,rows),tiers)[0]/len(cases)
        return total
    null=0.
    with torch.no_grad():
        for rows in rows_by_question.values():
            for row in rows:
                actual=scorer(model,row);expected=row['native_scores'].cuda()
                need(torch.allclose(actual,expected,atol=1e-5,rtol=1e-6),'native initial frozen-reference null')
                null=max(null,float((actual-expected).abs().max()))
                if scorer.kind!='HEAD':
                    zero=scorer.head(scorer.hidden(model,row,zero=True)).squeeze(-1)
                    need(torch.allclose(zero,expected,atol=1e-5,rtol=1e-6),'explicit zero native plumbing')
    scorer.zero_grad(set_to_none=True)
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            loss=objective();loss.backward();torch.cuda.synchronize()
        names=sorted({e.key for e in prof.key_averages()})
        need('aten::scaled_dot_product_attention' in names and 'aten::_scaled_dot_product_attention_math' in names and any('backward' in x.lower() and ('attention' in x.lower() or 'softmax' in x.lower()) for x in names),'actual explicitmath attention forward and backward')
    else:
        loss=objective();loss.backward();names=[]
    grads={n:float(p.grad.abs().max()) for n,p in scorer.named_parameters() if p.requires_grad and p.grad is not None}
    need(len(grads)==sum(p.requires_grad for p in scorer.parameters()) and all(torch.isfinite(p.grad).all() for p in scorer.parameters() if p.requires_grad),'complete finite gradient ownership')
    need(grads['head.weight']>0,'actual head loss gradient')
    report=dict(loss=float(loss.detach()),gradient_maxima=grads,max_null_error=null,native_null_checked=True,actual_training_questions=len(cases),kernel_events=names,representative_sampling_only=True,forced_backend=(scorer.kind!='HEAD'),backend_policy='SDPA_MATH for biased variants; original native anchor unchanged',full_trace_gate='DEFERRED_NOT_PASSED')
    if scorer.kind!='HEAD':
        for name in ('projection','bias')+(() if scorer.kind=='MLP' else ('theta',)):
            need(grads['adapter.'+name]>0,'active fidelity projection/bias/critic')
        p=scorer.adapter.projection if scorer.kind=='MLP' else scorer.adapter.theta
        at=int(p.grad.abs().argmax());expected=float(p.grad.flatten()[at]);old=float(p.detach().flatten()[at]);values=[]
        for sign in (-1,1):
            with torch.no_grad():p.flatten()[at]=old+sign*.001;values.append(float(objective()))
        with torch.no_grad():p.flatten()[at]=old
        fd=(values[1]-values[0])/.002
        need(abs(fd-expected)<=1e-4+.05*abs(expected),'actual native fidelity loss finite difference')
        # Independent of the head update: perturb only this adapter and verify
        # it changes valid-key contrasts rather than a softmax-null row offset.
        row=next(iter(rows_by_question.values()))[0]
        with torch.no_grad():
            reference_before=scorer.adapter.kernel(row['q'],row['k'],True).clone()
            p.flatten()[at]=old+.001
            delta=scorer.adapter(row['q'],row['k'])
            valid=torch.ones_like(delta,dtype=torch.bool) if row['mask'] is None else (row['mask'][:,:,0,:] if row['mask'].dtype==torch.bool else row['mask'][:,:,0,:]==0).expand_as(delta)
            means=(delta*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True)
            centered=(delta-means)*valid
            centered_max=float(centered.abs().max())
            p.flatten()[at]=old
            need(torch.equal(reference_before,scorer.adapter.kernel(row['q'],row['k'],True)),'independently frozen full reference')
        need(centered_max>0,'nonconstant valid-key bias sensitivity')
        # Observe real bias and mask gradients through native SDPA on a fixed
        # training chunk; no attention probabilities are substituted.
        captured={};scorer.zero_grad(set_to_none=True)
        h=scorer.hidden(model,row,capture=captured);scorer.head(h).sum().backward()
        need(captured['delta'].grad is not None and captured['mask'].grad is not None and bool(torch.isfinite(captured['delta'].grad).all()),'native mask and delta backward observed')
        dgrad=float(captured['delta'].grad.abs().max());need(dgrad>0,'nonzero real bias backward')
        report.update(finite_difference=fd,autograd=expected,finite_difference_error=abs(fd-expected),perturbed_centered_bias_max=centered_max,observed_delta_gradient_max=dgrad,actual_mask_backward=True)
        captured.clear();del h
    scorer.zero_grad(set_to_none=True)
    need(not model.span_predictor.encoder.bert_model.encoder.layer[-1].attention.self._forward_pre_hooks,'scoped hook cleanup')
    return report


def circuit_diagnostics(seed):
    # Independent scalar Kronecker implementation; admitted P4 only.
    g=torch.Generator().manual_seed(seed)
    theta=torch.randn(12,generator=g,dtype=torch.float64)*.2
    z=torch.randn(4,3,generator=g,dtype=torch.float64)*.5
    def encode_dense(zrow):
        gates=[]
        for a in zrow:
            c,s=torch.cos(a/2),torch.sin(a/2)
            ry=torch.stack((torch.stack((c,-s)),torch.stack((s,c)))).to(torch.complex128)
            rz=torch.diag(torch.stack((torch.exp(-.5j*a),torch.exp(.5j*a))))
            gates.append(rz@ry)
        return torch.kron(torch.kron(gates[2],gates[1]),gates[0])
    result={}
    for kind in ('QUANTUM','PRODUCT','GIVENS'):
        planes=((0,3),(1,6),(2,5),(0,7),(1,4),(3,6),
                (0,5),(1,7),(2,6),(3,4),(0,6),(2,7))
        unitary=torch.eye(8,dtype=torch.complex128)
        for n,angle in enumerate(theta):
            slot=n%6;q=slot if slot<3 else ((0,1),(1,2),(2,0))[slot-3][1]
            control=None if slot<3 else ((0,1),(1,2),(2,0))[slot-3][0]
            pairs=[planes[n]] if kind=='GIVENS' else [(i,i+(1<<q)) for i in range(8) if not i&(1<<q) and (control is None or kind=='PRODUCT' or i&(1<<control))]
            gate=torch.eye(8,dtype=torch.complex128)
            for i,j in pairs:
                gate[i,i]=gate[j,j]=torch.cos(angle/2)
                gate[i,j]=-torch.sin(angle/2);gate[j,i]=torch.sin(angle/2)
            unitary=gate@unitary
        expected=[]
        for zr in z:
            e=encode_dense(zr);expected.append(e@unitary@e[:,0])
        initial=torch.zeros((4,8),dtype=torch.complex128);initial[:,0]=1
        actual=encoding(critic(encoding(initial,z),theta,kind),z)
        error=float((actual-torch.stack(expected)).abs().max())
        need(error<1e-12,'scalar Kronecker encoding and critic oracle')
        result[kind]=dict(scalar_encoding_error=error,unitarity_error=float((unitary.conj().T@unitary-torch.eye(8)).abs().max()),nominal_critic_angles=12,effective_product_angles=3 if kind=='PRODUCT' else None)
    return result
