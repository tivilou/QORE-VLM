import numpy as np
import pytest
import torch
from applications.rag.quantum_reader_rotation import make_rotation_head,RotationConfig
from applications.rag.protected_update_feasibility import score_jacobian,bounded_proposal,boundary_system,membership_counts,rescue_targets


@pytest.mark.parametrize('family',['quantum','classical'])
@pytest.mark.parametrize('nonzero',[False,True])
def test_adjoint_matches_autograd_all_scores_and_parameters(family,nonzero):
    torch.manual_seed(7)
    head=make_rotation_head(torch.randn(8,dtype=torch.float64),.2,family=family,config=RotationConfig(layers=2))
    if nonzero:
        with torch.no_grad():head.theta.copy_(torch.randn_like(head.theta)*.2)
    feature={'cls':torch.randn(5,8,dtype=torch.float64),'pooled':torch.randn(5,40,dtype=torch.float64)}
    output,jac=score_jacobian(head,feature)
    native=head(feature['cls'],feature['pooled'])['scores']
    expected=torch.stack([torch.autograd.grad(native[i],head.theta,retain_graph=True)[0].flatten() for i in range(5)])
    torch.testing.assert_close(jac,expected,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(output['scores'],native)


def test_lp_feasible_and_infeasible_are_distinct():
    report,delta=bounded_proposal(np.array([[1.,0.],[0.,1.]]),np.array([.2,.1]),.3,5)
    assert report['success'] and np.all(delta>0) and np.isclose(abs(delta).max(),.2)
    report,delta=bounded_proposal(np.array([[1.],[-1.]]),np.ones(2),.3,5)
    assert not report['success'] and delta is None


def test_template_preserves_set_without_forcing_rescue_below_kept():
    base=np.linspace(5,-5,50)[None,:]
    target={'baseline_top5_indices':[0,1,2,3,4],'immutable_selected_indices':[0,1,2],
        'eligible_positive_indices':[5]}
    proposal=rescue_targets(base,[target],1)[0]
    assert proposal['removed']==4
    jac=np.eye(50)[None,:,:]
    a,b,lineage=boundary_system(base,jac,base,[target],proposal,protected=True,margin=.01)
    assert (0,0,5) not in map(tuple,lineage)  # no unnecessary intra-set ordering
    candidate=base.copy();candidate[0,5]=10
    assert np.min(a@(candidate-base).ravel()-b)>0
    counts=membership_counts(candidate,[target],[list(range(1,51))])
    assert counts['rescued']==1 and counts['lost_protected']==0


def test_global_constraints_include_other_questions():
    base=np.tile(np.linspace(5,-5,50),(2,1)); jac=np.tile(np.eye(50),(2,1,1))
    target={'baseline_top5_indices':[0,1,2,3,4],'immutable_selected_indices':[0,1,2],'eligible_positive_indices':[5]}
    proposal=rescue_targets(base,[target,target],1)[0]
    _,_,local=boundary_system(base,jac,base,[target,target],proposal,protected=False,margin=.01)
    _,_,global_=boundary_system(base,jac,base,[target,target],proposal,protected=True,margin=.01)
    assert set(local[:,0])=={0} and set(global_[:,0])=={0,1}


def test_complete_fixture_replays_native_rounding_and_rejects_false_witness(tmp_path):
    import argparse,json
    from scripts.collab.five_ideas.run_reader_protected_update_feasibility import run,validate_numeric_trace
    directory=run(argparse.Namespace(synthetic=True,output=tmp_path))
    assert validate_numeric_trace(directory)['status']=='passed'
    path=directory/'selector_trace.json'
    trace=json.loads(path.read_text())
    trial=trace['checkpoints'][0]['trials'][0]['nonlinear'][0]
    trial['safe_witness']=not trial['safe_witness']
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='false safe witness'):validate_numeric_trace(directory)
