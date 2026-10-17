"""Train-only local feasibility diagnostics, never an online selector.

LP infeasibility applies only to a fixed sufficient Top5 witness template and
trust box. It is not proof that safe shared parameters cannot exist.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import linprog


@torch.no_grad()
def score_jacobian(head, feature):
    """Exact adjoint for per-passage scores vs shared rotation coefficients.

    Each passage has its own state and angle but the same theta. Reverse gate
    propagation avoids a cohort-sized autograd graph/Jacobian of hidden states.
    """
    output = head(feature['cls'], feature['pooled'])
    n, gates = output['angles'].shape
    adjoint = torch.nn.functional.pad(head.weight, (0, head.state_size-head.hidden_size)).expand(n, -1).clone()
    angle_grad = torch.empty_like(output['angles'])
    for gate in range(gates-1, -1, -1):
        low, high = getattr(head, f'low_{gate}'), getattr(head, f'high_{gate}')
        before = output['states'][:, gate] * output['norm']
        a, z = before[:, low], before[:, high]
        gl, gh = adjoint[:, low].clone(), adjoint[:, high].clone()
        c = torch.cos(output['angles'][:, gate:gate+1]/2)
        s = torch.sin(output['angles'][:, gate:gate+1]/2)
        angle_grad[:, gate] = .5*(gl*(-s*a-c*z)+gh*(c*a-s*z)).sum(-1)
        adjoint[:, low], adjoint[:, high] = c*gl+s*gh, -s*gl+c*gh
    conditions = torch.cat((torch.ones_like(output['conditioning'][:, :1]), output['conditioning']), dim=1)
    jacobian = (angle_grad[:, :, None]*conditions[:, None, :]).reshape(n, -1)
    if not bool(jacobian.isfinite().all()): raise ValueError('nonfinite Jacobian')
    return output, jacobian


def rescue_targets(base, targets, limit):
    """Fixed smallest baseline entry gap; ties use source question/index order."""
    choices=[]
    for case, target in enumerate(targets):
        baseline=target['baseline_top5_indices']
        disposable=[i for i in baseline if i not in target['immutable_selected_indices']]
        if not disposable: continue
        removed=min(disposable, key=lambda i: (float(base[case,i]), -i))
        scale=np.subtract(*np.percentile(base[case], [75,25]))/1.349
        if scale <= 1e-8: raise ValueError('collapsed baseline scale')
        for added in target['eligible_positive_indices']:
            if added in baseline: raise ValueError('eligible candidate is already selected')
            choices.append({'case':case,'added':added,'removed':removed,
                'baseline_gap':float((base[case,removed]-base[case,added])/scale)})
    return sorted(choices,key=lambda r:(r['baseline_gap'],r['case'],r['added']))[:limit]


def boundary_system(scores, jacobians, base, targets, proposal, *, protected, margin):
    """A delta >= b, with no ordering imposed within a target Top5 set.

    On the rescue question require all5 target members to beat outsiders. On
    other questions require protected baseline members to beat baseline
    outsiders. This is sufficient, not necessary, for safe membership.
    """
    matrices, rhs, lineage=[],[],[]
    for case, target in enumerate(targets):
        is_rescue=case==proposal['case']
        if not protected and not is_rescue: continue
        baseline=set(target['baseline_top5_indices'])
        if is_rescue:
            members=sorted((baseline-{proposal['removed']})|{proposal['added']})
            outsiders=sorted(set(range(50))-set(members))
        else:
            members=target['immutable_selected_indices']
            outsiders=sorted(set(range(50))-baseline)
        scale=float(np.subtract(*np.percentile(base[case],[75,25]))/1.349)
        for kept in members:
            for outside in outsiders:
                matrices.append((jacobians[case,kept]-jacobians[case,outside])/scale)
                rhs.append(margin-(scores[case,kept]-scores[case,outside])/scale)
                lineage.append((case,kept,outside))
    return np.asarray(matrices,dtype=np.float64),np.asarray(rhs,dtype=np.float64),np.asarray(lineage,dtype=np.int64)


def bounded_proposal(matrix, rhs, radius, time_limit):
    """Minimum infinity-norm shared delta in the fixed box, using CPU HiGHS."""
    if matrix.ndim!=2 or rhs.shape!=(len(matrix),) or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError('invalid constraints')
    p=matrix.shape[1]
    # delta <= rho and -delta <= rho; minimizing rho avoids oversized steps.
    identity=np.eye(p)
    a=np.vstack((np.column_stack((-matrix,np.zeros(len(matrix)))),
        np.column_stack((identity,-np.ones(p))),np.column_stack((-identity,-np.ones(p)))))
    b=np.concatenate((-rhs,np.zeros(2*p)))
    result=linprog(np.r_[np.zeros(p),1.],A_ub=a,b_ub=b,
        bounds=[(-radius,radius)]*p+[(0,radius)],method='highs',
        options={'time_limit':time_limit,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    report={'status':int(result.status),'message':result.message,'success':bool(result.success)}
    if not result.success:return report,None
    delta=result.x[:-1]
    slack=matrix@delta-rhs
    if slack.min() < -2e-7 or np.abs(delta).max()>radius+1e-8:
        raise ValueError('solver returned invalid feasible proposal')
    report.update(infinity_norm=float(np.abs(delta).max()),minimum_linear_slack=float(slack.min()))
    return report,delta


def membership_counts(score_matrix, targets, ranks):
    selected=[sorted(range(50),key=lambda j:(-float(s[j]),ranks[i][j]))[:5] for i,s in enumerate(score_matrix)]
    return {'rescued':sum(len(set(s)&set(t['eligible_positive_indices'])) for s,t in zip(selected,targets)),
        'lost_protected':sum(len(set(t['immutable_selected_indices'])-set(s)) for s,t in zip(selected,targets)),
        'selected_indices':selected}
