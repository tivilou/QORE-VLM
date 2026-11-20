"""Aggregate diagnostic transfers; preserve per-question backward/AdamW order."""
from applications.rag.qvalue_portable_v001 import need


def step_diagnostics(losses, counts, parameters, denominator, *, fusion=False, empty=0):
    """One host packet for loss/count summaries and pre-update gradient gates.

    Reject every nonfinite loss and gradient BEFORE optimizer update. FP32 loss values
    become Python doubles exactly as before, then accumulate in original order;
    no GPU reduction substitutes for that host accumulation. No live graphs are
    retained. Clip/AdamW are called only after all original gradient gates pass.
    """
    import torch
    need(denominator>0 and len(losses)==len(counts),'complete diagnostic batch')
    parameters=list(parameters);grads=[p.grad for p in parameters if p.grad is not None]
    if not losses:
        need(fusion and not grads,'empty fusion batch only')
        return 0.,0,empty
    # Each integer count is <=50*49; promotion to double is exact.
    values=[x.detach().reshape(()).to(torch.float64) for x in losses]
    values += [x.detach().reshape(()).to(torch.float64) for x in counts]
    if grads:
        finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
        maximum=torch.stack([g.abs().max() for g in grads]).max()
    else:
        finite=values[0].new_tensor(1.);maximum=values[0].new_tensor(0.)
    values += [finite.to(torch.float64),maximum.to(torch.float64)]
    packet=torch.stack(values).cpu().tolist()
    n=len(losses);pairs=sum(int(x) for x in packet[n:2*n])
    import math
    need(all(math.isfinite(x) for x in packet[:n]),'finite fusion objective' if fusion else 'finite training-only objective')
    need(bool(packet[-2]),'finite fusion gradients' if fusion else 'finite data gradients')
    need((bool(grads) or fusion) and (pairs==0 or packet[-1]>0),
         'active fusion objective' if fusion else 'nonzero actual objective gradient when pairs exist')
    total=0.
    for value in packet[:n]:total+=value/denominator
    without=empty+sum(int(x==0) for x in packet[n:2*n])
    return total,pairs,without


def finite_parameters(parameters):
    """Single host predicate; keep the post-update nonfinite gate."""
    import torch
    flags=[torch.isfinite(p).all() for p in parameters]
    return not flags or bool(torch.stack(flags).all())


def head_sequence_rows(record, features, checkpoint):
    """Readout-only HEAD, same retained positions and original chunk sizes.

    Full hidden/mask/QKV are unused by this arm. Native FP32 anchor values come
    from the already SHA-bound cache manifest, indexed by ORIGINAL positions.
    The trainable head is still evaluated per original chunk, never as a merged
    full-question matrix multiplication. Full-attention arms keep their reader.
    """
    import torch
    from applications.rag.top8_portable_v001 import gpu_row
    offset=0;readout=features['readout'];keep=record['retained_indices']
    need(len(readout)==len(keep),'complete HEAD readout/retained alignment')
    for entry in record['sequences']:
        checkpoint();positions=entry['positions'];n=len(positions)
        need(keep[offset:offset+n]==positions,'complete HEAD sequence/feature alignment')
        row=gpu_row({'readout':readout},slice(offset,offset+n))
        native=torch.tensor([record['native_scores'][i] for i in positions],
                            dtype=row['readout'].dtype,device=row['readout'].device)
        yield row,native
        offset+=n
    need(offset==len(keep),'full HEAD chunk reader exhausted')
