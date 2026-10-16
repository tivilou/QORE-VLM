# V100s CPU experiments; 3080Ti artifact exchange

- Date: 2026-10-16
- Scope: rag-selector
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: Verified key authentication, adequate CPU memory/environment and existing trace/replay readiness before real fitting. A GPU is not required.
- Authorization: User 2026-10-16 permits experiments on V100s and explicitly clarifies this session is CPU-only. Earlier collaborator-only execution has this exception. No service migration or termination of unrelated workloads is authorized.

## Decision

Use the registered machine historically named V100s as a CPU experiment option. Bind it to this project's `cpu-experiments` role; keep development and GPU priority on 3080Ti. Keep 18083 on 3080Ti.

## Reason

The user offered this machine for experiments and then clarified its current CPU-only capacity. Cached Reader head fitting can use CPU; historical GPU inventory must not override the actual session or the user's clarification.

## Consequences

Verify CPU environment and available memory before launching. No dataset/model download, unrelated-process termination or exchange migration.

## Verified runtime and boundary

- serverctl project development retains 3080Ti first and adds `hcqns-117-50-194-228`; `cpu-experiments` resolves that machine, while `gpu-training` resolves 3080Ti. Registration, authorization and current availability remain separate.
- Key-only strict-host-check SSH succeeds to both registered hosts.3080Ti public18083 health reports`qore-upload-v4` and10GiB/object. The service and external token store stay there; no token is transferred to V100s.
- Current session is confirmed CPU-only by the user. A stdlib-only probe found Python 3.10.12, no Torch/NumPy in default Python, a 1,073,741,824-byte cgroup limit, 1,062,998,016-byte usage, and 681,873,408 bytes of anonymous memory. Some usage is reclaimable cache, but the existing anonymous allocation leaves insufficient headroom for the observed Torch workload. No new experiment or heavy import was launched there.
- Do not reuse existing directories, inspect unrelated project content, terminate workloads or install packages blindly. No project code path has been established on this CPU machine.
- Disk-backed trace storage and sequential CPU engineering on 3080Ti now pass actual frozen-cache and synthetic optimizer replay without new OOM kills. The fixed real screen is ready for a suitably provisioned CPU environment; the V100s option awaits sufficient free memory/dependencies, not GPU restoration.
