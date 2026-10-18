# Separate orchestration, aggregation and live compute roles

- Date: 2026-10-18
- Scope: shared
- Status: active
- Supersedes: 2026-09-05-project-execution-topology.md
- Superseded By: None
- Gate: serverctl role resolves one explicitly selected host; live SSH, accelerator visibility, memory and disk checks pass before a compute job
- Authorization: authorizes project role binding, read-only capacity discovery, and in-scope experiment execution on the selected compute host; it does not authorize silent failover, unrelated data movement, service restarts, or destructive cleanup

## Decision

Use the historically named 3080Ti server as the stable orchestration, code-coordination, 18083 artifact-exchange and data-aggregation host. Do not depend on it for GPU execution. Bind actual accelerator work to the `gpu-training` role and select a currently available registered remote GPU host after live capacity checks; the current selection is the Titan host. Keep collaborator-local execution as a separate handoff path rather than the only experiment path.

## Reason

The 3080Ti endpoint remained reachable while its active container exposed no GPU and only a 1 GiB cgroup. Treating an inventory label as runtime capacity repeatedly stalled Reader inference. Registered compute hosts are also ephemeral: cached capabilities can differ from currently visible devices. Explicit role separation preserves the stable services while letting experiments use verified live capacity.

## Consequences

`development`, `orchestration`, and `artifact-exchange` resolve the 3080Ti host. `gpu-training` currently resolves `gpushare-i1-54996-titan5`; before every GPU job, verify SSH, `nvidia-smi`, free device memory, disk, environment, model/input availability, and exact target path. Rebinding to another GPU host is an explicit project action. Durable job and artifact paths belong in the owning workstream state/session, while host, port, key and fingerprint remain only in serverctl.
