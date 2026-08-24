# Multi-Host Execution Note

This note records an execution-boundary decision; it is not yet a simulator contract.

- The canonical Git repository, public datasets, WSL environments, MuJoCo work, audit artifacts, and statistical evaluation remain under `/mnt/g/CryoLocoManip`.
- MuJoCo in WSL is the preferred path for early mechanism tests, regression tests, and lightweight dynamics experiments.
- Isaac Sim / Isaac Lab already installed in the native Ubuntu dual-boot system may be used later for high-fidelity GPU-parallel simulation.
- Before Stage 0D begins, a dedicated multi-host contract must define native-Ubuntu code checkout/storage, commit synchronization, simulator and asset identities, run manifests, result transfer, and cross-simulator comparison semantics.
- Isaac Sim / Isaac Lab must not be forced to run from NTFS/DrvFS when that would compromise performance or simulator support.
- No Stage 0B.2 result depends on Isaac Sim, Isaac Lab, MuJoCo, or the native-Ubuntu host.
