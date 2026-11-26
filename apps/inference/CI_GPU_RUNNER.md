# GPU CI runner note

Short operator note for the `Inference - GPU Tests (vLLM + CUDA)` workflow.

- Trigger paths: `apps/inference/**` and the workflow file itself
- Runner label comes from the repository variable `INFERENCE_GPU_RUNS_ON`
- The first job step runs `nvidia-smi` so a missing driver fails fast
