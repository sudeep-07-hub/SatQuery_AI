# ChangeMamba Provenance Document

## Source Information
- **Source**: ChenHongruixuan / ChangeMamba
- **Repository**: [https://github.com/ChenHongruixuan/ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba)
- **Model Architecture**: Visual State-Space Model (VSS) / Siamese encoder-decoder tailored for bi-temporal remote sensing.
- **License**: Unknown / Typically Research-Only (to be verified upon instantiation)
- **Integration Date**: 2026-09-14 (Attempted)

## Dependencies & Hardware
- **Framework**: PyTorch (`torch`)
- **State-Space Primitives**: `mamba-ssm`, `causal-conv1d`
- **Hardware Requirements**: NVIDIA GPU + CUDA + NVCC compiler. (The `mamba-ssm` package requires CUDA-accelerated causal convolution kernels and selective scan kernels to compile).

## Checkpoint Details
- **Checkpoint**: N/A
- **Revision**: N/A
- **Status**: **NOT LOADED**. The execution environment (macOS arm64 MPS) mechanically lacks CUDA, blocking the compilation of `mamba-ssm` and the instantiation of the genuine architecture.

## Contract Capabilities
- **Input Contract**: Two co-registered `[B, C, H, W]` image tensors.
- **Output Contract**: Binary change masks, changed region bounding boxes, and uncalibrated change confidence.

## Limitations
- Execution is strictly impossible on non-CUDA architectures without heavily mocked/compiled kernel stubs which would violate the research-validity constraints.
