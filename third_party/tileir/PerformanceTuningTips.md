# Performance Tuning Tips for CUDA TileIR Backend Optimization

This document provides a practical tutorial for optimizing Triton scripts to achieve better performance when running with the CUDA Tile IR backend.

## Autotune Configurations

### New Hints for CUDA TileIR Backend

#### **occupancy** (Critical)

The **occupancy** hint accepts an integer N from 1 to 10, indicating that the programmer expects N thread blocks to run simultaneously on a single SM. This hint is 1 by default and is worth tuning for many compute-intensive kernels.

#### Numerical Precision Options (approx & ftz)

Unlike the Triton PTX backend, the CUDA TileIR Backend backend disables approx and ftz by default. Setting `CUTILE_ENABLE_APPROX=1` and `CUTILE_ENABLE_FTZ=1` can provide performance improvements in certain workloads (with precision degradation within acceptable ranges), such as **`attention`** and its variant kernels. 

Note that the Tile IR compiler shipping in CUDA 13.1 does not automatically optimize `exp.approx -> ex2 +  mulf`.  For performance and precision parity with the Triton PTX backend, please explicitly rewrite `expOp` to use `ex2 + mulf` instead. 

#### opt-level

The default optimization level is currently `opt-level=3`. At this stage, adjusting this parameter is unnecessary.

### Existing Triton Hints

#### **num_ctas** (Critical)

Setting **num_ctas=2** is critical for dense dot-related workloads, as it enables 2CTA mode MMA on Blackwell architecture.

#### num_warps

The CUDA TileIR Backend currently ignores the `num_warps` hint, leaving tileiras to determine the optimal number of warps automatically. Therefore, autotuning `num_warps` is unnecessary. While the default is 4, the tileiras compiler will analyze and decide the specific num_warps after optimization.

#### num_stages

Unlike the PTX backend, the CUDA TileIR Backend treats the `num_stages` hint (whether per-kernel or per-loop) as a cost hint rather than a strict directive. This means a matmul kernel with `num_stages=3` won't necessarily have 3 stage buffers for pipelining. Instead, tileiras analyzes the impact of the `num_stages=3` operation from whole program perspective and determines the optimal pipeline configuration.

Since `num_stages` is a cost semantic hint, it is strongly recommended to expand the tuning range of `num_stages` during autotune, especially for dot-related kernels, where larger values can be tried.

The compiler should never produce SMEM or TMEM out-of-memory errors for any value of `num_stages` (or other hints).  These errors are always compilers bugs and should be reported as such.

#### warp_specializes

The CUDA TileIR Backend does not consider this hint.

#### Manual Slicing

Manual slicing approaches (such as `EPILOGUE_SUBTILE` in `python/tutorials/09-persistent-matmul.py`) may not provide positive benefits for CUDA TileIR Backend.

## Optimization Tips

- **CGA-Level Tile Representation**: The CUDA TileIR Backend treats tiles as CGA-level representations. When autotuning `BLOCK_SIZE`, consider increasing the block size appropriately to avoid missing high-performance program solutions.

- **2CTA Mode**: When using 2CTA mode, experiment with relatively larger `BLOCK_SIZE` values.

- **TMA API Preference**: The Tile IR compiler shipping in CUDA 13.1 has a known performance issue with the `tl.load` API (for example, running `03-matrix-multiplication.py` is `20%+` slower than when using the Triton PTX backend). It is recommended to use TMA APIs for all data loading scenarios. The tileiras compiler will automatically fall back to alternative instructions when TMA requirements are not met.

## Performance Benchmarks on B200 850W

```bash
sudo nvidia-smi -i 0 -pm 1; sudo nvidia-smi -i 0 -pl 850; sudo nvidia-smi -i 0 -lgc 1800
```

### Fused Attention fwd (06-fused-attention.py)

#### NVIDIA Backend

fused-attention-batch4-head32-d64-fwd-causal=True-warp_specialize=best
> `best` means for NVIDIA backend we choose best one in warp_specialize={true, false}

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 198.280                | 197.741               | 206.284          |
| 2048  | 390.542                | 457.815               | 279.894          |
| 4096  | 489.490                | 551.605               | 326.547          |
| 8192  | 551.109                | 600.419               | 352.358          |
| 16384 | 576.047                | 625.586               | 366.590          |

fused-attention-batch4-head32-d64-fwd-causal=False-warp_specialize=best

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 392.177                | 390.433               | 310.767          |
| 2048  | 551.897                | 569.906               | 351.237          |
| 4096  | 585.546                | 615.966               | 365.433          |
| 8192  | 601.009                | 636.474               | 371.975          |
| 16384 | 605.503                | 643.413               | 377.912          |

fused-attention-batch4-head32-d128-fwd-causal=True-warp_specialize=best

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 378.035                | 395.222               | 249.238          |
| 2048  | 556.625                | 707.721               | 324.878          |
| 4096  | 619.252                | 869.288               | 368.185          |
| 8192  | 684.843                | 1003.829              | 391.344          |
| 16384 | 732.929                | 1081.164              | 403.942          |

fused-attention-batch4-head32-d128-fwd-causal=False-warp_specialize=best

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 612.734                | 771.660               | 373.411          |
| 2048  | 709.941                | 981.171               | 386.271          |
| 4096  | 794.757                | 1045.812              | 409.654          |
| 8192  | 786.659                | 1097.696              | 412.452          |
| 16384 | 798.544                | 1146.955              | 409.147          |

#### CUDA TileIR Backend (enable approx & ftz)

fused-attention-batch4-head32-d64-fwd-causal=True-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 323.992                | 328.174               | 205.284          |
| 2048  | 478.665                | 490.317               | 278.807          |
| 4096  | 606.265                | 608.377               | 325.730          |
| 8192  | 712.298                | 702.676               | 352.680          |
| 16384 | 723.844                | 755.193               | 366.260          |

fused-attention-batch4-head32-d64-fwd-causal=False-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  | 578.115                | 576.234               | 310.600          |
| 2048  | 754.713                | 737.778               | 350.195          |
| 4096  | 806.960                | 782.496               | 365.149          |
| 8192  | 802.074                | 799.497               | 371.884          |
| 16384 | 736.156                | 806.969               | 377.525          |

fused-attention-batch4-head32-d128-fwd-causal=True-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  |   565.369              |  650.019              | 248.430          |
| 2048  |   859.711              |  982.496              | 324.684          |
| 4096  |   966.729              | 1237.101              | 374.295          |
| 8192  |  1047.048              | 1396.364              | 401.603          |
| 16384 |   984.629              | 1501.998              | 414.547          |

fused-attention-batch4-head32-d128-fwd-causal=False-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|-----------------------|------------------|
| 1024  |  956.991               | 1082.928              | 376.641          |
| 2048  | 1229.973               | 1353.094              | 406.701          |
| 4096  | 1198.051               | 1494.965              | 419.046          |
| 8192  | 1130.381               | 1564.618              | 422.727          |
| 16384 | 1028.020               | 1599.562              | 413.952          |

### Fused Attention bwd (06-fused-attention.py)

#### CUDA TileIR Backend (add `_attention_bwd_tma` kernel in python/tutorials/cutile/attention_tma.py, see `TMA` column)

fused-attention-batch4-head32-d64-bwd-causal=True-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [TMA FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Triton [TMA FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|----------------------------|-----------------------|---------------------------|------------------|
| 1024  |  92.102                | 174.920                    |  92.100               | 174.924                   | 172.861          |
| 2048  | 126.018                | 263.572                    | 126.132               | 263.524                   | 231.176          |
| 4096  | 146.523                | 337.504                    | 146.550               | 337.504                   | 276.981          |
| 8192  | 159.636                | 386.520                    | 159.716               | 386.495                   | 305.167          |
| 16384 | 166.291                | 416.442                    | 166.286               | 416.486                   | 320.319          |

fused-attention-batch4-head32-d64-bwd-causal=False-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [TMA FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Triton [TMA FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|----------------------------|-----------------------|---------------------------|------------------|
| 1024  | 184.524                | 258.195                    | 184.657               | 258.160                   | 255.243          |
| 2048  | 252.163                | 319.706                    | 252.369               | 319.714                   | 293.299          |
| 4096  | 293.022                | 354.036                    | 293.076               | 354.062                   | 316.000          |
| 8192  | 319.539                | 371.674                    | 319.189               | 371.674                   | 328.177          |
| 16384 | 332.417                | 383.150                    | 332.469               | 383.150                   | 336.770          |


fused-attention-batch4-head32-d128-bwd-causal=True-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [TMA FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Triton [TMA FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|----------------------------|-----------------------|---------------------------|------------------|
| 1024  |  82.152                | 233.501                    |  82.026               | 233.606                   | 183.572          |
| 2048  | 103.700                | 354.613                    | 103.784               | 354.586                   | 242.087          |
| 4096  | 119.233                | 459.481                    | 119.279               | 458.385                   | 285.513          |
| 8192  | 128.656                | 522.558                    | 128.576               | 524.870                   | 312.566          |
| 16384 | 132.782                | 556.285                    | 132.996               | 552.434                   | 326.871          |

fused-attention-batch4-head32-d128-bwd-causal=False-warp_specialize=False

| N_CTX | Triton [FP16] (TFLOPS) | Triton [TMA FP16] (TFLOPS) | Triton [FP8] (TFLOPS) | Triton [TMA FP8] (TFLOPS) | Flash-2 (TFLOPS) |
|-------|------------------------|----------------------------|-----------------------|---------------------------|------------------|
| 1024  | 164.558                | 389.693                    | 164.504               | 389.870                   | 286.835          |
| 2048  | 207.501                | 497.595                    | 207.624               | 497.772                   | 327.189          |
| 4096  | 238.310                | 555.552                    | 238.673               | 557.022                   | 349.583          |
| 8192  | 257.251                | 569.955                    | 257.030               | 565.657                   | 361.132          |
| 16384 | 265.002                | 558.477                    | 265.395               | 555.335                   | 369.739          |

### Persistent Matmul (09-persistent-matmul.py) 

> TFLOPS by proton

#### NVIDIA Backend 

| Kernel Name | K=512 | K=1024 | K=1536 | K=2048 | K=2560 | K=3072 | K=3584 | K=4096 | K=4608 | K=5120 | K=5632 | K=6144 | K=6656 | K=7168 | K=7680 | K=8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul_kernel | 410.535 | 485.939 | 508.868 | 523.959 | 523.860 | 517.353 | 509.405 | 503.433 | 457.957 | 462.662 | 466.334 | 467.583 | 465.737 | 468.807 | 467.914 | 474.498 |
| matmul_kernel_descriptor_persistent | 439.707 | 500.525 | 531.170 | 553.606 | 564.037 | 556.934 | 559.873 | 524.308 | 515.534 | 519.169 | 520.699 | 520.417 | 552.134 | 521.023 | 518.283 | 516.987 |
| matmul_kernel_descriptor_persistent_ws | 424.881 | 492.736 | 536.487 | 554.557 | 566.113 | 566.654 | 560.431 | 525.796 | 523.949 | 523.864 | 525.539 | 524.556 | 519.728 | 524.902 | 521.294 | 520.290 |
| matmul_kernel_persistent | 437.177 | 490.192 | 505.463 | 526.356 | 495.549 | 502.120 | 492.795 | 509.629 | 464.547 | 492.138 | 461.204 | 473.903 | 456.420 | 459.663 | 482.381 | 476.654 |
| matmul_kernel_tma | 453.171 | 510.479 | 540.693 | 554.571 | 550.412 | 547.197 | 537.709 | 504.863 | 495.738 | 495.422 | 501.529 | 500.631 | 502.919 | 504.600 | 503.772 | 505.822 |
| matmul_kernel_tma_persistent | 457.762 | 526.818 | 541.512 | 562.336 | 569.793 | 552.891 | 560.229 | 509.174 | 516.811 | 549.679 | 522.550 | 519.533 | 515.688 | 539.053 | 512.148 | 509.444 |
| matmul_kernel_tma_persistent_ws | 443.856 | 519.320 | 553.608 | 574.412 | 578.525 | 579.166 | 569.080 | 534.047 | 532.451 | 532.137 | 533.668 | 530.485 | 554.178 | 524.998 | 522.821 | 550.687 |
| matmul_kernel_tma_ws | 421.550 | 502.304 | 537.107 | 551.843 | 551.784 | 541.865 | 532.079 | 495.340 | 495.921 | 494.918 | 492.878 | 496.289 | 502.044 | 503.006 | 501.350 | 504.051 |

#### CUDA TileIR Backend

| Kernel Name | K=512 | K=1024 | K=1536 | K=2048 | K=2560 | K=3072 | K=3584 | K=4096 | K=4608 | K=5120 | K=5632 | K=6144 | K=6656 | K=7168 | K=7680 | K=8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul_kernel | 372.083 | 478.821 | 515.220 | 523.229 | 536.626 | 538.881 | 540.379 | 540.189 | 536.922 | 496.812 | 527.281 | 527.333 | 545.069 | 551.638 | 556.737 | 546.898 |
| matmul_kernel_descriptor_persistent | 449.608 | 566.495 | 592.396 | 615.399 | 621.022 | 625.198 | 633.241 | 632.614 | 633.009 | 629.261 | 632.138 | 637.709 | 641.277 | 644.160 | 648.690 | 648.044 |
| matmul_kernel_descriptor_persistent_ws | 448.865 | 566.048 | 592.297 | 616.102 | 620.858 | 628.390 | 637.610 | 640.445 | 634.553 | 631.684 | 647.245 | 639.895 | 641.622 | 645.320 | 650.257 | 646.576 |
| matmul_kernel_persistent | 386.227 | 472.954 | 502.894 | 512.529 | 523.132 | 530.562 | 535.570 | 538.549 | 538.180 | 538.355 | 541.091 | 541.664 | 547.022 | 549.228 | 548.273 | 552.914 |
| matmul_kernel_tma | 447.497 | 557.842 | 579.246 | 584.937 | 579.374 | 562.360 | 590.016 | 596.886 | 605.709 | 574.770 | 578.394 | 608.760 | 612.595 | 615.713 | 616.805 | 618.996 |
| matmul_kernel_tma_persistent | 450.121 | 566.328 | 594.972 | 614.759 | 620.405 | 628.140 | 635.045 | 635.619 | 630.554 | 629.911 | 646.355 | 636.326 | 639.891 | 645.985 | 644.748 | 644.186 |
| matmul_kernel_tma_persistent_ws | 442.042 | 566.433 | 591.798 | 616.341 | 621.496 | 628.013 | 636.439 | 633.790 | 633.202 | 629.759 | 631.215 | 630.826 | 641.347 | 643.391 | 649.245 | 646.864 |
| matmul_kernel_tma_ws | 446.199 | 557.764 | 581.963 | 588.196 | 580.131 | 558.987 | 590.458 | 599.535 | 607.182 | 608.649 | 611.659 | 611.689 | 614.381 | 617.276 | 619.827 | 620.500 |
