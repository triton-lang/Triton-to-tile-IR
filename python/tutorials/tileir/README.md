# TileIR tutorials

This directory owns the tutorials currently supported by TileIR. Tutorials
without TileIR-specific behavior are kept as exact copies of their counterparts
in the parent directory. TileIR-specific tuning stays here so upstream tutorial
syncs cannot silently overwrite it.

Tutorials 07 and 11 and the z0/z1 examples are not included because they are not
currently supported by TileIR.

`06-fused-attention.py` restores the tuning space recorded before the upstream
sync: 144 raw configurations spanning `BLOCK_M` 64/128/256, four pipeline
depths, and occupancy 1/2. It also restores the head-dimension prune and includes
`STAGE` in the autotune cache key.

`09-persistent-matmul.py` restores two TileIR configuration factories. Each
produces 64 raw configurations spanning `BLOCK_SIZE_M` 128/256, four pipeline
depths, and `num_ctas` 1/2.
