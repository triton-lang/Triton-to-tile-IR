// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts)" | FileCheck %s

