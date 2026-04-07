# precision-recovery
PyTorch benchmark and reference implementation for recovering high-precision continuous targets from low-precision interfaces using chunked representations, coarse-to-fine decoding, and quantized regression baselines.

## Overview

This project studies whether a model operating through an `N`-bit interface can recover or preserve a `2N`-bit continuous target more effectively with structured output heads than with standard scalar regression.

The first-pass version focuses on synthetic regression tasks:

- linear
- polynomial
- multiplicative
- oscillatory

It includes baseline models for:

- scalar regression
- two-word output
- coarse-plus-residual output
- sequential high/low decoding
- bitwise output
