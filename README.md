# precision-recovery
PyTorch benchmark and reference implementation for studying how an N-bit-constrained model predicts 2N-bit continuous targets, comparing plain scalar regression, chunked/structured output methods, and an optional higher-precision 2N-bit reference baseline.

The central comparison in this repository is **not** between low-precision and high-precision computing by default. The main study compares **two different ways of using an `N`-bit-constrained system** on a `2N`-bit target problem:

1. a **plain scalar regression head**, which predicts the target as a single continuous value
2. a **chunked or structured output head**, which predicts the target in multiple parts such as high/low words or coarse/fine components

The repository also includes a **higher-precision reference baseline** that uses `2N`-bit or full-precision computation internally, serving as an upper-bound comparison point.

## Overview

This project studies whether a model operating through an `N`-bit interface can recover or preserve a `2N`-bit continuous target more effectively with structured output heads than with a standard scalar regression head.

The intended interpretation is:

- the system interface, visible inputs, or compute path is constrained to `N` bits
- the prediction target carries `2N`-bit information
- the main experiment compares alternative **model/output designs under the same low-precision constraint**
- an optional higher-precision baseline is included only as a reference ceiling

## Main comparison

The core study compares these two settings under matched experimental conditions:

### 1. Plain scalar regression baseline
The model sees `N`-bit-constrained inputs and predicts a single scalar output directly.

This asks:

> How well can an ordinary regression model recover a `2N`-bit target without explicitly modeling its internal high/low precision structure?

### 2. Chunked or structured output model
The model still sees `N`-bit-constrained inputs, but predicts the target in multiple components that are recombined before evaluation or loss computation.

Examples include:

- high word + low word
- coarse prediction + residual
- sequential high-chunk then low-chunk decoding
- bitwise or wordwise output representations

This asks:

> Does adding structure to the output representation help an `N`-bit-constrained model better recover a `2N`-bit target?

## Reference baseline

The repository also includes:

### 3. Higher-precision reference baseline
A model that uses `2N`-bit or standard higher-precision computation internally and predicts the same target.

This baseline is included to answer a different question:

> How close do the `N`-bit-constrained methods get to a less constrained or higher-precision model?

This reference should be interpreted as an **upper bound**, not as one of the main low-precision methods being compared.

## First-pass scope

The first-pass version focuses on synthetic regression tasks:

- linear
- polynomial
- multiplicative
- oscillatory

It includes baseline model families for:

- scalar regression
- two-word output
- coarse-plus-residual output
- sequential high/low decoding
- bitwise output
- optional higher-precision reference model

