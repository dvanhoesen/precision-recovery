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

## First-pass condition names (plain language)

The script `experiments/first_pass_study.py` runs named conditions. These names are intentionally descriptive:

- `scalar_fp32_reference`
  - Model: `scalar`
  - Mode: `float_full`
  - Meaning:
    - Input is not reduced to N-bit levels before model compute.
    - Model hidden compute (weights/activations) runs in floating point.
    - Output is a single scalar prediction that is compared directly against the 2N-bit target value.
    - This is the reference ceiling (upper bound).

- `scalar_nbit_io_hidden_compute`
  - Model: `scalar`
  - Mode: `quant_full`
  - Meaning:
    - Input is reduced to N-bit levels using uniform quantization before entering the model.
    - Hidden compute uses N-bit fake quantization for weights and activations.
    - Output is a single scalar (no explicit 2-word output head).
    - The scalar prediction is compared against the original 2N-bit target value.

- `two_word_nbit_io_hidden_compute`
  - Model: `two_word`
  - Mode: `quant_full`
  - Meaning:
    - Input is reduced to N-bit levels with uniform quantization.
    - Hidden compute uses N-bit fake quantization for weights and activations.
    - Output head predicts two normalized N-bit-style words: `hi` and `lo`.
    - `hi` and `lo` are decoded and recombined into a 2N-bit-equivalent scalar.
    - Reconstructed scalar is compared to the 2N-bit target.

- `two_word_nbit_io_only`
  - Model: `two_word`
  - Mode: `quant_io`
  - Meaning:
    - Input is reduced to N-bit levels via uniform quantization.
    - Hidden compute stays floating-point (no hidden fake quantization).
    - Output predicts `hi`/`lo` words, then reconstructs a 2N-bit-equivalent scalar for evaluation.
    - This isolates the effect of structured output without hidden quantized compute.

## What the plotted error bars mean

Plots produced by `experiments/plot_summary.py` use `outputs/first_pass_study/summary.json`.

- Bar height = mean metric across repeated runs (usually different seeds).
- Error bar = sample standard deviation (`stdev`) across those runs.
- The number of runs per condition is stored in `n`.

Interpretation:
- Small error bars mean performance is stable across seeds.
- Large error bars mean performance is more seed-sensitive.

## summary.json keys by condition

Each condition in `summary.json` has these keys:

- `condition_name`: one of the condition labels above
- `experiment_mode`: precision mode actually used by training/eval
- `model_name`: model family (`scalar`, `two_word`, etc.)
- `task`: synthetic task name (for example `linear`)
- `n`: number of runs aggregated (typically number of seeds)
- `rmse_mean`, `rmse_std`: mean/std of RMSE over runs
- `mae_mean`, `mae_std`: mean/std of MAE over runs
- `hi_word_accuracy_mean`, `hi_word_accuracy_std`: high-word exact-match mean/std (word models only)
- `lo_word_accuracy_mean`, `lo_word_accuracy_std`: low-word exact-match mean/std (word models only)

For scalar conditions (`scalar_fp32_reference`, `scalar_nbit_io_hidden_compute`), the hi/lo word accuracy fields are `null` because scalar models do not emit word-level outputs.

## Model equivalence across conditions

For the first-pass conditions, the core MLP body is equivalent unless intentionally changed by the condition:

- `scalar_*` conditions use the same backbone architecture (`width`, `depth`, activations) and only differ by precision mode.
- `two_word_*` conditions use the same backbone architecture and only differ by precision mode.
- The intentional model-level difference between scalar and two-word conditions is the output interface/head:
  - scalar predicts one continuous value (`y`)
  - two-word predicts high/low normalized words (`hi`, `lo`) that are decoded back to scalar space

So yes: aside from the required input/output representation choices and quantization mode, model capacity and backbone structure are kept matched by config.

## Notes on training/validation behavior

To make validation metrics meaningful across epochs and splits, current training now:

- samples one task definition per run (for example one shared linear weight vector `w`) and reuses it for train/val/test
- normalizes targets with a train-derived scale and reuses that same scale for val/test
- logs both train and validation RMSE/MAE each epoch (in addition to train loss)

This reduces split mismatch and makes it easier to diagnose true overfitting vs optimization noise.
