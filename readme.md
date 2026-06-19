# FixSim: Fixed-Point Simulation for On-Device LLM Accelerators

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Systems%20Architecture-blue)](https://doi.org/10.1016/j.sysarc.2025.103548)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.sysarc.2025.103548-green)](https://doi.org/10.1016/j.sysarc.2025.103548)

**FixSim** is a fixed-point simulation script for exploring node-wise arithmetic bit-widths in LLaMA3-style large language models.

This repository supports the methodology presented in:

> **A new fixed-point simulation methodology for on-device AI based on large language models**
> Jung-Woo Kim, Seung-Hwan Yoon, Dong-Kyeong Kang, Seong-Won Lim, Hak-Bum Lee, Su-Min Oh, and Young-Ho Seo
> *Journal of Systems Architecture*, 168, 103548, 2025
> DOI: [10.1016/j.sysarc.2025.103548](https://doi.org/10.1016/j.sysarc.2025.103548)

---

## Overview

Conventional LLM quantization mainly reduces weight precision or matrix multiplication cost.
However, many methods still rely on floating-point restoration around operations such as Softmax, layer normalization, and activation functions.

FixSim targets a hardware-oriented question:

> **How many integer and fractional bits are required for each arithmetic node when an LLM is implemented as a fixed-point accelerator?**

The goal is not standalone LLM serving.
FixSim is intended for **accuracy-aware fixed-point precision exploration** for:

* on-device LLM acceleration,
* low-area NPU or ASIC design,
* FPGA prototyping,
* fixed-point datapath sizing,
* RTL test-vector generation.

FixSim assumes that a local LLaMA3-style implementation already exists.
The script should be connected to the model's inference interface so that intermediate arithmetic nodes can be simulated with fixed-point formats during evaluation.

---

## Method

FixSim partitions the LLM computation graph into **RONs**, or **Repeatable and One-to-One Nodes**.

A RON is a hardware-oriented computation unit that can reuse the same fixed-point configuration across repeated transformer layers. Examples include:

* matrix multiplication,
* scalar multiplication,
* Softmax sub-operations,
* SiLU activation,
* layer normalization sub-operations,
* division and square root,
* constants such as epsilon.

Each RON is assigned a fixed-point format using two parameters:

```text id="d96xv2"
IL: integer length, which determines numeric range
FL: fractional length, which determines fractional precision
```

Floating-point values are simulated as fixed-point values using:

```text id="l67vxw"
x_fix = clamp(round(x_fp * 2^FL), min, max) * 2^(-FL)
```

Unlike ordinary PTQ, this process does not rely on zero-point based dequantization.
Once `IL` and `FL` are selected, the operation can be mapped to integer datapaths.

---

## Coarse-to-Fine Simulation

FixSim searches the bit-width of each RON sequentially along the LLM computation path.
A downstream RON is not fixed before the upstream RON, because early precision loss can propagate to later nodes.

The search consists of two stages.

### 1. Integer-bit simulation

First, FixSim determines `IL`.

This step checks how many integer bits are required to cover the dynamic range of each RON without overflow.
Candidate `IL` values are evaluated while preserving enough fractional precision.

### 2. Fractional-bit simulation

After `IL` is fixed, FixSim determines `FL`.

The fractional search is divided into coarse and fine stages:

```text id="z0l0np"
Coarse search: 0, 8, 16, 24, ...
Fine search:   one-bit search around the first passing coarse point
```

For example, if `FL = 16` first satisfies the target metric in the coarse stage, FixSim searches nearby bit-widths one bit at a time and selects the smallest passing `FL`.

At each step, the model is evaluated using a quality metric such as perplexity or MMLU-style accuracy.
A bit-width is accepted only when the model remains within the user-defined safety margin.

---

## Accuracy Results

FixSim is designed to search fixed-point bit-widths while preserving model-level accuracy.
The reported results evaluate accuracy degradation and perplexity stability after fixed-point simulation.

### Accuracy Degradation

<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Method</th>
      <th align="center">Setting</th>
      <th align="center">Accuracy Degradation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="5">LLaMA3-8B</td>
      <td align="center">GPTQ</td>
      <td align="center">W4A16</td>
      <td align="center">1.25 %p</td>
    </tr>
    <tr>
      <td align="center">SmoothQuant</td>
      <td align="center">W8A8</td>
      <td align="center">8.70 %p</td>
    </tr>
    <tr>
      <td align="center">AWQ</td>
      <td align="center">W4A16</td>
      <td align="center">1.84 %p</td>
    </tr>
    <tr>
      <td align="center">SpinQuant</td>
      <td align="center">W4A8KV8</td>
      <td align="center">0.80 %p</td>
    </tr>
    <tr>
      <td align="center"><strong>FixSim (Ours)</strong></td>
      <td align="center"><strong>Fixed-point</strong></td>
      <td align="center"><strong>0.00 %p</strong></td>
    </tr>
  </tbody>
</table>

FixSim preserved the baseline accuracy after fixed-point simulation, showing **0.00 %p accuracy degradation**.

### Perplexity

|         Model        |   Method  | WikiText-2 ↓ |   C4 ↓  |  PTB ↓  |
| :------------------: | :-------: | :----------: | :-----: | :-----: |
| LLaMA3.2-1B-Instruct | FP16/FP32 |    12.3879   | 23.8186 | 23.4222 |
| LLaMA3.2-1B-Instruct |   FixSim  |    12.3819   | 23.7714 | 23.3014 |

Lower perplexity is better.
The fixed-point simulation maintained comparable perplexity to the FP16/FP32 baseline across WikiText-2, C4, and PTB.

---

## Output

FixSim records the simulation results and produces an `analyzer` tensor containing the selected bit-widths for each node.

The output can be used for:

* fixed-point adder sizing,
* multiplier and accumulator width selection,
* LUT precision selection,
* Softmax approximation design,
* layer normalization datapath design,
* hardware test-vector extraction.

---

## Quick Start

Prepare a local LLaMA3-style model implementation and connect FixSim to its inference interface.

```bash id="ht742u"
python fixsim.py run-main \
  --ckpt_dir /path/to/Llama3.2-1B-Instruct \
  --max_seq_len 2048 \
  --max_batch_size 1 \
  --num_samples 16
```

Model weights are not included in this repository.

---

## Requirements

A typical environment requires:

```bash id="zli3li"
pip install torch sentencepiece numpy tqdm
```

Additional dependencies may be required depending on the LLaMA3 implementation used as the backend.

---

## Citation

If you use this repository or the fixed-point simulation methodology, please cite:

```bibtex id="s4h445"
@article{kim2025new,
  title   = {A new fixed-point simulation methodology for on-device AI based on large language models},
  author  = {Kim, Jung-Woo and Yoon, Seung-Hwan and Kang, Dong-Kyeong and Lim, Seong-Won and Lee, Hak-Bum and Oh, Su-Min and Seo, Young-Ho},
  journal = {Journal of Systems Architecture},
  volume  = {168},
  pages   = {103548},
  year    = {2025},
  doi     = {10.1016/j.sysarc.2025.103548},
  url     = {https://doi.org/10.1016/j.sysarc.2025.103548}
}
```

---

## Disclaimer

This repository is research code for fixed-point simulation and hardware-oriented precision exploration.
It is not a standalone LLM inference engine and does not redistribute LLaMA model weights.
