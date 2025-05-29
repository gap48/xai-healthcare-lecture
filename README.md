# Explainable AI for Medical Imaging ✨  
_Comprehensive XAI toolbox & lecture materials_

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---
## Table of Contents
1. [Overview](#overview)
2. [Why Explainability Matters](#why-explainability-matters)
3. [Implemented Techniques](#implemented-techniques)  
   &nbsp;&nbsp;3.1&nbsp; [LIME](#lime) | 3.2&nbsp; [SHAP](#shap) | 3.3&nbsp; [Grad-CAM](#grad-cam) | 3.4&nbsp; [Guided Backprop & Guided Grad-CAM](#guided-grad-cam) | 3.5&nbsp; [SmoothGrad](#smoothgrad) | 3.6&nbsp; [Layer-wise Relevance Propagation](#lrp) | 3.7&nbsp; [Attention Visualisation](#attention) | 3.8&nbsp; [Wasserstein Adversarial Robustness](#wass)  
4. [Dataset — CBIS-DDSM](#dataset-cbis-ddsm)  
6. [Installation & Quick Start](#installation--quick-start)

---

## Overview
This repository accompanies a guest lecture on **Explainable AI (XAI) in Healthcare** delivered by **Ganesh Puthiaraju** for the graduate course **BIOENG 2195 – Practicum in Neuroimage Analysis (Spring 2025)**.  
It provides **fully-working Python notebooks, slides, and utility scripts** that demonstrate state-of-the-art interpretability techniques on a mammography-classification pipeline trained with the **CBIS-DDSM** dataset.

---

## Why Explainability Matters
> “In medical imaging, AI decisions can be life-changing. We need to understand **not just _what_ the model predicts, but _why_ it makes those predictions**.” :contentReference[oaicite:1]{index=1}

* **Build Trust** – transparent evidence for clinicians and patients  
* **Enable Clinical Adoption** – integrate seamlessly into radiologist workflow  
* **Improve Robustness** – uncover data artefacts & hidden biases  
* **Regulatory Compliance** – satisfy FDA / MDR audit trails  
* **Enhance Patient Safety** – reduce misdiagnoses from spurious correlations

---

## Implemented Techniques

### LIME  <a id="lime"></a>
_Local Interpretable Model-agnostic Explanations_ (LIME) fits an interpretable surrogate \(g\) around a point of interest \(x\):

\[
\underset{g \in G}{\arg\min}\; \mathcal{L}\bigl(f,\,g,\,\pi_x\bigr) + \Omega(g)
\]

* \(f\): black-box model * \(\pi_x\): locality kernel * \(\Omega\): complexity penalty  
* Perturb ↦ Predict ↦ Weight ↦ Fit ↦ Explain ﬂow :contentReference[oaicite:2]{index=2}

---

### SHAP  <a id="shap"></a>
_Game-theoretic Shapley values_ decompose a prediction into additive feature contributions:

\[
\phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,
\bigl(f_{S\cup\{i\}} - f_S\bigr)
\]

* Unifies six earlier attribution methods; satisfies local accuracy, consistency & additivity :contentReference[oaicite:3]{index=3}  

---

### Grad-CAM  <a id="grad-cam"></a>
_Gradient-weighted Class Activation Mapping_ produces a heat-map:

\[
\alpha_k^{c} = \frac{1}{Z}\sum_{i,j}\frac{\partial y^{c}}{\partial A^{k}_{ij}},\quad
L^{c} = \text{ReLU}\!\left(\sum_k \alpha_k^{c} A^{k}\right)
\]

where \(A^{k}\) are feature maps of a chosen conv-layer. Highlights class-discriminative regions while retaining spatial context.   

---

### Guided Backprop & Guided Grad-CAM  <a id="guided-grad-cam"></a>
* **Guided Backprop** – masks negative gradients during back-prop to sharpen details.  
* **Guided Grad-CAM** – element-wise product of Guided-BP map and Grad-CAM heat-map → high-resolution attribution.

---

### SmoothGrad  <a id="smoothgrad"></a>
Reduces noisy saliency by averaging maps over \(n\) noisy copies:

\[
\widehat{M}(x) = \frac{1}{n}\sum_{k=1}^{n} M\!\bigl(x + \mathcal{N}(0,\sigma^2)\bigr)
\]

Produces cleaner explanations and is architecture-agnostic. :contentReference[oaicite:5]{index=5}  

---

### Layer-wise Relevance Propagation (LRP)  <a id="lrp"></a>
Back-propagates a **relevance score** \(R\) instead of gradients:

\[
R_j = \sum_k \frac{a_j w_{jk}}
         {\sum_{j'} a_{j'} w_{j'k} + \varepsilon\,\text{sign}\!\Bigl(\sum_{j'} a_{j'} w_{j'k}\Bigr)}
         \; R_k
\]

Applicable to CNNs, ViTs and hybrids; supports multiple propagation rules (\(\varepsilon\), \(\alpha\)-\(\beta\), etc.). :contentReference[oaicite:6]{index=6}  

---

### Attention Visualisation (ViTs / Swin)  <a id="attention"></a>
* Extract **self-attention weights** \(A^{(l)}\in\mathbb{R}^{h\times N\times N}\) across heads \(h\) & layers \(l\).  
* Roll-out or accumulate to show long-range patch dependencies, revealing which breast regions steer the ViT’s decision. :contentReference[oaicite:8]{index=8}  

---

### Wasserstein Metric Adversarial Robustness  <a id="wass"></a>
Distributionally-robust training objective:

\[
\min_{\theta}\;\max_{\delta : W_p(\delta,0)\le \rho}\;
\mathcal{L}\!\bigl(f_\theta(x+\delta),\,y\bigr)
\]

* Attacks shift **entire image distributions** (texture, geometry) instead of \(\ell_p\) pixels.  
* Empirically improves mammography robustness vs PGD/FGSM baselines. :contentReference[oaicite:9]{index=9}  

---

## Dataset — CBIS-DDSM  <a id="dataset-cbis-ddsm"></a>

### Dataset Information
| Property | Value |
|----------|-------|
| Total studies | ≈ 2 620 (≈ 10 k images) |
| Views | CC & MLO |
| Lesion types | Mass, Calcifications |
| Labels | **Benign (~70 %) / Malignant (~30 %)** |
| Resolution | ≥ 4 000 × 3 000, 12–16-bit grayscale |
| Size | ≈ 163 GB |

Sources: TCIA collection notes & Kaggle mirror :contentReference[oaicite:10]{index=10}  

### Download
```bash
# Option 1 — TCIA
#  Step 1: grab manifest from
#  https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
#  Step 2: use TCIA-Downloader or NBIA Data Retriever

# Option 2 — scripted
python scripts/download_data.py --output_dir ./data/cbis-ddsm
data/
└── cbis-ddsm/
    ├── manifest-*/               # TCIA download
    │   └── CBIS-DDSM/
    │       ├── Mass-Training_P_*/
    │       ├── Mass-Test_P_*/
    │       ├── Calc-Training_P_*/
    │       └── Calc-Test_P_*/
    ├── csv_files/
    │   ├── mass_case_description_train_set.csv
    │   ├── mass_case_description_test_set.csv
    │   ├── calc_case_description_train_set.csv
    │   └── calc_case_description_test_set.csv
    └── dicom_info/
        └── dicom_paths.txt
