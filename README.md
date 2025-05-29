# Explainable AI for Mammography  
*A full-stack demonstration notebook for seven XAI methods + Wasserstein-robust auditing*

---

## 1 Why do we need explainability in healthcare?

Diagnostic AI systems carry high-stakes consequences—from delayed cancer
detection to overtreatment.  Regulatory bodies (e.g. FDA’s “Predetermined
Change Control Plan”, EU AI Act) now expect **human-understandable
justifications** for each prediction before clinical deployment.  Post-hoc
interpretability answers three safety questions:

| Trust dimension | Goal | Example from the notebook |
|-----------------|------|---------------------------|
| **Localization** | *Did the model focus on the lesion?* | Grad-CAM heat-map aligns with radiologist ROI |
| **Causality proxy** | *Which features contributed most?* | SHAP bar-plot ranks mass density & margin |
| **Robustness** | *Is the decision stable under realistic perturbations?* | Wasserstein PGD shifts image pixels yet retains diagnosis |

---

## 2 Techniques implemented

Below, each subsection gives the **intuition**, a concise **mathematical
formulation**, **strengths / caveats**, and a link to the original paper.

### 2.1 Local surrogate methods

#### • LIME – *Locally Interpretable Model-agnostic Explanations*  
Learns a sparse linear model  
$
g(x) = \sum_{k=1}^{K} w_k z_k
$  
around the neighbourhood of a single input \(x_0\); weights \(w_k\) are fitted
on perturbed samples drawn from a locality kernel.  Faithfulness is measured
by a locality-weighted loss plus an \(L_0\) complexity term. :contentReference[oaicite:0]{index=0}

*Pros*: model-agnostic, intuitive feature ranking.  
*Cons*: explanations vary with kernel width; no global guarantee.

#### • SHAP – *SHapley Additive exPlanations*  
Represents any explanation as an **additive feature attribution**  
\[
\phi_0 + \sum_{i=1}^{M}\phi_i x_i
\]  
and proves a **unique** solution that satisfies *local accuracy*, *missingness*,
and *consistency*.  Shapley values approximate the expected marginal
contribution of each feature over all coalitions. :contentReference[oaicite:1]{index=1}

*Pros*: unified theory, local & global views.  
*Cons*: exact values are \(2^{M}\)-costly; needs sampling for images.

---

### 2.2 Gradient & CAM family

#### • Grad-CAM – *Gradient-weighted Class Activation Mapping*  
Back-propagates the class score \(y^c\) to the final conv feature map
\(A^k\); importance weights  
\[
\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{i j}^k}
\]  
generate a coarse heat-map \( \mathrm{ReLU}\bigl(\sum_k \alpha_k^c A^k\bigr)\).
Works for any CNN-style architecture. :contentReference[oaicite:2]{index=2}

#### • Guided Backprop × Grad-CAM  
Element-wise product of a **Guided Backprop** saliency map (positive
gradients only) with Grad-CAM, yielding high-resolution overlays that keep
class-specific focus.  Introduced in the Grad-CAM paper. :contentReference[oaicite:3]{index=3}

*Pros*: human-readable, minimal compute.  
*Cons*: layer choice sensitive; fails on non-conv nets unless adapted.

#### • SmoothGrad  
Reduces visual noise by averaging \(n\) gradient maps from Gaussian-noised
copies of the input:  
\[
\tilde{S}(x) = \frac{1}{n}\sum_{i=1}^{n} S\!\bigl(x + \mathcal{N}(0,\sigma^2)\bigr)
\]  
where \(S(x)\) is any saliency function. :contentReference[oaicite:4]{index=4}

*Pros*: clearer edges.  
*Cons*: adds \(n\)-fold compute; still inherits saliency shortcomings.

---

### 2.3 Relevance propagation

#### • Layer-wise Relevance Propagation (LRP)  
Starts with the output score \(f(x)\) and **conserves relevance** while
propagating layer-by-layer: \( \sum_i R_i = f(x) \).  For a linear layer,
\( R_i = \sum_j \frac{a_i w_{ij}}{\sum_i a_i w_{ij}} R_j \).  
Captures both positive and negative evidence. :contentReference[oaicite:5]{index=5}

*Pros*: handles zero-grad non-linearities (ReLU dead zones).  
*Cons*: multiple propagation rules; sensitive to weight initialisation.

---

### 2.4 Transformer-specific insight

#### • Attention Rollout (Chefer *et al.*)  
Computes *joint relevance* via a modified relevance-propagation scheme that
traverses self-attention layers, skip-connections, and MLP blocks:  
\[
R^{l} = \mathrm{diag}\bigl(A^{l}\bigr) \cdot R^{l+1}
\]  
aggregated from the CLS token back to image patches.  Supports Swin and ViT
without re-training. :contentReference[oaicite:6]{index=6}

*Pros*: works when raw attention weights are uninformative.  
*Cons*: still heat-map style; cannot isolate feature interactions.

---

### 2.5 Distribution-robust auditing

#### • Wasserstein Adversarial Examples (Projected Sinkhorn)  
Defines the threat model as a **Wasserstein ball**  
\[
\mathcal{B}_{\varepsilon}^{W}(x) = \{\,x' : W(x, x') \le \varepsilon\}
\]
where \(W\) is the optimal-transport distance.  Adversarial points are
generated via Sinkhorn-iterated projections; defence uses PGD training on the
same geometry. :contentReference[oaicite:7]{index=7}

*Pros*: captures realistic pixel-mass shifts (translation, distortion).  
*Cons*: slower than \(L_p\); attack hyper-parameters non-trivial.

---

## 3 Dataset — CBIS-DDSM

Curated subset of the Digital Database for Screening Mammography containing
• 2 620 studies • dicom + ROI masks • verified pathology labels.  
Used as the running example in the notebook. :contentReference[oaicite:8]{index=8}

---

## 4 Repository structure
```bash
.
├── notebooks/Explainability_in_AI.ipynb ← one-stop demo (runs in order above)
├── slides/Explainability_in_AI.pptx ← 66-slide deck (Git LFS)
├── src/ ← reusable helpers
│ ├── xai/ ← LIME, SHAP, Grad-CAM, LRP …
│ └── adversarial/ ← Wasserstein PGD utilities
├── data/README.md ← how to fetch CBIS-DDSM
├── requirements.txt • environment.yml
├── SETUP_AND_USAGE.md ← install, dataset paths, Colab badge
├── LICENSE (MIT) • CITATION.cff
└── .gitignore

```
---

## 5 Quick start

```bash
git clone https://github.com/gap48/xai-healthcare-lecture.git
cd xai-healthcare-lecture
conda env create -f environment.yml && conda activate xai_healthcare
jupyter lab notebooks/Explainability_in_AI.ipynb
```
