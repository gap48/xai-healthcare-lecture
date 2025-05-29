# Explainable AI for Medical Imaging ✨  

## Table of Contents
1. [Overview](#overview)
2. [Why Explainability Matters](#why-explainability-matters)
3. [Implemented Techniques](#implemented-techniques)  
   &nbsp;&nbsp;3.1&nbsp; [LIME](#lime) | 3.2&nbsp; [SHAP](#shap) | 3.3&nbsp; [Grad-CAM](#grad-cam) | 3.4&nbsp; [Guided Backprop & Guided Grad-CAM](#guided-grad-cam) | 3.5&nbsp; [SmoothGrad](#smoothgrad) | 3.6&nbsp; [Layer-wise Relevance Propagation](#lrp) | 3.7&nbsp; [Attention Visualisation](#attention) | 3.8&nbsp; [Wasserstein Adversarial Robustness](#wass)  
4. [Dataset — CBIS-DDSM](#dataset-cbis-ddsm)  
6. [Installation & Quick Start](#installation--quick-start)

---

## Overview
This repository accompanies a guest lecture on **Explainable AI (XAI) in Healthcare** delivered by **Ganesh Puthiaraju** for the graduate course **BIOENG 2195 – Practicum in Neuroimage Analysis (Spring 2025)**. It provides **fully-working Python notebooks, slides, and utility scripts** that demonstrate state-of-the-art interpretability techniques on a mammography-classification pipeline trained with the **CBIS-DDSM** dataset.

---

## Why Explainability Matters
> “In medical imaging, AI decisions can be life-changing. We need to understand **not just _what_ the model predicts, but _why_ it makes those predictions**.” 

* **Build Trust** – transparent evidence for clinicians and patients  
* **Enable Clinical Adoption** – integrate seamlessly into radiologist workflow  
* **Improve Robustness** – uncover data artefacts & hidden biases  
* **Regulatory Compliance** – satisfy FDA / MDR audit trails  
* **Enhance Patient Safety** – reduce misdiagnoses from spurious correlations

---

## Implemented Techniques
## LIME <a id="lime"></a>

Local Interpretable Model-agnostic Explanations (LIME) fits an interpretable surrogate model around a point of interest to explain individual predictions.

### Mathematical Formulation

LIME approximates the complex model $f$ locally around an instance $x$ with a simpler, interpretable model $g \in G$:

$$\xi(x) = \arg\min_{g \in G} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

where:
- $\pi_x(z)$ measures the proximity of a sample $z$ to the instance $x$ (e.g., via a kernel function decreasing with distance)
- $\mathcal{L}(f, g, \pi_x)$ is a measure of how "unfaithful" $g$ is to $f$ in the locality defined by $\pi_x$
- $\Omega(g)$ is the complexity of the interpretable model $g$ (a penalty to encourage simplicity)

### Implementation

**Segmentation:** Divide the image into superpixels

```python
def generate_superpixels(self, image_np):
    segments = slic(image_np, n_segments=self.n_segments,
                  compactness=10, sigma=1, start_label=0,
                  channel_axis=None)
    return segments
```

**Perturbation:** Generate samples by randomly masking superpixels

```python
# Create a random binary mask (1 = keep segment, 0 = mask segment)
active_segments = np.random.binomial(1, 0.5, n_segments)

# Create a mask where 1 = keep pixel, 0 = mask pixel
mask = np.zeros(segments.shape)
for i in range(n_segments):
    mask[segments == i] = active_segments[i]

# Create a perturbed image where masked areas are set to 0 (black)
perturbed = image_np.copy() * mask
```

**Prediction:** Get model predictions for perturbed samples

```python
outputs = self.model(batch_tensor)
# Get class probabilities
probs = F.softmax(outputs, dim=1).cpu().numpy()
```

**Local modeling:** Train a linear model to approximate local behavior

```python
# Apply kernel to get sample weights
weights = self.kernel_function(distances)

# Train ridge regression
regression = Ridge(alpha=1.0)
regression.fit(binary_masks, target_probs, sample_weight=weights)
```

**Explanation:** Extract feature importance from coefficients

```python
# Create explanation heatmap
explanation = np.zeros(segments.shape)
for i in range(len(feature_importance)):
    explanation[segments == i] = feature_importance[i]
```

### Pros and Cons

**Advantages:**
- Model-agnostic: Works with any black-box model
- Local explanations: Provides insights into individual predictions
- Relatively easy to understand and implement

**Limitations:**
- Instability: Explanations can vary depending on the perturbation strategy and the choice of the local model
- Linearity Assumption: May not accurately capture complex, non-linear relationships
- Defining "Locality": Choosing the right neighborhood size can be challenging
- Computational Cost: Can be computationally expensive for high-dimensional data

---

## SHAP <a id="shap"></a>

SHapley Additive exPlanations (SHAP) is a game-theoretic approach that assigns each feature a value representing its contribution to the prediction.

### Mathematical Formulation

The Shapley value for feature $i$ is calculated as:

$$\phi_i(f, x) = \sum_{S \subseteq \{1,...,M\} \setminus \{i\}} \frac{|S|! (M - |S| - 1)!}{M!} [f(x_{S \cup \{i\}}) - f(x_S)]$$

where:
- $M$ is the total number of features
- $S$ is a subset of all features except $i$
- $f(\cdot)$ is the model's prediction function
- $\phi_i$ is the SHAP value for feature $i$

SHAP unifies six earlier attribution methods and satisfies local accuracy, consistency & additivity properties.

### Pros and Cons

**Advantages:**
- Strong theoretical foundation in game theory
- Satisfies important mathematical properties like local accuracy and consistency
- Provides both local and global explanations

**Limitations:**
- Computationally expensive, especially for high-dimensional data
- Requires access to data distribution for accurate estimation
- Can be challenging to interpret for non-technical stakeholders

---

## Grad-CAM <a id="grad-cam"></a>

Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image.

### Mathematical Formulation

Let $A^k$ be the feature maps of the chosen convolutional layer (index $k = 1, \ldots, K$) and $y^c$ be the score (logit) for class $c$ before the softmax layer.

1. **Compute the gradient** of $y^c$ with respect to each feature map $A^k$

2. **Global-average-pool** these gradients to obtain a scalar weight $\alpha^c_k$ per feature map:
   $$\alpha^c_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$$
   where $Z$ is the number of pixels in $A^k$

3. **Combine** the feature maps $A^k$ and their weights $\alpha^c_k$, then apply ReLU:
   $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha^c_k A^k\right)$$

4. **Upsample** $L^c_{\text{Grad-CAM}}$ to the original image size to produce a heatmap.

### Implementation

**Forward pass:** Target a specific layer to extract feature maps

```python
def forward_hook(module, input, output):
    if isinstance(output, tuple):
        self.feature_maps = output[0].clone()
    else:
        self.feature_maps = output.clone()
```

**Gradients calculation:** Compute gradients of target class with respect to feature maps

```python
def backward_hook(module, grad_input, grad_output):
    self.gradient = grad_output[0].clone()

# Backward pass for target class
target_score = model_output[0][target_class]
target_score.backward(retain_graph=True)
```

**Weighting:** Use gradients to determine importance of each feature map

```python
# For CNN models - standard GRAD-CAM approach
if len(gradients.shape) == 4: # (B, C, H, W)
    weights = torch.mean(gradients, dim=[2, 3])
```

**Weighted combination:** Create a weighted sum of activation maps

```python
# Apply weights to feature maps
cam = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)
for i, w in enumerate(weights[0]):
    cam += w * feature_maps[0, i]
```

**Processing:** Apply ReLU and normalization

```python
# Apply ReLU to focus on features that have positive influence
cam = F.relu(cam)

# Normalize CAM
if cam.max() > 0:
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Gamma correction for better visualization
    cam = torch.pow(cam, 0.5)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
```

### Pros and Cons

**Advantages:**
- Visual Explanations: Provides intuitive heatmaps that highlight important regions
- Model-Specific (CNNs): Well-suited for convolutional neural networks, leveraging their architecture
- Relatively Simple: Easy to implement and computationally efficient

**Limitations:**
- Coarse Localization: Heatmaps can be low-resolution and may not precisely pinpoint the exact boundaries of relevant regions
- Layer Choice: The choice of the convolutional layer can affect the results
- Gradient Saturation: Gradients can saturate, leading to less informative heatmaps

---

## Guided Backprop & Guided Grad-CAM <a id="guided-grad-cam"></a>

Guided Backpropagation modifies the backward pass to only propagate positive gradients for positive activations, highlighting features that positively contribute to the target class.

Guided Grad-CAM combines Grad-CAM with Guided Backpropagation to produce higher-resolution, more detailed visualizations.

### Implementation

**Modify ReLU backpropagation:** Only pass positive gradients for positive activations

```python
def backward_hook(module, grad_in, grad_out):
    # Get the preactivation from our stored dictionary
    module_id = id(module)
    preact = self.activation_maps.get(module_id)
    
    if preact is not None:
        # Make sure to clone the gradients to avoid in-place issues
        grad_out_clone = grad_out[0].clone()
        
        # Only pass positive gradients for positive preactivations
        positive_grad_out = torch.clamp(grad_out_clone, min=0.0)
        positive_preact = (preact > 0).float()
        
        # Guided backpropagation: zero out gradients for negative preactivations
        guided_grad = positive_grad_out * positive_preact
        
        return (guided_grad,)
    else:
        return grad_in
```

**Combine methods:** Multiply Guided Backpropagation gradients with GRAD-CAM

```python
def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    # Ensure grad_cam_mask is 2D
    if len(grad_cam_mask.shape) > 2:
        grad_cam_mask = grad_cam_mask.squeeze()
    
    # Ensure guided_backprop_mask is 3D (C, H, W)
    if len(guided_backprop_mask.shape) == 2:
        guided_backprop_mask = guided_backprop_mask[np.newaxis, :, :]
    
    # Resize GRAD-CAM to match guided backprop if needed
    if grad_cam_mask.shape != guided_backprop_mask.shape[1:]:
        from scipy.ndimage import zoom
        # Calculate zoom factors for height and width
        h_factor = guided_backprop_mask.shape[1] / grad_cam_mask.shape[0]
        w_factor = guided_backprop_mask.shape[2] / grad_cam_mask.shape[1]
        # Resize using scipy's zoom
        grad_cam_mask = zoom(grad_cam_mask, [h_factor, w_factor], order=1)
    
    # Element-wise multiplication of guided backprop and GRAD-CAM
    guided_grad_cam = []
    for i in range(guided_backprop_mask.shape[0]):
        guided_grad_cam.append(guided_backprop_mask[i] * grad_cam_mask)
    
    guided_grad_cam = np.array(guided_grad_cam)
    
    return guided_grad_cam
```

### Pros and Cons

**Advantages:**
- Higher Resolution: Provides more detailed visualizations compared to Grad-CAM alone
- Better Feature Localization: Combines the best of both methods
- Fine-grained Attribution: Shows specific features that contribute to the classification

**Limitations:**
- Computationally More Expensive: Requires additional computation for guided backpropagation
- More Complex Implementation: Requires careful handling of hooks and gradient manipulation
- Potential for Instability: Multiple gradient operations can sometimes lead to numerical issues

---

## SmoothGrad <a id="smoothgrad"></a>

SmoothGrad reduces noise in gradient-based visualizations by averaging the explanations obtained from multiple noisy versions of the input image.

### Mathematical Formulation

Given a saliency map function $M(x)$ for an input $x$, SmoothGrad is defined as:

$$\widehat{M}(x) = \frac{1}{n}\sum_{k=1}^{n} M\bigl(x + \mathcal{N}(0,\sigma^2)\bigr)$$

where $n$ is the number of samples and $\mathcal{N}(0,\sigma^2)$ is Gaussian noise with standard deviation $\sigma$.

### Implementation

**Add noise:** Generate multiple noisy versions of the input

```python
# Calculate standard deviation of the noise
input_min = input_image.min().item()
input_max = input_image.max().item()
stdev = self.noise_level * (input_max - input_min)

# Loop over noisy samples
for i in range(self.n_samples):
    # Add noise to the input image
    noisy_image = input_image.clone().detach()
    noise = torch.normal(0, stdev, size=input_image.shape,
                       device=input_image.device)
    noisy_image += noise
```

**Compute gradients:** Calculate gradients for each noisy sample

```python
# Get gradients for this noisy sample
grad, _ = self.generate_gradients(noisy_image, target_class)
```

**Average:** Combine gradients from all noisy samples

```python
# Add to smooth gradients
smooth_gradients += grad
successful_samples += 1

# Average gradients over successful samples
smooth_gradients /= successful_samples
```

### Pros and Cons

**Advantages:**
- Noise Reduction: Produces cleaner and more visually interpretable saliency maps
- Compatible with Various Methods: Can be applied to any gradient-based visualization technique
- Architecture-Agnostic: Works with any neural network architecture

**Limitations:**
- Increased Computation: Requires multiple forward and backward passes
- Parameter Sensitivity: Results can be sensitive to the choice of noise level and number of samples
- Still Inherits Base Method Limitations: Underlying issues with the base visualization method remain

---

## Layer-wise Relevance Propagation <a id="lrp"></a>

Layer-wise Relevance Propagation (LRP) backpropagates a "relevance score" from the output layer down to the input to show which input features are most responsible for a given prediction.

### Mathematical Formulation (Example ε-rule)

For a neuron $j$ in the current layer, its relevance $R_j$ is:

$$R_j = \sum_k \frac{x_j w_{jk}}{\sum_{j'} x_{j'} w_{j'k} + \varepsilon\,\text{sign}\Bigl(\sum_{j'} x_{j'} w_{j'k}\Bigr)} R_k$$

where:
- $R_j$ is the relevance of neuron $j$ in the current layer
- $R_k$ is the relevance of neuron $k$ in the next layer
- $x_j$ is the activation (or input) at neuron $j$
- $w_{jk}$ is the weight from neuron $j$ to neuron $k$
- $\varepsilon$ is a small stabilizer to avoid division by zero

LRP is applicable to CNNs, ViTs and hybrids; supports multiple propagation rules (ε, α-β, etc.).

### Pros and Cons

**Advantages:**
- Interpretable at Pixel Level: Provides detailed attribution maps showing each pixel's contribution
- Conservation Principle: Ensures that relevance is conserved across layers
- Multiple Rules: Different propagation rules can be applied for different layer types

**Limitations:**
- Computationally Intensive: Requires layer-by-layer backpropagation
- Rule Selection: Different rules may lead to different explanations
- Model Access: Requires access to model internals (weights and activations)

---

## Attention Visualisation <a id="attention"></a>

Visualizes attention weights from transformer models to show which regions of the input image the model focuses on when making predictions.

### Implementation

**Locate attention blocks:** Find suitable attention mechanisms in the model

```python
def get_attention_block(self):
    # If model is wrapped, get the base model
    if hasattr(model, 'model'):
        model = model.model
    
    # Specifically for swin_base_patch4_window7_224 from timm
    if hasattr(model, 'layers') and len(model.layers) > 0:
        # Try to get the second-to-last stage for better interpretability
        target_layer_idx = -3 if len(model.layers) > 1 else -1
        target_layer = model.layers[target_layer_idx]
        
        if hasattr(target_layer, 'blocks') and len(target_layer.blocks) > 0:
            # Get the last block in this stage
            last_block = target_layer.blocks[-1]
            
            # Check for attn attribute (standard in Swin Transformer)
            if hasattr(last_block, 'attn'):
                return last_block.attn
```

**Extract attention weights:** Capture attention patterns during forward pass

```python
def hook_fn(module, input, output):
    # For Window Attention in Swin Transformer
    if isinstance(output, tuple):
        # Some attention modules return (attention, value) or similar
        # Try to find the attention matrix - typically first or last element
        attn_matrix = None
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                # Typically, attention matrix has 'heads' dimension
                if len(out.shape) >= 3:
                    attn_matrix = out
                    break
        
        if attn_matrix is not None:
            self.attention_values.append(attn_matrix)
        else:
            # Default to first output
            self.attention_values.append(output[0])
    else:
        # Simple case - just one output
        self.attention_values.append(output)
```

**Process attention maps:** Convert attention weights to visualization format

```python
def process_attention_weights(self, attn_output, img_shape):
    # Convert to CPU and detach from computation graph
    attn = attn_output.detach().cpu()
    
    # Calculate feature magnitudes based on attention shape
    if len(attn.shape) == 4 and attn.shape[1] > 1: # [batch, heads, seq_len, seq_len]
        # Average over attention heads
        attn = attn.mean(dim=1)
        
        # For self-attention, focus on the attention from CLS token or average across all tokens
        if attn.shape[1] == attn.shape[2]: # Square attention matrix (seq_len × seq_len)
            # Average over rows to get column-wise importance (how much each token is attended to)
            feature_magnitudes = attn.mean(dim=1)
        else:
            # Non-square case - use the first dimension
            feature_magnitudes = attn.mean(dim=2)
```

### Pros and Cons

**Advantages:**
- Native to Transformers: Directly accesses the attention mechanism that's inherent to the model
- Intuitive Visualization: Shows which image regions the model focuses on
- No Gradient Computation: Usually doesn't require backpropagation, can be faster than gradient-based methods

**Limitations:**
- Limited to Transformer Models: Not applicable to CNN-only architectures
- Attention ≠ Attribution: Attention weights don't always reflect feature importance
- Multi-head Challenge: Aggregating information from multiple attention heads can be complex

---

## Wasserstein Adversarial Robustness <a id="wass"></a>

Uses Wasserstein distance to create adversarial examples that are more realistic and robust against distributional shifts.

### Mathematical Formulation

**Distributionally-robust training objective:**

$$\min_{\theta}\ \max_{\delta : W_p(\delta,0)\le \rho}\ \mathcal{L}\bigl(f_\theta(x+\delta),y\bigr)$$

where:
- $\mu$ is a distribution over possible perturbations around $x$
- $U_\epsilon(\delta_x)$ is the set of distributions within Wasserstein distance $\epsilon$ of the original input distribution $\delta_x$

### Implementation

**Iterative optimization:** Maximize loss through perturbations

```python
# Forward pass and loss computation
outputs = model(x_adv)
loss = criterion(outputs, labels)
grad = torch.autograd.grad(self.eps * loss, x_adv, create_graph=True)[0]
```

**Apply regularization:** Keep adversarial example close to original

```python
# L2 regularization to keep adversarial example close to original
l2_loss = 0.5 * torch.sum((x_adv - images) ** 2)
grad_reg = torch.autograd.grad(l2_loss, x_adv, create_graph=True)[0]

# Combine gradients and update adversarial example
grad = grad - grad_reg

step_size = 1.0 / torch.sqrt(torch.tensor(t + 2.0, device=x_adv.device, dtype=torch.float))
x_adv = x_adv + step_size * grad.detach()
```

**For targeted attacks:**

```python
# Calculate probability margin - we want to minimize this
margin_loss = probs[0, orig_class] - probs[0, target_class]

# Combined loss (we want to minimize this)
combined_loss = self.alpha * margin_loss + l2_loss

# Calculate gradients
grad = torch.autograd.grad(combined_loss, x_adv)[0]

# Apply momentum for more stable updates
momentum = 0.9 * momentum + grad / (torch.norm(grad, p=1) + 1e-8)

# Adaptive step size that decreases over iterations
step_size = self.eps * (1.0 - t/self.steps)

# Update adversarial example
x_adv = x_adv - step_size * torch.sign(momentum)
```

### Pros and Cons

**Advantages:**
- Distributional Robustness: Handles broader transformations beyond pixel-level perturbations
- Realistic Perturbations: Produces adversarial examples that are more natural-looking
- Stability: More robust against common defense mechanisms

**Limitations:**
- Computational Complexity: More complex optimization compared to standard adversarial techniques
- Parameter Sensitivity: Performance depends on careful tuning of multiple hyperparameters
- Implementation Difficulty: Requires sophisticated optimization techniques

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
# Visit: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
# Download the manifest file and use TCIA's data retrieval tools
```
```markdown
cbis-ddsm/
├── manifest-*/               # TCIA download
│   └── CBIS-DDSM/                           # Dataset directory
│       ├── mass_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       └── full mammogram images/
│           ├── Mass-Training_P_00001_LEFT_CC/
│           ├── Mass-Training_P_00001_LEFT_MLO/
│           └── ...
├── dcm_files.txt                             # Optional: List of DICOM paths
```
