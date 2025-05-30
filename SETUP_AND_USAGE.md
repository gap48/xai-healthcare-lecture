# Setup and Usage Guide for XAI Healthcare Repository

This repository contains code for applying explainable AI (XAI) techniques to mammography images from the CBIS-DDSM dataset. The code implements a Swin Transformer model for breast cancer classification and includes various XAI methods to visualize and understand the model's decisions.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration Parameters](#configuration-parameters)
- [Running the Code](#running-the-code)
- [Available XAI Methods](#available-xai-methods)
- [Troubleshooting](#troubleshooting)

## Requirements

The code has been tested with the following dependencies:

- Python 3.8+
- PyTorch 1.10+
- MONAI 0.9+
- Timm 0.5.4+
- NumPy 1.26.4
- Pandas 1.5.3
- Matplotlib
- Scikit-learn
- OpenCV
- SciPy

## Installation

### Option 1: Using the Jupyter Notebook

The easiest way to run this code is through the provided Jupyter notebook:

1. **Clone the repository:**

```bash
git clone https://github.com/gap48/xai-healthcare-lecture.git
cd xai-healthcare-lecture
```

2. **Open the Jupyter notebook** in Google Colab or locally.

3. **The notebook will install the required dependencies** when run.

### Option 2: Manual Installation

1. **Clone the repository:**

```bash
git clone https://github.com/gap48/xai-healthcare-lecture.git
cd xai-healthcare-lecture
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the requirements:**

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This code uses the CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset. To set up the dataset:

1. **Download the CBIS-DDSM dataset** from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/).

2. **Extract the dataset** to a local directory.

3. **Ensure the dataset contains** the following CSV files:
   - `mass_case_description_train_set.csv`
   - `mass_case_description_test_set.csv`
   - `calc_case_description_train_set.csv`
   - `calc_case_description_test_set.csv`

4. **Make sure the "full mammogram images"** are properly organized in the dataset directory.

## Configuration Parameters

Before running the code, you need to configure several parameters in the script:

```python
# Configuration parameters
root_dir = "/content/drive/MyDrive/cbis-ddsm-data"  # Update this path to your DDSM dataset
cache_dir_cls = "/content/drive/MyDrive"  # Update to your cache directory
dicom_path = "/content/drive/MyDrive/dcm_files.txt"  # Path to file containing DICOM paths
save_dir = "/content/drive/MyDrive/"  # Directory to save model checkpoints
batch_size = 4
num_workers = 2
num_epochs = 3
model_path = "/content/drive/MyDrive/best_mammo_model.pth"
```

Here's what each parameter means:

| Parameter | Description |
|-----------|-------------|
| `root_dir` | Path to your CBIS-DDSM dataset. This should contain the CSV files and DICOM images. |
| `cache_dir_cls` | Directory where metadata cache will be stored. Set to `None` to generate metadata from scratch. |
| `dicom_path` | Path to a text file listing all DICOM file paths. If the file doesn't exist, it will be created. Set to `None` to scan for DICOM files every time. |
| `save_dir` | Directory where trained model checkpoints will be saved. |
| `batch_size` | Number of samples in each training batch. Adjust based on your GPU memory. |
| `num_workers` | Number of worker processes for data loading. Usually 2-4 is optimal. |
| `num_epochs` | Number of training epochs. Increase for better performance, decrease for faster training. |
| `model_path` | Path to a pre-trained model checkpoint. Set to `None` to train from scratch. |

## Running the Code

### Using the Jupyter Notebook

1. **Open the notebook** in Google Colab or Jupyter.
2. **Update the configuration parameters** with your dataset paths.
3. **Run all cells in order.**

### Using Python Script

If you're running the code as a Python script:

1. **Update the configuration parameters** at the top of the script.

2. **Run the script:**

```bash
python main.py
```

### DICOM Path Handling

The code includes smart handling of DICOM file paths:

- If `dicom_path` is provided and exists, it loads DICOM paths from this file.
- If `dicom_path` is provided but doesn't exist, it scans the directory for DICOM files and saves the paths to this file.
- If `dicom_path` is `None`, it scans for DICOM files every time the code runs.

Using a cached list of DICOM paths can significantly speed up initialization when working with large datasets.

## Available XAI Methods

This repository implements several explainable AI techniques for analyzing mammography images:

### Gradient-based Methods
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Guided Backpropagation**: Modified gradient computation for cleaner visualizations
- **Guided Grad-CAM**: Combination of Grad-CAM and Guided Backpropagation
- **SmoothGrad**: Noise reduction through averaging multiple gradient samples

### Model-agnostic Methods
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations

### Attention-based Methods
- **Attention Visualization**: For transformer-based models (Swin Transformer)

### Adversarial Methods
- **Wasserstein Adversarial Examples**: Robust adversarial attack generation

## Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory Error**
```
Solution: Reduce batch_size in the configuration parameters.
```

**2. Dataset Not Found Error**
```
Solution: Verify that root_dir points to the correct CBIS-DDSM dataset location.
```

**3. Missing CSV Files**
```
Solution: Ensure all required CSV files are present in the dataset directory:
- mass_case_description_train_set.csv
- mass_case_description_test_set.csv
- calc_case_description_train_set.csv
- calc_case_description_test_set.csv
```

**4. DICOM Loading Issues**
```
Solution: Check that DICOM files are properly organized and accessible. 
Try setting dicom_path to None to force a fresh scan of DICOM files.
```

**5. Dependency Installation Problems**
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install dependencies one by one if requirements.txt fails
pip install torch torchvision
pip install monai
pip install timm
pip install numpy pandas matplotlib scikit-learn opencv-python scipy
```

**6. Model Loading Issues**
```
Solution: If model_path points to a non-existent file, set it to None to train from scratch.
```

### Performance Tips

1. **Use GPU acceleration** when available for faster training and inference.
2. **Adjust num_workers** based on your system's CPU cores (typically 2-4).
3. **Use cached DICOM paths** for faster initialization on subsequent runs.
4. **Monitor GPU memory usage** and adjust batch_size accordingly.

### System Requirements

**Minimum Requirements:**
- 8 GB RAM
- 4 GB available disk space
- Python 3.8+

**Recommended Requirements:**
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM
- 20 GB available disk space
- Python 3.9+

## Contributing

We welcome contributions to improve the repository! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{xai-healthcare-lecture,
  title={XAI Healthcare Lecture: Explainable AI for Mammography Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/gap48/xai-healthcare-lecture}
}
```

## Acknowledgments

- CBIS-DDSM dataset from The Cancer Imaging Archive
- MONAI framework for medical imaging
- Timm library for vision transformers
- PyTorch ecosystem
