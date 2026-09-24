# Intel Image Classification Using PyTorch and a Custom CNN

A computer vision project implementing an end-to-end image classification pipeline using **PyTorch** and a custom Convolutional Neural Network (CNN).

The model is trained on the **Intel Image Classification** dataset to classify natural scene images into six categories.

## Classes

The model classifies images into:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

## Project Overview

This project was built to understand the complete workflow of a computer vision classification problem using PyTorch, starting from dataset preparation and preprocessing to CNN training and model evaluation.

### Pipeline

```text
Dataset
   ↓
Image Preprocessing
   ↓
DataLoader
   ↓
Custom CNN
   ↓
Training
   ↓
Evaluation
   ↓
Accuracy / Loss Analysis
```

## Technologies Used

* Python
* PyTorch
* Torchvision
* Matplotlib

## Dataset

The project uses the **Intel Image Classification** dataset available on Kaggle.

The dataset contains images belonging to six different natural scene categories.

Dataset source:

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset is organized into training and testing directories and loaded using `torchvision.datasets.ImageFolder`.

## Image Preprocessing

Images are resized to:

```text
64 × 64
```

Training images use:

* Resize
* Random Horizontal Flip
* Tensor conversion

Test images use:

* Resize
* Tensor conversion

The `RandomHorizontalFlip` augmentation is applied only to the training data.

## CNN Architecture

A custom CNN was implemented using PyTorch.

```text
Input Image
    ↓
Conv2D (3 → 16)
    ↓
ReLU
    ↓
MaxPool
    ↓
Conv2D (16 → 32)
    ↓
ReLU
    ↓
MaxPool
    ↓
Conv2D (32 → 64)
    ↓
ReLU
    ↓
MaxPool
    ↓
Flatten
    ↓
Linear (2304 → 128)
    ↓
ReLU
    ↓
Linear (128 → 6)
```

The final layer produces six class scores corresponding to the six categories in the dataset.

## Training

The model was trained using:

* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam
* **Learning Rate:** 0.002
* **Batch Size:** 32
* **Epochs:** 20

The model was trained using GPU acceleration when available.

## Results

The final training run achieved approximately:

| Metric             |     Result |
| ------------------ | ---------: |
| Training Accuracy  | **85.01%** |
| Test Accuracy      | **80.37%** |
| Best Test Accuracy | **80.53%** |
| Final Test Loss    | **0.5656** |

The best test accuracy was obtained around **epoch 8**, after which the training accuracy continued increasing while test performance fluctuated.

This provides a useful example of the difference between training and generalization performance in a CNN.

## Evaluation

Model performance was evaluated using:

* Training loss
* Test loss
* Training accuracy
* Test accuracy


## Project Structure

```text
intel-image-classification-pytorch/
│
├── notebook
        └── Image_Identification.ipynb
├── README.md
├── requirements.txt
├── Dataset
└── LICENSE

```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Sifat192/intel-image-classification-pytorch.git
cd intel-image-classification-pytorch
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib opendatasets
```

### 3. Open the notebook

Run:

```bash
jupyter notebook
```

or open the notebook using Google Colab.

### 4. Download the dataset

The notebook uses `opendatasets` to download the Intel Image Classification dataset from Kaggle.

A Kaggle account/API credential may be required.

## What I Learned

Through this project, I worked through the complete PyTorch computer vision workflow:

* Loading image datasets with `ImageFolder`
* Applying image transformations
* Creating `DataLoader`s
* Building a CNN from scratch
* Understanding convolution and pooling layers in practice
* Implementing the PyTorch training loop
* Using `CrossEntropyLoss`
* Using Adam optimization
* Switching between training and evaluation modes
* Evaluating classification performance
* Interpreting loss and accuracy

## Future Improvements

Possible improvements for future experiments include:

* Transfer learning with pretrained CNN architectures
* More advanced data augmentation
* Hyperparameter tuning
* Larger image resolutions
* Comparing different CNN architectures

These are intentionally kept outside the scope of this first custom-CNN implementation.

## Author

**Sifat Bhatia**

B.Tech CSE (AI & ML)
KIIT University

GitHub: [Sifat192](https://github.com/Sifat192)
