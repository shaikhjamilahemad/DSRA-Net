# DSRA-Net: Dual-Scale Residual Attention for Pneumonia Detection

This repository contains the notebooks for the paper **"DSRA-Net: Dual-Scale Residual Attention for Pneumonia Detection"**.

DSRA-Net is a compact CNN (about 1.3 million parameters) trained from scratch to detect pneumonia in chest X-rays. It uses a dual-scale gate (DSG) block with parallel 3×3 and 5×5 convolutions, followed by three residual blocks with squeeze-and-excitation (SE) attention.

## Notebooks

| Notebook | Dataset | What it does |
|---|---|---|
| `kermany.ipynb` | Kermany chest X-ray dataset (children, 5,856 images) | Trains and tests DSRA-Net on the official split (624 test images), shows the output of every block, Grad-CAM maps, model size and speed, and runs patient-wise 5-fold cross-validation |
| `rsna.ipynb` | RSNA Pneumonia Detection Challenge (adults, 14,863 images: normal vs. lung opacity) | Trains and tests DSRA-Net on three patient-level 70/15/15 splits and checks Grad-CAM against the radiologists' boxes (pointing game) |

Both notebooks are saved with their outputs, so the results can be seen without running them.

## Results

| Dataset | Protocol | Accuracy | AUC |
|---|---|---|---|
| Kermany | Patient-wise 5-fold cross-validation | 96.68 ± 0.95% | 0.993 |
| Kermany | Official test set (3 models + test-time augmentation) | 92.95% | 0.983 |
| RSNA | Three patient-level splits | 93.72 ± 0.32% | 0.9815 |

On RSNA, the most activated Grad-CAM pixel falls inside the radiologist's box in 68.8% of pneumonia images (14.0% expected by chance).

## Datasets

The datasets are not included in this repository. Download them from:

- **Kermany:** [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) or its [Kaggle copy](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **RSNA:** [RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

## How to run

1. Open the notebook on Kaggle.
2. Add the dataset with **Add Input**: the Kermany dataset for `kermany.ipynb`, the RSNA dataset for `rsna.ipynb`. The notebook finds the data folder by itself.
3. Turn on the GPU (the results were produced on a Tesla T4) and click **Run All**.

Tested with TensorFlow 2.20.
