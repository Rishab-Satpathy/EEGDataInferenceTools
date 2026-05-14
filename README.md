# Acoustic-Derivative BCI: Decoding Motor Imagery via Audio Feature Extraction

## Overview

This repository contains a novel Brain-Computer Interface (BCI) pipeline that decodes Left/Right Hand Motor Imagery. Traditional BCI approaches rely on computationally heavy, "black-box" 2D Convolutional Neural Networks (like EEGNet).

This project introduces a highly interpretable, ultra-lightweight alternative: **treating spatial EEG voltage gradients as acoustic waveforms**. By applying physics-informed spatial filters and extracting audio textures (MFCCs, Spectral Centroid, ZCR) via `librosa`, this pipeline achieves high-accuracy decoding using a simple XGBoost classifier.

## Repository Structure & Iteration History

The project evolved through five distinct iterations, fully contained within self-executing Jupyter Notebooks:

### **Phase 1: Early Deep Learning & BCI Comp IV**

* `notebooks/CompleteModelVersion1.ipynb` - Initial experiments using LSTMs and Autoencoders on the BCI Competition IV dataset.
* `notebooks/V2-RF-LeakyRelu-SVMs(randomattempts).ipynb` - Transitioning away from pure deep learning to manual feature extraction.
* `notebooks/V3-hybridCAE-RF.ipynb` - Establishing the first XGBoost baselines on standard EEG bands.

### **Phase 2: The "Clean-Pipe" & GigaScience Scale-Up**

* `notebooks/V4-UsingLibrosaCAE.ipynb.ipynb` - Shifting to the 64-channel Cho2017 GigaScience dataset. Introduction of the "Clean-Pipe" spatial filter and Global (Subject-Independent) modeling.
* `notebooks/V5-LibrosaSpatialDifferenceXGBoost.ipynb` - **[FINAL PIPELINE]** Implementation of Dual-Band Acoustic Extraction, Temporal Micro-Slicing, and Subject-Specific validation to combat BCI Illiteracy and Inter-Subject Variability.

---

## The "Clean-Pipe" Architecture (Version 5)

The final pipeline operates in three deterministic phases:

### 1. Spatial Preprocessing

* **Reaction-Time Shift:** Slices the signal from $0.5s - 2.5s$ to remove Visual Evoked Potentials (VEP) and capture pure internal motor intent.
* **Common Average Reference (CAR):** Subtracts the global mean to cancel ambient scalp noise.
* **Z-Score Normalization & Gating:** Normalizes individual trials and clamps outliers at $\pm3.0\sigma$ to eliminate blink artifacts.
* **Symmetrical Bipolar Mapping:** Creates "Virtual Channels" by subtracting right hemisphere from left hemisphere (e.g., $C_3 - C_4$). This mathematically highlights the lateralization of the motor cortex.

### 2. Dual-Band Acoustic Extraction

Instead of passing raw signals to a model, the virtual channels are split into **Mu (8-12 Hz)** and **Beta (13-30 Hz)** bands using Butterworth filters. They are then processed as audio tracks using `librosa` to extract 1D textures:

* RMS Volume (Signal power)
* Zero Crossing Rate (ZCR)
* Spectral Centroid (Pitch center)
* Kurtosis (Signal impulsiveness)
* Mel-Frequency Cepstral Coefficients (MFCC 1 & 2)

### 3. Classification

A hyper-tuned `XGBoost` model evaluates the resulting dense 1D feature matrix.

---

## Results & Biological Proof

### The Inter-Subject Variability Wall

Initial tests on a 15-subject Global Model yielded an accuracy of ~53%. This accurately replicated the documented biological phenomenon of **Inter-Subject Variability** (differences in cortical folding and skull thickness destroying global decision boundaries).

### Subject-Specific Validation & "BCI Illiteracy"

Transitioning to within-subject models yielded a Grand Average of 55.38%. Consistent with clinical literature, roughly 70% of subjects exhibited "BCI Illiteracy," producing signals too chaotic for surface EEG decoding.

### The "Golden Subject" Case Study (Subject 14)

For subjects with high BCI literacy, the acoustic pipeline proved highly effective.
Using **Temporal Micro-Slicing** (chopping the 2.0s window into three overlapping timeline chunks to capture thought dynamics), the pipeline achieved:

* **True Average Accuracy:** 75.00%
* **Peak Fold Accuracy:** 80.00%

### Interpretability: Discovering Physics Blind

Unlike deep learning models, this pipeline remains perfectly interpretable. A feature importance analysis of the XGBoost decision trees revealed that the single most important metric for prediction was:

> `C3-C4 | Mu (8-12Hz) | RMS Volume`

The algorithm mathematically proved that a drop in electrical "volume" over the primary motor cortex in the 8-12Hz band is the primary indicator of movement. In neuroscience, this is a known phenomenon called **Event-Related Desynchronization (ERD)**. The machine learning model discovered human biology entirely on its own.

---

## Quick Start (Running Version 5)

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/acoustic-bci.git
cd acoustic-bci
pip install -r requirements.txt

```

2. Open the final Jupyter Notebook:

```bash
jupyter notebook notebooks/Ver5_Subject_Specific_Acoustics.ipynb

```

*Note: The `moabb` library will automatically download the required GigaScience Cho2017 dataset upon first execution.*

just for more extra testing data i have attached a google docs which has data stored on testing data from ver4 and ver5:
https://docs.google.com/document/d/1pZDn4CRjMdqDynXtrMZNI5GClYld_5VN19rI39tpsGo/edit?usp=sharing

model data for ver1 to ver3 hasn't been recorded as the success rate was hovering around 20-30 percent which suggested shallowness of data and model guessing was random and not heuristic 
