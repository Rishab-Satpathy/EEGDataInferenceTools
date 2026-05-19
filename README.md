# Acoustic-Derivative BCI & The RADAR Architecture

## Overview
This repository contains a continuously evolving Brain-Computer Interface (BCI) framework. It initially began as an exploration into decoding Left/Right Hand Motor Imagery by treating EEG spatial gradients as acoustic waveforms. 

In its latest iteration, the pipeline has evolved into the **RADAR Architecture (Resilient Architecture for Dynamic Affect Recognition)**. RADAR is a hardware-agnostic, dual-stream fusion pipeline designed for industrial edge-computing. It decodes complex human emotions and cognitive stress while actively combatting **Concept Drift** (hardware degradation and biological fatigue) over continuous real-world deployments.

---

## Repository Evolution & Notebook Structure

The project has evolved through three distinct research phases, fully contained within self-executing Jupyter Notebooks.

### Phase 1: Deep Learning & BCI Comp IV (Motor Imagery)
* `notebooks/V1_CompleteModelVersion1.ipynb` - Initial LSTM/Autoencoder experiments.
* `notebooks/V2_RF_LeakyRelu_SVMs.ipynb` - Transition to manual feature extraction.
* `notebooks/V3_hybridCAE_RF.ipynb` - Establishing the first XGBoost baselines on standard EEG bands.

### Phase 2: The "Clean-Pipe" & GigaScience (Motor Imagery)
* `notebooks/V4_UsingLibrosaCAE.ipynb` - Shift to 64-channel Cho2017 dataset. 
* `notebooks/V5_LibrosaSpatialDifferenceXGBoost.ipynb` - Implementation of Dual-Band Acoustic Extraction (MFCCs, ZCR, Spectral Centroid) and Temporal Micro-Slicing. Proved Event-Related Desynchronization (ERD) mathematically using XGBoost feature importance.

### Phase 3: The RADAR Architecture [LATEST] (Affect & Emotion Recognition)
* `notebooks/V6_RADAR_ConsumerEdge_DREAMER.ipynb` - Testing the pipeline on the 14-channel dry-electrode DREAMER dataset for binary stress classification.
* `notebooks/V7_RADAR_ClinicalScale_SEED_IV.ipynb` - Scaling to the 62-channel wet-electrode SEED-IV dataset. Introduces Tri-Phasic Evaluation (Stratified, Chronological, and Global LOSO) to measure Concept Drift.

---

## The RADAR Architecture (Version 7)

The latest emotion-recognition pipeline abandons computationally heavy, static deep learning in favor of a lightweight edge-computing philosophy.

### 1. Hardware-Adaptive Filtration & MTI
* **Dynamic Power Gating:** Automatically scaling impedance ceilings (800 µV² for dry headsets, 5000 µV² for wet rigs) to destroy macro-motion artifacts (e.g., violent jaw clenching).
* **The Muscle Tension Indicator (MTI):** Instead of discarding all noise, the 35–50Hz high-gamma band is isolated. This captures subconscious facial micro-tension—a highly correlated physiological proxy for stress.

### 2. Acoustic & Spatial Feature Extraction
Treating the brainwave as a raw acoustic waveform, the pipeline extracts audio textures via `librosa` (MFCCs, Zero Crossing Rate, RMS) alongside traditional Differential Entropy (DE) and spatial asymmetry metrics (DASM, RASM).

### 3. The 85/15 Dual-Stream Fusion Engine
* **Stream A (Primary):** A hyper-tuned XGBoost model (85% weight) handles complex, non-linear classification boundaries.
* **Stream B (Helper):** A 1D-Convolutional Autoencoder (15% weight) constrained by a severe **32-node bottleneck**. This strict compression acts as a spatial regularizer, preventing the model from memorizing temporal noise and acting as a "confidence sharpener."

---

## Findings: The "Chronological Collapse"

Current BCI literature relies on randomized evaluation, which artificially inflates accuracy by peeking into the biological future. The RADAR pipeline was explicitly designed to expose the reality of industrial deployment across time using the 4-class SEED-IV dataset:

1. **The Laboratory Ceiling (Stratified K-Fold): `91.28%`** * When data is randomized, neutralizing drift, the dual-stream architecture achieves near-perfect multi-class mapping.
2. **The Temporal Drift (Chronological Split): `33.32%`**
   * When trained on Session 1 and tested sequentially on Session 3, performance catastrophically collapses. This proves that **Concept Drift** (gel drying, cognitive fatigue) destroys static AI models over time.
3. **The Out-of-the-Box Reality (Global LOSO): `41.49%`**
   * Testing on completely unseen human subjects clears the 25% random-chance threshold, but proves that inter-subject variability (skull thickness, cortical folding) limits zero-shot deployment.

**Conclusion:** Edge BCI cannot rely on static models. It requires continuous, *Recursive-Resilient* on-device weight updates to survive temporal drift.

---

## Quick Start

Clone the repository and install dependencies:
```bash
git clone [https://github.com/Rishab-Satpathy/EEGDataInferenceTools.git](https://github.com/Rishab-Satpathy/EEGDataInferenceTools.git)
cd EEGDataInferenceTools
pip install -r requirements.txt

Note: Datasets (MOABB / SEED-IV) may require local downloading or API authentication depending on the selected notebook.
