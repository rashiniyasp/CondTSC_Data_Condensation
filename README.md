# Dataset Condensation for Time Series Classification via Dual Domain Matching (CondTSC) - ECG5000 Analysis

## Project Overview

This Kaggle project implements and evaluates **Dataset Condensation for Time Series Classification via Dual Domain Matching (CondTSC)**, a novel framework proposed in a recent research paper. The core idea of CondTSC is to learn a very small, synthetic dataset that can replace a much larger original dataset for training deep learning models, drastically reducing training time while maintaining high predictive performance.

### Why Time Series Condensation?

Large time series datasets pose significant challenges for deep learning, particularly in terms of storage, processing, and training efficiency (e.g., for Neural Architecture Search or Continual Learning). Traditional data compression is often insufficient as it requires decompression before use. CondTSC addresses this by distilling the essential training dynamics.

### CondTSC's Novelty

Unlike previous dataset condensation methods primarily designed for image or graph data, CondTSC specifically targets time series by:
- Leveraging information from **both time and frequency domains**.
- Incorporating **multi-view data augmentation** in frequency-enhanced spaces.
- Utilizing **dual domain training** and **dual surrogate objectives** (gradient matching and embedding distribution matching) to ensure the synthetic data captures complex patterns.

## Project Structure & Experiments

This repository contains a series of Jupyter notebooks that document the step-by-step implementation and comprehensive evaluation of the CondTSC framework on the **ECG5000 heartbeat classification dataset**.

We conduct the following experiments:

1.  **Notebook 1: CondTSC Baseline Implementation **
    *   **Objective:** Establish the foundational CondTSC framework.
    *   **Configuration:** K-means initialization, standard FFT for frequency domain, L2-based mean difference for embedding matching.
    *   **Result (Cond-S on 1% data):** Achieved **78.46% accuracy** (vs. 93.14% for full data). Demonstrates significant training speedup.

2.  **Notebook 2: STFT Enhancement **
    *   **Objective:** Evaluate the impact of Short-Time Fourier Transform (STFT) for frequency domain representation.
    *   **Configuration:** K-means initialization, **STFT for frequency domain**, L2-based mean difference for embedding matching. Initial parameters.
    *   **Result (Cond-S on 1% data):** Achieved 78.68% accuracy (vs. 94.46% for full data using STFT).

3.  **Notebook 3: K-Medoids Initialization Enhancement **
    *   **Objective:** Evaluate the impact of K-Medoids for synthetic data initialization.
    *   **Configuration:** **K-Medoids initialization**, standard FFT for frequency domain, L2-based mean difference for embedding matching.
    *   **Result (Cond-S on 1% data):** Achieved 62.44% accuracy (vs. 94.18% for full data).

4.  **Notebook 4: MMD Loss Enhancement **
    *   **Objective:** Evaluate the impact of Maximum Mean Discrepancy (MMD) loss for embedding matching.
    *   **Configuration:** K-means initialization, standard FFT for frequency domain, **MMD loss for embedding matching**.
    *   **Result (Cond-S on 1% data):** Achieved 64.16% accuracy (vs. 94.18% for full data).

5.  **Notebook 5: All Enhancements Combined **
    *   **Objective:** Assess the synergistic effect of integrating all three enhancements.
    *   **Configuration:** **K-Medoids initialization**, **STFT for frequency domain**, **MMD loss for embedding matching**.
    *   **Result (Cond-S on 1% data):** Achieved 62.32% accuracy (vs. 94.64% for full data using STFT).

6.  **Notebook 2 (Rerun): STFT Enhancement **
    *   **Objective:** Demonstrate the critical role of hyperparameter tuning for STFT enhancement.
    *   **Configuration:** K-means initialization, **STFT for frequency domain**, L2-based mean difference for embedding matching. **Tuned parameters (fewer condensation/evaluation epochs, lower M_iters_T)**.
    *   **Result (Cond-S on 1% data):** Achieved an outstanding **80.64% accuracy** (vs. 94.84% for full data using STFT), representing the best performance for the condensed dataset in this study.

## Key Findings

-   The CondTSC framework successfully condenses the ECG5000 dataset by **100x** (40 samples from 4000).
-   Models trained on the condensed dataset demonstrate **orders of magnitude faster training times** compared to training on the full dataset.
-   **STFT is a highly effective feature representation** for ECG5000, significantly boosting the baseline performance of models.
-   **Hyperparameter tuning is paramount:** A carefully tuned STFT-enhanced configuration achieved the highest condensed dataset accuracy (**80.64%**), significantly narrowing the performance gap while retaining extreme efficiency.
-   Directly applying advanced techniques like K-Medoids or MMD without specific tuning can sometimes lead to reduced performance, highlighting their sensitivity and the need for careful parameter selection.

## How to Run the Notebooks

1.  Open the desired notebook on Kaggle.
2.  Ensure a GPU accelerator (e.g., T4) is enabled in the Notebook settings (`Accelerator` -> `GPU`).
3.  Run all cells in sequence.

## Repository Contents

-   `README.md`: This file.
-   `condtsc-base-model.ipynb`: Implementation of CondTSC baseline.
-   `condtsc-stft.ipynb`: Implementation of CondTSC with STFT enhancement (original params).
-   `condtsc-k-medoids.ipynb`: Implementation of CondTSC with K-Medoids initialization.
-   `condtsc-mmd.ipynb`: Implementation of CondTSC with MMD loss.
-   `condtsc-combined-model.ipynb`: Implementation of CondTSC with all combined enhancements.
-   `condtsc-stft-best-paras-model-80.ipynb`: Implementation of CondTSC with STFT enhancement and tuned parameters.
-   `ECG5000_...`: Data files for the ECG5000 dataset (typically provided as a Kaggle dataset input).

---
