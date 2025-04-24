# Effect of Pruning on Mutual Information in Deep Neural Networks

## Overview

This repository contains the code and report for a project investigating the impact of neural network pruning on the flow of information within a deep neural network. The analysis focuses on visualizing changes in the **Information Plane** (Mutual Information I(X;H) vs. I(Y;H)) during the training of a Multilayer Perceptron (MLP) on a PCA-reduced subset of the MNIST dataset.

Different pruning strategies (weight-based, neuron-based, layer-wise, global) applied *at initialization* are explored and compared against a baseline unpruned model.

## Files

*   `Code_For_Testing_Pruning_Effects.ipynb`: Jupyter Notebook containing the PyTorch implementation of the baseline model, pruned models, training loops, mutual information calculations, and plotting functions.
*   `Report_Analyzing_Observed_Effects_Of_Pruning.pdf`: The detailed project report providing background on the information bottleneck theory, pruning methods, experiment details, results analysis, and discussion.

## Key Concepts

*   **Information Plane:** Visualization of mutual information between layers and input (I(X;H)) vs. layers and output (I(Y;H)).
*   **Mutual Information (MI):** Estimated empirically using discretization (binning) of activations.
*   **Neural Network Pruning:** Techniques to remove weights or neurons to create sparser networks. Explored methods include:
    *   Unstructured L1 Weight Pruning (`torch.nn.utils.prune.L1Unstructured`)
    *   Structured L2 Neuron Pruning (`torch.nn.utils.prune.ln_structured`)
*   **Model:** 4-Hidden Layer MLP with `tanh` activations.
*   **Dataset:** MNIST digits '0' and '2', reduced to 12 dimensions via PCA.

## Findings Summary

The experiments demonstrate how different pruning strategies applied before training affect the information pathways and final performance of the network. Pruning often leads to significantly different MI dynamics compared to the baseline, sometimes resulting in lower overall MI while maintaining relatively high accuracy, suggesting alternative learning strategies in sparser networks. Neuron pruning generally had a more drastic impact than weight pruning. Please see the report for a full analysis.
