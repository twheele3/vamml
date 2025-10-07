# Volumetric Additive Manufacturing Machine Learning (VAMML) code base

## Overview
This repository is for code related to the VAMML method as described in (pending). This method is designed to generate 2D training data for VAM by creating batches of random shapes to print, then automatically aligns them for training data in an attention U-net model. A visual overview of VAMML:

![svg](assets/Overview.svg)

## Uses

This package is designed as a guided pipeline to automate the following functions:
- Geenerate random shapes and assemble a voxel array for printing.
- Process calibrated images acquired of prints to create aligned and scaled training pairs.
- Aggregate data between experiments/printing runs to assemble a training dataset. 
- Train a machine learning model to correct volumetric additive manufacturing shape fidelity.
- Correct voxel arrays to enhance shape fidelity based on material and print parameters.
- Iteratively produce training data.

## How to use

Install the package by entering the following command into the terminal of your preferred python environment.

```pip install git+https://github.com/twheele3/vamml.git```

Refer to the file ```example.ipynb``` for a walkthrough on how to use each step of the VAMML pipeline.

Also see the repository wiki for more details on individual processing steps (WIP).