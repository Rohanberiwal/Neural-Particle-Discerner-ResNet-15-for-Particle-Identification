# Particle Classification with ResNet15

This repository contains code for training and evaluating a custom ResNet15 model for classifying particle types based on their hit energy and time matrices. The particles are classified into two categories: Electrons and Photons. The code includes data generation, preprocessing, model definition, training with cross-validation, and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Generation](#data-generation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Definition](#model-definition)
6. [Training and Validation](#training-and-validation)
7. [Testing](#testing)
8. [How to Use](#how-to-use)
9. [Results](#results)
10. [License](#license)

## Overview

The code is designed to:

- Generate synthetic data representing hit energy and time matrices for Electrons and Photons.
- Preprocess and augment the data.
- Define and train a custom ResNet15 model with L1 regularization.
- Perform k-fold cross-validation to evaluate the model's performance.
- Evaluate the model on a separate test dataset and generate predictions for random test cases.

## Requirements

To run this code, you need:

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

You can install the required libraries using pip:

```bash
pip install torch numpy scikit-learn matplotlib
