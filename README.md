# Electron/Photon Classification Using ResNet-15

## Overview

This project aims to classify electrons and photons using a deep learning model based on the ResNet-15 architecture. The classification is performed on 32x32 matrices representing two channels: hit energy and time for two types of particles, electrons and photons, hitting the detector. The datasets are provided by CERN and are used for the ML4Sci organization.

## Dataset Description

- **Photon Dataset**: [Photon Data](https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc)
- **Electron Dataset**: [Electron Data](https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA)

Each dataset consists of 32x32 matrices with two channels: hit energy and time.

## Project Requirements

- Use a ResNet-15 model architecture (modifiable).
- Achieve the highest possible classification score.
- Train the model on 80% of the data.
- Evaluate the model on the remaining 20%.
- Avoid overfitting on the test dataset.

## Model Architecture

The ResNet-15 architecture used in this project is a modified version of the ResNet model. Below is the architecture:

