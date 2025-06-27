# Project: 3D Object Localization from Synthetic Data

This project is an end-to-end pipeline for training a deep learning model to solve a 3D computer vision task using synthetically generated data.

## High-Level Goal

The primary objective is to **train a model that can infer the 3D location of an object by looking at it from multiple different viewpoints.** This is a classic 3D reconstruction problem, and this pipeline provides the tools to generate data, load it, and run a training loop.

## Prerequisites

Before running the pipeline, it's important to set up the correct environments.

### 1. System Python Environment (for Training)

The training scripts (`train.py`, `dataset.py`, `model.py`) run on your system's Python.

- **Python Version**: It is highly recommended to use a stable Python version such as **Python 3.10 or 3.11**. As of early 2024, major libraries like PyTorch may not have stable releases for very new versions like Python 3.13.
- **Virtual Environment**: To avoid conflicts with other projects, you should create a virtual environment.

