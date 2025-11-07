# Explainable AI Pipeline for MNIST CNN

This repository trains a small CNN on MNIST and generates explainability visualizations using Grad-CAM, LRP-ϵ, and SmoothGrad.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Train the CNN

```bash
python -m src.train --epochs 5 --batch-size 128 --model-path artifacts/mnist_cnn.pth
```

## 2. Generate Explanations and Report

```bash
python -m src.explain --model-path artifacts/mnist_cnn.pth --output-dir outputs --num-samples 5 --report-path report/report.pdf
```

Running the second command will create per-sample figures in `outputs/` and produce `report/report.pdf` summarizing the observations.
