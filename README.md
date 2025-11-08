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

Running the second command will create a structured asset folder in `outputs/` and produce `report/report.pdf` summarizing the observations.

Each correctly classified test sample gets its own directory (`outputs/sample_<index>/`) containing:

- `overview.png` – the multi-panel collage used in the PDF report.
- `original.png` – the grayscale MNIST digit.
- `grad_cam.png` – Grad-CAM heatmap overlaid on the input.
- `lrp_epsilon.png` – ε-LRP relevance visualization.
- `gradient_saliency.png` – absolute gradient saliency map.
- `smoothgrad_sigma*.png` – SmoothGrad saliency maps for every requested noise level.

These standalone images make it easy to reuse the explainability figures in slide decks, papers, or dashboards without re-running the pipeline.
