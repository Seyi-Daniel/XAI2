# Deliverables

- Python script(s) (`.py`) implementing all three methods.
- A short PDF report with visualizations and analysis.

---

## Dataset, Model, and Methods

- One of the following depending on your computing resources: **MNIST** or **CIFAR-10**
- Use a **self-trained CNN (3–5 convolutional layers)**
- You may use existing implementations of **Grad-CAM, LRP, and SmoothGrad**

---

## 1. Grad-CAM (15 points)

- Compute class activation maps for at least **5 correctly-classified test images**
- Overlay the Grad-CAM heatmap on the original image
- Visualize and interpret which image regions contribute most to the prediction

---

## 2. LRP (15 points)

- Use the same five samples as in the previous task and generate explanations using simple **LRP-ϵ rule**
- Compare the relevance distribution with Grad-CAM results
- Discuss whether LRP identifies similar or different areas of importance

---

## 3. SmoothGrad (15 points)

- Generate standard gradient saliency maps for the same images used in Grad-CAM
- Apply SmoothGrad to the same inputs for various σ values (e.g., **0.1, 0.2, 0.3**)
- Compare the resulting saliency maps in terms of **noise, sharpness, and interpretability**
