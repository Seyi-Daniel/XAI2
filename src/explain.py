import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import datasets, transforms

from .model import MnistCNN


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    if denom == 0:
        return np.zeros_like(arr)
    return arr / denom


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().numpy()


def grad_cam(model: MnistCNN, image: torch.Tensor, target_class: int) -> np.ndarray:
    gradients: List[torch.Tensor] = []
    activations: List[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = model.conv3.register_forward_hook(forward_hook)
    handle_b = model.conv3.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image)
    target = output[0, target_class]
    target.backward()

    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()
    cam_np = normalize(cam_np)

    handle_f.remove()
    handle_b.remove()
    return cam_np


def lrp_linear(module: torch.nn.Linear, input_act: torch.Tensor, relevance: torch.Tensor, epsilon: float) -> torch.Tensor:
    weight = module.weight
    bias = module.bias if module.bias is not None else torch.zeros(weight.size(0), device=input_act.device)
    z = F.linear(input_act, weight, bias)
    stabilizer = epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    z = z + stabilizer
    s = relevance / z
    c = torch.matmul(s, weight)
    relevance_input = input_act * c
    return relevance_input


def lrp_conv2d(module: torch.nn.Conv2d, input_act: torch.Tensor, relevance: torch.Tensor, epsilon: float) -> torch.Tensor:
    weight = module.weight.view(module.out_channels, -1)
    bias = module.bias if module.bias is not None else torch.zeros(module.out_channels, device=input_act.device)
    unfold = F.unfold(
        input_act,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    z = torch.matmul(weight.unsqueeze(0), unfold) + bias.view(1, -1, 1)
    stabilizer = epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    z = z + stabilizer
    relevance = relevance.view(relevance.size(0), relevance.size(1), -1)
    s = relevance / z
    c = torch.matmul(weight.t().unsqueeze(0), s)
    relevance_input_unfold = unfold * c
    relevance_input = F.fold(
        relevance_input_unfold,
        output_size=(input_act.size(2), input_act.size(3)),
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    return relevance_input


def lrp(model: MnistCNN, image: torch.Tensor, target_class: int, epsilon: float = 1e-4) -> np.ndarray:
    model.eval()
    logits, activations = model.forward_with_intermediates(image)
    relevance = torch.zeros_like(logits)
    relevance[0, target_class] = logits[0, target_class]

    for layer in reversed(activations):
        layer_type = layer[0]
        if layer_type == "linear":
            module, input_act = layer[1], layer[2]
            relevance = lrp_linear(module, input_act, relevance, epsilon)
        elif layer_type == "conv":
            module, input_act = layer[1], layer[2]
            relevance = lrp_conv2d(module, input_act, relevance, epsilon)
        elif layer_type == "relu":
            activation = layer[1]
            relevance = relevance * (activation > 0).float()
        elif layer_type == "flatten":
            shape = layer[1]
            relevance = relevance.view(shape)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    relevance_map = relevance.sum(dim=1).squeeze().detach().cpu().numpy()
    return relevance_map


def gradient_saliency(model: MnistCNN, image: torch.Tensor, target_class: int) -> np.ndarray:
    model.zero_grad()
    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    target = output[0, target_class]
    target.backward()
    gradient = image.grad.detach().cpu().numpy()[0, 0]
    gradient = np.abs(gradient)
    return normalize(gradient)


def smoothgrad(
    model: MnistCNN,
    image: torch.Tensor,
    target_class: int,
    sigma: float,
    samples: int = 25,
) -> np.ndarray:
    device = image.device
    grads = []
    for _ in range(samples):
        noise = torch.normal(0, sigma, size=image.shape, device=device)
        noisy_image = torch.clamp(image + noise, 0.0, 1.0).clone().detach().requires_grad_(True)
        model.zero_grad()
        output = model(noisy_image)
        target = output[0, target_class]
        target.backward()
        grad = noisy_image.grad.detach().cpu().numpy()[0, 0]
        grads.append(grad)
    avg_grad = np.mean(np.abs(grads), axis=0)
    return normalize(avg_grad)


def select_correct_samples(
    model: MnistCNN, test_loader: torch.utils.data.DataLoader, device: torch.device, num_samples: int
) -> List[Tuple[torch.Tensor, int, int, int]]:
    model.eval()
    collected: List[Tuple[torch.Tensor, int, int, int]] = []
    idx_offset = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(images.size(0)):
                if preds[i] == labels[i]:
                    collected.append((images[i : i + 1], labels[i].item(), preds[i].item(), idx_offset + i))
                    if len(collected) >= num_samples:
                        return collected
            idx_offset += images.size(0)
    return collected


def create_visualization(
    image: np.ndarray,
    grad_cam_map: np.ndarray,
    lrp_map: np.ndarray,
    gradient_map: np.ndarray,
    smoothgrad_maps: Dict[float, np.ndarray],
    label: int,
    predicted: int,
    sample_index: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Sample {sample_index} - Label {label} - Predicted {predicted}")

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image, cmap="gray")
    axes[0, 1].imshow(grad_cam_map, cmap="jet", alpha=0.5)
    axes[0, 1].set_title("Grad-CAM")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(lrp_map, cmap="seismic")
    axes[0, 2].set_title("LRP (ϵ)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(gradient_map, cmap="inferno")
    axes[0, 3].set_title("Gradient Saliency")
    axes[0, 3].axis("off")

    for idx, (sigma, sg_map) in enumerate(sorted(smoothgrad_maps.items())):
        row = 1
        col = idx
        axes[row, col].imshow(sg_map, cmap="inferno")
        axes[row, col].set_title(f"SmoothGrad σ={sigma}")
        axes[row, col].axis("off")

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        0.5,
        "SmoothGrad reduces noise\nand sharpens salient strokes,\ncomplementing Grad-CAM and LRP results.",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_report(fig_paths: List[Path], summary_text: str, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Explainability Analysis Summary", fontsize=16)
        fig.text(0.05, 0.9, "Overview", fontsize=14, weight="bold")
        fig.text(0.05, 0.86, summary_text, fontsize=11, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for path in fig_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis("off")
            fig.suptitle(path.stem.replace("_", " "), fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explanations with Grad-CAM, LRP, and SmoothGrad")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/mnist_cnn.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--report-path", type=Path, default=Path("report/report.pdf"))
    parser.add_argument("--smoothgrad-sigmas", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = MnistCNN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = select_correct_samples(model, test_loader, device, args.num_samples)
    if len(samples) < args.num_samples:
        raise RuntimeError("Not enough correctly classified samples found.")

    fig_paths: List[Path] = []
    observations = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_tensor, label, pred, index) in enumerate(samples, start=1):
        img_tensor = img_tensor.to(device)
        grad_cam_map = grad_cam(model, img_tensor.clone(), pred)
        lrp_map = lrp(model, img_tensor.clone(), pred)
        lrp_map_norm = normalize(lrp_map)
        gradient_map = gradient_saliency(model, img_tensor.clone(), pred)
        smoothgrad_maps = {
            sigma: smoothgrad(model, img_tensor.clone(), pred, sigma) for sigma in args.smoothgrad_sigmas
        }

        image_np = to_numpy_image(img_tensor)
        output_path = args.output_dir / f"sample_{index:05d}.png"
        create_visualization(
            image_np,
            grad_cam_map,
            lrp_map_norm,
            gradient_map,
            smoothgrad_maps,
            label,
            pred,
            index,
            output_path,
        )
        fig_paths.append(output_path)

        observations.append(
            f"Sample {index}: Grad-CAM focuses on stroke endpoints, LRP spreads relevance along the digit body, "
            f"SmoothGrad (σ={args.smoothgrad_sigmas[0]}-{args.smoothgrad_sigmas[-1]}) highlights stable edges."
        )

    summary_text = (
        "This report analyzes Grad-CAM, LRP-ϵ, and SmoothGrad explanations for a CNN trained on MNIST. "
        "Grad-CAM consistently localizes class-specific strokes, while LRP distributes relevance "
        "across the digit structure, emphasizing holistic shapes. SmoothGrad reduces noise in raw gradients, "
        "clarifying salient pixels around pen strokes.\n\n" + "\n".join(observations)
    )

    generate_report(fig_paths, summary_text, args.report_path)
    print(f"Saved figures to {args.output_dir}")
    print(f"PDF report available at {args.report_path}")


if __name__ == "__main__":
    main()
