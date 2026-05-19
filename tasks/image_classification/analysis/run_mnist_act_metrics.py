import os
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.ctm import ContinuousThoughtMachine


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device, iterations):
    model = ContinuousThoughtMachine(
        iterations=iterations,
        d_model=128,
        d_input=128,
        heads=2,
        n_synch_out=128,
        n_synch_action=128,
        out_dims=10,
        synapse_depth=2,
        dropout=0.1,
        memory_length=15,
        deep_nlms=True,
        memory_hidden_dims=16,
        do_layernorm_nlm=False,
        backbone_type="resnet18-4",
        positional_embedding_type="none",
    ).to(device)

    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1, 1, 1)

    model.initial_rgb = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1).to(device)
    model.backbone = nn.Sequential(
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        Reshape(),
    ).to(device)
    model.compute_features = model.backbone
    return model


def load_data(full_eval=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_data = datasets.MNIST("data", train=False, transform=transform)
    if not full_eval:
        test_data = torch.utils.data.Subset(test_data, range(1000))
    batch_size = 256 if full_eval else 128
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    return test_loader


def maybe_train(model, device, iterations, model_path):
    if os.path.exists(model_path):
        return
    print(f"[train] training MNIST model because {model_path} is missing")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_data = torch.utils.data.Subset(train_data, range(5000))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(3):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions, certainties, _ = model(inputs)
            loss = torch.nn.functional.cross_entropy(predictions[..., -1], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("[train] training done and model saved")


def evaluate(model, loader, device, iterations, threshold):
    model.eval()
    total_correct = 0
    total_samples = 0
    exit_steps_all = []
    targets_all = []
    preds_all = []
    batch_latencies = []

    use_cuda_timer = device.type == "cuda"

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_cuda_timer:
                starter, ender = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                starter.record()
            start_time = time.perf_counter()

            if threshold is None:
                predictions, certainties, _ = model(inputs)
                exit_steps = torch.full((inputs.size(0),), iterations, device=device)
            else:
                predictions, certainties, _, exit_steps = model(
                    inputs, early_exit_threshold=threshold, return_exit_steps=True
                )

            if use_cuda_timer:
                ender.record()
                torch.cuda.synchronize()
                batch_latency_ms = starter.elapsed_time(ender)
            else:
                batch_latency_ms = (time.perf_counter() - start_time) * 1000

            batch_latencies.append(batch_latency_ms / inputs.size(0))

            final_logits = predictions[..., -1]
            preds = final_logits.argmax(dim=1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            exit_steps_all.append(exit_steps.detach().to("cpu"))
            targets_all.append(targets.detach().to("cpu"))
            preds_all.append(preds.detach().to("cpu"))

    exit_steps_all = torch.cat(exit_steps_all)
    targets_all = torch.cat(targets_all)
    preds_all = torch.cat(preds_all)

    accuracy = total_correct / total_samples
    mean_ticks = exit_steps_all.float().mean().item()
    std_ticks = exit_steps_all.float().std(unbiased=False).item()
    latency_mean = float(np.mean(batch_latencies))
    latency_std = float(np.std(batch_latencies, ddof=0))

    errors_mask = preds_all != targets_all
    error_exit_steps = exit_steps_all[errors_mask].tolist()

    return {
        "threshold": "baseline" if threshold is None else threshold,
        "accuracy": accuracy,
        "mean_ticks": mean_ticks,
        "std_ticks": std_ticks,
        "latency_per_image_ms_mean": latency_mean,
        "latency_per_image_ms_std": latency_std,
        "error_exit_steps": error_exit_steps,
        "exit_steps": exit_steps_all.tolist(),
    }


def save_tables(results, iterations, out_md, out_json):
    header = "| ACT Threshold | Accuracy | Mean Exit Tick +/- Std | Latency/image (ms) +/- Std | Proxy Tick Reduction |\n| :--- | :--- | :--- | :--- | :--- |\n"
    lines = [header]
    for r in results:
        tick_reduction = (iterations - r["mean_ticks"]) / iterations * 100
        line = f"| {r['threshold']} | {r['accuracy'] * 100:.2f}% | {r['mean_ticks']:.2f} +/- {r['std_ticks']:.2f} | {r['latency_per_image_ms_mean']:.3f} +/- {r['latency_per_image_ms_std']:.3f} | ~{tick_reduction:.1f}% |\n"
        lines.append(line)
    lines.append("\nProxy tick reduction is computed from per-sample exit ticks. The current implementation skips recurrent computation only after every sample in the batch has crossed the threshold.\n")
    with open(out_md, "w") as f:
        f.writelines(lines)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)


def plot_pareto(results, out_path):
    plt.figure(figsize=(7, 5))
    for r in results:
        label = str(r["threshold"])
        plt.scatter(r["mean_ticks"], r["accuracy"] * 100, s=80)
        plt.text(r["mean_ticks"] + 0.2, r["accuracy"] * 100, label)
    plt.xlabel("Mean Ticks (lower = cheaper)")
    plt.ylabel("Accuracy (%)")
    plt.title("MNIST Entropy Early Exit (Accuracy vs Mean Exit Tick)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_hist(results, iterations, out_path):
    thresholds = [r["threshold"] for r in results if r["threshold"] != "baseline"]
    n_cols = 3
    n_rows = math.ceil(len(thresholds) / n_cols)
    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    for idx, r in enumerate([res for res in results if res["threshold"] != "baseline"]):
        plt.subplot(n_rows, n_cols, idx + 1)
        bins = np.arange(0.5, iterations + 1.5, 1)
        plt.hist(r["error_exit_steps"], bins=bins, edgecolor="black", alpha=0.7)
        plt.title(f"Thr={r['threshold']} (errors: {len(r['error_exit_steps'])})")
        plt.xlabel("Exit Tick (misclassified)")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    iterations = 30
    model = build_model(device, iterations)
    model_path = "outputs/mnist/mnist_model.pth"
    maybe_train(model, device, iterations, model_path)

    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=device))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    loader = load_data(full_eval=True)

    thresholds = [None, 0.99, 0.95, 0.90, 0.80, 0.70, 0.50]
    results = []
    for thr in thresholds:
        r = evaluate(model, loader, device, iterations, thr)
        results.append(r)
        print(
            f"[ACT] thr={r['threshold']} acc={r['accuracy'] * 100:.2f}% ticks={r['mean_ticks']:.2f}+/-{r['std_ticks']:.2f} lat={r['latency_per_image_ms_mean']:.3f}ms"
        )

    os.makedirs("outputs/mnist", exist_ok=True)
    save_tables(
        results,
        iterations,
        "outputs/mnist/mnist_act_metrics.md",
        "outputs/mnist/mnist_act_metrics.json",
    )
    plot_pareto(results, "outputs/mnist/mnist_act_pareto.png")
    plot_error_hist(results, iterations, "outputs/mnist/mnist_act_error_hist.png")


if __name__ == "__main__":
    main()
