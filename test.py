import os
import time
import math
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
from transformers import AutoModelForImageClassification

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False
    print("WARNING: pynvml not installed, GPU power metrics will be unavailable.")


def get_nvml_handle(torch_dev_idx: int):
    """
    Map torch.cuda.current_device() to NVML index and return handle.
    pytorch device indices are affected by CUDA_VISIBLE_DEVICES, but NVML isn't—
    this naive assumes direct mapping. For advanced use, remap if CUDA_VISIBLE_DEVICES is set.
    """
    return pynvml.nvmlDeviceGetHandleByIndex(torch_dev_idx)


def safe_get_total_energy(handle):
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # in mJ
    except Exception:
        return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference on {device}")

if device.type == "cuda" and _HAS_NVML:
    pynvml.nvmlInit()
    nvml_handle = get_nvml_handle(torch.cuda.current_device())
else:
    nvml_handle = None

models = [
    "efficientvit_b1.r224_in1k",
    "levit_192", "levit_conv_192",
    "tiny_vit_5m_224.in1k",
    "tiny_vit_5m_224.dist_in22k_ft_in1k",
    "tiny_vit_11m_224.in1k",
    "tiny_vit_11m_224.dist_in22k_ft_in1k",
    "poolformerv2_s24.sail_in1k",
    "tiny_vit_21m_224.in1k",
    "tiny_vit_21m_224.dist_in22k_ft_in1k",
    "vit_small_patch16_224",
    "deit_small_patch16_224.fb_in1k",
    "deit_small_distilled_patch16_224.fb_in1k"
]

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.CIFAR10(
    root="data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, pin_memory=(device.type == "cuda"))

results = []
log_file = "inference_times.log"
csv_file = "cifar10_full_finetune_test_results.csv"

with open(csv_file, "w") as f:
    f.write("model_name,test_accuracy,time_seconds,total_energy_J,avg_power_W\n")

torch.cuda.empty_cache()
dummy = torch.randn(1, 3, 224, 224, device=device)
_ = dummy + dummy

for model_name in models:
    pth_path = f"{model_name}_cifar10_full.pth"
    print(f"\n-------------------------------------------")
    print(f"Testing model: {model_name}")

    if not os.path.exists(pth_path):
        msg = f"Model {model_name}: weights not found at {pth_path}, skipping\n"
        print("  " + msg.strip())
        with open(log_file, "a") as f:
            f.write(msg)
        continue

    if model_name == "vit_small_patch16_224":
        model = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-small-patch16-224")
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, 10)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=10)

    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start_time = time.time()
    print(
        f"  Started inference at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    if nvml_handle is not None:
        start_energy_mJ = safe_get_total_energy(nvml_handle)
    else:
        start_energy_mJ = None

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    end_time = time.time()
    print(
        f"  Finished at {time.strftime('%H:%M:%S', time.localtime(end_time))}")

    if nvml_handle is not None:
        end_energy_mJ = safe_get_total_energy(nvml_handle)
    else:
        end_energy_mJ = None

    test_acc = correct / total
    elapsed = end_time - start_time
    elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    if start_energy_mJ is not None and end_energy_mJ is not None:
        delta_energy_mJ = int(end_energy_mJ) - int(start_energy_mJ)
        total_energy_J = max(delta_energy_mJ / 1000.0, 0.0)
        avg_power_W = total_energy_J / elapsed if elapsed > 0 else float("nan")
    else:
        total_energy_J = None
        avg_power_W = None

    print(f"  Time elapsed           : {elapsed_fmt} ({elapsed:.2f}s)")
    print(f"  Accuracy               : {test_acc:.4f} ({100*test_acc:.2f}%)")
    if avg_power_W is not None:
        print(f"  GPU energy             : {total_energy_J:.1f} J")
        print(f"  Avg. GPU power         : {avg_power_W:.1f} W")
    else:
        print("  GPU energy/power metrics: unavailable (NVML/energy API unsupported)")

    with open(csv_file, "a") as f:
        f.write(f"{model_name},{test_acc:.4f},{elapsed:.2f},")
        if total_energy_J is not None and avg_power_W is not None:
            f.write(f"{total_energy_J:.1f},{avg_power_W:.1f}\n")
        else:
            f.write("N/A,N/A\n")

    with open(log_file, "a") as f:
        msg = (
            f"Model: {model_name}, Time: {elapsed_fmt} ({elapsed:.2f}s), "
            f"Accuracy: {test_acc:.4f} ({100*test_acc:.2f}%), "
        )
        if total_energy_J is not None and avg_power_W is not None:
            msg += f"Energy: {total_energy_J:.1f} J, AvgPower: {avg_power_W:.1f} W\n"
        else:
            msg += "Energy/Power: N/A\n"
        f.write(msg)

    results.append((model_name, test_acc, elapsed,
                   total_energy_J, avg_power_W))

print("\n✅ All models evaluated. Results saved to:")
print(f"  - {log_file}")
print(f"  - {csv_file}")

if nvml_handle is not None:
    pynvml.nvmlShutdown()
