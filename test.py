import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
from transformers import AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference on {device}")

models = [
    "efficientvit_b1.r224_in1k",
    "levit_192",
    "levit_conv_192",
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
    root='data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

results = []
log_file = "inference_times.log"
csv_file = "cifar10_full_finetune_test_results.csv"

for model_name in models:
    pth_path = f"{model_name}_cifar10_full.pth"
    print(f"\nTesting {model_name}")

    if not os.path.exists(pth_path):
        print(f"  Model weights not found: {pth_path} -- skipping")
        with open(log_file, "a") as f:
            f.write(f"Model: {model_name}, Status: weights not found\n")
        continue

    # Model creation
    if model_name == "vit-small-patch16-224":
        model = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-small-patch16-224")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 10)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=10)

    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()

    start_time = time.time()
    print(
        f"Started inference for {model_name} at {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print(
        f"Finished inference for {model_name} at {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    print(
        f"Inference for {model_name} took {elapsed_formatted} (HH:MM:SS) or {elapsed_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f} ({100 * test_accuracy:.2f}%)")

    # Logging
    with open(log_file, "a") as f:
        f.write(
            f"Model: {model_name}, Time: {elapsed_formatted} (HH:MM:SS) or {elapsed_time:.2f} seconds, "
            f"Test Accuracy: {test_accuracy:.4f} ({100 * test_accuracy:.2f}%)\n"
        )

    results.append((model_name, test_accuracy,
                   elapsed_formatted, elapsed_time))

# Save results to CSV
with open(csv_file, "w") as f:
    f.write("model_name,test_accuracy,time_formatted,time_seconds\n")
    for model_name, acc, elapsed_formatted, elapsed_time in results:
        f.write(f"{model_name},{acc:.4f},{elapsed_formatted},{elapsed_time:.2f}\n")

print(f"\nAll inference results saved to:\n  - {log_file}\n  - {csv_file}")
