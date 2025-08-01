import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import timm
from tqdm import tqdm
from transformers import AutoModelForImageClassification
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Fine-tuning on {device}")

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(
    root='data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(
    root='data', train=False, download=True, transform=val_transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])


batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

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


def fine_tune_model(model_name, total_steps=15000, initial_lr=0.01):
    print(f"Full fine-tuning {model_name}")

    if model_name == "vit-small-patch16-224":
        model = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-small-patch16-224")
    else:
        model = timm.create_model(model_name, pretrained=True)

    if model_name == "vit-small-patch16-224":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 10)
    else:
        model.reset_classifier(10)

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=initial_lr, momentum=0.9)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    global_step = 0
    epoch = 0
    while global_step < total_steps:
        model.train()
        running_loss = 0.0
        num_batches = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            if global_step >= total_steps:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            num_batches += 1
            global_step += 1
        if num_batches > 0:
            print(f'Epoch {epoch+1}, Train Loss: {running_loss / num_batches}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(
                    outputs, 'logits') else outputs
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        print(
            f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {val_accuracy}')

        epoch += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(
        f"Fine-tuning {model_name} took {elapsed_formatted} (HH:MM:SS) or {elapsed_time:.2f} seconds")

    torch.save(model.state_dict(), f'{model_name}_cifar10_full.pth')
    print(f"Saved fully fine-tuned model to {model_name}_cifar10_full.pth")

    log_file = "fine_tuning_times.log"
    with open(log_file, "a") as f:
        f.write(
            f"Model: {model_name}, Time: {elapsed_formatted} (HH:MM:SS) or {elapsed_time:.2f} seconds\n")


for model_name in models:
    fine_tune_model(model_name)
