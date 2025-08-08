import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import timm
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from transformers import AutoModelForImageClassification

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Fine-tuning on {device}")

BATCH_SIZE = 32
NUM_CLASSES = 10
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
generator = torch.Generator().manual_seed(SEED)

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
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

# Dataset and loaders
train_dataset = datasets.CIFAR10(
    root='data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(
    root='data', train=False, download=True, transform=val_transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Models to fine-tune
models = [
    "levit_192",
    "levit_conv_192"
]


def fine_tune_model(model_name, num_epochs=50, initial_lr=3e-4, warmup_epochs=5):
    print(f"\nFull fine-tuning: {model_name}")

    if model_name == "vit-small-patch16-224":
        model = AutoModelForImageClassification.from_pretrained(
            "WinKawaks/vit-small-patch16-224")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)
    else:
        model = timm.create_model(model_name, pretrained=True)
        model.reset_classifier(NUM_CLASSES)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # Scheduler with warmup
    num_steps = num_epochs * len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=1e-6,
        warmup_t=warmup_epochs * len(train_loader),
        warmup_lr_init=1e-6,
        cycle_limit=1
    )

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Debug: gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(
            f"Epoch {epoch+1} | Val Loss: {val_loss / len(val_loader):.4f} | Val Accuracy: {val_accuracy:.4f}")

    # Save model
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    model_path = f'{model_name}_cifar10_full.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Saved model to: {model_path}")
    print(
        f"🕒 Total Time: {elapsed_formatted} | Final Val Accuracy: {val_accuracy:.4f}")

    # Log to file
    with open("fine_tuning_times.log", "a") as f:
        f.write(
            f"Model: {model_name}, Time: {elapsed_formatted} "
            f"({elapsed:.2f}s), Final Val Accuracy: {val_accuracy:.4f}\n"
        )


# Run training
for model_name in models:
    fine_tune_model(model_name)
