DEBUG = False
DEBUG_SAMPLES = 1000

EPOCHS = 3
BATCH_SIZE = 32

RESULTS_DIR = "../results"
MODEL_PATH = "../models/resnet18_3epochs_best.pth"
HISTORY_PATH = "../results/training_history.csv"

import os
import csv

import torch
import torch.nn as nn
from tqdm import tqdm

from model import create_model
from data import load_data, create_splits
from dataset import create_dataloaders


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = outputs.argmax(
            dim=1
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        current_loss = (
            total_loss / total_samples
        )

        current_accuracy = (
            total_correct / total_samples
        )

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_accuracy * 100:.2f}%"
        )

    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )

    return epoch_loss, epoch_accuracy


def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            current_loss = (
                total_loss / total_samples
            )

            current_accuracy = (
                total_correct / total_samples
            )

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_accuracy * 100:.2f}%"
            )

    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )

    return epoch_loss, epoch_accuracy


def main():

    # ========================================
    # CONFIGURAÇÃO
    # ========================================
    learning_rate = 0.0001

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=== CONFIGURAÇÃO ===")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {learning_rate}")

    # ========================================
    # DADOS
    # ========================================

    print("\n=== CARREGANDO DATASET ===")

    dataset = load_data()

    print(f"Total: {len(dataset)}")

    # ========================================
    # SPLIT
    # ========================================

    print("\n=== CRIANDO SPLITS ===")

    (
        train_indices,
        validation_indices,
        test_indices
    ) = create_splits(dataset, debug=DEBUG, debug_samples=DEBUG_SAMPLES)

    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(validation_indices)}")
    print(f"Test: {len(test_indices)}")

    # ========================================
    # DATALOADERS
    # ========================================

    print("\n=== CRIANDO DATALOADERS ===")

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        dataset,
        train_indices,
        validation_indices,
        test_indices,
        batch_size=BATCH_SIZE
    )

    # ========================================
    # MODELO
    # ========================================

    print("\n=== CRIANDO MODELO ===")

    model = create_model()

    model = model.to(device)

    # ========================================
    # LOSS
    # ========================================

    criterion = nn.CrossEntropyLoss()

    # ========================================
    # OPTIMIZER
    # ========================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ========================================
    # HISTÓRICO
    # ========================================

    history = []

    best_validation_accuracy = 0.0

    # ========================================
    # TREINAMENTO
    # ========================================

    print("\n=== INICIANDO TREINAMENTO ===")

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        })

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "\n✓ Melhor modelo salvo!"
            )        

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        HISTORY_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy"
            ]
        )

        writer.writeheader()

        writer.writerows(history)

    print(f"\nHistórico salvo em: " f"{HISTORY_PATH}")

    print("\n=== TREINAMENTO FINALIZADO ===")

    print(
        f"Melhor Validation Accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )    


if __name__ == "__main__":
    main()