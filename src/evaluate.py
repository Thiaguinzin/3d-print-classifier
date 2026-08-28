import torch
import torch.nn as nn
from tqdm import tqdm

from model import create_model
from data import load_data, create_splits
from dataset import create_dataloaders

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc="Evaluating",
            leave=False
        ):

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

            # Guarda previsões
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Guarda labels reais
            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return (
        loss,
        accuracy,
        all_labels,
        all_predictions
    )


def main():

    # ========================================
    # CONFIGURAÇÃO
    # ========================================

    batch_size = 32

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=== CONFIGURAÇÃO ===")
    print(f"Device: {device}")

    # ========================================
    # DATASET
    # ========================================

    print("\n=== CARREGANDO DATASET ===")

    dataset = load_data()

    # IMPORTANTE:
    # Mesmo split utilizado no treinamento
    train_indices, validation_indices, test_indices = (
        create_splits(
            dataset,
            debug=False
        )
    )

    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(validation_indices)}")
    print(f"Test: {len(test_indices)}")

    # ========================================
    # DATALOADERS
    # ========================================

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        dataset,
        train_indices,
        validation_indices,
        test_indices,
        batch_size=batch_size
    )

    # ========================================
    # MODELO
    # ========================================

    print("\n=== CARREGANDO MODELO ===")

    model = create_model()

    model_path = "../models/resnet18_3epochs_best.pth"

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    print(
        f"Modelo carregado: {model_path}"
    )

    # ========================================
    # LOSS
    # ========================================

    criterion = nn.CrossEntropyLoss()

    # ========================================
    # TESTE
    # ========================================

    print("\n=== AVALIANDO TESTE ===")

    (
        test_loss,
        test_accuracy,
        test_labels,
        test_predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "3d_printed",
            "not_3d_printed"
        ]
    )
    
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )
    
    display.plot(
        ax=ax,
        values_format="d"
    )
    
    ax.set_title(
        "Matriz de Confusão - ResNet18"
    )
    
    plt.tight_layout()
    
    plt.savefig(
        "../results/confusion_matrix.png",
        dpi=300
    )
    
    plt.close()
    
    print(
        "\nMatriz de confusão salva em:"
        " ../results/confusion_matrix.png"
    )    

    print("\n=== MATRIZ DE CONFUSÃO ===")

    print(cm)

    report = classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "3d_printed",
            "not_3d_printed"
        ]
    )

    print("\n=== CLASSIFICATION REPORT ===")

    print(report)

if __name__ == "__main__":
    main()