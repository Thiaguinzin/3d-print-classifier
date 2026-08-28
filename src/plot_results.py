import os
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


# ========================================
# CONFIGURAÇÃO
# ========================================

RESULTS_DIR = "../results"
HISTORY_PATH = "../results/training_history.csv"
ERRORS_DIR = "../results/errors"


# Resultado obtido no conjunto de teste
CONFUSION_MATRIX = np.array([
    [3847, 17],
    [54, 3810]
])

CLASS_NAMES = [
    "3d_printed",
    "not_3d_printed"
]


# ========================================
# CARREGAR HISTÓRICO
# ========================================

def load_history():

    epochs = []
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []

    with open(
        HISTORY_PATH,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            epochs.append(
                int(row["epoch"])
            )

            train_loss.append(
                float(row["train_loss"])
            )

            train_accuracy.append(
                float(row["train_accuracy"]) * 100
            )

            validation_loss.append(
                float(row["validation_loss"])
            )

            validation_accuracy.append(
                float(row["validation_accuracy"]) * 100
            )

    return (
        epochs,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy
    )


# ========================================
# PLOT LOSS
# ========================================

def plot_loss(
    epochs,
    train_loss,
    validation_loss
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Treinamento"
    )

    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validação"
    )

    plt.xlabel("Época")
    plt.ylabel("Loss")

    plt.title(
        "Evolução da função de perda"
    )

    plt.xticks(epochs)

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "training_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfico salvo: {output_path}"
    )


# ========================================
# PLOT ACCURACY
# ========================================

def plot_accuracy(
    epochs,
    train_accuracy,
    validation_accuracy
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Treinamento"
    )

    plt.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validação"
    )

    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")

    plt.title(
        "Evolução da acurácia"
    )

    plt.xticks(epochs)

    plt.ylim(90, 100)

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "training_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Gráfico salvo: {output_path}"
    )


# ========================================
# MATRIZ DE CONFUSÃO
# ========================================

def plot_confusion_matrix():

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=CONFUSION_MATRIX,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=axis,
        values_format="d"
    )

    axis.set_title(
        "Matriz de Confusão - Conjunto de Teste"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Matriz de confusão salva: {output_path}"
    )


# ========================================
# MONTAGEM DOS ERROS
# ========================================

def plot_errors():

    error_files = [
        os.path.join(
            ERRORS_DIR,
            f"error_{i}.png"
        )
        for i in range(1, 11)
    ]

    existing_files = [
        file
        for file in error_files
        if os.path.exists(file)
    ]

    if not existing_files:

        print(
            "Nenhuma imagem de erro encontrada."
        )

        return

    columns = 2

    rows = (
        len(existing_files) + columns - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, rows * 5)
    )

    axes = np.array(axes).reshape(-1)

    for index, file in enumerate(
        existing_files
    ):

        image = plt.imread(file)

        axes[index].imshow(image)

        axes[index].set_title(
            f"Erro {index + 1}"
        )

        axes[index].axis("off")

    for index in range(
        len(existing_files),
        len(axes)
    ):

        axes[index].axis("off")

    figure.suptitle(
        "Exemplos de classificações incorretas",
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "errors_montage.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Montagem dos erros salva: {output_path}"
    )


# ========================================
# MAIN
# ========================================

def main():

    print(
        "=== GERANDO VISUALIZAÇÕES ==="
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print(
        "\nCarregando histórico..."
    )

    (
        epochs,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy
    ) = load_history()

    print(
        f"Épocas encontradas: {epochs}"
    )

    print(
        "\nGerando gráfico de Loss..."
    )

    plot_loss(
        epochs,
        train_loss,
        validation_loss
    )

    print(
        "\nGerando gráfico de Accuracy..."
    )

    plot_accuracy(
        epochs,
        train_accuracy,
        validation_accuracy
    )

    print(
        "\nGerando matriz de confusão..."
    )

    plot_confusion_matrix()

    print(
        "\nGerando montagem dos erros..."
    )

    plot_errors()

    print(
        "\n=== FINALIZADO ==="
    )


if __name__ == "__main__":
    main()