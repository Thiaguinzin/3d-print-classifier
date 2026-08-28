import os

import matplotlib.pyplot as plt
import torch

from data import load_data, create_splits
from dataset import create_dataloaders
from model import create_model


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "../models/resnet18_3epochs_best.pth"
RESULTS_DIR = "../results/errors"


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("Carregando dataset...")

    dataset = load_data()

    (
        train_indices,
        validation_indices,
        test_indices
    ) = create_splits(
        dataset,
        debug=False
    )

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        dataset,
        train_indices,
        validation_indices,
        test_indices,
        batch_size=32
    )

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    errors = []

    # ========================================
    # PREDIÇÕES
    # ========================================

    with torch.no_grad():

        for batch_idx, (
            images,
            labels
        ) in enumerate(test_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            for i in range(len(labels)):

                if predictions[i] != labels[i]:

                    errors.append({
                        "batch": batch_idx,
                        "position": i,
                        "real": labels[i].item(),
                        "predicted": predictions[i].item()
                    })

    print(
        f"\nTotal de erros: {len(errors)}"
    )

    # ========================================
    # MOSTRAR PRIMEIROS ERROS
    # ========================================

    for error_idx, error in enumerate(
        errors[:10]
    ):

        batch = error["batch"]
        position = error["position"]

        # Recupera batch novamente
        for current_batch_idx, (
            images,
            labels
        ) in enumerate(test_loader):

            if current_batch_idx == batch:

                image = images[position]

                break

        image = image.permute(
            1,
            2,
            0
        )

        image = image.numpy()

        # Normalização visual simples
        image = (
            image - image.min()
        ) / (
            image.max() - image.min()
            + 1e-8
        )

        real_label = (
            "3d_printed"
            if error["real"] == 0
            else "not_3d_printed"
        )

        predicted_label = (
            "3d_printed"
            if error["predicted"] == 0
            else "not_3d_printed"
        )

        plt.figure(figsize=(5, 5))

        plt.imshow(image)

        plt.title(
            f"Real: {real_label}\n"
            f"Predito: {predicted_label}"
        )

        plt.axis("off")

        filename = (
            f"{RESULTS_DIR}/"
            f"error_{error_idx + 1}.png"
        )

        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=200
        )

        plt.close()

        print(
            f"Erro salvo: {filename}"
        )


if __name__ == "__main__":
    main()