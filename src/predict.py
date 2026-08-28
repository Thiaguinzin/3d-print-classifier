import sys

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


# ========================================
# CONFIGURAÇÃO
# ========================================

MODEL_PATH = "../models/resnet18_3epochs_best.pth"

CLASS_NAMES = [
    "3d_printed",
    "not_3d_printed"
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ========================================
# TRANSFORMAÇÕES
# ========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ========================================
# PREDIÇÃO
# ========================================

def predict(image_path):

    print("Carregando modelo...")

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    # ------------------------------------
    # IMAGEM
    # ------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # ------------------------------------
    # PREDIÇÃO
    # ------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    # ------------------------------------
    # RESULTADO
    # ------------------------------------

    print("\n=== RESULTADO ===")

    print(
        f"Imagem: {image_path}"
    )

    print(
        f"Classe prevista: "
        f"{CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confiança: "
        f"{confidence * 100:.2f}%"
    )

    print("\n=== PROBABILIDADES ===")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name}: "
            f"{probabilities[0, index].item() * 100:.2f}%"
        )


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Uso:"
        )

        print(
            "python predict.py "
            "caminho/da/imagem.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict(
        image_path
    )