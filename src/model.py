import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model():

    # Carrega ResNet18 pré-treinada
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Número de características produzidas
    # pela última camada da ResNet
    num_features = model.fc.in_features

    # Substituímos a classificação original
    # pela nossa classificação binária
    model.fc = nn.Linear(
        num_features,
        2
    )

    return model