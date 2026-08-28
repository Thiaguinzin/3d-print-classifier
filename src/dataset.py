from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transforms():

    transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transform


class ThreeDPrintedDataset(Dataset):

    def __init__(self, dataset, indices, transform=None):

        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        dataset_index = self.indices[index]

        example = self.dataset[dataset_index]

        image = example["image"]
        label = example["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    dataset,
    train_indices,
    validation_indices,
    test_indices,
    batch_size=32
):

    transform = get_transforms()

    train_dataset = ThreeDPrintedDataset(
        dataset=dataset,
        indices=train_indices,
        transform=transform
    )

    validation_dataset = ThreeDPrintedDataset(
        dataset=dataset,
        indices=validation_indices,
        transform=transform
    )

    test_dataset = ThreeDPrintedDataset(
        dataset=dataset,
        indices=test_indices,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )