from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_data():

    dataset = load_dataset(
        "cmudrc/3d-printed-or-not"
    )

    return dataset["train"]


def create_splits(dataset, debug=False, debug_samples=1000):

    labels = dataset["label"]

    # ========================================
    # DEBUG
    # ========================================

    if debug:

        debug_indices, _ = train_test_split(
            range(len(dataset)),
            train_size=debug_samples,
            stratify=labels,
            random_state=42
        )

        # Dataset reduzido apenas para o experimento
        indices = debug_indices

        labels = [
            dataset["label"][i]
            for i in indices
        ]

    else:

        indices = range(len(dataset))

    # ========================================
    # TRAIN / TEMP
    # ========================================

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=42
    )

    # Labels correspondentes ao TEMP
    temp_labels = [
        dataset["label"][i]
        for i in temp_indices
    ]

    # ========================================
    # VALIDATION / TEST
    # ========================================

    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42
    )

    return (
        train_indices,
        validation_indices,
        test_indices
    )