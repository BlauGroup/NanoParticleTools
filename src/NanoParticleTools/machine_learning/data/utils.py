from NanoParticleTools.machine_learning.data import NPMCDataset
import torch

import warnings
import os

SUNSET_SPECIES_TABLE = {
    1: sorted(["Yb", "Er", "Xsurfacesix"]),
    2: sorted(["Yb", "Er"]),
    3: sorted(["Yb", "Er", "Xsurfacesix", "Tm"]),
    4: sorted(["Yb", "Er"]),
    5: sorted(["Yb", "Er", "Nd"]),
    6: sorted(['Yb', 'Er', "Xsurfacesix", 'Tm', 'Nd', 'Ho', 'Eu', 'Sm', 'Dy'])
}


def get_sunset_datasets(
    sunset_ids: int | list[int],
    feature_processor_cls,
    label_processor_cls,
    data_path: str = "./",
    feature_processor_kwargs=None,
    label_processor_kwargs=None,
    val_split_fraction: float = 0.15,
    random_split: int = 10,
    dataset_kwargs: dict = None,
) -> tuple[NPMCDataset, NPMCDataset, NPMCDataset, NPMCDataset]:
    """
    Get the training, validation, iid testing, and ood testing datasets for SUNSET.

    Args:
        sunset_ids: The SUNSET dataset id. If multiple are specified in list form, then they will
            be concatenated.
        feature_processor_cls: The class used to construct feature processor.
        label_processor_cls: The class used to construct label processor.
        feature_processor_kwargs: The kwargs passed to the feature processor.
            Exclude the possible_elements kwarg, since this will be automatically added
            based on the elements in the dataset.
        label_processor_kwargs: The kwargs passed to the label processor.
        random_split: The random split id for train-val splitting.
    """
    if isinstance(sunset_ids, int):
        sunset_ids = [sunset_ids]

    _dataset_kwargs = {'gpu_training': False,
                       'cache_in_memory': True}

    if dataset_kwargs is not None:
        _dataset_kwargs.update(dataset_kwargs)

    if feature_processor_kwargs is None:
        feature_processor_kwargs = {}
    if label_processor_kwargs is None:
        label_processor_kwargs = {}

    possible_elements = set()
    if 'possible_elements' in feature_processor_kwargs:
        warnings.warn(
            "Possible_elements found in feature processor kwargs, updating "
            "with sunset elements.")
        possible_elements.update(feature_processor_kwargs['possible_elements'])
    for i in sunset_ids:
        possible_elements.update(SUNSET_SPECIES_TABLE[i])

    feature_processor_kwargs['possible_elements'] = sorted(list(possible_elements))

    feature_processor = feature_processor_cls(**feature_processor_kwargs)
    label_processor = label_processor_cls(**label_processor_kwargs)

    train_datasets = []
    val_datasets = []
    iid_test_datasets = []
    ood_test_datasets = []

    for i in sunset_ids:
        training_dataset = NPMCDataset(os.path.join(data_path,
                                                    f"SUNSET-{i}.json"),
                                       feature_processor,
                                       label_processor,
                                       **_dataset_kwargs)

        # split the training dataset into a training and validation set
        val_size = int(len(training_dataset) * val_split_fraction)
        train_size = len(training_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            training_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_split))

        iid_test_dataset = NPMCDataset(os.path.join(data_path,
                                                    f"SUNSET-{i}-IID.json"),
                                       feature_processor,
                                       label_processor,
                                       **_dataset_kwargs)

        ood_test_dataset = NPMCDataset(os.path.join(data_path,
                                                    f"SUNSET-{i}-OOD.json"),
                                       feature_processor,
                                       label_processor,
                                       **_dataset_kwargs)

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        iid_test_datasets.append(iid_test_dataset)
        ood_test_datasets.append(ood_test_dataset)

    if len(sunset_ids) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        iid_test_dataset = torch.utils.data.ConcatDataset(iid_test_datasets)
        ood_test_dataset = torch.utils.data.ConcatDataset(ood_test_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
        iid_test_dataset = iid_test_datasets[0]
        ood_test_dataset = ood_test_datasets[0]

    return train_dataset, val_dataset, iid_test_dataset, ood_test_dataset
