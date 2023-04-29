from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset, Subset

from contact_dataset import ContactDataset


def load_datasets(  # pylint: disable=too-many-arguments
        num_clients: int = 10,
        iid: Optional[bool] = True,
        balance: Optional[bool] = True,
        val_ratio: float = 0.1,
        batch_size: Optional[int] = 32,
        seed: Optional[int] = 42,
) -> tuple[list[DataLoader[Any]], list[DataLoader[Any]], DataLoader[Any]]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between the
        clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """

    datasets, testset = _partition_data(num_clients, iid, balance, seed)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    # Split the dataset into num_clients parts.
    # num_items = len(datasets)
    # items_per_client = num_items // num_clients
    # client_data = []
    # for i in range(num_clients):
    #     start_idx = i * items_per_client
    #     # The last client gets the remaining items
    #     end_idx = (i + 1) * items_per_client if i < num_clients - 1 else num_items
    #     client_data.append(datasets[start_idx:end_idx])

    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def _partition_data(
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    trainset, testset = ContactDataset().partition(0.7, 0.3)
    train_length = len(trainset)
    partition_size = train_length // num_clients
    remainder = train_length % num_clients
    lengths = [partition_size] * num_clients
    for i in range(remainder):
        lengths[i] += 1
    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if balance:
            trainset = _balance_classes(trainset, seed)
            partition_size = int(len(trainset) / num_clients)
        shard_size = int(partition_size / 2)
        idxs = trainset.targets.argsort()
        sorted_data = Subset(trainset, idxs)
        tmp = []
        for idx in range(num_clients * 2):
            tmp.append(
                Subset(sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1)))
            )
        idxs_list = torch.randperm(
            num_clients * 2, generator=torch.Generator().manual_seed(seed)
        )
        datasets = [
            ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
            for i in range(num_clients)
        ]

    return datasets, testset


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in class_counts:
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled
