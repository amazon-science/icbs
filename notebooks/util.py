# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import random

import numpy as np
import torch
from datasets import load_dataset as load_dataset_


def set_num_threads(num_threads):
    """Set number of threads to be used."""
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    # LLM specific
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def set_seed(seed):
    """Sets the seed for all random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_dataset(dataset_name, **dataset_args):
    """Loads the train and valid datasets for an arbitrary dataset.

    Uses datasets.load_dataset() to load the dataset - see that function's documentation
    for which datasets are supported.
    """
    train_dataset = load_dataset_(
        dataset_name,
        split="train",
        **dataset_args,
    )
    try:
        valid_dataset = load_dataset_(
            dataset_name,
            split="validation",
            **dataset_args,
        )
    except ValueError:
        # Some debugging datasets only contain a train set
        valid_dataset = load_dataset_(
            dataset_name,
            split="train",
            **dataset_args,
        )

    return train_dataset, valid_dataset
