import os
import numpy as np


def splits_iteration(splits_path, models_path, f):
    splits = os.listdir(splits_path)

    losses = []

    for split in splits:
        input()
        split_path = f'{splits_path}/{split}'
        model_path = f'{models_path}/{split}.pth'

        loss = f(split, split_path, model_path)
        losses.append(loss)

    return np.array(losses).mean(axis=0)