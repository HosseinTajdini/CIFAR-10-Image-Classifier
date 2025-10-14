import torchvision
import torch 
import os
def find_normalize_number(trainset:torchvision.datasets,
                          num_workers:int = os.cpu_count()):
    loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # flatten H*W
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean,std
