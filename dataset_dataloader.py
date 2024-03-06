from typing import Union

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def load_MNIST_into_dataloaders(valid_split: Union[int, float, bool]=False, root: Union[str, None]=None, batch_size: int = 64, verbose: bool = False):
    
    if root:
        train_set = datasets.MNIST(root=root, train=True, transform=ToTensor(), download=True)
        test_set = datasets.MNIST(root=root, train=False, transform=ToTensor(), download=True)
        if verbose:
            print(f'Downloading MNIST dataset to "./{root}/" if not already downloaded')
    else:
        train_set = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
        test_set = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)
        if verbose:
            print('Downloading MNIST dataset to "./data" if not already downloaded')
    
    if valid_split:
        if valid_split == True:
            train_set, valid_set = random_split(train_set, [len(train_set) - len(test_set), len(test_set)], generator=torch.Generator().manual_seed(1))
        elif isinstance(valid_split, float): 
            train_set, valid_set = random_split(train_set, [int(len(train_set) * (1 - valid_split)), int(len(train_set) * valid_split)], generator=torch.Generator().manual_seed(1))
        elif isinstance(valid_split, int):
            train_set, valid_set = random_split(train_set, [len(train_set) - valid_split, valid_split], generator=torch.Generator().manual_seed(1))
        
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        if verbose:
            print(f'Training set contains {len(train_dl.dataset)} examples.\nValidation set contains {len(valid_dl.dataset)} examples\nTest set contains {len(test_dl.dataset)} examples\nAll with {batch_size} sized batches')
        return train_dl, valid_dl, test_dl      
            

    else:
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        if verbose:
            print(f'Training set contains {len(train_dl.dataset)} examples.\nTest set contains {len(test_dl.dataset)} examples\nAll with {batch_size} sized batches') 
        return train_dl, test_dl
    
if __name__ == '__main__':
    train, valid, test = load_MNIST_into_dataloaders(valid_split=0.3, root='test', verbose=True)
    print(train, valid, test)     
                  
      