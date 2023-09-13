import abc
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

class TorchDataPreprocessing(metaclass=abc.ABCMeta):

    def __init__(self, batch_size=64):
        # If there's a GPU available...
        if torch.cuda.is_available():

            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")

        # If not...
        else:
            self.device = torch.device("cpu")

        self.batch_size = batch_size

    @staticmethod
    def convert_arrays_to_tensors(input_array):
        return torch.tensor(input_array)

    def prepare_torch_dataloader(self, *args ):

        # Create the DataLoader for our training set.
        train_data = TensorDataset(*args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        return train_dataloader

    def convert_model_to_device(self, model):
        return model.to(self.device)


