from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd



def train_test_loaders(train_X, train_y, test_X, test_y, batch_size_train = 1000, batch_size_test = 1000):
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size_test)

    return train_loader, test_loader
