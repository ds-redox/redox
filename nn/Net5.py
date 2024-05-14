import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd


class Net5(nn.Module):
    def __init__(self, n_features, M):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_features, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 2),
            nn.ReLU(),
            nn.Linear(2, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, 2),
            nn.LogSoftmax(dim = -1)
    )
        
                
    def forward(self, x):
        output = self.fc(x)
        return output
    
    def train_round(self, device, optimizer, train_loader, loss_f):
        accumulator = 0
        total = 0
        correct = 0
        
        self.train()
        
        for data, targets in train_loader:
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            output = self.forward(data)
            
            loss = loss_f(output, targets)
            
            loss.backward()
            optimizer.step()
            accumulator += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).float().sum().item()

        train_loss = accumulator / total
        train_acc = correct / total
            
        return (train_loss, train_acc)
    
    def test(self, device, test_loader, loss_f):
        accumulator = 0 
        total = 0 
        correct = 0 

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0 
        
        self.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                output = self.forward(data)
                loss = loss_f(output, targets)
                accumulator += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).float().sum().item()

                true_positives += ((targets == 1) & (predicted == 1)).float().sum().item()
                false_positives += ((targets == 0) & (predicted == 1)).float().sum().item()
                true_negatives += ((targets == 0) & (predicted == 0)).float().sum().item()
                false_negatives += ((targets == 1) & (predicted == 0)).float().sum().item()
            
            test_loss = accumulator / total   
            test_acc = correct / total
            test_rec = true_positives / (true_positives + false_negatives)
            
        return (test_loss, test_acc, test_rec)
    


    def model_evaluation(self, device, lr, train_loader, test_loader, optimizer_f, epochs, weight):
            
        optimizer = optimizer_f(self.parameters(), lr = lr)
        loss_f = nn.NLLLoss(weight = weight)

        trajectory_train_loss = list()
        trajectory_train_acc = list()
        trajectory_train_rec = list()
        trajectory_test_loss = list()
        trajectory_test_acc = list()
        trajectory_test_rec = list()
        
        for epoch in tqdm(np.arange(epochs)):

            train_loss, train_acc = self.train_round(device, optimizer, train_loader, loss_f)
            
            if epoch%2 == 0:
                train_loss, train_acc, train_rec = self.test(device, train_loader, loss_f)
                test_loss, test_acc, test_rec = self.test(device, test_loader, loss_f)
            
            trajectory_train_loss.append(train_loss)
            trajectory_train_acc.append(train_acc)
            trajectory_train_rec.append(train_acc)
            
            trajectory_test_loss.append(test_loss)
            trajectory_test_acc.append(test_acc)
            trajectory_test_rec.append(test_rec)

        self.trajectory_train_loss = trajectory_train_loss
        self.trajectory_train_acc = trajectory_train_acc
        self.trajectory_train_rec = trajectory_train_rec
        self.trajectory_test_loss = trajectory_test_loss
        self.trajectory_test_acc = trajectory_test_acc 
        self.trajectory_test_rec = trajectory_test_rec

        train_loss, train_acc, train_rec = self.test(device, train_loader, loss_f)
        test_loss, test_acc, test_rec = self.test(device, test_loader, loss_f)
        
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.train_rec = train_rec
        self.test_loss = test_loss
        self.test_acc = test_acc
        self.test_rec = test_rec
        
        return None 
    
    def predict(self, test_X_2023):
        
        self.eval()
        _, predicted = torch.max(self.forward(test_X_2023), 1)
        predicted = predicted.detach().cpu().numpy().astype(bool)
        
        print(f"Number of predicted anomalies for 2023 {np.sum(predicted)}")    

        self.predictions = predicted

        return None
