from scipy.io import loadmat
import numpy as np
import pandas as pd
import copy
import time
import os
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

def load_Dataset(Train, Test):
    x_train = Train[:, 0:-1]
    y_train = Train[:, -1]

    x_test = Test[:, 0:-1]
    y_test = Test[:, -1]
    if False:
        y_train = np.expand_dims(y_train, axis = 1)
        y_train = np.hstack((y_train, (1 - y_train)))
        y_test = np.expand_dims(y_test, axis = 1)
        y_test = np.hstack((y_test, (1 - y_test)))

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    return x_train, y_train, x_test, y_test

def my_pca(x_train, x_test, k):
    x_train_mean = torch.mean(x_train, dim = 0)
    x_train_reduced = x_train - x_train_mean

    Sigma = torch.mm(x_train_reduced.t(), x_train_reduced) / len(x_train_reduced)

    u, s, v = torch.svd(Sigma)
    score = torch.sum(s[0:k]) / torch.sum(s)
    u_reduced = u[:, 0:k]
    
    x_train_reduced = torch.mm(x_train_reduced, u_reduced)
    
    x_test_mean = torch.mean(x_test, dim = 0)
    x_test_reduced = x_test - x_test_mean

    x_test_reduced = torch.mm(x_test_reduced, u_reduced)
    return x_train_reduced, x_test_reduced, score

# Accuracy
def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy

# Multilayer Perceptron 
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_classes)

    def forward(self, x_in, apply_softmax=False):
        a_1 = F.relu(self.fc1(x_in)) 
        a_2 = F.relu(self.fc2(a_1))
        y_pred = self.fc3(a_2)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
    
    
def train_model(pca_bool = False, pca_num = 10):
    # Load Dataset
    x_train, y_train, x_test, y_test = load_Dataset(Train, Test)
    score = 100
    if pca_bool:
        x_train, x_test, score = my_pca(x_train, x_test, pca_num)
    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Model configuration
    [m_train, n_train] = x_train.shape
    input_size = n_train
    hidden_size = [60, 30] 
    num_classes = 2
    # Train configuration
    num_epochs = 3000
    learning_rate = 0.01
    dropout_p = 0.5
    step_size = 500
    
    model = MLP(input_size = input_size, 
               hidden_size = hidden_size,
                num_classes = num_classes)

    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Optimization
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=0.5)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Training
    for t in range(num_epochs):

        scheduler.step()
        # Forward pass
        y_pred = model(x_train)

        # Accuracy
        _, predictions = y_pred.max(dim = 1)
        accuracy = get_accuracy(y_pred = predictions.long(), y_target = y_train)

        # Loss
        loss = loss_fn(y_pred, y_train)


        # Verbose
        if t%10==0: 
            _, pred_test = model(x_test, apply_softmax=True).max(dim=1)
            test_acc = get_accuracy(y_pred=pred_test, y_target=y_test)
            # deep copy the model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if t%500 == 0:
                print ("epoch: {0:4d} | loss: {1:.4f} | Train accuracy: {2:.1f}% | Test accuracy: {3:.1f}%" \
                       .format(t, loss, accuracy, test_acc))

        # Zero all gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    
        # Predictions
    _, pred_train = model(x_train, apply_softmax=True).max(dim=1)
    _, pred_test = model(x_test, apply_softmax=True).max(dim=1)

    # Train and test accuracies
    train_acc = get_accuracy(y_pred = pred_train, y_target=y_train)
    test_acc = get_accuracy(y_pred = pred_test, y_target=y_test)
    print ("train acc: {0:.1f}%, test acc: {1:.1f}%".format(train_acc, test_acc))

    y_true = y_test.cpu().numpy()
    y_pred = pred_test.cpu().numpy()

    cm_perf = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(report)
    acc = (cm_perf[1, 1] + cm_perf[0, 0]) / np.sum(cm_perf)
    recall = cm_perf[1, 1] / (cm_perf[1, 0] + cm_perf[1, 1])
    precision = cm_perf[1, 1] / (cm_perf[0, 1] + cm_perf[1, 1])
    score = 2 / ((1 / recall) + (1 / precision))
    model_perf = torch.tensor([acc, precision, recall, score])
    
    return model, model_perf