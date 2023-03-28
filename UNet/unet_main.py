import os
import numpy as np
import pandas as pd
import itertools
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import MultiplicativeLR
#from torchmetrics import F1, Recall, Precision, Accuracy, ConfusionMatrix
from sklearn.model_selection import train_test_split, StratifiedKFold

from model import UNET
from dataloader import Fire, classes

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def train_model(model, optimizer, epochs, criterion, scheduler, train_dataloader, date):
    model.train()
    loss_values = []
    acc_values = []
    total_samples = 0
    for epoch in range(epochs):
        running_loss = 0
        running_correct = 0
        epoch_samples = 0
        for batch, data in enumerate(train_dataloader):
            instances, labels = data
            instances = instances.to(device)
            optimizer.zero_grad()
            
            output = model(instances.type(torch.FloatTensor))
            
            labels = torch.tensor(labels)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(labels)
            _, predictions = torch.max(output, dim=1)
            running_correct += torch.sum(predictions == labels)
            epoch_samples += len(labels)
            total_samples += len(labels)
            
        epoch_loss = running_loss / epoch_samples
        epoch_acc = running_correct / epoch_samples
        
        loss_values.append(epoch_loss)
        acc_values.append(epoch_acc)
        
        scheduler.step()
        if epoch % 5 == 0:
            print('Epoch: %d \tAccuracy: %.5f \tLoss: %.5f' % (epoch, epoch_acc, epoch_loss))
           
    return loss_values, acc_values
        
def test_model(model, optimizer, criterion, test_dataloader):
    model.eval()
    y_pred = []
    y_true = []
    test_loss = 0
    total_samples = 0
    correct_predictions = 0
    features_df = pd.DataFrame()
    labels_df = pd.DataFrame()
    for batch, data in enumerate(test_dataloader):
        instances, labels = data
        instances = instances.to(device)
        
        label_pred = model(instances.type(torch.FloatTensor))
        labels = labels.squeeze().type(torch.LongTensor)
        loss = criterion(label_pred, labels)
        
        y_true.extend(labels)
        y_pred.extend(torch.exp(label_pred).detach().numpy())
        
        test_loss += loss.item()
        #probability -> exponent
        df = pd.DataFrame(torch.exp(label_pred).detach().numpy())
        df["predicted_label"] = df.idxmax(axis=1)
        df["true_label"] = labels.numpy()
                
        fdf = pd.DataFrame(instances.detach().numpy())
        labels_df = pd.concat([labels_df, df], ignore_index=True)
        features_df = pd.concat([features_df, fdf], ignore_index=True)
        
        test_loss += loss.item()
        _, predictions = torch.max(label_pred, dim=1)
        correct_predictions += torch.sum(predictions == labels)
        total_samples += len(labels)
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('/home/accelerator/UNet/cm.png', dpi=600, bbox_inches="tight")
    plt.close()
    
    test_accuracy = correct_predictions / total_samples
    test_loss = test_loss / len(test_dataloader)
    print('\nTest loss after Training', test_loss)
    
    return labels_df, features_df, test_loss, test_acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize,
        ]
    )
    dataset = Fire('/home/accelerator/bands_test/', transform)
    
    ################ HYPERPARAMETERS ################
    #random_seed = 42
    test_size = 0.2
    batch_size = 1024
    epochs = 100
    lmbda = lambda epoch: 0.95
    learning_rate = 0.0001
    ################ HYPERPARAMETERS ################
    
    labels_vector = []
    for i, j in enumerate(dataset):
        labels_vector.append(j[-1])
    
    train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                            test_size=test_size,
                                            train_size=1-test_size,
                                            shuffle=True)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=10)
    
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=10)
    
    model = UNET()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    print('Begin Training Trainset')
    train_loss, train_acc = train_model(model, optimizer, epochs, criterion, scheduler, train_dataloader, date)
    print('Finished Training Trainset')
    
    labels_df, features_df, test_loss, test_acc = test_model(model, optimizer, criterion, test_dataloader)
    
    
    plt.plot(np.arange(1,epochs+1), train_loss, 'r')
    plt.plot(np.arange(1,epochs+1), np.array([test_loss]*epochs), color='k', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Fire Data - Classification")
    plt.savefig('/home/accelerator/UNet/loss.png', dpi=600, bbox_inches="tight")
    plt.close()
    
    plt.plot(np.arange(1,epochs+1), train_acc, 'b')
    plt.plot(np.arange(1,epochs+1), np.array([test_acc]*epochs), color='k', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Fire Data - Classification")
    plt.savefig('/home/accelerator/UNet/accuracy.png', dpi=600, bbox_inches="tight")
    plt.close()
    
    torch.save(model.state_dict(), '/home/accelerator/UNet/UNet_checkpoint.pt')
    