import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def train(train_loader, valid_loader, model, criterion, optimizer, device, num_epochs, model_path):
    best_loss = 1e8
    train_losses = []
    valid_losses = []
    
    for i in range(num_epochs):
        print(f"Epoch {i+1} of {num_epochs}")
        valid_loss, train_loss = [], []
        model.train()

        for batch_labels, batch_data in tqdm(train_loader):
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']

            batch_labels = batch_labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = torch.squeeze(input_ids, 1)

            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)

            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for batch_labels, batch_data in tqdm(valid_loader):
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']

            batch_labels = batch_labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = torch.squeeze(input_ids, 1)

            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)

            loss = criterion(batch_output, batch_labels)
            valid_loss.append(loss.item())
            
        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        train_losses.append(t_loss)
        valid_losses.append(v_loss)
        
        print(f"Train loss: {t_loss}, Validation Loss: {v_loss}")
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), model_path)
        print(f"Best Validation Loss: {best_loss}")
    return train_losses, valid_losses

def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = []
    test_acc = []
    for batch_labels, batch_data in tqdm(test_loader):
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']

        batch_labels = batch_labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        input_ids = torch.squeeze(input_ids, 1)

        batch_output = model(input_ids, attention_mask)
        batch_output = torch.squeeze(batch_output)

        loss = criterion(batch_output, batch_labels)
        test_loss.append(loss.item())
        batch_preds = torch.argmax(batch_output, dim=1)

        if torch.cuda.is_available():
            batch_labels = batch_labels.cpu()
            batch_preds = batch_preds.cpu()
        
        test_acc.append(accuracy_score(batch_labels.detach().numpy(),
                                       batch_preds.detach().numpy(),))
        
    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


def plot_learning_curve(train_losses, valid_losses, save_path=None):
    """
    Plot the learning curve showing training and validation losses over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        valid_losses: List of validation losses per epoch
        save_path: Optional path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()