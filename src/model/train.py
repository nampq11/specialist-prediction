import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5, patience=2, save_path="best_model.pt"):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct_level1 = 0
        correct_combined = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training..."):
            reason_text_ids = batch['reason_text_ids'].to(device)
            reason_text_mask = batch['reason_text_mask'].to(device)
            user_info = batch.get('user_info', None)

            if user_info is not None:
                user_info = user_info.to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            level1_logits, level2_logits = model(
                reason_text_ids,
                reason_text_mask,
                user_info
            )

            _, level1_preds = torch.max(level1_logits, dim=1)
            correct_mask = (level1_preds == labels)
            loss = criterion(level1_logits, labels)

            if level2_logits is not None and (~correct_mask).any():
                incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]
                level2_loss = criterion(level2_logits[incorrect_indices], labels[incorrect_indices])
                loss += level2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_level1 += correct_mask.sum().item()

            if level2_logits is not None:
                _, level2_preds = torch.max(level2_logits, dim=1)
                final_preds = torch.where(correct_mask, level1_preds, level2_preds)
            else:
                final_preds = level1_preds

            correct_combined += (final_preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {running_loss / len(train_loader): .4f}")
        print(f"Level 1 Accuracy: {100 * correct_level1 / total:.2f}")
        print(f"Final Accuracy (with Level 2 fallback): {100 * correct_combined / total: .2f}%\n")

        # validation
        model.eval()
        val_loss = 0.0
        correct_level1 = 0
        correct_combined = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validate...'):
                reason_text_ids = batch['reason_text_ids'].to(device)
                reason_text_mask = batch['reason_text_mask'].to(device)
                user_info = batch.get('user_info', None)
                if user_info is not None:
                    user_info = user_info.to(device)

                    for p in model.hidden_layer2.parameters():
                        p.requires_grad = True
                    for p in model.level2_output.parameters():
                        p.requires_grad = True
                else:
                    for p in model.hidden_layer2.parameters():
                        p.requires_grad = False
                    for p in model.level2_output.parameters():
                        p.requires_grad = False
                labels = batch['labels'].to(device)

                level1_logits, level2_logits = model(
                    reason_text_ids,
                    reason_text_mask,
                    user_info
                )

                _, level1_preds = torch.max(level1_logits, dim=1)
                correct_mask = (level1_preds == labels)
                loss = criterion(level1_logits, labels)
                if level2_logits is not None and (~correct_mask).any():
                    incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]
                    level2_loss = criterion(level2_logits[incorrect_indices], labels[incorrect_indices])
                    loss += level2_loss

                val_loss += loss.item()
                correct_level1 += correct_mask.sum().item()

                if level2_logits is not None:
                    _, level2_preds = torch.max(level2_logits, dim=1)
                    final_preds = torch.where(correct_mask, level1_preds, level2_preds)
                else:
                    final_preds = level1_preds

                correct_combined += (final_preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation level 1 Accuracy: {100 * correct_level1 / total:.2f}%")
        print(f"Validation Final Accuracy (with Level 2 fallback): {100 * correct_combined / total:.2f}%\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break                

def evaluate(model, test_loader, device):
    model.to(device)
    
    model.eval()
    correct_level1 = 0
    correct_combined = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating...'):
            reason_text_ids = batch['reason_text_ids'].to(device)
            reason_text_mask = batch['reason_text_mask'].to(device)
            user_info = batch.get('user_info', None)
            if user_info is not None:
                user_info = user_info.to(device)
            labels = batch['labels'].to(device)

            level1_logits, level2_logits = model(
                reason_text_ids,
                reason_text_mask,
                user_info
            )

            _, level1_preds = torch.max(level1_logits, dim=1)
            correct_mask = (level1_preds == labels)

            if level2_logits is not None:
                _, level2_preds = torch.max(level2_logits, dim=1)
                final_preds = torch.where(correct_mask, level1_preds, level2_preds)
            else:
                final_preds = level1_preds

            correct_level1 += correct_mask.sum().item()
            correct_combined += (final_preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Level 1 Accuracy: {100 * correct_level1 / total:.2f}%")
    print(f"Test Final Accuracy (with Level 2 fallback): {100 * correct_combined / total:.2f}%\n")

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