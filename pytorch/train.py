import torch
import torch.nn as nn
import torch.optim as optim
from dataset import prepare_cmapss_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import os
train_dataset, test_dataset, scaler = prepare_cmapss_data()


# 2. Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class AeroSense(nn.Module):
    def __init__(self, input_size=24, hidden_size=64):
        super().__init__()
        
        # Define the layers of the Multi-Layer Perceptron
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1) # Single output node for binary classification
        )

    def forward(self, x):
        # The forward pass computes the prediction logits from the input data
        return self.network(x)


model=AeroSense()

# Loss Function and optimizer
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(params=model.parameters(),lr=0.01)

device="cuda" if torch.cuda.is_available() else "cpu"


def calculate_binary_accuracy(logits, labels):
    """
    Calculates accuracy for models outputting raw logits.
    """
    # 1. Apply Sigmoid to convert raw logits to probabilities (0.0 to 1.0)
    probabilities = torch.sigmoid(logits)
    
    # 2. Round to nearest integer (>= 0.5 becomes 1, < 0.5 becomes 0)
    predictions = torch.round(probabilities)
    
    # 3. Compare with true labels and calculate percentage
    correct = (predictions == labels).float() 
    accuracy = correct.sum() / len(correct)
    return accuracy.item()



    
def train(model,train_loader,epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for features,labels in train_loader:
            features=features.to(device)
            labels=labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            logits=model(features)

            loss=criterion(logits,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_accuracy += calculate_binary_accuracy(logits, labels)
            
        # Calculate epoch averages
        avg_loss = running_loss / len(train_loader)
        avg_acc = running_accuracy / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {avg_acc*100:.2f}%")

    print("Training Complete!")
    return model


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval() 
    
    all_predictions = []
    all_labels = []
    
    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Convert logits to probabilities, then to 0 or 1
            probabilities = torch.sigmoid(logits)
            predictions = torch.round(probabilities)
            
            # Move back to CPU and store for sklearn metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Print Evaluation Metrics
    print("\n--- AeroSense Watchdog Evaluation ---")
    print(confusion_matrix(all_labels, all_predictions))
    print("\n")
    print(classification_report(all_labels, all_predictions, target_names=["Healthy (0)", "Failure Imminent (1)"]))

# --- Execution Flow ---
# 1. Prepare data and scaler


trained_model = train(model, train_loader, epochs=15)
tested_model=test(model,test_loader)


torch.save(model.state_dict(), "aerosense.pth")