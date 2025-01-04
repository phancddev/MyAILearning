import torch

def train_model(model, dataloader, criterion, optimizer, epochs, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")
    
    # Lưu mô hình
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")