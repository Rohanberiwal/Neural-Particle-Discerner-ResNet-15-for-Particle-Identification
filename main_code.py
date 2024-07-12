import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply, RandomChoice, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, RandomGrayscale

# Function to generate random matrices
def generate_random_matrices(num_samples, img_size=32):
    electrons_hit_energy = np.random.rand(num_samples, img_size, img_size)
    electrons_time = np.random.rand(num_samples, img_size, img_size)
    
    photons_hit_energy = np.random.rand(num_samples, img_size, img_size)
    photons_time = np.random.rand(num_samples, img_size, img_size)
    
    return electrons_hit_energy, electrons_time, photons_hit_energy, photons_time

# Number of datasets to generate
num_datasets = 1000  # Reduced for demonstration
num_samples_per_dataset = 1

# Lists to store all generated matrices
all_electrons_hit_energy = []
all_electrons_time = []
all_photons_hit_energy = []
all_photons_time = []

# Generate datasets
for _ in range(num_datasets):
    electrons_hit_energy, electrons_time, photons_hit_energy, photons_time = generate_random_matrices(num_samples_per_dataset)
    all_electrons_hit_energy.append(electrons_hit_energy)
    all_electrons_time.append(electrons_time)
    all_photons_hit_energy.append(photons_hit_energy)
    all_photons_time.append(photons_time)

# Convert lists to numpy arrays
all_electrons_hit_energy = np.array(all_electrons_hit_energy)
all_electrons_time = np.array(all_electrons_time)
all_photons_hit_energy = np.array(all_photons_hit_energy)
all_photons_time = np.array(all_photons_time)

# Data division
num_total_datasets = len(all_electrons_hit_energy)
num_train = int(0.8 * num_total_datasets)
num_val = int(0.1 * num_total_datasets)
num_test = num_total_datasets - num_train - num_val

# Split into training, validation, and test sets
train_electrons_hit_energy = all_electrons_hit_energy[:num_train]
train_electrons_time = all_electrons_time[:num_train]
train_photons_hit_energy = all_photons_hit_energy[:num_train]
train_photons_time = all_photons_time[:num_train]

val_electrons_hit_energy = all_electrons_hit_energy[num_train:num_train+num_val]
val_electrons_time = all_electrons_time[num_train:num_train+num_val]
val_photons_hit_energy = all_photons_hit_energy[num_train:num_train+num_val]
val_photons_time = all_photons_time[num_train:num_train+num_val]

# Convert data to PyTorch tensors
train_electrons_hit_energy_tensor = torch.FloatTensor(train_electrons_hit_energy)
train_electrons_time_tensor = torch.FloatTensor(train_electrons_time)
train_photons_hit_energy_tensor = torch.FloatTensor(train_photons_hit_energy)
train_photons_time_tensor = torch.FloatTensor(train_photons_time)

val_electrons_hit_energy_tensor = torch.FloatTensor(val_electrons_hit_energy)
val_electrons_time_tensor = torch.FloatTensor(val_electrons_time)
val_photons_hit_energy_tensor = torch.FloatTensor(val_photons_hit_energy)
val_photons_time_tensor = torch.FloatTensor(val_photons_time)

# Concatenate hit energy and time matrices along the channel dimension
train_electrons_data = torch.stack((train_electrons_hit_energy_tensor, train_electrons_time_tensor), dim=1)
train_photons_data = torch.stack((train_photons_hit_energy_tensor, train_photons_time_tensor), dim=1)

val_electrons_data = torch.stack((val_electrons_hit_energy_tensor, val_electrons_time_tensor), dim=1)
val_photons_data = torch.stack((val_photons_hit_energy_tensor, val_photons_time_tensor), dim=1)

# Combine electrons and photons data into one tensor
train_data = torch.cat((train_electrons_data, train_photons_data), dim=0)
val_data = torch.cat((val_electrons_data, val_photons_data), dim=0)

# Create labels (0 for electrons, 1 for photons)
train_labels = torch.cat((torch.zeros(train_electrons_data.size(0)), torch.ones(train_photons_data.size(0))))
val_labels = torch.cat((torch.zeros(val_electrons_data.size(0)), torch.ones(val_photons_data.size(0))))

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

# Define aggressive data augmentations
transformations = transforms.Compose([
    RandomApply([RandomChoice([
        RandomRotation(15),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        RandomGrayscale(p=0.1)
    ])], p=0.8)
])

# Apply transformations to datasets
train_dataset.transform = transformations
val_dataset.transform = transformations

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class ResNet15(nn.Module):
    def __init__(self):
        super(ResNet15, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128 * 8 * 8, 2)  # Assuming 32x32 input size and 2 output classes
    
    def forward(self, x):
        # Reshape input to remove the extra dimension if present
        if x.dim() == 5:
            x = x.squeeze(2)  # Remove the extra dimension
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Initialize model, criterion, and optimizer
model = ResNet15()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adding weight decay for regularization

# Training loop
num_epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    total_train = 0
    
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_loss = train_loss / total_train
    train_accuracy = train_correct / total_train
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation
    val_loss = 0.0
    val_correct = 0
    total_val = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    val_loss = val_loss / total_val
    val_accuracy = val_correct / total_val
    
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Plotting function
def plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot training and validation metrics
plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

