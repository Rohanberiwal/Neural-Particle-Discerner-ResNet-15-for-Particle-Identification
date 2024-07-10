import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define paths to your HDF5 files
photons_path = r"C:\Users\rohan\Downloads\SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"
electrons_path = r"C:\Users\rohan\Downloads\SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

# Function to print HDF5 file contents
def print_hdf5_file_contents(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Contents:")
        print_group_contents(f)

# Recursive function to print group contents
def print_group_contents(group, indent=0):
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            print(f"{' ' * indent}Group: {key}")
            print_group_contents(group[key], indent + 4)
        elif isinstance(group[key], h5py.Dataset):
            print(f"{' ' * indent}Dataset: {key} | Shape: {group[key].shape} | Dtype: {group[key].dtype}")
        else:
            print(f"{' ' * indent}Unknown: {key}")

# Print contents of both photon and electron HDF5 files
print_hdf5_file_contents(photons_path)
print_hdf5_file_contents(electrons_path)

class HDF5Dataset(Dataset):
    def __init__(self, file_path, label, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.label = label

        # Load HDF5 dataset
        with h5py.File(self.file_path, 'r') as hf:
            self.data = hf['X'][:]
            self.targets = hf['y'][:].astype(int)  # Convert to integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(target, dtype=torch.long)  # Ensure target is torch.long

# Load and preprocess data
def load_data(photons_path, electrons_path, transform):
    photon_dataset = HDF5Dataset(photons_path, label=0, transform=transform)
    electron_dataset = HDF5Dataset(electrons_path, label=1, transform=transform)
    
    dataset = torch.utils.data.ConcatDataset([photon_dataset, electron_dataset])
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# Transformation to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Load data using defined function
train_loader, test_loader = load_data(photons_path, electrons_path, transform)

# Define ResNet-15 model
class ResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet15, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4)
        self.layer3 = self._make_layer(128, 256, 6)
        self.layer4 = self._make_layer(256, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Train and evaluate the model
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    model.train()
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy

# Define the model
model = ResNet15()

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses, test_accuracies = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

# Plot training loss and test accuracy
plt.figure(figsize=(12, 5))

# Plotting loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save model weights
torch.save(model.state_dict(), 'resnet15_particle_classification.pth')


# Save model weights
torch.save(model.state_dict(), 'resnet15_particle_classification.pth')
