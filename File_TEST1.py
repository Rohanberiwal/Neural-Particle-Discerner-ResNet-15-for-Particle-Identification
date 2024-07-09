import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

photons_path = r"C:\Users\rohan\Downloads\SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"
electrons_path = r"C:\Users\rohan\Downloads\SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)


def print_hdf5_file_contents(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Contents:")
        print_group_contents(f)

def print_group_contents(group, indent=0):
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            print(f"{' ' * indent}Group: {key}")
            print_group_contents(group[key], indent + 4)
        elif isinstance(group[key], h5py.Dataset):
            print(f"{' ' * indent}Dataset: {key} | Shape: {group[key].shape} | Dtype: {group[key].dtype}")
        else:
            print(f"{' ' * indent}Unknown: {key}")

print_group_contents(photons_path)
print_group_contents(electrons_path)
# Custom dataset class for HDF5 data
class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        
        # Load HDF5 dataset
        with h5py.File(self.file_path, 'r') as hf:
            if 'data' in hf and 'labels' in hf:
                self.data = hf['data'][:]
                self.targets = hf['labels'][:]
            elif 'images' in hf and 'labels' in hf:
                self.data = hf['images'][:]
                self.targets = hf['labels'][:]
            else:
                raise KeyError(f"Dataset 'data' or 'images' and 'labels' not found in HDF5 file: {self.file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# Load and preprocess data
def load_data(photons_path, electrons_path):
    train_dataset = HDF5Dataset(photons_path, transform=transforms.ToTensor())
    test_dataset = HDF5Dataset(electrons_path, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data(photons_path, electrons_path)

# Define ResNet-15 like model
class ResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet15, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet15().to(device)

# Training and evaluation functions
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

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
    print(f'Test Accuracy: {accuracy:.2f}%')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate(model, test_loader)

# Save model weights
torch.save(model.state_dict(), 'resnet15_particle_classification.pth')
