import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# Define the transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the data from the data folder
data_dir = 'data'

# define transformations
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

# create the dataset
train_data = datasets.ImageFolder(os.path.join(data_dir), transform=transform)

# create the data loader
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True)

# Define the model architecture
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
# 58 is the number of labels in the CSV file
model.fc = nn.Linear(num_features, 58)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
model.train()
for epoch in range(10):  # Train for 10 epochs
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Print loss every 100 batches
            print('[Epoch %d, Batch %d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

print("Training finished!")
"""
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=58)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define a custom dataset

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label_file = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img_folder = self.label_file.iloc[idx, 0]
        img_folder = str(self.img_labels[idx])
        img_name = os.path.join(self.root_dir, img_folder, random.choice(
            os.listdir(os.path.join(self.root_dir, img_folder))))
        image = Image.open(img_name).convert('RGB')
        # label = self.label_file.iloc[idx, 1]
        label = int(self.label_file.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label


# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Define the train and validation datasets
train_dataset = TrafficSignDataset(
    root_dir='\data', label_file='labels.csv', transform=transform)


# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Define the model, loss function, and optimizer
model = TrafficSignNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
for epoch in range(10):
    train_loss = 0.0
    valid_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, train_loss))

"""
