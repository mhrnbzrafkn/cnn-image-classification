import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class CNN(nn.Module):
    def __init__(self, image_shape, deep=64, num_classes=2):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # input: [3, 64, 64]
            nn.Conv2d(image_shape[0], deep, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            # [64, 32, 32]

            nn.Conv2d(deep, deep*2, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(deep*2),
            nn.ReLU(),
            # [128, 16, 16]

            nn.Conv2d(deep*2, deep*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*4),
            nn.ReLU(),
            # [256, 8, 8]

            nn.Conv2d(deep*4, deep*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*8),
            nn.ReLU(),
            # [512, 4, 4]
            
            nn.Flatten(),
            # [8192]

            nn.Linear(deep*8*4*4, deep*8),
            nn.ReLU(),
            nn.Dropout(0.5),
            # [512]

            nn.Linear(deep*8, deep*8),
            nn.ReLU(),
            nn.Dropout(0.5),
            # [512]

            nn.Linear(deep*8, num_classes),
            nn.Softmax(dim=1)
            # ouptut: [num_classes]
        )
    
    def forward(self, inputs):
        return self.model(inputs)

def save_model(model, base_path, epoch):
    model_path = f'{base_path}/CNN-{epoch+1}.pt'
    model_scripted_path = f'{base_path}/CNN-scripted-{epoch+1}.pt'

    # Save new models
    torch.save(model.state_dict(), model_path)
    scripted_generator = torch.jit.script(model)
    scripted_generator.save(model_scripted_path)

    prev_epoch = epoch
    prev_model_path = f'{base_path}/CNN-{prev_epoch}.pt'
    prev_model_scripted_path = f'{base_path}/CNN-scripted-{prev_epoch}.pt'

    if os.path.exists(prev_model_path):
        os.remove(prev_model_path)
    if os.path.exists(prev_model_scripted_path):
        os.remove(prev_model_scripted_path)

# HyperParameters
image_size = (3, 64, 64)
deep = 64
batch_size = 16
num_classes = 8

# Loading Data
training_images_path = './dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'- Using {device} to train the model.')

transform = transforms.Compose([
    transforms.Resize((image_size[1], image_size[2])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root=training_images_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Building Model
model = CNN(image_size, deep, num_classes).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Process
max_epochs = 20
losses = []

print('- Training loop started.')
for current_epoch in range(max_epochs):
    epoch_losses = []
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.view(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        # Logging
        max_epoch_length = len(str(max_epochs))
        max_batch_length = len(str(len(data_loader)))
        max_loss_length = 8
        print(f'[Epoch: {current_epoch+1:0{max_epoch_length}d}/{max_epochs}] '
              f'[Batch: {i+1:0{max_batch_length}d}/{len(data_loader)}] '
              f'[Loss: {loss.item():<{max_loss_length}.4f}]')

    # Save losses
    losses.append(sum(epoch_losses)/len(epoch_losses))

    # Save model
    save_model(model, './saved_models', current_epoch)