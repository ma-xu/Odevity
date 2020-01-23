import torch
import torch.nn as nn
import torchvision
from odevity import *
import torchvision.transforms as transforms
from NumberDataset import NumberDataset

# Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_dataset=NumberDataset(length=1000)
val_dataset=NumberDataset(length=500)

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=10,
                                          shuffle=False)

# Logistic regression model
model = odevityNet_1()
print(model.state_dict())

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)


        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            print(model.state_dict())

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
print(model.state_dict())
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
