import torch
from odevity import *
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from NumberDataset import NumberDataset
from torch.utils.data import Dataset, DataLoader

train_dataset=NumberDataset(length=50000)
val_dataset=NumberDataset(length=500)

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=10,
                                          shuffle=False)

# Logistic regression model
model = odevityNet_6()
# print(model)
# print(model.state_dict())


# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the model
total_step = len(train_loader)
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):

        # input_var = Variable(images)
        # target_var = Variable(labels)
        #
        # # compute output
        # outputs = model(input_var)
        # loss = criterion(outputs, target_var)


        # # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100000==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, 5, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        outputs = model(images)
        # print(outputs)
        outputs[outputs>=0.5]=1
        outputs[outputs < 0.5] = 0
        total += labels.size(0)
        correct += (outputs == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
# print(model.state_dict())
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
