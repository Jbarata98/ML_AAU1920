import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot
import numpy as np

def split_indicies(n, val_pct):
    # Size of validation set
    n_val = int(n*val_pct)
    # Random permutation
    idxs = np.random.permutation(n)
    # Return first indexes for the validation set
    return idxs[n_val:], idxs[:n_val]

# Load the data
train_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Get the indicies for the training data and test data (the validation set will consists of 20% of the data)
train_idxs, val_idxs = split_indicies(len(train_dataset), 0.2)

# Define samplers (used by Dataloader) to the two sets of indicies
train_sampler = SubsetRandomSampler(train_idxs)
val_sampler = SubsetRandomSampler(val_idxs)

# Specify data loaders for our training and test set (same functionality as in the previous self study)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=val_sampler)

print(f"Number of training examples: {len(train_idxs)}")
print(f"Number of validation examples: {len(val_idxs)}")

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True)

#  Network Structure

class FC_NN(nn.Module):

    def __init__(self):
        super().__init__()

        # Define a FCNN
        self.fc1 = nn.Linear(784,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 100)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100,10)

    def forward(self, fc):

        # Reshape the input tensor; '-1' indicates that PyTorch will fill-in this
        # dimension, whereas the '1' indicates that we only have one color channel.
        fc = fc.view(-1, 784)
        fc = F.relu(self.fc1(fc))
        # Reshape the representation
        fc = F.relu(self.fc2(fc))
        fc = self.fc3(fc)

        self.out = F.softmax(fc)

        return fc


train_losses = []
def train(model, train_loader, loss_fn, epoch):
    # Tell PyTorch that this function is part of the training
    model.train()
    # As optimizer we use stochastic gradient descent as defined by PyTorch. PyTorch also includes a variety
    # of other optimizers
    learning_rate = 0.0001
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Iterate over the training set, one batch at the time, as in the previous self sudy
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get the prediction
        y_pred = model(data)
        # Remember to zero the gradients so that they don't accumulate
        opt.zero_grad()
        # Calculate the loss and and the gradients
        loss = loss_fn(y_pred, target) #+1 *1/2*sum(torch.norm(w) for w in model.parameters())  regularization
        loss.backward()
        # Optimize the parameters by taking one 'step' with the optimizer
        opt.step()
        train_losses.append(loss)
        # For every 10th batch we output a bit of info
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx * len(data) / len(train_loader.sampler), loss.item()))


def test_model(model, data_loader, loss_fn):
    # Tell PyTorch that we are performing evaluation
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nTest/validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.sampler),
        100. * correct / len(data_loader.sampler)))


def save_model(file_name, model):
    torch.save(model, file_name)

def load_model(file_name):
    model = torch.load(file_name)
    model.eval()
    return model

# The number of passes that will be made over the training set
num_epochs = 10

# torch.nn defines several useful loss-functions, which we will take advantage of here (see Lecture 1, Slide 11, Log-loss).
loss_fn = nn.CrossEntropyLoss()

# Instantiate the model class
model = FC_NN()
# and get some information about the structure
print('Model structure:')
print(model)


for i in range(num_epochs):
    train(model, train_loader, loss_fn, i)
    # Evaluate the model on the test set
    test_model(model, val_loader, loss_fn)


plt.plot(range(len(train_losses)), train_losses,'b')
plt.show()

# Evaluate the model on the test set
test_model(model, test_loader, loss_fn)

