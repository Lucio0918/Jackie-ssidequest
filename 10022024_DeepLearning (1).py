import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json

# load the human activity dataset from project 3
with open('Project3.json','rt') as f:
    dt = json.load(f)

# try 1,.1,.01,.001,.0001
lr = 0.01 # learning rate
# bs = 500
bs = np.size(dt['train_y']) # batch size
epochs = 100

# use GPU if cuda/pytorch properly installed.
# After installing the correct version of CUDA, may need to do something like this:
# >>pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
#   I am using Cuda 11.8 so in the command I use (cu118) for the packages and the pytorch wheel
#   If you use 12.1 it would be cu121
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# create torch tensors for training and validation
D = torch.tensor(np.array(dt['train_D'], dtype=np.float32)) # typically 64
y = torch.tensor(np.array(dt['train_y'], dtype=np.longlong) - 1)
# D = torch.tensor(np.array(dt['train_D'], dtype=np.float32)[0:500,:]) # typically 64
# y = torch.tensor(np.array(dt['train_y'], dtype=np.longlong)[0:500] - 1)

Dv = torch.tensor(np.array(dt['validation_D'], dtype=np.float32)).to(dev) # move to GPU
yv = torch.tensor(np.array(dt['validation_y'], dtype=np.longlong)-1).to(dev)


# create a base class with tools for Fully Supervised learning
class DLN_FS(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.dev = dev
###change alot
    def loss_batch(self, loss_func, xb, yb, opt=None):
        # xb will be [bs x 60] in our dataset
        # yb will be [bs x 1]
        loss = loss_func(self(xb), yb)
        if opt is not None:
            loss.backward() # performs gradient backpropogation
            opt.step() # takes an optimization step in the gradient descent direction
            opt.zero_grad() # zeros out gradient estimates
##change a little bit
        return loss.item() # retrieve from gpu

    def fit(self, epochs, loss_func, opt, train_D, train_y, valid_D, valid_y, bs, savebest=[]):
        N = train_y.size()[0] # tensor.size() like np.shape()
        NB = np.ceil(N / bs).astype(np.longlong)

        tlosslist = []
        vlosslist = []
        best_val_loss = np.inf

        for epoch in range(epochs):
            self.train() # put network in training mode
            losslist = []
            for i in range(NB): # loop over batches
                start_i = i * bs
                end_i = start_i + bs
                xb = train_D[start_i:end_i].to(self.dev) # move training batch to GPU
                yb = train_y[start_i:end_i].to(self.dev)
                loss = self.loss_batch(loss_func, xb, yb, opt)
                losslist.append(loss)

            self.eval() # put network in evaluation mode to run validation data without updating weights/gradients
            with torch.no_grad():
                val_loss = self.loss_batch(loss_func, valid_D, valid_y)

            # if we have found the best network we can save it
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if np.size(savebest)>0:
                    torch.save(self, savebest)

            tlosslist.append(np.mean(losslist))
            vlosslist.append(val_loss)
            # plot current progress
            self.plot(epoch, tlosslist, vlosslist, best_epoch, best_val_loss, valid_D)

    def plot(self,epoch, tlosslist, vlosslist, best_epoch, best_val_loss, valid_D=[]):
        plt.cla()
        plt.plot(np.arange(1,epoch+2), tlosslist, 'r', label='Training')
        plt.plot(np.arange(1,epoch + 2),vlosslist,'g',label='Validation')
        plt.plot(best_epoch+1, best_val_loss, 'b*', label='Best result')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.pause(.01) # comment if live plotting not necessary

##
# custom class for network so it can inherit from base class
class DLN_custom(DLN_FS):
    # initialize network layers with trainable parameters
    def __init__(self, dev):
        super().__init__(dev)

    # define how layers are connected and data sent through
    def forward(self, xb):
        pass


class DLN(DLN_FS):
    def __init__(self, dev):
        super().__init__(dev)
        # initialize 5 fully connected layers
        self.fc1 = nn.Linear(60,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,5)
        self.rm1 = torch.tensor(np.zeros(64, dtype=np.float32)).to(dev)
        self.rv1 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.b1 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.w1 = torch.tensor(np.ones(64,dtype=np.float32)).to(dev)

        self.rm2 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.rv2 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.b2 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.w2 = torch.tensor(np.ones(64,dtype=np.float32)).to(dev)

        self.rm3 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.rv3 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.b3 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.w3 = torch.tensor(np.ones(64,dtype=np.float32)).to(dev)

        self.rm4 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.rv4 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.b4 = torch.tensor(np.zeros(64,dtype=np.float32)).to(dev)
        self.w4 = torch.tensor(np.ones(64,dtype=np.float32)).to(dev)

    def forward(self, xb):
        #define how data pass through the layers and lead to network output
        xb = F.relu(F.batch_norm(self.fc1(xb), running_mean=self.rm1,
                    running_var=self.rv1, weight=self.w1, bias=self.b1, training=self.training))
        xb = F.relu(F.batch_norm(self.fc2(xb),running_mean=self.rm2,
                    running_var=self.rv2,weight=self.w2,bias=self.b2,training=self.training))
        xb = F.relu(F.batch_norm(self.fc3(xb),running_mean=self.rm3,
                    running_var=self.rv3,weight=self.w3,bias=self.b3,training=self.training))
        xb = F.relu(F.batch_norm(self.fc4(xb),running_mean=self.rm4,
                    running_var=self.rv4,weight=self.w4,bias=self.b4,training=self.training))
        xb = F.softmax(self.fc5(xb), dim=1)
        # output of softmax is 5 probabilities
        return xb

    def myLoss(self, ypred, y):# example of a custom loss function with a self supervised term
        v1 = 0.5
        f = torch.nn.functional.cross_entropy
        ce = f(ypred, y)
        # sharpness = torch.sum(ypred > 0.5) # not differentiable due to gt operation
        sharpness = torch.sum(torch.log(ypred + 1e-12))
        return v1*ce + (1-v1)*sharpness # want to check each term to make sure each is generating gradients

net = DLN(dev)
net.to(dev)
print(net)

# choose an optimizer and a loss function
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

loss_func = torch.nn.functional.cross_entropy
# loss_func = net.myLoss
print(loss_func(net(Dv), yv)) # sanity check that we can pass data through network and generate loss

fig, ax = plt.subplots()
# perform training
net.fit(epochs, loss_func, opt, D, y, Dv, yv, bs, 'FCN_HumAct.pth')
# load network that performed best on validation data from file
net = torch.load('FCN_HumAct.pth')
#put network in evaluation mode to try on test data
net.eval()
D = torch.tensor(np.array(dt['test_D'], dtype=np.float32)).to(dev)
y = torch.tensor(np.array(dt['test_y'], dtype=np.longlong)-1).to(dev)

# compute accuracy on test set
yr = torch.argmax(net(D), axis=1)
acc = 100*torch.sum(yr == y).item()/y.size()[0]
print(acc)

# Print network weights to console
print("Optimizer's state_dict")
for var_name in net.state_dict():
    print(var_name, '\t', net.state_dict()[var_name])

# load a predefined architecture
import torchvision
resnet = torchvision.models.resnet18()
print(resnet)

# Change from 3 input channels to 1
resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

# Change from 1000 output classifcation labels to binary
resnet.fc = torch.nn.Linear(512, 2)

print(resnet)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from six import print_
from skimage.io import imread
import statsmodels.api as sm
from PCA import *
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import skimage.io
import torchvision
from torchvision import models
import torch.optim as optim

f = open('Project3_4.json','rt')
dt = json.load(f)
f.close()
D = dt['D']
y = np.array(dt['y']).astype(np.longlong) - 1

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = skimage.io.imread(path, as_gray=True)  # Load as grayscale
        images.append(img)
    images = np.array(images).astype(np.float32)  # Convert to float32 for PyTorch compatibility
    return images


# Split the dataset into training, validation, and test sets (80%, 10%, 10%) with stratification by class
train_D, test_D, train_y, test_y = train_test_split(D, y, test_size=0.2, stratify=y, random_state=42)
valid_D, test_D, valid_y, test_y = train_test_split(test_D, test_y, test_size=0.5, stratify=test_y, random_state=42)

# Load images into numpy arrays
train_images = load_images(train_D)
valid_images = load_images(valid_D)
test_images = load_images(test_D)

# Reshape the images to match CNN input [N x 1 x 28 x 28]
train_images = train_images.reshape(-1, 1, 28, 28)
valid_images = valid_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

# Convert the numpy arrays to PyTorch tensors
train_images_tensor = torch.tensor(train_images)
valid_images_tensor = torch.tensor(valid_images)
test_images_tensor = torch.tensor(test_images)

train_labels_tensor = torch.tensor(train_y)
valid_labels_tensor = torch.tensor(valid_y)
test_labels_tensor = torch.tensor(test_y)

print(f'Training data shape: {train_images_tensor.shape}')
print(f'Validation data shape: {valid_images_tensor.shape}')
print(f'Test data shape: {test_images_tensor.shape}')



# Instantiate the resnetnn model instead of ModifiedResNet18
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class resnetnn(nn.Module):
    def __init__(self, dev, num_classes=1):
        super().__init__()
        self.dev = dev

        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel (grayscale) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modify the final fully connected layer to match the number of output classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.to(self.dev)

    def forward(self, x):
        return self.resnet(x)

    def loss_batch(self, loss_func, xb, yb, opt=None):
        # xb will be [bs x C x H x W] in ResNet (for image data)
        loss = loss_func(self(xb), yb)
        if opt is not None:
            loss.backward()  # Gradient backpropagation
            opt.step()  # Optimization step
            opt.zero_grad()  # Zero gradients for the next step

        return loss.item()  # Retrieve loss from GPU

    def fit(self, epochs, loss_func, opt, train_D, train_y, valid_D, valid_y, bs, savebest=[]):
        N = train_y.size()[0]  # tensor.size() like np.shape()
        NB = np.ceil(N / bs).astype(np.longlong)

        tlosslist = []
        vlosslist = []
        best_val_loss = np.inf

        for epoch in range(epochs):
            self.train()  # put network in training mode
            losslist = []

            for i in range(NB):  # loop over batches
                start_i = i * bs
                end_i = start_i + bs
                xb = train_D[start_i:end_i].to(self.dev)  # move training batch to GPU
                yb = train_y[start_i:end_i].to(self.dev)
                loss = self.loss_batch(loss_func, xb, yb, opt)
                losslist.append(loss)

            # Validation after each epoch
            self.eval()  # put network in evaluation mode to run validation data without updating weights/gradients
            with torch.no_grad():
                val_loss = self.loss_batch(loss_func, valid_D, valid_y)
                val_preds = torch.argmax(self(valid_D), dim=1)
                val_acc = 100 * torch.sum(val_preds == valid_y).item() / valid_y.size(0)

            # if we have found the best network, save it
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if np.size(savebest) > 0:
                    torch.save(self, savebest)

            # Calculate and store losses
            tlosslist.append(np.mean(losslist))
            vlosslist.append(val_loss)

            # Print metrics for the current epoch
            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(losslist):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

            # Plot progress
            self.plot(epoch, tlosslist, vlosslist, best_epoch, best_val_loss, valid_D)

    def plot(self, epoch, tlosslist, vlosslist, best_epoch, best_val_loss):
        plt.cla()
        plt.plot(np.arange(1, epoch + 2), tlosslist, 'r', label='Training')
        plt.plot(np.arange(1, epoch + 2), vlosslist, 'g', label='Validation')
        plt.plot(best_epoch + 1, best_val_loss, 'b*', label='Best result')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        plt.pause(.01)



# Instantiate the resnetnn model instead of ModifiedResNet18

model = resnetnn(dev=device, num_classes=10)



# Print the modified ResNet18 architecture to the console
print(model)



# Loss function (Cross-Entropy for multi-class classification)
loss_func = nn.CrossEntropyLoss()

# Mini-batch size
batch_size = 1000


# Store validation accuracies for each learning rate
validation_accuracies = {}

lrs = [0.1, 0.01, 0.001, 0.0001]
results = []
epochs=20

for lr in lrs:
    print(f"\nTraining with learning rate: {lr}")

    net = resnetnn(device)
    net.to(device)

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.fit(epochs, loss_func, opt, train_images_tensor, train_labels_tensor, valid_images_tensor, valid_labels_tensor,
            batch_size)

    # Validation accuracy after training
    net.eval()
    with torch.no_grad():
        val_preds = torch.argmax(net(valid_images_tensor.to(device)), dim=1)
        val_acc = 100 * torch.sum(val_preds == valid_labels_tensor.to(device)).item() / valid_labels_tensor.size(0)

    results.append((lr, val_acc))
    print(f"Learning rate: {lr}, Validation Accuracy: {val_acc:.2f}%")

# Compare results
for lr, acc in results:
    print(f"Learning rate: {lr}, Final Validation Accuracy: {acc:.2f}%")
lrs = [x[0] for x in results]
accuracies = [x[1] for x in results]

plt.figure()
plt.plot(lrs, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy vs Learning Rate')
plt.show()
