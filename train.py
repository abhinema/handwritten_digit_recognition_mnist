from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import fc_model

import cv2
import matplotlib.pyplot as plt
import os
import os.path

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             #checkpoint['hidden_layers']
                             )
    model.load_state_dict(checkpoint['state_dict'])

    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', type=str, default="./model_checkpoint/checkpoint.pth",
                        help='Path to save Check point')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    devname = "cuda" if use_cuda else "cpu"    
    print("\n\n----------------\nDevice Used to process:",devname)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) ] )


    train_loader = torch.utils.data.DataLoader( datasets.MNIST('./data', train=True, download=True,transform= transform),
                                                batch_size= args.batch_size, shuffle=True,**kwargs)

    test_loader = torch.utils.data.DataLoader(  datasets.MNIST('./data', train=False, download=True,transform= transform),
                                                batch_size= args.test_batch_size, shuffle=True,**kwargs)

    cwd = os.getcwd()
    cwd = cwd + "/model_checkpoint/" + args.checkpoint
#    print("\n\ncwd :{}\n\n".format(cwd))

    if os.path.isfile(cwd):
        print("File is already present")
    else:
        print("\n----------------\nFile is not present...creating one with name:{}\n----------------\n\n".format(cwd))
        model = fc_model.Network(784, 10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        #Train and validation
        fc_model.train(model, train_loader, test_loader, criterion, optimizer, device, args.epochs)

        print("Our model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())

        checkpoint = {'input_size': 784,
                  'output_size': 10,
                  #'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}

        torch.save(checkpoint, args.checkpoint)

    model1 = load_checkpoint(args.checkpoint)
    model1.to(device)
    print("\n\nloaded model\n\n", model1)


    # Test out your network!

    model1.eval()

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    #img = images[0]
    # Convert 2D image to 1D vector

    #img = img.view(1, 784)
    img, labels = images.to(device), labels.to(device)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model1.forward(img)

    ps = torch.exp(output)
    equality = (labels.data == ps.max(1)[1])

    print(ps)
    print(equality)

if __name__ == '__main__':
    main()


"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
"""
