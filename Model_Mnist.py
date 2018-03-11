import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

#from torch.utils.data.dataset import random_split


######## NEURAL NETWORK ##########

class CNN(nn.Module):
    """
    Neural network with:
                    -convolutional -> Max Pool -> RELU
                    -convolutional -> Max Pool -> RELU
                    -FC -> RELU
                    -dropout
                    -FC -> Softmax
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



class Model_Mnist():

    def __init__(self, use_cuda, loss_metric, lr, momentum):
        self.use_cuda = use_cuda
        self.loss_metric = loss_metric

        self.model = CNN()
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def retrain(self, path_file):
        if self.use_cuda:
            self.model.load_state_dict(torch.load(path_file))
        else:
            self.model.load_state_dict(torch.load(path_file, map_location=lambda storage, loc: storage))
            self.model.cpu()


    def train(self, epochs, train_loader, val_loader):

        for epoch_idx in range(1, epochs+1):
            # trainning
            self.on_epoch_train(epoch_idx, train_loader)
            self.end_epoch_train(epoch_idx, val_loader)

    def on_epoch_train(self, epoch_idx, train_loader):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = Variable(x), Variable(y)
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            y_pred = self.model(x)
            loss = self.loss_metric(y_pred, y).data[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.end_batch_train(batch_idx, len(train_loader), epoch_idx, loss)

    def end_epoch_train(self, epoch_idx, val_loader):
        correct_cnt= 0
        total_cnt = 0
        for batch_idx, (x, y) in enumerate(val_loader):
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            x, target = Variable(x, volatile=True), Variable(y, volatile=True)
            y_pred = self.model(x)
            loss = self.loss_metric(y_pred, target).data[0]
            _, pred_label = torch.max(y_pred.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            self.end_batch_val(batch_idx, len(val_loader), epoch_idx, loss, correct_cnt, total_cnt)

    @staticmethod
    def end_batch_train(batch_idx, len_train_loader, idx_epoch, loss):
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len_train_loader:
            print("epoch: {}, batch index: {}, train loss: {:.6f}".format(idx_epoch,
                                                                          batch_idx + 1,
                                                                          loss
                                                                          )
                  )

    @staticmethod
    def end_batch_val(batch_idx, len_val_loader, idx_epoch, loss, correct_cnt, total_cnt):
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len_val_loader:
            print("epoch: {}, batch index: {}, validation loss: {:.6f}, acc: {:.3f}".format(idx_epoch,
                                                                                            batch_idx + 1,
                                                                                            loss,
                                                                                            correct_cnt * 1.0 / total_cnt
                                                                                            )
                  )