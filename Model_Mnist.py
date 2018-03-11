import sys
import torch
import shutil
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

    def __name__(self):
        return "CNN"


class Model_Mnist():

    def __init__(self, use_cuda, loss_metric, lr, momentum, root_models):
        self.use_cuda = use_cuda
        self.loss_metric = loss_metric
        self.root_models = root_models

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
        val_loss = self.on_train_begin()

        for epoch_idx in range(1, epochs+1):
            self.on_epoch_begin()

            for batch_idx, (x, y) in enumerate(train_loader):

                x,y = self.on_batch_begin(x, y)

                y_pred = self.model(x)
                loss = self.loss_metric(y_pred, y)

                self.on_batch_end(loss, batch_idx, epoch_idx, len(train_loader))

            val_loss = self.on_epoch_end(epoch_idx, val_loader, val_loss)

        self.on_train_end()


    def on_train_begin(self):
        sys.stdout.write("Comenzó el entrenamiento ...")
        val_loss = 1e5
        return val_loss

    def on_train_end(self):
        sys.stdout.write("Terminó el entrenamiento ...")
        self.save_model(self.root_models + self.model.__name__() + "_last.tar")
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, epoch_idx, val_loader, val_loss_prev):
        ## Calculamos Accuracy y perdida en Validation Set
        correct_cnt = 0
        total_cnt = 0
        loss = 1e5
        for batch_idx, (x, y) in enumerate(val_loader):
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            x, target = Variable(x, volatile=True), Variable(y, volatile=True)
            y_pred = self.model(x)
            loss = self.loss_metric(y_pred, target)
            _, pred_label = torch.max(y_pred.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

        print("epoch: {}, validation loss: {:.6f}, acc: {:.3f}".format(epoch_idx,
                                                                       loss.data[0],
                                                                       correct_cnt * 1.0 / total_cnt
                                                                       )
             )
        val_loss = loss.data[0]
        is_best = val_loss < val_loss_prev
        self.save_checkpoint({'epoch': epoch_idx + 1,
                              'name': self.model.__name__(),
                              'state_dict': self.model.state_dict(),
                              'best_prec1': val_loss,
                             }, is_best,
                             self.root_models + self.model.__name__() + "_epoch{}_vallos{}.tar".format(epoch_idx+1, val_loss)
                             )

        return val_loss if  val_loss < val_loss_prev else val_loss_prev

    def on_batch_begin(self, x, y):
        x, y = Variable(x), Variable(y)
        if self.use_cuda:
            x, y = x.cuda(), y.cuda()
        return x, y

    def on_batch_end(self, loss, batch_idx, epoch_idx, len_train_loader):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len_train_loader:
            print("epoch: {}, batch index: {}, train loss: {:.6f}".format(epoch_idx,
                                                                          batch_idx + 1,
                                                                          loss.data[0]
                                                                          )
                  )

    def save_model(self, path_to_file):
        torch.save(self.model.state_dict(), path_to_file)

    @staticmethod
    def save_checkpoint(state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.tar')