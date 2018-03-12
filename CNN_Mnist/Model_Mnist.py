import sys
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from progressbar import ProgressBar, Percentage, Bar


######## NEURAL NETWORK ##########

class CNN(nn.Module):
    """
    Neural network with:
                    -Convolutional -> Max Pool -> RELU
                    -Convolutional -> Max Pool -> RELU -> Dropout
                    -Flatten
                    -Linear
                    -Linear
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1,
                                             out_channels = 10,
                                             kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size = 2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 10,
                                             out_channels = 20,
                                             kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size = 2),
                                   nn.Dropout2d(p = 0.3)
                                   )

        self.out = nn.Sequential(nn.Linear(in_features =320,
                                           out_features = 50),
                                 nn.Linear(in_features = 50,
                                           out_features = 10)
                                 )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.out(x)

        return F.log_softmax(x,  dim=1)

    def __name__(self):
        return "CNN"


class Model_Mnist():

    def __init__(self, use_cuda, loss_metric, lr, momentum, root_models, verbose = False):
        self.use_cuda = use_cuda
        self.loss_metric = loss_metric
        self.root_models = root_models
        self.verbose = verbose
        self.total_batch_number = -1

        self.model = CNN()
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def load_to_retrain(self, path_file):
        if self.use_cuda:
            self.model.load_state_dict(torch.load(path_file))
        else:
            self.model.load_state_dict(torch.load(path_file, map_location=lambda storage, loc: storage))
            self.model.cpu()

    def train(self, epochs, train_loader, val_loader):
        total_batch_number = len(train_loader)
        val_loss = self.on_train_begin()

        for epoch_idx in range(1, epochs+1):
            self.on_epoch_begin(epoch_idx, epochs)
            pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = total_batch_number).start()

            for batch_idx, (x, y) in enumerate(train_loader):

                x,y = self.on_batch_begin(x, y)

                y_pred = self.model(x)
                loss = self.loss_metric(y_pred, y)

                pbar.update(batch_idx + 1)

                self.on_batch_end(loss, batch_idx, epoch_idx, len(train_loader))

            pbar.finish()

            val_loss = self.on_epoch_end(epoch_idx, val_loader, val_loss)

        self.on_train_end()


    def on_train_begin(self):
        sys.stdout.write("\nComenzó el entrenamiento ...\n")
        val_loss = 1e5
        return val_loss

    def on_train_end(self):
        sys.stdout.write("\nTerminó el entrenamiento ...\n")
        self.save_checkpoint({'name': self.model.__name__(),
                              'state_dict': self.model.state_dict(),
                              }, False,
                             self.root_models + self.model.__name__() + "_finalmodel.tar"
                             )

    def on_epoch_begin(self, epoch_idx, epochs):
        sys.stdout.write("epoch {}/{}\n".format(epoch_idx, epochs))

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

        ## Imprimimos la perdida y el accuracy en el conjunto de validacion
        sys.stdout.write("\nepoch: {}, validation loss: {:.6f}, acc: {:.3f}\n\n".format(epoch_idx,
                                                                       loss.data[0],
                                                                       correct_cnt * 1.0 / total_cnt
                                                                       )
             )
        val_loss = loss.data[0]
        ## Guardamos el modelo si es el mejor
        is_best = val_loss_prev > val_loss
        self.save_checkpoint({'epoch': epoch_idx + 1,
                              'name': self.model.__name__(),
                              'state_dict': self.model.state_dict(),
                              'best_prec1': val_loss,
                             }, is_best,
                             self.root_models + self.model.__name__() + "_epoch{}_val_loss{:.6f}.tar".format(epoch_idx+1, val_loss)
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
        ## Imprimimos la perdida de cada 100 batches o en el batch final
        if ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len_train_loader) and self.verbose:
            sys.stdout.write("epoch: {}, batch index: {}, train loss: {:.6f}\n".format(epoch_idx,
                                                                                           batch_idx + 1,
                                                                                           loss.data[0]
                                                                                           )
                                )

    @staticmethod
    def save_checkpoint(state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.tar')