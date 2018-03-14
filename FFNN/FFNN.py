import sys
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from progressbar import ProgressBar, Percentage, Bar


class FFNN(torch.nn.Module):

    # En el inicializador debemos crear los parámetros de la red
    # así como todo lo necesario para hacer las predicciones.
    def __init__(self, input_size, 
                 hidden_size_weight, 
                 output_size, 
                 number_hidden_layers):
      
        super(FFNN, self).__init__()
        
        if len(dict_weight_hidden_layers) is not (number_hidden_layers+1):
            raise ValueError("dimension hidden layers is {} and dictionary weight have {} elements".format(len(dict_weight_hidden_layers),
                                                                                                           len(number_hidden_layers)
                                                                                                          )
                            )
        
        self.hidden_layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_weight = hidden_size_weight
        
        self.input_layer = nn.Linear(in_features = input_size,
                                     out_features = hidden_size_weight[0],
                                     bias = True)
        
        self.output_layer = nn.Linear(in_features = hidden_size_weight[-1],
                                      out_features = output_size,
                                      bias = True)
        
        if number_hidden_layers > 0:
          for i in range(number_hidden_layers):
            self.hidden_layers.append(nn.Linear(in_features = hidden_size_weight[i],
                                                out_features = hidden_size_weight[i+1],
                                                bias = True
                                               )
                                     )
 
        self.tanh = nn.Tanh()
        
        self.sigmoid = nn.Sigmoid()        
        
    def forward(self, x):
        # reshape
        x = x.view(-1, self.input_size)
        
        #input
        x = self.input_layer(x)
        x = self.tanh(x)
        
        #hidden_layers
        for hlayer in self.hidden_layers:
          x = hlayer(x)
          x = self.tanh(x)
        
        #output
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        #reshape
        y_pred = x.view(-1, self.output_size)
        
        return y_pred
        
    def __name__(self):
      return "FFNN2L"


class FFNN_Mnist():

    def __init__(self, 
                 use_cuda, 
                 loss_metric, 
                 lr, 
                 momentum, 
                 root_models, 
                 input_size, 
                 hidden_size_weight, 
                 output_size, 
                 number_hidden_layers,
                 verbose = False):
      
        self.use_cuda = use_cuda
        self.loss_metric = loss_metric
        self.root_models = root_models
        self.verbose = verbose
        self.total_batch_number = -1

        self.model = FFNN(input_size, 
                          hidden_size_weight, 
                          output_size, 
                          number_hidden_layers)

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
        val_loss_prev = 1e5

        for epoch_idx in range(1, epochs + 1):
            sys.stdout.write("epoch {}/{}\n".format(epoch_idx, epochs))

            with ProgressBar(widgets=[Percentage(), Bar()], maxval = total_batch_number) as pbar:
              
              for batch_idx, (x, y) in enumerate(train_loader):
                  x, y = Variable(x), Variable(y)
                  if self.use_cuda:
                      x, y = x.cuda(), y.cuda()

                  y_pred = self.model(x)
                  loss = self.loss_metric(y_pred, y)
                  self.optimizer.zero_grad()
                  loss.backward()
                  self.optimizer.step()

                  pbar.update(batch_idx+ 1)

            sys.stdout.write("epoch: {}, train loss: {:.6f}\n".format(epoch_idx,
                                                                      loss.data[0]
                                                                     )
                            )

            correct_cnt = 0
            total_cnt = 0
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
            ## Guardamos el modelo si es mejor
            is_best = val_loss_prev > val_loss
            self.save_checkpoint({'epoch': epoch_idx + 1,
                                  'name': self.model.__name__(),
                                  'state_dict': self.model.state_dict(),
                                  'best_prec1': val_loss,
                                  }, is_best,
                                 self.root_models + self.model.__name__() + "_epoch{}_val_loss{:.6f}.pt".format(
                                     epoch_idx + 1, val_loss)
                                 )


            val_loss_prev = val_loss if  val_loss < val_loss_prev else val_loss_prev

        sys.stdout.write("\nTerminó el entrenamiento ...\n")
        self.save_checkpoint({'name': self.model.__name__(),
                              'state_dict': self.model.state_dict(),
                              }, False,
                             self.root_models + self.model.__name__() + "_finalmodel.pt"
                             )

    @staticmethod
    def save_checkpoint(state, is_best, filename):
        torch.save(state["state_dict"], filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pt')