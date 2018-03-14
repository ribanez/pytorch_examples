import sys
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from progressbar import ProgressBar, Percentage, Bar


class FFNN(nn.Module):

    def __init__(self, input_size, 
                 hidden_size,
                 output_size, 
                 number_hidden_layers
                 ):
        """
        Feed-Forward neural network
        Estructura de la red:
                    -Linear_1
                    -Linear_2
                    ...
                    -Linear_n

        Dimensiones de las capas:
                    -Linear_1(input_size, hidden_size[0])
                    -Linear_2(hidden_size[0], hidden_size[1])
                    ...
                    -Linear_k(hidden_size[k-2], hidden_size[k-1]
                    ...
                    -Linear_n(hidden_size[n-2], output_size)
        :param input_size: (int) dimension del entrada
        :param hidden_size: (array) con las dimensiones de las capas ocultas
        :param output_size: (int) dimension de la salida
        :param number_hidden_layers: (int) número de capas ocultas
        """

        super(FFNN, self).__init__()

        if len(hidden_size) is not (number_hidden_layers+1):
            raise ValueError("dimension hidden layers is {} and dictionary weight have {} elements".format(len(hidden_size),
                                                                                                           len(number_hidden_layers)
                                                                                                          )
                            )

        self.hidden_layers = []
        self.input_size = input_size
        self.output_size = output_size

        # Definimos la capa de entrada
        self.input_layer = nn.Linear(in_features = input_size,
                                     out_features = hidden_size[0],
                                     bias = True)

        # Definimos la capa de salida
        self.output_layer = nn.Linear(in_features = hidden_size[-1],
                                      out_features = output_size,
                                      bias = True)

        # Definimos las capas ocultas
        if number_hidden_layers > 0:
            for i in range(number_hidden_layers):
                self.hidden_layers.append(nn.Linear(in_features = hidden_size[i],
                                                    out_features = hidden_size[i+1],
                                                    bias = True
                                                   )
                                         )

        # Usaremos Tagente hiperbólica y sigmoide como funciones de activación
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()        

    def forward(self, x):
        # redimensionamos la entrada, pasamos la imagen a un tensor
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

        #redimensionamos la salida para que tenga la dimensión del número de clases
        y_pred = x.view(-1, self.output_size)

        return y_pred

    def cuda(self):
        """
        Metodo para usar GPU con modelo FFNN
        """
        self.input_layer.cuda()
        self.output_layer.cuda()
        for hlayer in self.hidden_layers:
            hlayer.cuda()

        self.tanh.cuda()
        self.sigmoid.cuda()

    def __name__(self):
      return "FFNN2L"


class Model_Mnist():

    def __init__(self, 
                 use_cuda, 
                 loss_metric, 
                 lr, 
                 momentum, 
                 root_models, 
                 input_size, 
                 hidden_size,
                 output_size, 
                 number_hidden_layers):
        """
        Utilizamos una feed-forward neural network

        :param use_cuda: (boolean) indica si está usando o no GPU
        :param loss_metric: (Loss function) función de perdida para el modelo.
               ref: http://pytorch.org/docs/0.3.1/nn.html#id38
        :param lr: (float) learning rate para el optimizador SGD (stochastic gradient descent)
               ref: http://pytorch.org/docs/0.3.1/optim.html
        :param momentum: (float) momentum para el optimizador SGD (stochastic gradient descent)
               ref: http://pytorch.org/docs/0.3.1/optim.html
        :param root_models: (string) carpeta donde se guardarán los modelos
        :param input_size: (int) dimension del entrada
        :param hidden_size: (array) con las dimensiones de las capas ocultas
        :param output_size: (int) dimension de la salida
        :param number_hidden_layers: (int) número de capas ocultas
        """

        self.use_cuda = use_cuda
        self.loss_metric = loss_metric
        self.root_models = root_models
        self.total_batch_number = -1

        # Creamos un modelo de red feed-forward
        self.model = FFNN(input_size, 
                          hidden_size,
                          output_size, 
                          number_hidden_layers
                          )
        if self.use_cuda:
            self.model.cuda()

        ## Creamos el optimizador SGD (stochastic gradient descent)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def load_to_retrain(self, path_file):
        """
        Método para leer un modelo ya entrenado para poder re-entrenarlo
        :param path_file: (string) ruta al archivo a re-entrenar
        """
        if self.use_cuda:
            self.model.load_state_dict(torch.load(path_file))
        else:
            self.model.load_state_dict(torch.load(path_file, map_location=lambda storage, loc: storage))
            self.model.cpu()

    def train(self, epochs, train_loader, val_loader):
        """
        Método para entrenar en modelo
        :param epochs: (int) numero de épocas que se entrenará el modelo
        :param train_loader: (Data.Loader) Iterador con los datos provenientes del data_set
               ref: http://pytorch.org/docs/0.3.1/data.html#torch.utils.data.DataLoader
        :param val_loader: (Data.Loader) Iterador con los datos que se utilizarán para validar el entrenamiento
               ref: http://pytorch.org/docs/0.3.1/data.html#torch.utils.data.DataLoader
        """

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

            # Calculamos la perdida y accuracy en el conjunto de validación
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

            sys.stdout.write("\nepoch: {}, validation loss: {:.6f}, acc: {:.3f}\n\n".format(epoch_idx,
                                                                                            loss.data[0],
                                                                                            correct_cnt * 1.0 / total_cnt
                                                                                            )
                             )

            ## Guardamos el modelo
            val_loss = loss.data[0]
            is_best = val_loss_prev > val_loss

            ## Podríamos no guardar en cada época sino que cada K epocas ...
            self.save_checkpoint(self.model.state_dict(),
                                 is_best,
                                 self.root_models + self.model.__name__() + "_epoch{}_val_loss{:.6f}.pt".format(epoch_idx + 1, val_loss))

            val_loss_prev = val_loss if  val_loss < val_loss_prev else val_loss_prev

        sys.stdout.write("\nTerminó el entrenamiento ...\n")

        ## Guardamos el ultimo modelo, sea mejor o no, por si se quiere continuar el entrenamiento
        self.save_checkpoint(self.model.state_dict(), False,
                             self.root_models + self.model.__name__() + "_finalmodel.pt"
                             )

    @staticmethod
    def save_checkpoint(state_dict, is_best, filename):
        """
        Metodo para guardar los pesos
        :param state_dict: (dictionary) pesos del modelo
        :param is_best: (boolean) condicion de si el modelo es mejor a los anteriores o no, en caso de ser True
        sobre-escribirá el modelo "model_best.pt"
        :param filename: (string) ruta donde se guardará el modelo
        :return:
        """
        torch.save(state_dict , filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pt')