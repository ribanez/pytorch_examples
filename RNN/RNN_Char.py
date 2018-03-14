import torch
import torch.nn as nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, voc_size, num_layers, recurrent_dropout, dropout):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.num_layers = num_layers
        self.recurrent_dropout = recurrent_dropout

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Embedding(num_embeddings =self.voc_size,
                                    embeddings_dim = self.input_size)

        self.lstm = nn.LSTM(input_size = self.input_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            dropout = self.recurrent_dropout)

        self.decoder = nn.Linear(in_features = self.hidden_size,
                                 out_features = self.voc_size)

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = F.log_softmax(self.decoder(output[0], dim = 1))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_())
               )