import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence


class BiLSTM(torch.nn.Module):

    def __init__(self,
                 input_size=20,
                 number_layers=1,
                 hidden_dim=64,
                 embedding_dim=64,
                 dropout=0.5,
                 output_size=12
                 ):

        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_size = output_size

        self.embedding = torch.nn.Embedding(input_size, embedding_dim)
        self.bilstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_dim // 2,
                                    num_layers = number_layers,
                                    bias = True,
                                    batch_first=True,
                                    dropout= dropout,
                                    bidirectional=True)


        self.hidden2target = torch.nn.Linear(self.hidden_dim, self.output_size)

        self.hidden = self.init_hidden()

    def init_hidden(self, minibatch_size=32):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(2, minibatch_size, self.hidden_dim // 2)).cuda(),
                    Variable(torch.zeros(2, minibatch_size, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(torch.zeros(2, minibatch_size, self.hidden_dim // 2)),
                    Variable(torch.zeros(2, minibatch_size, self.hidden_dim // 2)))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        if type(lstm_out) == PackedSequence:
            target = PackedSequence(self.hidden2target(lstm_out.data), lstm_out.batch_sizes)
        else:
            target = self.hidden2target(lstm_out)

        return target

    def forward(self, sentence):
        return self._get_lstm_features(sentence)