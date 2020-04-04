import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class AdaptiveSoftmaxRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, cutoffs=[20000, 50000]):
        super(AdaptiveSoftmaxRNN, self).__init__()
        ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = AdaptiveInput(ninp, ntoken, cutoffs)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # self.decoder = nn.Linear(nhid, ntoken)
        self.decoder = nn.AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=cutoffs, div_value=2.0)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        
        # weight sharing as described in the paper
        for i in range(len(cutoffs)):
            self.encoder.tail[i][0].weight = self.decoder.tail[i][1].weight
            
            # sharing the projection layers
            self.encoder.tail[i][1].weight = torch.nn.Parameter(self.decoder.tail[i][0].weight.transpose(0,1))

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, targets):
        emb = self.encoder(input) # (seq_len, bsz, ninp)
        output, hidden = self.rnn(emb, hidden) # (seq_len, bsz, ninp)
        output = self.drop(output)
        output = output.view(-1,output.size(2)) # (seq_len*bsz, ninp)
        # output = output.transpose(0,1)
        # targets = targets.view(targets.size(0) * targets.size(1)) # (seq_len * bsz)
        # targets = targets.transpose(0,1)
        output, loss = self.decoder(output, targets)
        return output, hidden, loss

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
    
  

class AdaptiveInput(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs=None,
                 div_value=2.0, head_bias=False):
        super(AdaptiveInput, self).__init__()
        if not cutoffs:
            cutoffs = [5000, 10000]
        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

#         self.head = nn.Sequential(nn.Embedding(self.head_size, self.in_features),
#                                   nn.Linear(self.in_features, self.in_features, bias=self.head_bias))
        
        self.head = nn.Embedding(self.head_size, self.in_features)
#                                   nn.Linear(self.in_features, self.in_features, bias=self.head_bias))
        
        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Embedding(osz, hsz),
                nn.Linear(hsz, self.in_features, bias=False),
            )

            self.tail.append(projection)


    def forward(self, input):
        used_rows = 0
        input_size = list(input.size())

        output = input.new_zeros([input.size(0) * input.size(1)] + [self.in_features]).float()
        input = input.view(-1)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            out = self.head(input[input_mask] - low_idx) if i == 0 else self.tail[i - 1](input[input_mask] - low_idx)
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        # if used_rows != input_size[0] * input_size[1]:
        #     raise RuntimeError("Target values should be in [0, {}], "
        #                        "but values in range [{}, {}] "
        #                        "were found. ".format(self.n_classes - 1,
        #                                              input.min().item(),
        #                                              input.max().item()))
        return output.view(input_size[0], input_size[1], -1)
