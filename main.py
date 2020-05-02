# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from millify import millify
import data
import model
import pickle
import wandb
import json

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--min_freq', type=int, default=2,
                    help='minimum frequency for a word in the corpus to get included in the vocabulary')
parser.add_argument('--add_eos', action='store_true',
                    help='whether to add and <eos> token after a training sample')                    
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to embedding')
parser.add_argument('--rnn_dropout', type=float, default=0.2,
                    help='dropout applied to rnn layers')
parser.add_argument('--out_dropout', type=float, default=0.3,
                    help='dropout on the output of rnn')
parser.add_argument('--tail_dropout', type=float, default=0.3,
                    help='dropout applied to tail clusters')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--no_log', action='store_true',
                    help='whether to log in wandb')
parser.add_argument('--no_save', action='store_true',
                    help='whether to save models or not')                                      
parser.add_argument('--save', type=str, default='checkpoints',
                    help='path to save the final model')
parser.add_argument('--patience', type=int, default=0,
                    help='LR Scheduler patience')


args = parser.parse_args()

adaptive = True

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# save the arguments
with open(os.path.join(args.save, 'options.json'), 'w') as f:
    json.dump(args.__dict__, f)

###############################################################################
# Load data
###############################################################################
vocab_cache = f'{args.save}/vocab.pickle'

if(os.path.exists(vocab_cache)):
    print('[#] Found vocab cache in the corpus directory')
    print('[#] Loading the cache...')
    with open(vocab_cache, 'rb') as f:
        corpus = pickle.load(f)
else:
    print('[#] No vocab cache found!')
    print('[#] Building the vocabulary...')
    corpus = data.Corpus(args.data, args.min_freq, args.add_eos)
    with open(vocab_cache, 'wb') as f:
        pickle.dump(corpus, f)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if not adaptive:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.emb_dropout, args.tied).to(device)
elif args.model == 'AWD':
    model = model.AdaptiveSoftmaxRNNImproved(
        ntokens,
        args.emsize,
        args.nhid, 
        args.nlayers, 
        emb_dropout = args.emb_dropout,
        out_dropout = args.out_dropout,
        rnn_dropout = args.rnn_dropout,
        tail_dropout = args.tail_dropout,
        cutoffs=[20000, 50000],
        tie_weights = args.tied
      ).to(device)
elif args.model == 'LSTM':
    model = model.AdaptiveSoftmaxRNN(
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        emb_dropout = args.emb_dropout,
        rnn_dropout = args.rnn_dropout,
        tail_dropout = args.tail_dropout,
        cutoffs = [20000, 50000],
        tie_weights = args.tied,
        adaptive_input=True
    ).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min',
    factor = 0.5,
    patience = args.patience,
    verbose = True,
    min_lr = 1.0
)

total_tokens = corpus.dictionary.total_tokens
vocabulary = len(corpus.dictionary)
print(f'total tokens: {total_tokens} ({millify(total_tokens)})')
print(f'vocabulary size: {vocabulary} ({millify(vocabulary)})')
print('-' * 89)
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total params: {total_params} ({millify(total_params)})')
print(f'trainable params: {trainable_params} ({millify(trainable_params)})')
print('-' * 89)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if adaptive:
                output, hidden, loss = model(data, hidden, targets)
            else:
                output, hidden = model(data, hidden)
            
            hidden = repackage_hidden(hidden)

            if adaptive:
                total_loss += len(data) * loss
            else:
                total_loss += len(data) * criterion(output, targets).item()

    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
   
        hidden = repackage_hidden(hidden)
        if not adaptive:
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
        else:
            output, hidden, loss = model(data, hidden, targets)
        
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            ppl = math.exp(loss)
           
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, ppl))
            
            if not args.no_log:
                wandb.log({'Perplexity': ppl, 'Loss': cur_loss})
            
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    if not args.no_log:
        name = f'b{args.batch_size}_lr{args.lr}_L{args.nlayers}_h{args.nhid}_em{args.emsize}_drp{args.rnn_dropout}_bptt{args.bptt}'
        wandb.init(name=name, project="5m_line_shuffled")
        wandb.config.update(args)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        val_ppl = math.exp(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, val_ppl))
        print('-' * 89)
        
        if not args.no_log:
            wandb.log({'val_ppl': val_ppl, 'val_loss': val_loss})
        
        # Save the model if the validation loss is the best we've seen so far.
        if not args.no_save:
            # create the destination directory if it doesn't exist
            if not os.path.exists(args.save):
                os.mkdir(args.save)
            
            # check if the current loss is the best validation loss
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss

                # save the best model
                print('saving model...')
                torch.save(model, f'{args.save}/best_model.pt')

                # also save the checkpoint for the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl
                }, f'{args.save}/best_model_checkpoint.pt')
            else:
                # this saves the checkpoint for every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl':val_ppl
                }, f'{args.save}/checkpoint.pt')
        
        scheduler.step(val_loss)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')



# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


