
# import generate_similar_user_and_item as generate_data
import numpy as np
import random
import sys
import csv
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sys

hidden_size = 64
learning_rate = 0.001
optimizer_option = 2
print_val = 1000
repeat_num = 2
num_iter = 500
batch_size = 64
seq_len = 10

use_cuda = torch.cuda.is_available()
# use_cuda = 0

class embedding_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(embedding_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, item_id_var):
        embedded_item = self.embedding(item_id_var)
        return embedded_item

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.eye(m.weight)
#         m.bias.data.fill_(0.01)

class neural_adding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, batch_size):
        super(neural_adding, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        # self.weight = torch.nn.Parameter(data=torch.Tensor(output_size), requires_grad=True)
        # self.weight.data.uniform_(-1, 1)
        # self.weight_mat = nn.Linear(output_size, output_size)
        # # torch.nn.init.eye_(self.weight_mat.weight)
        # # init_mat = torch.eye(int(output_size))
        # # self.weight_mat.weight.data.copy_(init_mat)
        # # self.weight_mat.bias.data.fill_(0)
        # self.weight_mat_state = nn.Linear(self.output_size, self.output_size)
        # # self.mlp1 = nn.Linear(self.input_size, self.hidden_size)
        # self.mlp2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.embedding_mat = torch.nn.Parameter(data=torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.embedding_mat.data.uniform_(-1, 1)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.gru = nn.GRU(self.input_size, self.hidden_size, 1)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1)

    def forward(self, input, hidden_state):

        # input_var = Variable(torch.zeros(self.output_size).view(1,1,-1))
        # input_var[0,0,idx] = 1
        # if use_cuda:
        #     input_var = input_var.cuda()
        # addition = self.weight_mat(input_var)
        # input_embedding = torch.mm(torch.squeeze(input), self.embedding_mat)
        # output, hidden_state = self.gru(input_embedding.view(1, self.batch_size, -1), hidden_state)

        output, hidden_state = self.gru(input, hidden_state)
        # input_vec = np.zeros(self.output_size)
        # input_vec[idx] = 1

        # updated_state = self.weight_mat_state(memory)
        # memory = updated_state + addition
        # memory[idx] += self.weight_mat[idx]
        # hidden = F.relu(self.mlp1(concatenated_input))
        # hidden = F.relu(self.mlp2(hidden))
        # hidden = F.relu(self.mlp2(hidden))
        # hidden = F.relu(self.mlp2(hidden))
        # hidden = F.relu(self.mlp2(hidden))
        # output = F.relu(self.mlp2(hidden))

        # hidden = self.mlp1(concatenated_input)
        # hidden = self.mlp2(hidden)
        # hidden = self.mlp2(hidden)
        # hidden = self.mlp2(hidden)
        # output = self.mlp2(hidden)
        out = self.out(hidden_state)
        # out = None
        return out, hidden_state


    def initHidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        memory = Variable(torch.zeros(self.output_size))
        if use_cuda:
            return result.cuda(), memory.cuda()
        else:
            return result, memory



class neural_adding_linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, batch_size):
        super(neural_adding_linear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        # self.weight = torch.nn.Parameter(data=torch.Tensor(output_size), requires_grad=True)
        # self.weight.data.uniform_(-1, 1)
        # self.weight_mat = nn.Linear(output_size, output_size)
        # # torch.nn.init.eye_(self.weight_mat.weight)
        # # init_mat = torch.eye(int(output_size))
        # # self.weight_mat.weight.data.copy_(init_mat)
        # # self.weight_mat.bias.data.fill_(0)
        # self.weight_mat_state = nn.Linear(self.output_size, self.output_size)
        self.mlp1 = nn.Linear(self.input_size, self.hidden_size)
        self.mlp2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # self.gru = nn.GRU(self.output_size, self.output_size, 1)

    def forward(self, input_var, hidden_state):

        # input_var = Variable(torch.zeros(self.output_size))
        # input_var[idx] = 1
        # if use_cuda:
        #     input_var = input_var.cuda()
        # # addition = self.weight_mat(input_var)

        # output, hidden_state = self.gru(input_var, hidden_state)
        # input_vec = np.zeros(self.output_size)
        # input_vec[idx] = 1

        # updated_state = self.weight_mat_state(memory)
        # memory = updated_state + addition
        # memory[idx] += self.weight_mat[idx]
        hidden1 = self.mlp1(input_var)
        hidden2 = self.mlp2(hidden_state)
        hidden_state = hidden1 + hidden2
        output = self.out(hidden_state)
        return output, hidden_state

    def initHidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        memory = Variable(torch.zeros(self.output_size))
        if use_cuda:
            return result.cuda(), memory.cuda()
        else:
            return result, memory

class representation_decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(representation_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.out2 = nn.Linear(self.hidden_size, self.output_size)
        # self.out3 = nn.Linear(self.output_size, self.output_size)
        # self.alpha = torch.nn.Parameter(data=torch.Tensor(output_size), requires_grad=True)
        # self.alpha.data.uniform_(-1, 1)

    def forward(self, input, memory):

        # hidden = F.relu(self.out1(input))
        # hidden = F.relu(self.out1(hidden))
        # hidden = F.relu(self.out1(hidden))
        # hidden = F.relu(self.out1(hidden))
        # hidden = F.relu(self.out1(hidden))
        # output1 = F.relu(self.out2(hidden))
        # output2 = self.out3(memory)
        # ones = Variable(torch.ones(self.output_size))
        # if use_cuda:
        #     ones = ones.cuda()
        # output = output1*self.alpha + output2*(ones - self.alpha)
        output = memory
        # hidden = self.out1(input)
        # hidden = self.out1(hidden)
        # hidden = self.out1(hidden)
        # hidden = self.out1(hidden)
        # output = self.out2(hidden)

        return output




class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self):
        super(custom_MultiLabelLoss_torch, self).__init__()

    def forward(self, pred, target):
        mseloss = torch.sum(torch.pow((pred - target), 2))

        loss = mseloss
        return loss


def one_hot_vec(idx, K):
    one_vecs = torch.nn.functional.one_hot(idx.to(torch.long), K)
    one_vecs_var = Variable(one_vecs.float())
    if use_cuda:
        one_vecs_var = one_vecs_var.cuda()
    return one_vecs_var



def train(batch_user_seq, embedding, encoder, embedding_optimizer, encoder_optimizer,  criterion, dim_vec):
    hidden_state, memory = encoder.initHidden()

    embedding_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()

    num_ins, num_user, seq_len = batch_user_seq.shape
    batch_user_seq = np.transpose(batch_user_seq, (1, 0, 2))
    input_var = Variable(torch.from_numpy(batch_user_seq)).float()
    if use_cuda:
        input_var = input_var.cuda()

    # batch_idx_var = torch.argmax(input_var, dim=-1, keepdim=False)
    # batch_idx_var = torch.unsqueeze(batch_idx_var, 1)
    input_var = torch.unsqueeze(input_var, 1)
    loss = 0
    # torch.autograd.set_detect_anomaly(True)
    # left_embedding = embedding(user_seq[0])

    # real_sum_vec_var = Variable(torch.zeros(dim_vec).view(1,-1))
    # if use_cuda:
    #     real_sum_vec_var = real_sum_vec_var.cuda()

    #real_sum_vec = np.cumsum(batch_user_seq, axis=0)
    real_sum_vec_var = Variable(torch.from_numpy(batch_user_seq)).float()
    if use_cuda:
        real_sum_vec_var = real_sum_vec_var.cuda()
    real_sum_vec_var = torch.cumsum(real_sum_vec_var, dim=0)
    for ei in range(num_user):

        # right_embedding = embedding(batch_idx_var[ei])
        # out, hidden_state = encoder(right_embedding, hidden_state)
        out, hidden_state = encoder(input_var[ei], hidden_state)
        # decoded_sum_vec_var = decoder(hidden_state, memory)
        decoded_sum_vec_var = out


        tt = criterion(decoded_sum_vec_var, real_sum_vec_var[ei])
        loss += tt
        if ei == num_user - 1:
            last_loss = tt/batch_size

    loss.backward()

    embedding_optimizer.step()
    encoder_optimizer.step()
    # decoder_optimizer.step()

    return loss.item() / (num_ins*num_user), last_loss.item()


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(data_set, dim_vec,embedding,encoder,  num_iter,print_every=300):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # encoder_pathes = []
    # decoder_pathes = []
    partition_point = int(len(data_set)/2)

    # partition_point = 200
    training_set = data_set[:partition_point]
    test_set = data_set[partition_point:]
    num_users = len(training_set)
    num_batch = int(num_users/batch_size)
    print('Number of training instances: ' + str(num_users))
    # print('Num of users: '+str(num_users))

    if optimizer_option == 1:
        embedding_optimizer = optim.SGD(embedding.parameters(), lr=learning_rate)
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    elif optimizer_option == 2:
        #encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09, weight_decay=0)
        #encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.88, 0.95), eps=1e-08, weight_decay=0)
        embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11, weight_decay=0)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11, weight_decay=0)
        # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11, weight_decay=0)
    elif optimizer_option == 3:
        embedding_optimizer = torch.optim.RMSprop(embedding.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    elif optimizer_option == 4:
        encoder_optimizer = torch.optim.Adadelta(encoder.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
        # decoder_optimizer = torch.optim.Adadelta(decoder.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)

    criterion = custom_MultiLabelLoss_torch()
    # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    total_iter = 0
    n_iters = num_iter
    # keys_list = list(training_set.keys())
    for j in range(n_iters):

        permutaed_training_set = np.random.permutation(training_set)

        error_of_iteration = []
        last_error_of_iteration = []
        for iter in range(num_batch):

            batch_user_seq = permutaed_training_set[iter*batch_size:(iter+1)*batch_size]

            loss, last_loss = train(batch_user_seq, embedding, encoder,
                         embedding_optimizer, encoder_optimizer,
                                     criterion, dim_vec)

            error_of_iteration.append(loss)
            last_error_of_iteration.append(last_loss)

            print_loss_total += loss
            plot_loss_total += loss

            total_iter += 1

        print_loss_avg = print_loss_total / (num_batch*batch_size)
        print_loss_total = 0
        print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * num_users)),
                                     total_iter, total_iter / (n_iters * num_users) * 100, print_loss_avg))

        print('Mean error: '+str(np.mean(error_of_iteration)))
        print('Mean last error: ' + str(np.mean(last_error_of_iteration)))
        print('Std: '+str(np.std(error_of_iteration)))
        # filepath = './models/embedding'+ str(model_id) + '_model_epoch' + str(int(j))
        # torch.save(embedding, filepath)
        # filepath = './models/encoder'+ str(model_id) + '_model_epoch' + str(int(j))
        # torch.save(encoder, filepath)
        # filepath = './models/decoder'+ str(model_id)  + '_model_epoch' + str(int(j))
        # torch.save(decoder, filepath)
        print('Finish epoch: '+str(j))
        # print('Model is saved.')
        # if j % 1 == 0:
        #     evaluate(test_set, embedding, encoder, decoder, dim_vec, batch_size, criterion)

        sys.stdout.flush()






training = 1


model_version = 100
path = './'



np.random.seed(0)

import csv

# filename = 'generated_sequences2500_10_100.csv'

filename = 'generated_sequences_5K5000_10_100.csv'

# filename = 'generated_sequences.csv'

sequential_records = []
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|')
    for row in reader:
        sequential_records.append(list(map(int, row)))

sequential_records = np.asarray(sequential_records)
sequential_records = sequential_records[:, :seq_len]

# sequential_records = sequential_records[:, :seq_len]

# test_sequential_records = sequential_records[int(len(sequential_records)/2):, :seq_len]

random.seed(10)

for j in range(repeat_num):
    for i in range(len(sequential_records)):
        idx = 10
        while idx >= 10:
            idx = random.randint(0, 10 - repeat_num)
        sequential_records[i, 9-j] = sequential_records[i, idx]

dim_vec = np.max(sequential_records) + 1
#
# print('shape: '+str(np.shape(sequential_records)))
#
# a, num_vec = np.shape(sequential_records)

# dim_vec = np.amax(sequential_records) + 1
dim_vec = int(dim_vec)
embedding = embedding_layer(dim_vec, hidden_size)
encoder = neural_adding(dim_vec, hidden_size, dim_vec, seq_len, batch_size)
# encoder = neural_adding_linear(dim_vec, hidden_size, dim_vec, seq_len, batch_size)
# encoder = neural_adding(hidden_size, hidden_size, dim_vec, seq_len, batch_size)
# decoder = representation_decoder(hidden_size, dim_vec)

if use_cuda:
    embedding = embedding.cuda()
    encoder = encoder.cuda()
    # decoder = decoder.cuda()


# training_set = sequential_records
# training_set = {}
# for idx in range(len(sequential_records)):
#     training_set[idx] = sequential_records[idx]

sequential_records = np.asarray(sequential_records)

# Lhot = np.transpose(np.eye(dim_vec)[sequential_records], (1, 2, 0))
Lhot = (np.arange(sequential_records.max()+1) == sequential_records[...,None]).astype(float)



trainIters(Lhot, dim_vec, embedding, encoder,  num_iter, print_every=print_val)


