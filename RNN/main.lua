require 'torch'
require 'nn'
require 'nngraph'

ModelUtil = require 'util/ModelUtil'
CharLMMinibatchLoader = require 'util/CharLMMinibatchLoader'
RNN = require 'model/RNN'
LSTM = require 'model/LSTM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
opt = cmd:parse(arg)

seq_size = 5
rnn_size = 10

loader = CharLMMinibatchLoader('data/data.t7', 'data/vocab.t7', 1, seq_size)

input_size = loader.vocab_size

if opt.model == 'rnn' then
    model = RNN.create(input_size, rnn_size, seq_size)
elseif opt.model == 'lstm' then
    model = LSTM.create(input_size, rnn_size, seq_size)
end

parameters, gradParameters = model:getParameters()

criterion = nn.ParallelCriterion()
for i = 1, seq_size do
    criterion:add(nn.ClassNLLCriterion(), 1.0 / seq_size)
end

function train(iter, lr)
    for i = 1, iter do
        print('epoch:', i)
        local timer = torch.Timer()
        local shuffle = torch.randperm(loader.nbatches)
        local mloss = 0
        for j = 1, loader.nbatches do
            print('batch:', j)
            local id = shuffle[j]
            local inputs = {}
            inputs[1] = torch.zeros(rnn_size)
            for k = 1, seq_size do
                inputs[k+1] = loader:vectorize(loader.x_batches[id][{1,k}])
            end
            local labels = {}
            for k = 1, seq_size do
                --labels[k] = loader:vectorize(loader.y_batches[id][{1,k}])
                labels[k] = loader.y_batches[id][{1,k}]
            end
            model:zeroGradParameters()
            local outputs = model:forward(inputs)
            print('inputs:', loader:decode_batch(loader.x_batches[id]))
            print('labels:', loader:decode_batch(loader.y_batches[id]))
            print('outputs:', loader:decode_outputs(outputs))
            print('loss:', mloss)
            mloss = mloss + criterion:forward(outputs, labels)
            local df_do = criterion:backward(outputs, labels)
            local df_di = model:backward(inputs, df_do)
            model:updateParameters(lr)
        end
        print('iter:', i, 'loss:', mloss, 'time:', timer:time().real)
    end
end

function test()
    local shuffle = torch.randperm(loader.nbatches)
    local id = shuffle[1]
    local batch = loader.x_batches[id]
    local inputs = {}
    inputs[1] = torch.zeros(rnn_size)
    for k = 1, seq_size do
      inputs[k+1] = loader:vectorize(batch[{1,k}])
    end
    local outputs = model:forward(inputs)
    print(loader:decode_batch(batch))
    print(loader:decode_outputs(outputs))
end

