-- OMP_NUM_THREADS=1 th exec
-- à exec, car ça peut speedup si nos matrice sont petites

require 'torch'
require 'nn'
require 'nngraph'

require 'gUtils'

model_utils = require 'model_utils'

CharLMMinibatchLoader = require 'CharLMMinibatchLoader'

v = CharLMMinibatchLoader.create("data.t7","vocab.t7",1,5)

v.number_mapping = {}
for key,value in pairs(v.vocab_mapping) do
  v.number_mapping[value] = key
end

v.vectorize = function (key)
  local vector = torch.zeros(v.vocab_size)
  vector[key] = 1
  return vector
end

v.scalarize = function (vector)
  local _, key = torch.max(vector, 1)
  return key
end

v.decode_outputs = function (outputs)
  local string = ''
  for i = 1, #outputs do
    local key = torch.squeeze(v.scalarize(outputs[i])) -- tensor1 to scalar
    string = string .. v.number_mapping[key]
  end
  return string
end

v.decode_batch = function (batch)
  local string = ''
  for i = 1, v.seq_length do
    local key = batch[{1,i}]
    string = string .. v.number_mapping[key]
  end  
  return string
end

dim_x = v.vocab_size
dim_h = 10
seqSize = v.seq_length

function module_LSTM(dim_x, dim_h):
  local input_x = nn.Identity()()
  local input_c = nn.Identity()()
  local input h = nn.Identity()()

  local lin_xi = nn.Linear(dim_x, dim_h)(input_x)
  local lin_hi = nn.Linear(dim_h, dim_h)(input_h)
  local lin_ci = nn.CMul(dim_h)(input_c)
  local biais_i = nn.Linear(1, dim_h)()
  local output_i = nn.Sigmoid()(nn.CAddTable()({lin_xi, lin_hi, lin_ci, biais_i}))
  
  local lin_xf = nn.Linear(dim_f, dim_h)(input_x)
  local lin_hf = nn.Linear(dim_h, dim_h)(input_h)
  local lin_cf = nn.CMul(dim_h)(input_c)
  local biais_f = nn.Linear(1, dim_h)()
  local output_f = nn.Sigmoid()(nn.CAddTable()({lin_xf, lin_hf, lin_cf, biais_f}))

  local mult1_c = torch.cmul(output_f, input_c) -- avoid gradient vanishing
  local lin_xc = nn.Linear(dim_f, dim_h)(input_x)
  local lin_hc = nn.Linear(dim_h, dim_h)(input_h)
  local biais_c = nn.Linear(1, dim_h)()
  local tanh_c = nn.Tanh()(nn.CAddTable(){lin_xc, lin_hc, biais_c})
  local mult2_c = nn.CMulTable()({output_i, tanh_c})
  local output_c = nn.CAddTable()({mult1_c, mult2_c})

  local lin_xo = nn.Linear(dim_f, dim_h)(input_x)
  local lin_ho = nn.Linear(dim_h, dim_h)(input_h)
  local lin_co = nn.CMul(dim_h)(input_c)
  local biais_o = nn.Linear(1, dim_h)()
  local output_f = nn.Sigmoid()(nn.CAddTable()({lin_xo, lin_ho, lin_co, biais_o}))

  local output_h = nn.CMulTable()({output_o, output_c})

  return nn.gModule({input_h, input_c, input_x}, {output_h, output_c, output_x})
end

LSTM = module_LSTM(dim_x, dim_h)

graph.dot(LSTM.fg, 'LSTM', 'myLSTM')

modules_H = model_utils.clone_many_times(h, seqSize)
modules_G = model_utils.clone_many_times(g, seqSize)

inputs  = {}
outputs = {}
list_h  = {}
inputs[1] = nn.Identity()()
list_h[1] = inputs[1]
for i = 1, seqSize do
  inputs[i+1] = nn.Identity()()
  list_h[i+1] = modules_H[i]({list_h[i],inputs[i+1]})
  outputs[i] = modules_G[i](list_h[i+1])
end
model = nn.gModule(inputs, outputs)

model

criterion = nn.ParallelCriterion()
for i = 1, seqSize do
  criterion:add(nn.ClassNLLCriterion(), 1.0/seqSize)
end

parameters, gradParameters = model:getParameters()

function train(nbIter, lr)
  for i = 1, nbIter do
    print('epoch:', i)
    local timer = torch.Timer()
    local shuffle = torch.randperm(v.nbatches)
    local mloss = 0
    for j = 1, v.nbatches do
      print('batch:', j)
      local id = shuffle[j]
      local inputs = {}
      inputs[1] = torch.zeros(dim_h) -- input_h
      inputs[2] = torch.zeros(dim_h) -- input_c
      for k = 1, seqSize do
        inputs[k+2] = v.vectorize(v.x_batches[id][{1,k}])
      end
      local labels = {}
      for k = 1, seqSize do
        --labels[k] = v.vectorize(v.y_batches[id][{1,k}])
        labels[k] = v.y_batches[id][{1,k}]
      end
      model:zeroGradParameters()
      local outputs = model:forward(inputs)
      print('inputs:', v.decode_batch(v.x_batches[id]))
      print('labels:', v.decode_batch(v.y_batches[id]))
      print('outputs:', v.decode_outputs(outputs))
      print('loss:', mloss)
      mloss = mloss + criterion:forward(outputs, labels)
      local df_do = criterion:backward(outputs, labels)
      local df_di = model:backward(inputs, df_do)
      model:updateParameters(lr)
    end
    print('iter:', i, 'loss:', mloss, 'time:', timer:time().real)
  end
end

-- train(3000,1e-2)

torch.save('model.t7', model)

model = torch.load('model.t7')

-- 5 --

shuffle = torch.randperm(v.nbatches)
id = shuffle[1]
batch = v.x_batches[id]
inputs = {}
inputs[1] = torch.zeros(dim_h)
for k = 1, seqSize do
  inputs[k+1] = v.vectorize(batch[{1,k}])
end
outputs = model:forward(inputs)

print(v.decode_batch(batch))
print(v.decode_outputs(outputs))









--[[
moduleComplet = {}
moduleComplet.listH = listH
moduleComplet.listG = listG
moduleComplet.dim_h = dim_h
moduleComplet.seqSize = seqSize


print(v)
print(v.x_batches[1])
print(v.y_batches[1])

v.vocab_mapping]]

