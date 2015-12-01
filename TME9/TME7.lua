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

v.vectorize = function (id)
  local out = torch.zeros(v.vocab_size)
  out[id] = 1
  return out
end

dimx = v.vocab_size
dimh = 10
seqSize = v.seq_length

-- 1 --
g_linear = nn.Linear(dimh, dimx)()
g_softMax = nn.SoftMax()(g_linear)
g = nn.gModule({g_linear}, {g_softMax})
graph.dot(g.fg, 'G', 'myG')

-- 2 --
h_linH = nn.Linear(dimh, dimh)()
h_linW = nn.Linear(dimx, dimh)()
h_sum = nn.CAddTable()({h_linH, h_linW})
h_tanh = nn.Tanh()(h_sum)
h = nn.gModule({h_linH, h_linW}, {h_tanh})
graph.dot(h.fg, 'H', 'myH')

-- 3 --
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
graph.dot(model.fg, 'model', 'myModel')

-- 4 --
criterion = nn.ParallelCriterion()
for i = 1, seqSize do
  criterion:add(nn.ClassNLLCriterion(), .1/seqSize)
end

function train(nbIter, lr)
  for i = 1, nbIter do
    local timer = torch.Timer()
    local shuffle = torch.randperm(v.nbatches)
    local mloss = 0
    for j = 1, v.nbatches do
      local id = shuffle[j]
      local inputs = {}
      inputs[1] = torch.zeros(dimh)
      for k = 1, seqSize do
        inputs[k+1] = v.vectorize(v.x_batches[id][{1,k}])
      end
      local labels = {}
      for k = 1, seqSize do
        --labels[k] = v.vectorize(v.y_batches[id][{1,k}])
        labels[k] = v.y_batches[id][{1,k}]
      end
      model:zeroGradParameters()
      local outputs = model:forward(inputs)
      mloss = mloss + criterion:forward(outputs, labels)
      local df_do = criterion:backward(outputs, labels)
      local df_di = model:backward(inputs, df_do)
      model:updateParameters(lr)
    end
    print('iter:', i, 'loss:', mloss, 'time:', timer:time().real)
  end
end


-- 5 --

shuffle = torch.randperm(v.nbatches)
id = shuffle[1]
inputs = {}
inputs[1] = torch.zeros(dimh)
for k = 1, seqSize do
  print(v.number_mapping( v.x_batches[id][{1,k}] ))
  inputs[k+1] = v.vectorize(v.x_batches[id][{1,k}])
end
outputs = model:forward(inputs)










moduleComplet = {}
moduleComplet.listH = listH
moduleComplet.listG = listG
moduleComplet.dimh = dimh
moduleComplet.seqSize = seqSize


print(v)
print(v.x_batches[1])
print(v.y_batches[1])

v.vocab_mapping

