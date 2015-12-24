nn = require 'nn'
require 'nngraph'

function create_rnn_many(dim_h, dim_x)
  local h = nn.Identity()()
  local x = nn.Identity()()
  local jt = nn.JoinTable(2,2){h, x}
  local h_out = nn.Tanh()(nn.Linear(dim_h + dim_x, dim_h)(jt))
  local model = nn.gModule({h, x}, {h_out})
  model.params, model.gradParams = model:getParameters()
  return model
end

function create_rnn_one(dim_h, dim_y)
  local net_one = nn.Linear(dim_h, dim_y)
  net_one.params, net_one.gradParams = net_one:getParameters()
  return net_one
end

function clone_many_times(net, T)
  local tab_clone = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])   
    end
    tab_clone[t] = clone
    collectgarbage()
  end
  mem:close()
  return tab_clone
end

function make_dataset(dim_x, size, seq_len_max)
  local dataset = {}
  function dataset:size()
    return size
  end
  function dataset:vocab_size()
    return dim_x
  end
  function dataset:seq_len_max()
    return seq_len_max
  end
  function dataset:one_hot(tab_x)
    local tab_x_new = {}
    for i=1,#tab_x do
      local x = torch.zeros(1,dataset:vocab_size())
      x[{1,tab_x[i]}] = 1
      tab_x_new[i] = x
    end
    return tab_x_new
  end
  
  for i=1,dataset:size() do
    dataset[i] = {}
    dataset[i][1] = {}
    local seq_len = math.random(1,dataset:seq_len_max())
    --local values = {-1,1}
    --local rand_id = math.random(1, 2)
    for j=1,seq_len do
      -- dataset[i][1][j] = values[rand_id]
      dataset[i][1][j] = math.random(1, dataset:vocab_size())
    end
    --dataset[i][2] = torch.Tensor{values[rand_id]}:view(1,1)
    dataset[i][2] = torch.Tensor{math.random(0,1)}:view(1,1)
  end
  --[[
  dataset[1] = {}
  dataset[1][1] = {}
  dataset[1][1][1] = 1
  dataset[1][2] = torch.Tensor{-1}:view(1,1)
  dataset[2] = {}
  dataset[2][1] = {}
  dataset[2][1][1] = 2
  dataset[2][2] = torch.Tensor{1}:view(1,1) --]]
  return dataset
end

function trainer_many2one(net_many, net_one, criterion)
  local trainer = {}
  trainer.learning_rate = 1e-1
  trainer.max_iteration = 100
  trainer.net_many = net_many
  trainer.net_one = net_one
  trainer.criterion = criterion

  function trainer:train(dataset)
    local shuffle = torch.randperm(dataset:size(), 'torch.LongTensor')
    self.net_many.params:uniform(-0.02, 0.02)
    self.net_one.params:uniform(-0.02, 0.02)

    local tab_clone = clone_many_times(self.net_many, dataset:seq_len_max())

    local current_iter = 0
    while true do
      local current_err = 0
      for id = 1, dataset:size() do
        local example = dataset[shuffle[id]]
        local input = example[1]
        local y = example[2]
        local seq_length = #input
        local tab_x = dataset:one_hot(input)

        -- print(tab_x[1])
        -- print(y)

        self.net_many.gradParams:zero()
        self.net_one.gradParams:zero()

        local tab_h = {}
        tab_h[1] = torch.zeros(1, dim_h)

        for i=1,seq_length do
          tab_h[i+1] = tab_clone[i]:forward{tab_h[i], tab_x[i]} 
        end
        local y_pred = self.net_one:forward(tab_h[seq_length+1])

        local loss = criterion:forward(y, y_pred)
        loss = loss / seq_length
        current_err = current_err + loss

        local dloss = criterion:backward(y, y_pred)
        local gradOutput = self.net_one:backward(tab_h[seq_length+1], dloss)
        for i=seq_length,1,-1 do
          gradOutput = tab_clone[i]:backward({tab_h[i], tab_x[i]}, gradOutput)
          gradOutput = gradOutput[1]
        end

        self.net_one.params:add(self.learning_rate, self.net_one.gradParams) -- what did I do wrong
        self.net_many.params:add(self.learning_rate, self.net_many.gradParams)
      end
      current_iter = current_iter + 1
      current_err = current_err / dataset:size()
      print(current_iter, current_err)

      collectgarbage()

      if current_iter > self.max_iteration then
        print("# you have reached the maximum number of iterations")
        print("# training error = " .. current_err)
        break
      end
    end
  end
  return trainer
end

dim_h = 2
dim_x = 2
dim_y = 1
size = 2
seq_len_max = 1

net_many = create_rnn_many(dim_h, dim_x)
net_one = create_rnn_one(dim_h, dim_y)
criterion = nn.MSECriterion()
trainer = trainer_many2one(net_many, net_one, criterion)
dataset = make_dataset(dim_x, size, seq_len_max)

trainer:train(dataset) 






