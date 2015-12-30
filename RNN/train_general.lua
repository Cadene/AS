nn = require 'nn'
require 'nngraph'
posix = require 'posix'

cmd = torch.CmdLine()

-- set global variables 
opt = {}
cmd:option('-model', 'rnn', '[mlp|rnn|lstm]')
cmd:option('-dataset', 'mnist', '[toy|mnist]')
cmd:option('-cuda', 'true', '[true|false]')
cmd:option('-seed', 1337, 'int : seed cpu and gpu')
cmd:option('-batch_size', 40, 'int')
cmd:option('-dim_h', 15, 'int')
cmd:option('-max_iteration', 1, 'int')
cmd:option('-learning_rate', 0.02, 'float')
opt = cmd:parse(arg or {})

--[[ Pixel-by-pixel MNIST, task suggested by Le et al. (2015)
(A simple way to initialize recurrent networks of rectified linear units.)
and reused in Arjovsky et al. (2016) Unitary evolution recurrent
neural networks ]]

path2mnist = '/home/cadene/data/mnist_lecunn/'
-- path2mnist = 'data/mnist/'

batch_size = opt.batch_size
dim_h = opt.dim_h
dropout = .5
max_iteration = opt.max_iteration
learning_rate = opt.learning_rate
if opt.cuda == 'true' then
  opt.cuda = true
else
  opt.cuda = false
end

print("# lunching using pid = "..posix.getpid("pid"))
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.cuda then
  print('# switching to CUDA')
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(1)
  cutorch.manualSeed(opt.seed)
end

if opt.dataset == 'toy' then
  csv = require 'util/csv'
  toy = require 'util/toy'

  gendata = torch.load('data/toy_gendata.t7')
  data = torch.load('data/toy.t7')
  n = data:size(1)
  len = gendata.len

  x = data:narrow(2, 1, len)
  y = data:narrow(2, len+1, 1)

  pc_train = .7
  n_train = torch.floor(n * pc_train)
  n_test = n - n_train

  x_train = x:narrow(1, 1, n_train)
  y_train = y:narrow(1, 1, n_train)

  x_test = x:narrow(1, n_train+1, n_test)
  y_test = y:narrow(1, n_train+1, n_test)

elseif opt.dataset == 'mnist' then
  mnist = require 'util/mnist'
  tensor_type = torch.getdefaulttensortype()
  len = 28*28 -- 784
  
  trainset = mnist.traindataset(path2mnist)
  n_train = trainset.size -- 60000
  x_train = trainset.data:reshape(n_train,28*28):type(tensor_type)
  y_train = trainset.label:view(n_train,1):type(tensor_type)
  trainset = nil
  
  testset = mnist.testdataset(path2mnist)
  n_test = testset.size -- 10000
  x_test = testset.data:reshape(n_test,28*28):type(tensor_type)
  y_test = testset.label:view(n_test,1):type(tensor_type)
  testset = nil
  
  collectgarbage()  
end
-- normalize trainset and testset
x_train_mean = x_train:mean()
x_train_std = x_train:std()
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std

-- prepare data structure (trainer)
function create_h0(batch_size, dim_h)
  return torch.zeros(batch_size, dim_h)
end
function create_c0(batch_size, dim_h)
  return torch.zeros(batch_size, dim_h)
end
function create_input(batch_size, dim_h, x)
  local input
  if opt.model == 'mpl' then
    input = x
  elseif opt.model == 'rnn' then
    input = {create_h0(batch_size, dim_h), x}
  elseif opt.model == 'lstm' then
    input = {create_h0(batch_size, dim_h), create_c0(batch_size, dim_h), x}
  end
  if opt.cuda then
    if type(input) == 'table' then
      for i=1,#input do
        input[i] = input[i]:cuda()
      end
    else
      input = input:cuda()
    end
  end
  return input
end
function create_label(y)
  if opt.cuda then
    return y:cuda()
  end
  return y
end

dataset = {}
function dataset:size()
  return torch.floor(n_train / batch_size)
end
for i=1,dataset:size() do
  local start = (i-1)*batch_size + 1
  dataset[i] = {}
  dataset[i][1] = create_input(batch_size, dim_h, x_train:narrow(1, start, batch_size))
  dataset[i][2] = create_label(y_train:narrow(1, start, batch_size))
end

if opt.model == 'mlp' then
  mlp = require 'model/mlp'
  model = mlp.create(len, dim_h, 1, dropout)
elseif opt.model == 'rnn' then
  rnn = require 'model/rnn'
  model = rnn.create(len, dim_h)
elseif opt.model == 'lstm' then
  lstm = require 'model/lstm'
  model = lstm.create(len, dim_h, 1)
end

if opt.model == 'rnn' or opt.model == 'lstm' then
  function model:accUpdateGradParameters(input, gradOutput, lr)
    self.gradParameters:zero()
    self:accGradParameters(input, gradOutput, 1)
    self.parameters:add(-lr, self.gradParameters)
  end
end

criterion = nn.MSECriterion()

if opt.cuda then
  model = model:cuda()
  criterion = criterion:cuda()
end

print(torch.type(model.parameters))

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = max_iteration
trainer.learningRate = learning_rate
function trainer:hookIteration(iter)
  local input = create_input(n_test, dim_h, x_test)
  local label = create_label(y_test)
  print(iter.."# test error = " .. criterion:forward(model:forward(input), label))
end
hookExample_count = 1
function trainer:hookExample(example)
  print(hookExample_count, '/', dataset:size(), 'batch')
  hookExample_count = hookExample_count + 1
  if hookExample_count == dataset:size() then
    hookExample_count = 1
  end
end



model.parameters:uniform(-0.1, 0.1) -- view article for initialization
model:zeroGradParameters()
print("parameter count: " .. model.parameters:size(1))
print(x_test:size())
input = create_input(n_test, dim_h, x_test)
label = create_label(y_test)
loss = criterion:forward(model:forward(input), label)
print("initial error before training = " .. loss) 
trainer:train(dataset)
input = create_input(n_test, dim_h, x_test)
label = create_label(y_test)
loss = criterion:forward(model:forward(input), label)
print("# testing error = " .. loss)

torch.save('data/'.. opt.model ..'.t7', model)

-- TODO: cuda=true not implemented
if opt.dataset == 'toy' then
  -- for displaying
  grid_size = 100
  data_disp = toy.generate_data(grid_size, gendata.max_y, 0.0, gendata.len, gendata.delta, 0)
  x_disp = data_disp:narrow(2, 1, len)
  y_true = data_disp:narrow(2, len+1, 1)
  x_feed = (x_disp - x_train_mean) / x_train_std
  input = create_input(grid_size, dim_h, x_feed)
  y_pred = model:forward(input)
  disp = nn.JoinTable(2):forward{x_disp, y_true, y_pred}
  label = {}
  for i=1,len do
    label[i] = 'x'..i
  end
  label[len+1] = 'y_true'
  label[len+2] = 'y_pred'
  csv.tensor_to_csv(disp, 'data/toy_disp.csv', label)

  print("# display error = " .. y_pred:dot(y_true:t())/grid_size)
end

-- END
