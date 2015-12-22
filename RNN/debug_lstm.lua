nn = require 'nn'
require 'nngraph'
lstm = require 'model/lstm'
csv = require 'util/csv'
toy = require 'util/toy'

-- debug
nngraph.setDebug(true)

-- set global variables 
batch_size = 20
dim_h = 26
dropout = .5
max_iteration = 30
learning_rate = 0.02
cuda = false

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

-- normalize trainset and testset
x_train_mean = x_train:mean()
x_train_std = x_train:std()
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std

-- prepare data structure (trainer)
h0 = torch.zeros(batch_size, dim_h)
c0 = torch.zeros(batch_size, dim_h)

dataset = {}
function dataset:size()
  return torch.floor(n_train / batch_size)
end
for i=1,dataset:size() do
  local start = (i-1)*batch_size + 1
  dataset[i] = {}
  dataset[i][1] = {}
  dataset[i][1][1] = h0
  dataset[i][1][2] = c0
  dataset[i][1][3] = x_train:narrow(1, start, batch_size)
  dataset[i][2] = y_train:narrow(1, start, batch_size)
end

-- model = lstm.create(len, dim_h, 1)

function create(dim_x, dim_h, dim_xi)
  local h0 = nn.Identity()()
  local c0 = nn.Identity()()
  local x = nn.Identity()()
  local xs = nn.SplitTable(2)(x)
  local h = h0
  local c = c0
  for i=1,3 do
    local xi = nn.Reshape(1)(nn.SelectTable(i)(xs))
    -- print(dim_xi)
    gate_i = nn.Sigmoid()(nn.CAddTable()({
      nn.Linear(dim_xi, dim_h)(xi),
      nn.Linear(dim_h, dim_h)(h),
      nn.Linear(dim_h, dim_h)(c)
    }))
    gate_f = nn.Sigmoid()(nn.CAddTable()({
      nn.Linear(dim_xi, dim_h)(xi),
      nn.Linear(dim_h, dim_h)(h),
      nn.Linear(dim_h, dim_h)(c)
    }))
    learning = nn.Tanh()(nn.CAddTable()({
      nn.Linear(dim_xi, dim_h)(xi),
      nn.Linear(dim_h, dim_h)(h)
    }))
    c = nn.CAddTable()({
      nn.CMulTable()({gate_f, c}),
      nn.CMulTable()({gate_i, learning})        
    })
    gate_o = nn.Sigmoid()(nn.CAddTable()({
      nn.Linear(dim_xi, dim_h)(xi),
      nn.Linear(dim_h, dim_h)(h),
      nn.Linear(dim_h, dim_h)(c)
    }))
    h = nn.CMulTable()({gate_o, nn.Tanh()(c)})
  end
  local y = nn.Linear(dim_h, 1)(h)
  nngraph.annotateNodes()
  return nn.gModule({h0, c0, x}, {y})
end

model = create(len, dim_h, 1)
model.name = 'lstm_debug'

example = dataset[1][1]
pcall(function()
  label = model:forward(example)
  print(example)
  print(label)
end)

os.execute('open -a Safari lstm_debug.svg')
os.execute('sleep 1; rm -f lstm_debug.svg')
os.execute('sleep 1; rm -f lstm_debug.dot')

-- END