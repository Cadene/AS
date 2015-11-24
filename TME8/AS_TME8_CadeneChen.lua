require 'nn'
require 'nngraph'

dim = 10

-- Model 1 --
siglin = nn.Sequential()
siglin:add(nn.Linear(dim,dim))
siglin:add(nn.Sigmoid())

concat = nn.ConcatTable()
concat:add(siglin)
concat:add(nn.Identity())

model1 = nn.Sequential()
model1:add(concat)
model1:add(nn.CAddTable())
----------------

-- gModel 1 --
gInput = nn.Identity()()

gLin = nn.Linear(dim,dim)(gInput)
gSig = nn.Sigmoid()(gLin)

gAdd = nn.CAddTable()({gSig, gInput})

gModel1 = nn.gModule({gInput}, {gAdd})
-------------

-- gModel 2 --
gX = nn.Identity()()

gGamma = nn.Linear(1,dim)()
gL1Gamma = nn.L1Penalty(1e-4)(gGamma)
gReLUL1Gamma = nn.ReLU()(gL1Gamma)

gMul = nn.CMulTable()({gX, gReLUL1Gamma})

gLin2 = nn.Linear(dim,dim)(gMul)
gSig2 = nn.Sigmoid()(gLin2)

gAdd2 = nn.CAddTable()({gSig2, gMul})

gModel2 = nn.gModule({gX, gGamma}, {gAdd2})
--------------

-- Données --
nExemples = 5000
x = torch.rand(nExemples, dim)
y = x:clone()

xtest = torch.rand(nExemples, dim)
ytest = y:clone()
-------------


-------------


criterion = nn.MSECriterion()

function training(model, criterion, x, y, nbIter, lr)
  for i = 1, nbIter do
    local timer = torch.Timer()
    local shuffle = torch.randperm(x:size(1))
    local mloss = 0
    for j = 1, x:size(1) do
      local id = shuffle[j]
      local input = x[id]
      local label = y[id]
      model:zeroGradParameters()
      local output = model:forward(input)
      mloss = mloss + criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      local df_di = model:backward(input, df_do)
      model:updateParameters(lr)
    end
    print('loss:', i, mloss, timer:time().real)
  end
end

function training2(model, criterion, x, y, nbIter, lr)
  for i = 1, nbIter do
    local timer = torch.Timer()
    local shuffle = torch.randperm(x:size(1))
    local mloss = 0
    local one = torch.ones(1)
    for j = 1, x:size(1) do
      local id = shuffle[j]
      local input = x[id]
      local label = y[id]
      model:zeroGradParameters()
      local output = model:forward({input,one})
      mloss = mloss + criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      local df_di = model:backward({input,one}, df_do)
      model:updateParameters(lr)
    end
    print('loss:', i, mloss, timer:time().real)
  end
end


----------------- Version algorithmique ------------------
lin0 = nn.Linear(dim,dim)
sig0 = nn.Sigmoid()
add0 = nn.CAddTable()

function training0(criterion, x, y, nbIter, lr)
  for i = 1, nbIter do
    local timer = torch.Timer()
    local shuffle = torch.randperm(x:size(1))
    local mloss = 0
    for j = 1, x:size(1) do
      local id = shuffle[j]
      local input = x[id]
      local label = y[id]
      lin0:zeroGradParameters()
      local x1 = lin0:forward(input)
      local x2 = sig0:forward(x1)
      local output = add0:forward({x2, input})
      mloss = mloss + criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      local df_diAdd = add0:backward({x2, input}, df_do)
      local df_diSig = sig0:backward(x1, df_diAdd[1])
      local df_di = lin0:backward(input, df_diSig)
      lin0:updateParameters(lr)
    end
    print('loss:', i, mloss, timer:time().real)
  end
end

function forward0(input)
    local x1 = lin0:forward(input)
    local x2 = sig0:forward(x1)
    local output = add0:forward({x2, input})
    return output
end
--[[
print("Version Algorithmique:")
print ("Score Test sans apprentissage:", criterion:forward(forward0(xtest), ytest))
training0(criterion, x, y, 100, 3e-2)
print("Score Test:", criterion:forward(forward0(xtest), ytest))

print()

print("Version Conteneurs")
print(model1)
print ("Score Test sans apprentissage:", criterion:forward(model1:forward(xtest), ytest))
training(model1, criterion, x, y, 100, 3e-2)
print("Score Test:", criterion:forward(model1:forward(xtest), ytest))

print()

print("Version nnGraph")
print(gModel1)
--graph.dot(gModel1.fg, 'graph1')
print ("Score Test sans apprentissage:", criterion:forward(gModel1:forward(xtest), ytest))
training(gModel1, criterion, x, y, 100, 3e-2)
print("Score Test:", criterion:forward(gModel1:forward(xtest), ytest))

print()
]]--
print("Model 2")
print(gModel2)
--graph.dot(gModel2.fg, 'graph2')
ones = torch.ones(nExemples,1)
print ("Score Test sans apprentissage:", criterion:forward(gModel2:forward({xtest,ones}), ytest))
training2(gModel2, criterion, x, y, 100, 3e-2)
print("Score Test:", criterion:forward(gModel2:forward({xtest,ones}), ytest))