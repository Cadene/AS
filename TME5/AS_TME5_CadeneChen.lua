require 'nn'
require 'image'
mnist = require 'mnist'

---------------------------------------------------------------------------
-- Apprentissage par descente de gradient
function train(mlp, criterion, data, labels, lr, nIter)
   local lr = lr or 1e-5
   local nIter = nIter or 100
   for i = 1,nIter do
      mlp:zeroGradParameters()
      y = mlp:forward(data)
      loss = criterion:forward(y,labels)
      df_do = criterion:backward(y,labels)
      df_di = mlp:backward(data, y)
      mlp:updateParameters(lr)
      print(loss)
   end
end

---------------------------------------------------------------------------
-- Données mnist
trainset = mnist.traindataset()
testset = mnist.testdataset()

-- Constitution d'un ensemble d'apprentissage et de test à l'arrache
nEx = 1000
classes = {6,8}
nClass = #classes

trainData = torch.zeros(nEx,28*28)
testsData = torch.zeros(nEx,28*28)
trainLabels = torch.zeros(nEx)
testsLabels = torch.zeros(nEx)
i = 1
j = 1
while i <= nEx do
   for k = 1,nClass do
      if trainset.label[j] == classes[k] then
	 trainLabels[i] = k
	 trainData[i] = trainset.data[j]
	 i = i + 1
	 break
      end
   end
   j = j + 1
end
i = 1
j = 1
while i <= nEx do
   for k = 1,nClass do
      if testset.label[j] == classes[k] then
	 testsLabels[i] = k
	 testsData[i] = testset.data[j]
	 i = i + 1
	 break
      end
   end
   j = j + 1
end
trainData = (trainData / 128) - 1 --On mets les données entre -1 et 1
testsData = (testsData / 128) - 1

----------------------------------------------------------------------------
-- Liste des tailles successives
layerSize = {28*28,
	     14*14,
	     50,
	     25,
	     28*28}

-- Constitution des layers
layers = {}
for i=1,(#layerSize)-1 do
   table.insert(layers, nn.Linear(layerSize[i],layerSize[i+1]))
end

-- Constitution des autoencoders
autoEncoders = {}
for i=1,#layers do
   autoEncoder = nn.Sequential()
   autoEncoder:add(layers[i])
   autoEncoder:add(nn.Tanh())
   autoEncoder:add(nn.Linear(layerSize[i+1], layerSize[i]))
   --autoEncoder:add(nn.Tanh())
   table.insert(autoEncoders, autoEncoder)
end

-- Entrainement des AutoEncodeurs et stacking des couches
mse = nn.MSECriterion()
stack = nn.Sequential()
--stack:add(nn.Reshape(28*28,1))
--stack:add(nn.MulConstant(1/128))
--stack:add(nn.AddConstant(-1))
for i=1,(#autoEncoders)-1 do
   print(i)
   x = stack:forward(trainData)
   train(autoEncoders[i], mse, x, x)
   stack:add(layers[i])
   stack:add(nn.Tanh())
end
i = (#autoEncoders)
x = stack:forward(trainData)
train(autoEncoders[i], mse, x, x)

--Entrainement du classifieur final
classifieur = nn.Linear(layerSize[i], nClass)
nll = nn.ClassNLLCriterion()
x = stack:forward(trainData)
train(classifieur, nll, x, trainLabels)
stack:add(classifieur)

--FineTuning
train(stack, nll, trainData, trainLabels, 1e-5, 100)

-- Evaluation en test
pred = stack:forward(testsData)
__, pred = torch.max(pred,2)
print("score:")
print(torch.add(testsLabels:long(),-pred):eq(0):double():mean())

