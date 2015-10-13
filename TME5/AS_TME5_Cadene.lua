require 'torch'
require 'nn'

-- Importation de MNIST

local mnist = require 'mnist'

trainset = mnist.traindataset()
testset  = mnist.testdataset()

print(trainset.size)
print(testset.size)

-- Création du dataset MNIST minifié

dim = 14
dimSq = dim^2

avg_pool = nn.SpatialAveragePooling(2,2,2,2)
trainset.data = nn.Reshape(dimSq):forward(avg_pool:forward(trainset.data:double()))
testset.data  = nn.Reshape(dimSq):forward(avg_pool:forward(testset.data:double()))

-- Création du dataset MNIST bruité

trainmaskg = torch.randn(trainset.data:size(1),dimSq) -- mean=0, var=1
testmaskg  = torch.randn(testset.data:size(1),dimSq) -- mean=0, var=1

trainsetg = {data=trainset.data+trainmaskg, label=trainset.label, size=trainset.data:size(1)}
testsetg  = {data=testset.data+testmaskg, label=testset.label, size=testset.data:size(1)}

-- Création auto encoder

encoders = {}
encoders[1] = nn.Linear(dimSq,dimSq)
encoders[2] = nn.Linear(dimSq,dimSq)
encoders[3] = nn.Linear(dimSq,dimSq)
encoders[4] = nn.Linear(dimSq,dimSq)

decoders = {}
decoders[1] = nn.Linear(dimSq,dimSq)--add(nn.Reshape(dim,dim))
decoders[2] = nn.Linear(dimSq,dimSq)
decoders[3] = nn.Linear(dimSq,dimSq)
decoders[4] = nn.Linear(dimSq,dimSq)


--[[
encoders = {}
encoders[1] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
encoders[2] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
encoders[3] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
encoders[4] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)

decoders = {}
decoders[1] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
decoders[2] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
decoders[3] = nn.SpatialConvolution(1,1,1,1,1,1,0,0)
decoders[4] = nn.SpatialConvolution(1,1,1,1,1,1,0,0) ]]

criterion = nn.MSECriterion()

-- Entrainement des encoders et des decoders denoising

-- Premier auto-encoder

function training_autoencoder_1(nbIter, lr, wd)
  for i = 1, nbIter do
    local timer = torch.Timer()
    local shuffle = torch.randperm(trainsetg.size)
    local mloss = 0
    for j = 1, trainsetg.size do
      local id = shuffle[j]
      local inputg = trainsetg.data[id]
      local input  = trainset.data[id]
      encoders[1]:zeroGradParameters()
      decoders[1]:zeroGradParameters()
      local outputE = encoders[1]:forward(inputg)
      local outputD = decoders[1]:forward(outputE)
      mloss = mloss + criterion:forward(outputD, input)
      local df_do = criterion:backward(outputD, input)
      local df_diE = encoders[1]:backward(inputg, df_do)
      local df_diG = decoders[1]:backward(outputE, df_do)
      encoders[1]:updateParameters(lr)
      decoders[1]:updateParameters(lr)
    end
    print('loss:', i, mloss, timer:time().real)
  end
end

training_autoencoder_1(10, 1e-5, 0)

--[[
for i = 1, nbIter do
   shuffle = torch.randperm(inputs:size(1))
   for j = 1, inputs:size(1) do
      id = shuffle[j]
      input = inputs[id]
      label = torch.Tensor{y[id]}
      model:zeroGradParameters()
      output = model:forward(input)
      loss = criterion:forward(output, label)
      df_do = criterion:backward(output, label)
      df_di = model:backward(input, df_do)
      model:updateParameters(lr)
      print(i,loss)
   end
end ]]



-- Visualisation avec (1,0,0,0,...) dans l'espace latent


