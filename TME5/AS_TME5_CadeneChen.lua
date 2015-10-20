require 'nn'
require 'image'
mnist = require 'mnist'

---------------------------------------------------------------------------
-- Apprentissage par descente de gradient
function train(mlp, criterion, data, labels, lr, nIter)
   local lr = lr or 1e-1
   local nIter = nIter or 1000
   local choices = torch.LongTensor((#data)[1])
   for i = 1,nIter do
      mlp:zeroGradParameters()
      choices:random((#data)[1])
      local x = data:index(1,choices)
      local y = labels:index(1,choices)
      local pred = mlp:forward(x)
      local loss = criterion:forward(pred,y)
      local df_do = criterion:backward(pred,y)
      local df_di = mlp:backward(x, df_do)
      mlp:updateParameters(lr)
      if i % 10 == 0 then
	 print(i,loss)
      end
   end
end

function penalizedFineTuning(mlp, criterion, data, labels, l, lr, nIter)
   local lr = lr or 1e-1
   local nIter = nIter or 1000
   local choices = torch.LongTensor((#data)[1])
   for i = 1,nIter do
      mlp:zeroGradParameters()
      choices:random((#data)[1])
      local x = data:index(1,choices)
      local y = labels:index(1,choices)
      local pred = mlp:forward(x)
      local loss = criterion:forward(pred,y)
      local df_do = criterion:backward(pred,y)
      local df_di = mlp:backward(x, df_do)
	  newGradParameters:add(torch.add(newParameters, -encodersParameters) * 2 * l)
	  mlp:updateParameters(lr)
      if i % 10 == 0 then
	 print(i,loss)
      end
   end
end

function buildDeepDecoder(decoders, depth)
   local decoder = nn.Sequential()
   for i = depth,1,-1 do
      decoder:add(decoders[i])
--      decoder:add(nn.Tanh())
   end
   return decoder
end

function visualizeAutoEncoding(deepEncoder, deepDecoder, data)
   local depth =  (#deepEncoder.modules)/2
   local imgSize = math.sqrt((#data)[2])
   for i = 1,(#data)[1] do
      local img = data[i]:reshape(imgSize,imgSize)
      image.save("input".. depth .."_" .. i .. ".png", img)
      img = deepDecoder:forward(deepEncoder:forward(data[i]))
      img = image.scale((img/img:abs():max() + 1):abs():reshape(imgSize,imgSize),28,28)
      image.save("output".. depth .. "_" .. i ..".png", img)
   end
end

function visualizeDecoding(deepDecoder, code)
   local img = deepDecoder:forward(code)
   local imgSize = math.sqrt((#img)[1])
   img = img:reshape(imgSize,imgSize)
   return img
end

---------------------------------------------------------------------------
-- Données mnist
trainset = mnist.traindataset()
testset = mnist.testdataset()

---------------------------------------------------------------------------
---- Constitution d'un ensemble d'apprentissage et de test à l'arrache
nEx = 500
--classes = {6,8}
classes = {6,7,8}
nClass = #classes
trainData = torch.zeros(nEx,14*14)
testsData = torch.zeros(nEx,14*14)
trainLabels = torch.zeros(nEx)
testsLabels = torch.zeros(nEx)
i = 1
j = 1
while i <= nEx do
	for k = 1,nClass do
		if trainset.label[j] == classes[k] then
	trainLabels[i] = k
	trainData[i] = image.scale(trainset.data[j],14,14)
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
			testsData[i] = image.scale(testset.data[j],14,14)
			i = i + 1
			break
		end
	end
	j = j + 1
end
--trainData = (trainData / 128) - 1 --On mets les données entre -1 et 1
--testsData = (testsData / 128) - 1
trainData = (trainData / 256)
testsData = (testsData / 256)
---------------------------------------------------------------------------
--[[
classes = torch.range(0,9)
nClass = (#classes)[1]

nEx = testset.size
testsData = torch.zeros(nEx, 14*14)
testsLabels = torch.zeros(nEx)
for i=1,nEx do
   testsLabels[i] = testset.label[i] + 1
   testsData[i] = image.scale(testset.data[i],14,14)
end

nEx = trainset.size
trainData = torch.zeros(nEx, 14*14)
trainLabels = torch.zeros(nEx)
for i=1,nEx do
   trainLabels[i] = trainset.label[i] + 1
   trainData[i] = image.scale(trainset.data[i],14,14)
end
trainData = (trainData / 256)
testsData = (testsData / 256)
]]--
----------------------------------------------------------------------------
-- Liste des tailles successives
layerSize = {(#trainData)[2],
	     100,
	     80,
	     60,
	     40,
	     20
             }
	     

-- Constitution des layers
encoders = {}
decoders = {}
for i=1,(#layerSize)-1 do
   table.insert(encoders, nn.Linear(layerSize[i],layerSize[i+1]))
   table.insert(decoders, nn.Linear(layerSize[i+1],layerSize[i]))
end

-- Constitution des autoencoders
autoEncoders = {}
for i=1,#encoders do
   autoEncoder = nn.Sequential()
   autoEncoder:add(encoders[i])
   autoEncoder:add(nn.Tanh())
   autoEncoder:add(decoders[i])
   --autoEncoder:add(nn.Tanh())
   table.insert(autoEncoders, autoEncoder)
end

-- Entrainement des AutoEncodeurs et stacking des couches
mse = nn.MSECriterion()
deepEncoder = nn.Sequential()
for i=1,(#autoEncoders) do
   print("AutoEncodeur ", i)
   x = deepEncoder:forward(trainData)
   train(autoEncoders[i], mse, x, x, 1e-1, 500)
   deepEncoder:add(encoders[i])
   deepEncoder:add(nn.Tanh())
   visualizeAutoEncoding(deepEncoder, buildDeepDecoder(decoders, i), trainData[{{1,5}}])
end

--Entrainement du classifieur lineaire
print("Clf:")
classifier = nn.Linear(layerSize[#layerSize], nClass)
nll = nn.CrossEntropyCriterion()
x = deepEncoder:forward(trainData)
train(classifier, nll, x, trainLabels, 1e-1, 500)

--Consitution du classifieur final
deepClassifier = nn.Sequential()
deepClassifier:add(deepEncoder)
deepClassifier:add(classifier)

--Visualiser un decodage de la dernière couche:
codeSize = layerSize[#layerSize]
for i = 1,codeSize do
   code = torch.zeros(codeSize)
   code[i] = 1
   img = visualizeDecoding(buildDeepDecoder(decoders, #decoders), code)
   img = image.scale(img, 28,28)
   image.save("decoding" .. i .. ".png", img)
end

--FineTuning
--Extraction des paramètres des encodeurs
newParameters, newGradParameters = deepEncoder:getParameters()
encodersParameters = newParameters:clone()
print("Fine tuning:")
penalizedFineTuning(deepClassifier, nll, trainData, trainLabels, 0.3, 1e-1, 200)

-- Evaluation en train
pred = deepClassifier:forward(trainData)
__, pred = torch.max(pred,2)
print("score train:")
print(torch.add(trainLabels:long(),-pred):eq(0):double():mean())

-- Evaluation en test
pred = deepClassifier:forward(testsData)
__, pred = torch.max(pred,2)
print("score test:")
print(torch.add(testsLabels:long(),-pred):eq(0):double():mean())


--[[ trois méthodes d'entrainement de l'encodeur
1. sans Fine Tuning
2. avec Fine Tuning 
3. avec Fine Tuning et contrainte : lambda * ||Teta_1 - Teta_1^*||

Visualisation

D'autres bases de données

-- Pour le 17 novembre
]]--
