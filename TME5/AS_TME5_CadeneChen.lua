require 'nn'
require 'image'
mnist = require 'mnist'

---------------------------------------------------------------------------
-- Apprentissage par descente de gradient
function train(mlp, criterion, data, labels, lr, nIter, miniBatchSize, decayTime)
   local lr = lr or 1e-1
   local nIter = nIter or 1000
   local miniBatchSize = miniBatchSize or 200
   local choices = torch.LongTensor((#data)[1])
   for i = 1,nIter do
      mlp:zeroGradParameters()
      choices:randperm((#data)[1])
      local j = 1
      local loss = 0
      while j <= (#data)[1] do
	 local k = math.min((#data)[1], j+miniBatchSize-1)
	 local x = data:index(1,choices[{{j,k}}])
	 local y = labels:index(1,choices[{{j,k}}])
	 local pred = mlp:forward(x)
	 loss = loss + criterion:forward(pred,y)
	 local df_do = criterion:backward(pred,y)
	 local df_di = mlp:backward(x, df_do)
	 mlp:updateParameters(lr)
	 j = j + miniBatchSize
      end
      if i % 1 == 0 then
	 print(i,loss)
	 if i % decayTime == 0 then
	    lr = lr / 2
	    print('new learning rate = ', lr)
	 end
      end
   end
end

--[[
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
]]--

function buildDeepDecoder(decoders, depth)
   local decoder = nn.Sequential()
   for i = depth,1,-1 do
      decoder:add(decoders[i])
      decoder:add(nnActivationFunctionDec())
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
      img = image.scale((img / img:max()):reshape(imgSize,imgSize),28,28)
      img[img:lt(0)] = 0
      image.save("output".. depth .. "_" .. i ..".png", img)
   end
end

function visualizeDecoding(deepDecoder, code)
   local img = deepDecoder:forward(code)
   if ((not imgZero) or ((#img)[1] ~= (#imgZero)[1])) then
      local codeZ = torch.zeros(#code)
      imgZero = deepDecoder:forward(codeZ):clone()
      visualizeDecoding(deepDecoder, codeZ)
      img = deepDecoder:forward(code)
   end
   local imgSize = math.sqrt((#img)[1])
   --local visu = torch.zeros(imgSize,imgSize)
   local visu = (img - imgZero):reshape(imgSize, imgSize)
   visu = visu / torch.abs(visu):max()
   return visu
end

---------------------------------------------------------------------------
-- Données mnist
trainset = mnist.traindataset()
testset = mnist.testdataset()

---------------------------------------------------------------------------
---- Constitution d'un ensemble d'apprentissage et de test à l'arrache
--[[
nEx = 10000
classes = {1,2,3,4,5,6,7,8,9,0}
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
]]
---------------------------------------------------------------------------
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
trainData = (trainData / 255)
testsData = (testsData / 255)

----------------------------------------------------------------------------
-- CONFIG --
----------------------------------------------------------------------------
-- Liste des tailles successives
layerSize = {(#trainData)[2],
	     50,
	     50,
	     50,
	     50,
	     50
}

lrAutoEnc = 1e-2
lrClassif = 1e-3
lrFineTune = 1e-6
-- score de 0.9173 avec lrFineTune = 5e-09
nEpochAutoEnc = 200
nEpochClassif = 200
nEpochFineTune = 200
decayTime = 30
miniBatchSize = 20

lambdaL1 = {1e-4,
	    1e-4,
	    1e-6,
	    1e-8,
	    5e-10}



nnActivationFunctionEnc = nn.Tanh
nnActivationFunctionDec = nn.Identity
--nnActivationFunctionDec = nn.Tanh  -- Pas bon

-----------------------------------------------------------------------------


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
   autoEncoder:add(nn.L1Penalty(lambdaL1[i]))
   autoEncoder:add(nnActivationFunctionEnc())
   autoEncoder:add(decoders[i])
   autoEncoder:add(nnActivationFunctionDec())
   table.insert(autoEncoders, autoEncoder)
end

-- Entrainement des AutoEncodeurs et stacking des couches
deepEncoder = nn.Sequential()
for i=1,(#autoEncoders) do
   print("AutoEncodeur ", i)
   x = deepEncoder:forward(trainData)
   train(autoEncoders[i], nn.MSECriterion(), x, x, lrAutoEnc, nEpochAutoEnc, miniBatchSize, decayTime)
   deepEncoder:add(encoders[i])
   deepEncoder:add(nnActivationFunctionEnc())
   visualizeAutoEncoding(deepEncoder, buildDeepDecoder(decoders, i), trainData[{{1,10}}])
end

--Entrainement du classifieur lineaire
print("Clf:")
classifier = nn.Linear(layerSize[#layerSize], nClass)
x = deepEncoder:forward(trainData)
train(classifier, nn.CrossEntropyCriterion(), x, trainLabels, lrClassif, nEpochClassif, miniBatchSize, decayTime)

--Consitution du classifieur final
deepClassifier = nn.Sequential()
deepClassifier:add(deepEncoder)
deepClassifier:add(classifier)

--Visualiser un decodage:
for d = 1,#decoders do
   deepDecoder = buildDeepDecoder(decoders, d)
   codeSize = layerSize[d+1]
   for i = 1,codeSize do
      code = torch.zeros(codeSize)
      code[i] = 1
      img = visualizeDecoding(deepDecoder, code)
      img = image.scale(img,28,28)
      local decoding3 = torch.zeros(3,28,28)
      decoding3[1] = img
      decoding3[2] = -img
      decoding3[1][img:lt(0)] = 0
      decoding3[2][(-img):lt(0)] = 0
--      image.save("decoding" .. d .. "_" .. i .. ".png", img)
      image.save("decodingColor" .. d .. "_" .. i .. ".png", decoding3)
   end
end

-- Evaluations sans fine-tuning---------------------------------
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



--FineTuning----------------------------------------------------
--Extraction des paramètres des encodeurs
--newParameters, newGradParameters = deepEncoder:getParameters()
--encodersParameters = newParameters:clone()
print("Fine tuning:")
train(deepClassifier, nn.CrossEntropyCriterion(), trainData, trainLabels, lrFineTune, nEpochFineTune, miniBatchSize, decayTime)
--penalizedFineTuning(deepClassifier, nn.CrossEntropyCriterion(), trainData, trainLabels, 0.3, 1e-1, 200)

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
