require 'nn'

dimInput = 6
dimOutput = 3
learningRate = 1e-2

input = torch.Tensor{1,2,3,4,5,6}
label = torch.Tensor{0,3,2}

model = nn.Linear(dimInput, dimOutput)
criterion = nn.MSECriterion()

model:zeroGradParameters()
output = model:forward(input)
loss = criterion:forward(output, label)
df_do = criterion:backward(output, label)
df_di = model:backward(input, df_do)
model:updateParameters(learningRate)

print(df_do:size())
print(df_di:size())