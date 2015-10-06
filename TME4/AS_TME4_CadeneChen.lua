require 'nn'

local ReQU, parent = torch.class('nn.ReQU', 'nn.Module')

function ReQU:__init()
   parent.__init(self)
end
 
function ReQU:updateOutput(input)
   self.output = torch.pow(input,2)
   self.inputIsPositive = input:ge(0):double()
   self.output:cmul(self.inputIsPositive)
   return self.output
end
 
function ReQU:updateGradInput(input, gradOutput)
   self.gradInput = input * 2
   self.gradInput:cmul(self.inputIsPositive)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end

local Linear, parent = torch.class('nn.MyLinear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self:reset()
end
 
function Linear:updateOutput(input) 
   self.output = self.weight * input
   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   self.gradInput = self.weight:transpose(1,2) * gradOutput
   return self.gradInput
end
 
function Linear:accGradParameters(input, gradOutput)
   self.gradWeight = self.gradWeight + gradOutput:reshape(gradOutput:size(1),1) * input:reshape(1,input:size(1))
   return self.gradWeight
end
 
function Linear:reset()
    self.weight:uniform()
end

lin = nn.Linear(5,10)
lin.bias:zero()
lin:zeroGradParameters()

mylin = nn.MyLinear(5,10)
mylin.weight = lin.weight:clone()
mylin:zeroGradParameters()

q = torch.linspace(-9,10,5)

d = torch.rand(10)

lin:forward(q)
mylin:forward(q)

lin:accGradParameters(q,d)
mylin:accGradParameters(q,d)

print(lin.weight)
print(mylin.weight)
