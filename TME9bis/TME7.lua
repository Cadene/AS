require 'torch'
require 'nn'
require 'nngraph'

require 'gUtils'

local model_utils = require 'model_utils'
local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

local v=CharLMMinibatchLoader.create("data.t7","vocab.t7",1,5)

dimx = v.vocab_size
dimh = 10
seqSize = v.seq_length

-- 2 --
h_linH = nn.Linear(dimh, dimh)()
h_linW = nn.Linear(dimx, dimh)()

h_sum = nn.CAddTable()({h_linH, h_linW})

h_tanh = nn.Tanh()(h_sum)

h_graph = nn.gModule({h_linH, h_linW}, {h_tanh})

-- 1 --
g = nn.Sequential()
g:add(nn.Linear(dimh,dimx))
g:add(nn.SoftMax())

H_Module, Parent = torch.class('nn.H_Module', 'nn.Module')

H_Module = graphToModule(h_graph, H_Module, Parent)

h = H_Module()

-- 3 --
--[[
listH = model_utils.clone_many_times(h, seqSize)
listG = model_utils.clone_many_times(g, seqSize)

moduleComplet = {}
moduleComplet.listH = listH
moduleComplet.listG = listG
moduleComplet.dimh = dimh
moduleComplet.seqSize = seqSize

function moduleComplet:forward(wt)
   self.ht = torch.zeros(self.dimh)
   for i=1,seqSize do
      self.ht = self.listH[i]:forward(self.ht, wt[i])
   end
   self.p = 
end

print(v)
print(v.x_batches[1])
print(v.y_batches[1])
]]--
