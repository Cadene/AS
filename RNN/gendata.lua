nn = require 'nn'
toy = require 'util/toy'
csv = require 'util/csv'

-- http://kbullaughey.github.io/lstm-play/toy/

n = 100
std = 0.2
max_output = 10
output = torch.rand(n,1):mul(max_output)
input = toy.output_to_input(output, std)

data = nn.JoinTable(2):forward{input,output}

csv.tensor_to_csv(data, 'data/gendata.csv', {'n','n-1','n-2','output'})
