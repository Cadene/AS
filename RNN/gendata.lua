nn = require 'nn'
toy = require 'util/toy'
csv = require 'util/csv'

-- http://kbullaughey.github.io/lstm-play/toy/

g = {} -- gendata 
g.min_y = -2
g.max_y = 10
g.std = .2
g.len = 3
g.delta = 1
torch.save('data/toy_gendata.t7', g)

data_n = 70000
data = toy.generate_data(data_n, g.max_y, g.std, g.len, g.delta)
label = {}
for i=1,g.len do
  label[i] = 'x'..i
  label[i+g.len] = 'y'..i
end
csv.tensor_to_csv(data, 'data/toy.csv', label)
torch.save('data/toy.t7', data)

curve_n = 100
curve = toy.generate_curve(curve_n, g.min_y, g.max_y)
csv.tensor_to_csv(curve, 'data/toy_curve.csv', {'x','y'})

-- END