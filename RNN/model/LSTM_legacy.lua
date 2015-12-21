local class = require 'class'

local LSTM = class('LSTM')

function LSTM.create(input_size, rnn_size, seq_size, dropout, graph2fig)

    local dropout = dropout or 0
    local graph2fig = graph2fig or false

    local input_x = nn.Identity()()
    local input_c = nn.Identity()()
    local input_h = nn.Identity()()

    local lin_xi = nn.Linear(input_size, rnn_size)(input_x)
    local lin_hi = nn.Linear(rnn_size, rnn_size)(input_h)
    local lin_ci = nn.CMul(rnn_size)(input_c)
    local biais_i = nn.Linear(1, rnn_size)()
    local output_i = nn.Sigmoid()(nn.CAddTable()({lin_xi, lin_hi, lin_ci, biais_i}))
    

    local lin_xf = nn.Linear(input_size, rnn_size)(input_x)
    local lin_hf = nn.Linear(rnn_size, rnn_size)(input_h)
    local lin_cf = nn.CMul(rnn_size)(input_c)
    local biais_f = nn.Linear(1, rnn_size)()
    local output_f = nn.Sigmoid()(nn.CAddTable()({lin_xf, lin_hf, lin_cf, biais_f}))

    local mult1_c = torch.cmul(output_f, input_c) -- avoid gradient vanishing
    local lin_xc = nn.Linear(input_size, rnn_size)(input_x)
    local lin_hc = nn.Linear(rnn_size, rnn_size)(input_h)
    local biais_c = nn.Linear(1, rnn_size)()
    local tanh_c = nn.Tanh()(nn.CAddTable(){lin_xc, lin_hc, biais_c})
    local mult2_c = nn.CMulTable()({output_i, tanh_c})
    local output_c = nn.CAddTable()({mult1_c, mult2_c})

    local lin_xo = nn.Linear(input_size, rnn_size)(input_x)
    local lin_ho = nn.Linear(rnn_size, rnn_size)(input_h)
    local lin_co = nn.CMul(rnn_size)(input_c)
    local biais_o = nn.Linear(1, rnn_size)()
    local output_f = nn.Sigmoid()(nn.CAddTable()({lin_xo, lin_ho, lin_co, biais_o}))

    local output_h = nn.CMulTable()({output_o, output_c})

    local model = nn.gModule({input_h, input_c, input_x}, {output_h, output_c, output_x})

    --[[
    local modules_H = ModelUtil.clone_many_times(h, seq_size)
    local modules_G = ModelUtil.clone_many_times(g, seq_size)

    local inputs  = {}
    local outputs = {}
    local list_h  = {}
    inputs[1] = nn.Identity()()
    list_h[1] = inputs[1]
    for i = 1, seq_size do
      inputs[i+1] = nn.Identity()()
      list_h[i+1] = modules_H[i]({list_h[i],inputs[i+1]})
      outputs[i] = modules_G[i](list_h[i+1])
    end
    local model = nn.gModule(inputs, outputs)

    if graph2fig then
        graph.dot(model.fg, 'LSTM', 'LSTM')
    end]]

    graph.dot(model.fg, 'LSTM', 'LSTM')

    return model
end

return LSTM