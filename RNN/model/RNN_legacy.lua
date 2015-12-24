local RNN = torch.class('RNN')

function RNN.create(input_size, rnn_size, seq_size, dropout, graph2fig)

    local dropout = dropout or 0
    local graph2fig = graph2fig or true

    local g_linear = nn.Linear(rnn_size, input_size)()
    local g_softmax = nn.SoftMax()(g_linear)
    local g = nn.gModule({g_linear}, {g_softmax})

    local h_linH = nn.Linear(rnn_size, rnn_size)()
    local h_linW = nn.Linear(input_size, rnn_size)()
    local h_sum = nn.CAddTable()({h_linH, h_linW})
    local h_tanh = nn.Tanh()(h_sum)
    local h = nn.gModule({h_linH, h_linW}, {h_tanh})

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
        graph.dot(g.fg, 'RNN_G', 'RNN_G')
        graph.dot(h.fg, 'RNN_H', 'RNN_H')
        graph.dot(model.fg, 'RNN', 'RNN')
        graph.dot(modules_H.fg, 'RNN_mod_H', 'RNN_mod_H')
        graph.dot(modules_G.fg, 'RNN_mod_G', 'RNN_mod_G')
    end
    
    return model
end

return RNN