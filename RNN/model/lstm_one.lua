local lstm_variable = {}

function lstm_variable.one(dim_x, dim_h)--, dropout, graph2fig)

  local h = nn.Identity()()
  local c = nn.Identity()()

  local x = nn.Identity()()

  local gate_i = nn.Sigmoid()(nn.CAddTable()({
    MapLinear(dim_x, dim_h)(x),
    MapLinear(dim_h, dim_h)(h),
    MapLinear(dim_h, dim_h)(c)
  }))
  local gate_f = nn.Sigmoid()(nn.CAddTable()({
    MapLinear(dim_x, dim_h)(x),
    MapLinear(dim_h, dim_h)(h),
    MapLinear(dim_h, dim_h)(c)
  }))
  local learning = nn.Tanh()(nn.CAddTable()({
    MapLinear(dim_x, dim_h)(x),
    MapLinear(dim_h, dim_h)(h)
  }))
  local c_out = nn.CAddTable()({
    nn.CMulTable()({gate_f, c}),
    nn.CMulTable()({gate_i, learning})        
  })
  local gate_o = nn.Sigmoid()(nn.CAddTable()({
    MapLinear(dim_x, dim_h)(x),
    MapLinear(dim_h, dim_h)(h),
    MapLinear(dim_h, dim_h)(c_out)
  }))
  local h_out = nn.CMulTable()({gate_o, nn.Tanh()(c_out)})
 
  local y = nn.Linear(dim_h, 1)(h)
  local model = nn.gModule({h, c, x}, {h_out, h_out})

  model.name = 'lstm_one'
  
  return model
end

return lstm