local lstm = {}

local linear_map = {} -- beware of global
local MapLinear = function(dim_i, dim_o)
  local l = nn.Linear(dim_i, dim_o)
  table.insert(linear_map, l)
  return l
end

function lstm.create(seq_len, dim_h, dim_xi)--, dropout, graph2fig)

  linear_map = {}

  local h0 = nn.Identity()()
  local c0 = nn.Identity()()

  local x = nn.Identity()()
  local xs = nn.SplitTable(2)(x)

  local h = h0
  local c = c0
  for i=1,seq_len do
    --local xi = nn.Reshape(1)(nn.SelectTable(i)(xs)) -- Reshape(1) for batch
    local xi = nn.Reshape(1)(nn.SelectTable(i)(xs))

    local gate_i = nn.Sigmoid()(nn.CAddTable()({
      MapLinear(dim_xi, dim_h)(xi),
      MapLinear(dim_h, dim_h)(h),
      MapLinear(dim_h, dim_h)(c)
    }))
    local gate_f = nn.Sigmoid()(nn.CAddTable()({
      MapLinear(dim_xi, dim_h)(xi),
      MapLinear(dim_h, dim_h)(h),
      MapLinear(dim_h, dim_h)(c)
    }))
    local learning = nn.Tanh()(nn.CAddTable()({
      MapLinear(dim_xi, dim_h)(xi),
      MapLinear(dim_h, dim_h)(h)
    }))
    c = nn.CAddTable()({
      nn.CMulTable()({gate_f, c}),
      nn.CMulTable()({gate_i, learning})        
    })
    local gate_o = nn.Sigmoid()(nn.CAddTable()({
      MapLinear(dim_xi, dim_h)(xi),
      MapLinear(dim_h, dim_h)(h),
      MapLinear(dim_h, dim_h)(c)
    }))
    h = nn.CMulTable()({gate_o, nn.Tanh()(c)})
  end
  local y = nn.Linear(dim_h, 1)(h)
  local model = nn.gModule({h0, c0, x}, {y})

  model.name = 'lstm'

  local linear_count = #linear_map / seq_len
  for i=2,seq_len do
    for l=1,linear_count do
      local first_linear = linear_map[i] -- first instance
      local current_linear = linear_map[(i-1)*linear_count+i]
      local fl_param, fl_grad_param = first_linear:parameters()
      local cl_param, cl_grad_param = current_linear:parameters()
      cl_param[1]:set(fl_param[1]) -- share weigths
      cl_param[2]:set(fl_param[2]) -- share biais
      cl_grad_param[1]:set(fl_grad_param[1])
      cl_grad_param[2]:set(fl_grad_param[2])
    end
  end

  model.parameters, model.gradParameters = model:getParameters()
  
  return model
end

return lstm