local mlp = {}

function mlp.create(dim_x, dim_h, dim_y, dropout)

  local model = nn.Sequential()
  model:add(nn.Linear(dim_x, 30))
  --model:add(nn.BatchNormalization(dim_h))
  model:add(nn.Tanh())
  -- model:add(nn.Dropout(dropout))
  model:add(nn.Linear(30, 20))
  model:add(nn.Tanh())
  -- model:add(nn.Dropout(dropout))
  model:add(nn.Linear(20, dim_y))

  return model
end

return mlp

-- END