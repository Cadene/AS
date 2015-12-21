
local toy = {}

-- period 8 pi = 25.13 (max_y)
toy.f = function(x)
  if not torch.isTensor(x) then
    x = torch.Tensor(1):fill(x)
  end
  return (
    torch.sin(x/2 - 1):mul(0.5) +
    torch.sin(x) +
    torch.sin(x*2 + 2) +
    torch.sin(x/4 + 1) + 2)
end

toy.generate_data = function(n, max_y, std, len, delta, rand)
  local std = std or 0
  local len = len or 1
  local delta = delta or 1
  local rand = rand or 1
  local seq_y = {}
  if rand == 1 then
    seq_y[1] = torch.rand(n,1):mul(max_y)
  else
    seq_y[1] = torch.linspace(0, max_y, n):view(n, 1)
  end
  for i=2,len do
    seq_y[i] = seq_y[i-1]-delta
  end
  local y = nn.JoinTable(2):forward(seq_y)
  local x = toy.f(y) + torch.randn(n,len):mul(std)
  local data = nn.JoinTable(2):forward{x, y}
  return data
end

toy.generate_curve = function(n, min_y, max_y)
  local y = torch.linspace(min_y, max_y, n):view(n, 1)
  local x = toy.f(y)
  local curve = nn.JoinTable(2):forward{x, y}
  return curve
end

return toy

-- END