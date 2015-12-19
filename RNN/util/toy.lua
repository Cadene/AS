
local toy = {}

toy.max_output = 10

toy.f = function(x)
  if not torch.isTensor(x) then
    x = torch.Tensor(1):fill(x)
  end
  -- period 8 pi = 25.13
  return (
    torch.sin(x/2 - 1):mul(0.5) +
    torch.sin(x) +
    torch.sin(x*2 + 2) +
    torch.sin(x/4 + 1) + 2)
end

toy.output_to_input = function(output, std, len)
  local std = std or 0.2
  local len = len or 3
  local seq = {}
  for i=1,len do
    seq[i] = output-i+1
  end
  local input = toy.f(nn.JoinTable(2):forward(seq))
  local n = output:size(1)
  return input + torch.randn(n,len):mul(std) 
end

return toy
