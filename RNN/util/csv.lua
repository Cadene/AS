
local csv = {}

-- tensor 2D
-- first dim  : nb exemples
-- second dim : nb features
csv.tensor_to_csv = function(tensor, path, columns, sep)
  local columns = columns or nil
  local sep = sep or ','
  local file = assert(io.open(path, 'w'))
  if columns ~= nil then
    if #columns ~= tensor:size(2) then
      error("not enough columns")
    end
    file:write(columns[1])
    for i=2,#columns do 
      file:write(sep,columns[i])
    end
    file:write("\n")
  end
  for i=1,tensor:size(1) do
    file:write(tensor[i][1])
    for j=2,tensor:size(2) do
      file:write(sep,tensor[i][j])
    end
    file:write("\n")
  end
  file:close()
end

return csv
