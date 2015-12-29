function create_dico()
  local dico = {}
  dico.word2id = {}
  dico.id2word = {}
  local file,err = io.open('data/aclImdb/imdb.vocab')
  local id = 1
  while true do
    local word = file:read()
    if word == nil then
      break
    end
    dico.word2id[word] = id
    dico.id2word[id] = word
    id = id + 1
  end
  return dico
end

function create_loader()
  local loader = {}
  loader.path2pos = '/Users/remicadene/Dropbox/_Docs/UPMC/AS/RNN/data/aclImdb/train/pos/'
  loader.path2neg = '/Users/remicadene/Dropbox/_Docs/UPMC/AS/RNN/data/aclImdb/train/neg/'
  local path2esc = {}
  path2esc['.'] = true
  path2esc['..'] = true
  path2esc['.DS_Store'] = true
  path2esc['._.DS_Store'] = true
  loader.txt = {}
  loader.label = {}
  local id = 1
  for _, txt in pairs(paths.dir(loader.path2pos)) do
    if path2esc[txt] == nil then
      loader.txt[id] = txt
      loader.label[id] = 1
    end
  end
  for _, txt in pairs(paths.dir(loader.path2neg)) do
    if path2esc[txt] == nil then
      loader.txt[id] = txt
      loader.label[id] = -1
    end
  end
  return loader
end

-- word2id, id2word = create_dico()
loader = create_loader()
for _, a in pairs(loader.txt) do
  if a == '.' then
    print('fuuuck')
    break
  end
end

