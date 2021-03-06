require 'xlua'
require 'optim'
require 'nn'
require 'image'

local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --layers                   (default 4)             Number of layers (in case of simple model)
   --max_epoch                (default 100)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   -n,--num_of_nodes          (default 1)             Number of nodes to run.
   -i,--id                    (default 0)             id of the node.
   -f, --merge_freq           (default 130)             SESOP frequancy.
   --sesop_batch_size         (default 1000)            SESOP base batch size.
   --port                     (default 8080)            port for communication between solvers.
   --optimizer                (default "sgd")            internal optimizer fuinction.
   -h,--history_size          (default 5)            sesop history size.
   -m,--merger                (default "sesop")            internal optimizer fuinction.
]]

opt.backend = 'cudnn' 
local ipc = require 'libipc'
local sys = require 'sys'
require 'cunn'

if (opt.num_of_nodes > 1) then
  require 'seboost_parallel'
else
  require 'seboost'
end

local optimizer = optim.sgd
if (opt.optimizer == 'adagrad') then
  optimizer = optim.adagrad
end
if (opt.optimizer == 'adam') then
  optimizer = optim.adam
end
if (opt.optimizer == 'adadelta') then
  optimizer = optim.adadelta
end
if (opt.optimizer == 'rprop') then
  optimizer = optim.rprop
end

--opt.save = opt.save..'/n_'..opt.num_of_nodes..'_lrd_'..opt.epoch_step..'/id_'..opt.id
--opt.epoch_step = math.ceil(opt.epoch_step/opt.num_of_nodes) --we mult the epoch_step to acount for multiple nodes.

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    --print('start BatchFlip:updateOutput')
    if false and self.train and opt.model ~= 'simple' then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      --print(flip_mask)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then 
        --  print('about to call image.hflip')
          --print(input:size())
          --print(input[i]:size())
          --print('i='..i)
          image.hflip(input[i], input[i]) 
          --print('after call image.hflip')
        end
      end
    end
    self.output:set(input)
    --print('done BatchFlip:updateOutput')
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

--make sure all experiments have the same init weights.
torch.manualSeed(8765467)

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.benchmark=true
   cudnn.convert(model:get(3), cudnn)
end

print(model)

--make sure every rank uses different permutations.
torch.manualSeed(8765467*(opt.id + 1))

print(c.blue '==>' ..' loading data')
--SV DEBUG

if (opt.model == 'simple') then
  require 'cubic_provider'
  --provider = Provider()
  --torch.save('cubic_provider.t7', provider)
  provider = torch.load('cubic_provider.t7')
else
  dofile './provider.lua'
  provider = torch.load 'provider.t7'
end


provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
--SV DEBUG
if (opt.model == 'simple') then
  criterion = cast(nn.MSECriterion())
else
  criterion = cast(nn.CrossEntropyCriterion())
end


function copy2(obj)
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[copy2(k)] = copy2(v) end
  return res
end

print(c.blue'==>' ..' configuring optimizer')

optimState = { }
sesopConfig = {
        optMethod=optimizer, 
        sesopData=provider.trainData.data,
        sesopLabels=provider.trainData.labels,
        sesopUpdate=opt.merge_freq,
        isCuda=true,
        save=opt.save,
        optConfig={
          learningRate = opt.learningRate,
          weightDecay = opt.weightDecay,
          momentum = opt.momentum,
          learningRateDecay = opt.learningRateDecay,
        }, --placeholder state for the inner optimization function.
        
        sesopBatchSize=opt.sesop_batch_size,
        numNodes=opt.num_of_nodes,
        nodeIters=opt.merge_freq,
        histSize=opt.history_size,
        merger=opt.merger,
        model=model
}  
print(sesopConfig)



if(opt.num_of_nodes > 1) then
  if (opt.id == 0) then
    sesopConfig.master = Master(parameters, opt.num_of_nodes - 1, opt.port) 
  end

  if (opt.id > 0) then
    sesopConfig.worker = Worker(opt.id, opt.port)
  end
  
end

inEpochTrainError = torch.zeros((opt.max_epoch + 1)*(provider.trainData.data:size(1)/opt.batchSize))
inEpochTestError = torch.zeros((opt.max_epoch + 1)*(provider.trainData.data:size(1)/opt.batchSize))

trainLoss = torch.zeros(opt.max_epoch + 1)
trainError = torch.zeros(opt.max_epoch + 1)
testError = torch.zeros(opt.max_epoch + 1)
times = torch.Tensor(opt.max_epoch + 1)

function train()
  model:training()
  iter = iter or 1
  
  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
    sesopConfig.optConfig.learningRate = sesopConfig.optConfig.learningRate/2 
  end
  
  --We pretend as we have opt.num_of_nodes nodes.
  print(c.red'Number of nodes = ' .. opt.num_of_nodes)
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = nil
  local tic = torch.tic()
  
  --every node chooses its own permutation
  indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil
  
  for t,v in ipairs(indices) do --epoch
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))
    --print(inputs:size())
    local feval = function(x, finputs, ftargets)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local _inputs = finputs or inputs --take inputs in case of sesop and x incase of baseMethod.
      local _targets = ftargets or targets
        
      local outputs = model:forward(_inputs)
      
      --SV DEBUG
      local f = criterion:forward(outputs, _targets:cuda())
      --print(f)
      --save the full train loss to evaluate SESOP
      --SV DEBUG
      local df_do = criterion:backward(outputs, _targets:cuda())
      model:backward(_inputs, df_do)
      
      return f,gradParameters
    end
      
    optim.seboost(feval, parameters, sesopConfig, optimState)
    
    --inEpochTrainError[iter] = messureError(provider.trainData)
    --inEpochTestError[iter] = messureError(provider.testData)
    torch.save(opt.save..'/inEpochTrainError.txt', inEpochTrainError)
    torch.save(opt.save..'/inEpochTestError.txt', inEpochTestError)
  
    --print('Train error:', (inEpochTrainError[iter]))
    --print('Test error:', (inEpochTestError[iter]))
    iter = iter + 1
  
  end
  
end

--return the Test/Train Error (not the loss unless its MSE!)
function messureError(data)
  model:evaluate()
  confusion:zero()
  --print('testing... model = ')
  --print(parameters)
  --print(':::::::::::::::::::::::::::')
  local regressionError = 0
  local bs = 125
  for i=1,data.data:size(1),bs do
    local outputs = model:forward(data.data:narrow(1,i,bs))
    --SV DEBUG
    if (opt.model ~= 'simple') then
      confusion:batchAdd(outputs, data.labels:narrow(1,i,bs))
    else
      currError = criterion:forward(outputs, data.labels:narrow(1,i,bs):cuda())
      regressionError = regressionError + currError
    end
  end
  
  if (opt.model == 'simple') then
    --print (regressionError/(data.data:size(1)/bs))
    return regressionError/(data.data:size(1)/bs)
  else
    confusion:updateValids()
    return 1 - confusion.totalValid
  end
  
end

function test()
  -- disable flips, dropouts and batch normalization
  print(c.blue '==>'.." testing")

  testError[epoch] = messureError(provider.testData)
  trainError[epoch] = messureError(provider.trainData)
  --trainLoss[epoch] =  trainLoss[epoch]/(#indices)
  
  print('Train accuracy:', (1 - trainError[epoch]) * 100)
  print('Test accuracy:', (1 - testError[epoch]) * 100)
    
  torch.save(opt.save..'/trainLoss.txt', trainLoss)
  torch.save(opt.save..'/trainError.txt', trainError)
  torch.save(opt.save..'/testError.txt', testError)
  
  
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, (1 - testError[epoch]) * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  confusion:zero()
end

local timer = torch.Timer()
epoch = 1
for i=1,opt.max_epoch do
  timer:reset()
  train()
  times[i] = timer:time().real
  torch.save(opt.save..'/epochTimes.txt', times)
  
  test()
  
  torch.save(opt.save..'/epoch.txt', epoch)
  epoch = epoch + 1
end


