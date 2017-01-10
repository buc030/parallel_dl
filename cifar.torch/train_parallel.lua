require 'xlua'
require 'optim'
require 'nn'
require 'image'
dofile './provider.lua'
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
    if self.train then
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
provider = torch.load 'provider.t7'
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
criterion = cast(nn.CrossEntropyCriterion())

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
        isCuda=true,
        optConfig={
          learningRate = opt.learningRate,
          weightDecay = opt.weightDecay,
          momentum = opt.momentum,
          learningRateDecay = opt.learningRateDecay,
        }, --placeholder state for the inner optimization function.
        
        sesopBatchSize=sesop_batch_size,
        numNodes=opt.num_of_nodes,
        nodeIters=opt.merge_freq,
        histSize=opt.history_size
        merger=opt.merger
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


trainLoss = torch.Tensor(opt.max_epoch + 1)
trainError = torch.Tensor(opt.max_epoch + 1)
testError = torch.Tensor(opt.max_epoch + 1)

function train()
  model:training()
  epoch = epoch or 1
  iter = iter or 1
  trainLoss[epoch] = 0
  trainError[epoch] = 0
  
  
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
      local f = criterion:forward(outputs, _targets)
      
      --save the full train loss to evaluate SESOP
      iter = iter + 1
      
      trainLoss[epoch] = trainLoss[epoch] + f
      local df_do = criterion:backward(outputs, _targets)
      model:backward(_inputs, df_do)
        
      --we only add the non sesop batches
      if finputs == nil then
        confusion:batchAdd(outputs, _targets)
      end
      
      return f,gradParameters
    end
      
    optim.seboost(feval, parameters, sesopConfig, optimState)
  end
  
  confusion:updateValids()

  trainLoss[epoch] =  trainLoss[epoch]/(#indices)
  trainError[epoch] = 1 - confusion.totalValid
  torch.save(opt.save..'/trainLoss.txt', trainLoss)
  torch.save(opt.save..'/trainError.txt', trainError)
  torch.save(opt.save..'/epoch.txt', epoch)
  
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")

  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end
  
  confusion:updateValids()
  
  testError[epoch] = 1 - confusion.totalValid
  torch.save(opt.save..'/testError.txt', testError)
  
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
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

times = torch.Tensor(opt.max_epoch + 1)

for i=1,opt.max_epoch do
  timer:reset()
  train()
  times[i] = timer:time().real
  torch.save(opt.save..'/epochTimes.txt', times)
  
  test()
end


