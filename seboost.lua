--[[ A implementation of seboost

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX.
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.optMethod`         : The base optimizaion method
- `config.momentum`          : weight for SEBOOST's momentum direction
- `config.histSize`          : number of previous directions to keep in memory
- `config.anchorPoints`      : A tensor of values, each describing the number of             iterations between an update of an anchor point
- `config.sesopUpdate`       : The number of regular optimization steps between each boosting step
- `config.sesopData`         : The training data to use for the boosting stage
- `config.sesopLabels`       : The labels to use for the boosting stage
- `config.sesopBatchSize`    : The number of samples to use for each optimization step
- `config.isCuda`            : Whether to train using cuda or cpu
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.sesopLastX`        : The last point from which the boosting was ran
- `state.itr`               : The current optimization iteration
- `state.dirs`              : The set of directions to optimize in
- `state.anchors`           : The current anchor points
- `state.aOpt`              : The current set of optimal coefficients
- `state.dirIdx`            : The next direction to override

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

require 'optim'
require 'hf'


function dumpAlphas(opfunc, xInit, state, config, sesopInputs, sesopTargets)
  --[[
  --SV DEBUG Dump h(alpha) for the subspace
  local h = function(a, j)
    --A function of the coefficients
    local dirMat = state.dirs[{ {}, j }] -- only the j-th vector! (just for plot)
    --Note that opfunc also gets the batch
    
    local afx, adfdx = opfunc(xInit+dirMat*a, sesopInputs, sesopTargets)
    
    --local adfda = (dirMat:t()*adfdx)
    return afx, nil
  end
  
  
  local anchorsSize = 0
  if config.anchorPoints then --If anchors exist
    anchorsSize = config.anchorPoints:size(1)
  end
  
  if state.h_of_alphas == nil then
    state.h_of_alphas = {}
    state.alphas = {}
  end
  for j = 1, config.histSize + anchorsSize do
    local samples = 2000
    local low_x = -10
    local high_x = 10
    local tmp_h_of_alphas = torch.zeros(1, samples)
    local tmp_alphas = torch.zeros(1, samples)
    for i = 1,samples do 
      local samplePoint = low_x + (i - 1)*((high_x - low_x)/samples)
      tmp_alphas[1][i] = samplePoint
      --print(samplePoint)
      tmp_h_of_alphas[1][i] = h(samplePoint, j)
    end
    
    if state.h_of_alphas[j] == nil then
      state.h_of_alphas[j] = tmp_h_of_alphas 
      state.alphas[j] = tmp_alphas
    else
      state.h_of_alphas[j] = torch.cat(state.h_of_alphas[j], tmp_h_of_alphas, 1)
      state.alphas[j] = torch.cat(state.alphas[j], tmp_alphas, 1)
      
    end
  
    torch.save(config.save..'/h_of_alphas'..j..'.txt', state.h_of_alphas[j])
    torch.save(config.save..'/alphas'..j..'.txt', state.alphas[j])
  end
  ]]
end

--create the ith minibatch
function createMinibatch(sesopIteration, sesopBatchSize, i)
  -- create mini batch
  local subT = (sesopIteration - 1) * sesopBatchSize + 1
  subT = subT % (sesopData:size(1) - sesopBatchSize) --Calculate the next batch index
  local sesopInputs = sesopData:narrow(1, subT, sesopBatchSize)
  local sesopTargets = sesopLabels:narrow(1, subT, sesopBatchSize)
  if isCuda then
    --sesopInputs = sesopInputs:cuda()
    --sesopTargets = sesopTargets:cuda()
  end
  return sesopInputs, sesopTargets
end


function optim.seboost(opfunc, x, config, state)

  -- get/update state
  local config = config or {}
  local state = state or config
  local momentum = config.momentum or 0.9
  local histSize = config.histSize
  local anchorPoints = config.anchorPoints or nil
  local isCuda = config.isCuda or false
  local sesopUpdate = config.sesopUpdate or 100
  local sesopData = config.sesopData
  local sesopLabels = config.sesopLabels
  local sesopBatchSize = config.sesopBatchSize or 100
  local eps = 1e-5 --Minimal norm of a direction
  state.sesopLastX = state.sesopLastX or x:clone() --Never forget to clone!
  state.itr = state.itr or 0
  state.itr = state.itr + 1
  
  local timer = torch.Timer()

  x,fx = config.optMethod(opfunc, x, config.optConfig) -- Apply regular optimization method, changing the model directly
  --[[
  _,dfx = opfunc(x)
  
  state.sum_dfx_norm = state.sum_dfx_norm or 0
  state.sum_dfx_norm = state.sum_dfx_norm + dfx:norm()
  state.count_dfx_norm = state.count_dfx_norm or 0
  state.count_dfx_norm = state.count_dfx_norm + 1
  ]]
  
  --state.prev_dfx_norm = state.prev_dfx_norm or 0.00001 --very small!
  --state.curr_dfx_norm = dfx:norm()
  if (state.itr % sesopUpdate) ~= 0 or histSize == 0 then -- Not a sesop iteration.
  --if state.curr_dfx_norm/state.prev_dfx_norm > 0.05 or histSize == 0 then  --Do sesop iteration if we are near minima!
  --if (state.sum_dfx_norm/state.count_dfx_norm > 0.05 or state.count_dfx_norm < 10) or histSize == 0 then 
    --state.prev_dfx_norm = state.curr_dfx_norm
    --print ('SEBOOST')
    --print(fx)
    return x,fx
  end
  --state.prev_dfx_norm = 0.0000001
  --state.sum_dfx_norm = 0
  --state.count_dfx_norm = 0
  --print('SESOP in epoch '..epoch..' iter '..state.itr)
  ------------------------- SESOP Part ----------------------------

  --Set some initial values
  local lastDirLocation = histSize  -- The last location of a direction that is not the momentum.

  --Set size of history to include anchors and momentum

  local anchorsSize = 0
  if anchorPoints then --If anchors exist
    anchorsSize = anchorPoints:size(1)
  end

  local momentumIdx = 0
  if momentum > 0 then --If momentum is used
    histSize = histSize + 1  -- To include momentum vector
    momentumIdx = histSize
  end

  local sesopIteration = state.itr / sesopUpdate --Calculate the current
  local newDir = x - state.sesopLastX -- Current Direction

  state.dirs = state.dirs or torch.zeros(x:size(1), histSize+anchorsSize)
  state.anchors = state.anchors or torch.zeros(x:size(1), anchorsSize)
  state.aOpt = torch.zeros(histSize+anchorsSize)
  state.norms = state.norms or torch.zeros(1)
  state.f_vals = state.f_vals or torch.zeros(1)
  state.starting_norms = state.starting_norms or torch.zeros(1)
  state.sesop_indeces = state.sesop_indeces or torch.zeros(1)
  
  if (isCuda) then
    state.dirs = state.dirs:cuda()
    state.anchors = state.anchors:cuda()
    state.aOpt = state.aOpt:cuda()
  end

  --Update anchor points
  for i = 1, anchorsSize do
    if sesopIteration % anchorPoints[i] == 1 then
      state.anchors[{ {}, i }] = x:clone() --Set new anchor
    end
    state.dirs[{ {}, histSize + i }] = x - state.anchors[{ {},i }]
    if (state.dirs[{ {}, histSize + i }]:norm() > eps) then
      --Normalize directions
      state.dirs[{ {}, histSize + i }] = state.dirs[{ {}, histSize + i }] / state.dirs[{ {}, histSize + i }]:norm()
    end
  end

  state.dirIdx = state.dirIdx or 1
  if (newDir:norm() > eps) then
    --Override direction only if not small
    state.dirs[{ {},state.dirIdx }]:copy(newDir)
  else
    print('New gradient is too small!')
     --Keep using old directions
  end

  local xInit = x:clone() --Save the starting point

  -- create mini batch
  local subT = (sesopIteration - 1) * sesopBatchSize + 1
  subT = subT % (sesopData:size(1) - sesopBatchSize) --Calculate the next batch index
  local sesopInputs = sesopData:narrow(1, subT, sesopBatchSize)
  local sesopTargets = sesopLabels:narrow(1, subT, sesopBatchSize)
  if isCuda then
    --sesopInputs = sesopInputs:cuda()
    --sesopTargets = sesopTargets:cuda()
  end

  -- Create inner opfunc for finding a*
  local feval = function(a)
    --A function of the coefficients
    local dirMat = state.dirs
    --Note that opfunc also gets the batch
    
    local afx, adfdx = opfunc(xInit+dirMat*a, sesopInputs, sesopTargets)
    

    --local afx, adfdx = opfunc(xInit+dirMat*a)
    local adfda = (dirMat:t()*adfdx)
    
    --[[
    --calc gradient with finite diffs
    local finite = torch.zeros(adfda:size()):cuda()
    
    for i=1,a:size(1) do
      local e = torch.zeros(a:size()):cuda()
      
      --local eps_machine = 2e-6
      --local epsilon = torch.pow(eps_machine, 1/3)*torch.max(a)
      local epsilon = 1e-6
      e[i] = epsilon
      local afx2,_ = opfunc(xInit+dirMat*(a + e), sesopInputs, sesopTargets)
      local afx3,_ = opfunc(xInit+dirMat*(a - e), sesopInputs, sesopTargets)
      
      finite[i] = (afx2 - afx3)/(2*epsilon)
    end
    print('finite diffrence:')
    print(finite)
    print('backward:')
    print(adfda)
    ]]
    
    return afx, adfda
  end
  
  
  --randomize a fixed p for the dropout~
  --call the model forward one time in order to set a new dropout noise 
  state.starting_norms = torch.cat(state.starting_norms, config.model:forward(sesopInputs):norm()*torch.ones(1), 1)
  torch.save(config.save..'/starting_norms.txt', state.starting_norms)
  old_p = {}
  for k,v in pairs(config.model:findModules('nn.Dropout')) do
    v.fixNoise = true
  end
  
  
  --SV DEBUG: plot h(alpha) to see ho it looks like when changing depth.
  --here alpha is of dim one.
  --SV DEBUG Dump h(alpha) for the subspace
  dumpAlphas(opfunc, xInit, state, config, sesopInputs, sesopTargets)
  
  
  
  config.maxIter=200

  lbfgsConfig = {
    lineSearch = optim.lswolfe,
    maxIter=200,
    nCorrection=state.aOpt:size(1),
    tolFun=1e-6,
    tolX=1e-10,
    lineSearchOptions = {
      c2 = 0.01,
      c1 = 0.01/2
    }
  }
  
--config.max_iter
--config.eps


  hfConfig = {
    max_iter=5,
    eps=2e-6
  }
  
  state.lbfgsState = state.lbfgsState or {}
  
  --do 10 steps with big step size to escape from flat areas!
  --[[
  local oldLr = config.optConfig.learningRate
  sgdState = {
    learningRate = 0.1
  }
  for i=1,10 do
    cg_alphas, fHist = optim.adadelta(feval, state.aOpt, sgdState)
  end
  config.optConfig.learningRate = oldLr
  ]]
  --local cg_alphas, fHist = optim.lbfgs(feval, state.aOpt, lbfgsConfig, state.lbfgsState) --Apply optimization using inner function
  --local cg_alphas, fHist,__,dfHist = optim.cg(feval, state.aOpt, config, state) --Apply optimization using inner function
  --for i=1,100 do
  --local  cg_alphas, fHist,__,dfHist = optim.rprop(feval, state.aOpt, config, state.lbfgsState) --Apply optimization using inner function
  --end
  local cg_alphas, fHist,__,dfHist = optim.hf(feval, state.aOpt, hfConfig, state) --Apply optimization using inner function
  --print(fHist)
  for k,v in pairs(config.model:findModules('nn.Dropout')) do
    v.fixNoise = false
  end
  
  --print(state.aOpt)
  state.sesop_indeces = torch.cat(state.sesop_indeces, #fHist*torch.ones(1), 1)
  torch.save(config.save..'/sesop_indeces.txt', state.sesop_indeces)
  
  
  --dump f values during cg.
  for i = 1, #fHist do
    state.f_vals = torch.cat(state.f_vals, fHist[i]*torch.ones(1), 1)
  end
  torch.save(config.save..'/cg_f_vals.txt', state.f_vals)
  
  --[[
  --dump f gradient norms during cg.
  for i = 1, #dfHist do
    state.norms = torch.cat(state.norms, dfHist[i]:norm()*torch.ones(1), 1)
  end
  torch.save(config.save..'/cg_grad_norms.txt', state.norms)
  ]]

  --print(fHist)
  --print(state.aOpt)

  --Apply a step in the direction
  x:copy(xInit)
  local sesopDir = state.dirs*state.aOpt
  x:add(sesopDir)

  --Add direction to history
  state.dirs[{ {}, state.dirIdx }]:add(sesopDir) --Save newDir+sesopDir in the subspace

  -- Update Momentum
  if momentum > 0 then
      state.dirs[{ {},momentumIdx }] = state.dirs[{ {},momentumIdx }]:mul(momentum) + state.dirs[{ {}, state.dirIdx }]
  end

  state.dirIdx = (state.dirIdx % lastDirLocation) + 1 --Update next direction to override

  state.sesopLastX:copy(x) --Update the last point
  --print('sesop Time ' .. timer:time().real)

  return x,fHist
end

return optim

