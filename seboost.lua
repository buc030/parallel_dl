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
  if (state.itr % sesopUpdate) ~= 0 or histSize == 0 then -- Not a sesop iteration.
    return x,fx
  end
  
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
    
    local adfda = (dirMat:t()*adfdx)
    local norm_test = adfda:norm()*torch.ones(1)
    
    state.norms = torch.cat(state.norms, norm_test, 1)
    torch.save(config.save..'/cg_grad_norms.txt', state.norms)
    
    state.f_vals = torch.cat(state.f_vals, afx*torch.ones(1), 1)
    torch.save(config.save..'/cg_f_vals.txt', state.f_vals)
    
    
    return afx, adfda
  end
  config.maxIter=200
  
  local _, fHist = optim.cg(feval, state.aOpt, config, state) --Apply optimization using inner function

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

