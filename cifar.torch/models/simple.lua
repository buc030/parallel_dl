require 'nn'
require 'ThresholdTanh'

local vgg = nn.Sequential()
vgg:add(nn.View(4))
--vgg:add(nn.Tanh())
--vgg:add(nn.MulConstant(0.01)) 
for i = 1,opt.layers do
  --the inputs of the 16th layer should be BIGEST.
    --vgg:add(nn.MulConstant(math.pow(3, i)))
    --
    vgg:add(nn.Linear(4, 4))
    
    --vgg:add(nn.BatchNormalization(4, 1e-3, 0.1))
    --vgg:add(nn.ReLU(true))
    --vgg:add(nn.ThresholdTanh())
    vgg:add(nn.Tanh())
    
    --vgg:add(nn.Dropout(0.5))
    
end
--vgg:add(nn.Reshape(2,2))
--vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
--vgg:add(nn.View(1))
--vgg:add(nn.MulConstant(100)) 
vgg:add(nn.Linear(4, 1))
  

--vgg:add(nn.Tanh())
--vgg:add(nn.Linear(1, 1))
  
  --vgg:add(nn.BatchNormalization(10 ,1e-3))
--vgg:add(nn.View(32*32*3))
--vgg:add(nn.Linear(32*32*3, 32*32*3))
--vgg:add(nn.BatchNormalization(10 ,1e-3))



--vgg:add(nn.Linear(32*32*3, 1))



--vgg:add(nn.ReLU(true))

--vgg:add(nn.Dropout(0.5))





return vgg

