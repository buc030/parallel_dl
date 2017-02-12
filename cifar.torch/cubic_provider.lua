require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init()
  local trsize = 5000
  local tesize = 1000
  
  local randmodel = nn.Sequential()
  randmodel:add(nn.View(4))
  
  for i = 1,1 do
  --  randmodel:add(nn.Linear(32*32*3, 32*32*3))
  --  randmodel:add(nn.Tanh())
  end
  randmodel:add(nn.Linear(4, 1))

  
  
  local trainData = torch.rand(trsize, 4)
  local trainLabels = torch.Tensor(trsize)
  trainLabels:copy(randmodel:forward(trainData))
  
  local testData = torch.rand(tesize, 4)
  local testLabels = torch.Tensor(tesize)
  testLabels:copy(randmodel:forward(testData))
  
  -- load dataset

  self.trainData = {
     data = trainData,
     --labels = randmodel:forward(trainData),
     labels=trainLabels,
     size = function() return trsize end
  }


  

  self.testData = {
       data = testData,
       --labels = randmodel:forward(testData),
       labels=testLabels,
       size = function() return tesize end
  }

  print('self.trainData.labels 1')
  print(self.trainData.data:size())
  
  print('self.testData.labels 2')
  print(self.testData.data:size())
  
  
  
  
  trainData = self.trainData
  testData = self.testData
  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  --trainData.data = trainData.data:reshape(trsize,3,32,32)
  --testData.data = testData.data:reshape(tesize,3,32,32)
end
