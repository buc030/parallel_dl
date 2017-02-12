local ThresholdTanh, Parent = torch.class('nn.ThresholdTanh', 'nn.Module')

function ThresholdTanh:__init()
  Parent.__init(self)
end

function ThresholdTanh:updateOutput(input)
   input.THNN.Tanh_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
  
   return self.output
end

function ThresholdTanh:updateGradInput(input, gradOutput)
  --gradOutput * (1 - *output * *output);
  --self.gradInput:set(gradOutput)
  --self.gradInput:cmul(self.output)
  --print('gradOutput = ')
  --print(gradOutput:norm())

  
  self.gradInput:set(self.output) --self.output
  self.gradInput:cmul(self.output) --self.output**2
  
  local one = torch.ones(self.gradInput:size()):cuda()
  self.gradInput:add(-one) --self.output**2 - 1
  self.gradInput:cmul(-one) --1 - self.output**2
  self.gradInput:cmul(gradOutput) --gradOutput*(1 - self.output**2)
  
  local scale = 1e-2/gradOutput:norm()
  self.gradInput:cmul(scale*one)
  
   return self.gradInput
end
