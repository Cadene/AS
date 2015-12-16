require 'nn'
require 'nngraph'

function graphToModule(gModule, NewModule, Parent)
    
   function NewClass:__init()
      Parent.__init(self)
   end
   
   function NewModule:updateOutput(input)
      return gModule:updateOutput(input)
   end
   
   function NewModule:updateGradInput(input, gradOutput)
      return gModule:updateGradInput(input, gradOutput)
   end
   
   function NewModule:accGradParameters(input, gradOutput)
      return gModule:accGradParameters(input, gradOutput)
   end
   
   function NewModule:reset()
      return gModule:reset()
   end

   return NewModule
end
