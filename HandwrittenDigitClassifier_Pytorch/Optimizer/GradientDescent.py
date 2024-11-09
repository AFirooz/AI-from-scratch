import torch

class Optimizer:
    def __init__(self, model_parms):
        self.model_parms = model_parms

    def zero_grad(self):
        for param in self.model_parms:
            param.grad = torch.zeros_like(param)
        return self.model_parms


class GD (Optimizer):
    """ Assumptions:
        1. `model_parms` will hold the parameters of the model.
        2. Each parameter, is an object with two attributes: `value` and `grad`.
        3. `value` is the current parameter value.
        4. `grad` is the partial derivative of the loss function with respect to the parameter,
           evaluated at the current parameter value using the current batch of data.

        Note that our (previously) coded ReverseAutoDiff (and Pytorch) provide such implementation for the parameters.
    """

    def __init__(self, model_parms, learning_rate):
        self.super()
        self.model_parms = model_parms
        self.learning_rate = learning_rate

    def step(self, model, ):
        """ Update weights & biases of the model
        """
        
        for param in model_parms:
            param_grad = param.detach().numpy()
            param = param.detach().numpy()

            param -= param_grad * self.learning_rate
            
            param = torch.from_numpy(param)
            param.requires_grad = True

        return model_parms