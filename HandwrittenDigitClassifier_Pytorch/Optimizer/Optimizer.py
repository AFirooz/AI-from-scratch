import torch


class Base:
    def zero_grad(self, model):
        for layer in model:
            if any(p.requires_grad for p in layer.parameters()):
                layer.weight.grad = torch.zeros_like(layer.weight)
                layer.bias.grad = torch.zeros_like(layer.bias)
        return model


class GD(Base):
    """Assumptions:
    1. `model_parms` will hold the parameters of the model.
    2. Each parameter, is an object with two attributes: `value` and `grad`.
    3. `value` is the current parameter value.
    4. `grad` is the partial derivative of the loss function with respect to the parameter,
       evaluated at the current parameter value using the current batch of data.

    Note that our (previously) coded ReverseAutoDiff (and Pytorch) provide such implementation for the parameters.
    """

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, model):
        """Update weights & biases of the model"""
        for layer in model:
            if any(p.requires_grad for p in layer.parameters()):
                # grab all the weights and biases and their gradients
                weight = layer.weight.detach().numpy()
                weight_grad = layer.weight.grad.detach().numpy()

                bias = layer.bias.detach().numpy()
                bias_grad = layer.bias.grad.detach().numpy()

                # update the parameters
                weight -= weight_grad * self.learning_rate
                bias -= bias_grad * self.learning_rate

                # create new weight & bias torch object
                weight = torch.from_numpy(weight)
                # weight.requires_grad = True
                # weight.grad = torch.zeros_like(weight)

                bias = torch.from_numpy(bias)
                # bias.requires_grad = True
                # bias.grad = torch.zeros_like(bias)

                # update the layer to the new parameters
                layer.weight = torch.nn.Parameter(weight, requires_grad=True)
                layer.bias = torch.nn.Parameter(bias, requires_grad=True)
        return model
