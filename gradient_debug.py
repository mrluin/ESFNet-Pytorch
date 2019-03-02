import torch


def get_printer(msg):
    """
    returns a printer function, that prints information about a tensor's gradient
    Used by register_hook in the backward pass.
    :param msg:
    :return: printer function
    """
    def printer(tensor):
        if tensor.nelement == 1:
            print("{} {}".format(msg, tensor))
        else:
            print("{} shape: {}"
                  "max: {} min: {}"
                  "mean: {}"
                  .format(msg, tensor.shape, tensor.max(), tensor.min(), tensor.mean()))
    return printer

def register_hook(tensor, msg):
    """
    Utility function to call retain_grad and register_hook in a single line
    :param tensor:
    :param msg:
    :return:
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))

if __name__ == '__main__':

    x = torch.randn((1,1), requires_grad=True)
    y = 3*x
    z = y**2
    register_hook(y, 'y')
    z.backward()

