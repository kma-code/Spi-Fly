import torch
import math

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 15.0
        height = .15
        lens = 0.5
        gamma = .8  # gradient scale
        #temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
                - gaussian(input, mu=lens, sigma=scale * lens) * height \
                - gaussian(input, mu=-lens, sigma=scale * lens) * height
        #print("gamma in ActFun:backward", gamma)
        return grad_input * temp.float() * gamma

act_fun = ActFun.apply

### NEURON MODEL
def mem_update_adp(x, mem, thr, decay_neu):
    mem = decay_neu * mem + x       # leaky integration of x
    inputs_ = mem - thr             # input to spike act func
    spike = act_fun(inputs_)        # spike if mem > thr
    mem = mem * (1 - spike)         # reset spiking neurons to zero
    negative_ = (mem < 0).float()   # Check all the negative elements of mem and convert into floating number (1.0 or 0.0)
    mem = (mem * (1 - negative_)) - (0 * negative_)
    return mem, spike

### NEURON MODEL
def noisy_mem_update_adp(x, mem, thr, decay_neu, noise_lvl=0.0):
    mem = decay_neu * mem + x       # leaky integration of x
    # sample gaussian noise
    mem = mem + torch.normal(0, noise_lvl, x.size())
    inputs_ = mem - thr             # input to spike act func
    spike = act_fun(inputs_)        # spike if mem > thr
    mem = mem * (1 - spike)         # reset spiking neurons to zero
    negative_ = (mem < 0).float()   # Check all the negative elements of mem and convert into floating number (1.0 or 0.0)
    mem = (mem * (1 - negative_)) - (0 * negative_)
    return mem, spike