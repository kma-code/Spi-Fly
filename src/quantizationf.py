from collections import namedtuple
import torch
import torch.nn as nn
import math

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale

  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def calcScaleZeroPointSym(min_val, max_val,num_bits=8):

    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    if min_val >= 0. or max_val <= 0.:
        # if all entries are same sign, we don't need a sign bit
        qmax = 2.**(num_bits) - 1.
    else:
        qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax

    return scale, 0

def quantize_tensor(x, num_bits=4, min_val=None, max_val=None):

    if not min_val and not max_val:
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    if torch.all(x >= 0.) or torch.all(x <= 0.):
        # if all entries are same sign, we don't need a sign bit
        qmax = 2.**(num_bits) - 1.
    else:
        qmax = 2.**(num_bits-1) - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

# num_bits=4 means 3 bits for positive and negative side respectively
def quantize_tensor_sym(x, num_bits=4, min_val=None, max_val=None):

    if (not min_val) and ((not max_val) or x.max() <= max_val * 0.9):
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    if torch.all(x >= 0.) or torch.all(x <= 0.):
        # if all entries are same sign, we don't need a sign bit
        qmax = 2.**(num_bits) - 1.
    else:
        qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax

    q_x = x/scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)

def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())

def exp_scaler(exp_bits, a, b, c, max_val):
    x_max = (math.log(max_val / a) - c) / b
    unit_x = x_max / (2. ** exp_bits - 1)
    x_ticks = torch.tensor([unit_x * item for item in range(1, int(2. ** exp_bits))])
    y_ticks = a * torch.exp(b * x_ticks + c)

    y_range = torch.zeros(len(y_ticks)) 

    # Ensure y_range has the correct size
    y_range[1:] = (y_ticks[:-1] + y_ticks[1:]) / 2  # Midpoints
    y_range[0] = y_ticks[0] / 2  # Lower bound
    return y_range, y_ticks

def exp_quantizer(x, num_bits, max_val, a, b, c):
    device = x.device
    if max_val is None:
        max_val = x.max()

    if torch.all(x >= 0.) or torch.all(x <= 0.):
        # if all entries are same sign, we don't need a sign bit
        exp_bits = num_bits
    else:
        exp_bits = num_bits-1

    y_range, y_ticks = exp_scaler(exp_bits, a, b, c, max_val)

    y_range = y_range.to(device)
    y_ticks = y_ticks.to(device)
    
    abs_x = torch.abs(x)
    sign = torch.sign(x)

    # Find bin index where abs_x falls in y_range
    idx = torch.searchsorted(y_range, abs_x) - 1  # Find the index in y_range
    idx = torch.clamp(idx, 0, len(y_ticks) - 1)  # Ensure valid index

    # Assign quantized values
    q_x = torch.where(abs_x < y_range[0], torch.tensor(0.0, device=x.device), y_ticks[idx])
    return q_x * sign  # Restore the sign to the output

def exp_quantizer_probabilistic(x, num_bits, max_val, a, b, c, base_temp=0.02, scale_factor=0.5):
    device = x.device
    y_range, y_ticks = exp_scaler(num_bits, a, b, c, max_val)

    y_range = y_range.to(device)
    y_ticks = y_ticks.to(device)

    abs_x = torch.abs(x)
    sign = torch.sign(x)

    # **Prepend 0 to y_ticks**
    y_ticks = torch.cat([torch.tensor([0.0], device=device), y_ticks])

    # **Normalize y_ticks by its max value**
    max_y_ticks = y_ticks.max()
    normalized_y_ticks = y_ticks / max_y_ticks  # Now in range [0, 1]

    # **Adaptive temperature based on normalized y_ticks**
    adaptive_temperature = base_temp * (scale_factor + normalized_y_ticks)

    # Compute distances between abs_x and each y_tick
    distances = torch.abs(abs_x.unsqueeze(-1) - y_ticks)  # Shape: [batch_size, num_ticks]

    # Convert distances to probabilities using the adaptive temperature
    probs = torch.exp(-distances / adaptive_temperature)  # Exponential decay
    probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize

    # Sample from the probability distribution
    idx = torch.multinomial(probs, 1).squeeze(-1)  # Get sampled indices

    # Assign quantized values
    q_x = y_ticks[idx]

    return q_x * sign  # Restore sign

class FakeQuantOp_exp_probabilistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base_temp, scale_factor, num_bits=3, min_val=None, max_val=None):
        x = exp_quantizer_probabilistic(x, num_bits, max_val, a=0.01, b=0.5, c=0, base_temp=base_temp, scale_factor=scale_factor)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

class FakeQuantOp_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=4, min_val=None, max_val=None):
        x = exp_quantizer(x, num_bits, max_val, a=0.01, b=0.5, c=0)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None
    
class FakeQuantOp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=3, min_val=None, max_val=None):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

class FakeQuantOp_linear_sym(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=3, min_val=None, max_val=None):
        x = quantize_tensor_sym(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor_sym(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None