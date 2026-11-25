from contextlib import ExitStack

import torch

from comfy_kitchen import list_backends
from comfy_kitchen.backends import cuda as kitchen_cuda
from comfy_kitchen.tensor.int8_utils import _build_hadamard

from comfy.ops import CastBiasWeightContext
from comfy.quant_ops import QuantizedTensor

from . import decode_kernels


GROUP_SIZE = 64


def available(device, dtype):
    if device.type != "cuda" or torch.version.cuda is None or dtype != torch.bfloat16:
        return False
    backend = list_backends().get("cuda", {})
    return (backend.get("available", False) and not backend.get("disabled", False)
            and {"int8_linear", "quantize_and_rotate_rowwise"}.issubset(backend.get("capabilities", ())))


def supports(linear):
    weight = linear.weight
    if (not isinstance(weight, QuantizedTensor) or getattr(linear, "quant_format", None) != "int8_tensorwise"
            or linear._full_precision_mm or getattr(linear, "comfy_force_cast_weights", False)
            or getattr(linear, "comfy_force_forward", False) or getattr(linear, "pre_quant_scale", None) is not None
            or linear.weight_function or linear.bias_function or getattr(linear, "weight_lowvram_function", None) is not None
            or getattr(linear, "bias_lowvram_function", None) is not None):
        return False
    params = weight._params
    return (params.convrot and params.convrot_groupsize == GROUP_SIZE and not params.tiled and not params.transposed
            and weight._qdata.is_contiguous() and params.scale.is_contiguous() and params.scale.numel() == linear.out_features)


def quantize_rows(x):
    rotation = _build_hadamard(GROUP_SIZE, device=x.device, dtype=x.dtype)
    return kitchen_cuda.quantize_and_rotate_rowwise(x.contiguous(), rotation, GROUP_SIZE)


def gemv(linear, x, codes, scale):
    # Use the same AIMDO residency and casting contract as Kitchen's regular Linear.
    with CastBiasWeightContext(linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True) as (weight, bias):
        return kitchen_cuda.int8_gemv_dequant(codes, weight._qdata, scale, weight._params.scale.reshape(-1), bias, x.dtype)


def grouped_gemv(projections, activations, codes, scales, shared_input, relu_squared=False):
    with ExitStack() as casts:
        weights = [casts.enter_context(CastBiasWeightContext(
            linear, activations, offloadable=True, compute_dtype=activations.dtype, want_requant=True,
        ))[0] for linear in projections]
        return decode_kernels.expert_gemv(codes, scales, [w._qdata for w in weights],
                                          [w._params.scale for w in weights], shared_input, relu_squared)


def routed_outputs(experts, x, q_input, grouped):
    if grouped:
        hidden = grouped_gemv([expert.up_proj for expert in experts], x, *q_input, shared_input=True, relu_squared=True)
        codes, scales = quantize_rows(hidden)
        return grouped_gemv([expert.down_proj for expert in experts], hidden, codes, scales, shared_input=False)
    hidden = torch.cat([gemv(expert.up_proj, x, *q_input) for expert in experts]).relu_().square_()
    codes, scales = quantize_rows(hidden)
    return torch.cat([gemv(expert.down_proj, hidden[index:index + 1], codes[index:index + 1], scales[index:index + 1])
                      for index, expert in enumerate(experts)])


def resident_outputs(routes, q_input, bank, hidden_size, expert_size):
    up, down = bank
    hidden = decode_kernels.expert_bank_gemv(*q_input, routes, up, expert_size, shared_input=True, relu_squared=True)
    return decode_kernels.expert_bank_gemv(*quantize_rows(hidden), routes, down, hidden_size)
