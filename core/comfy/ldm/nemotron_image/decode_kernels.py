"""Single-token INT8 expert GEMVs and Mamba state update on NVIDIA CUDA."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _select(slot, a, b, c, d, e, f):
        return tl.where(slot < 3,
                        tl.where(slot == 0, a, tl.where(slot == 1, b, c)),
                        tl.where(slot == 3, d, tl.where(slot == 4, e, f)))

    @triton.jit
    def _expert_gemv(X, XScale, W0, W1, W2, W3, W4, W5, S0, S1, S2, S3, S4, S5, Y,
                     N: tl.constexpr, K4: tl.constexpr, TILE_N: tl.constexpr, TILE_K4: tl.constexpr,
                     SAME_INPUT: tl.constexpr, RELU_SQUARED: tl.constexpr):
        slot = tl.program_id(1)
        weight = _select(slot, W0, W1, W2, W3, W4, W5)
        scale = _select(slot, S0, S1, S2, S3, S4, S5)
        _gemv_lanes(X, XScale, weight, scale, Y, slot, N, K4, TILE_N, TILE_K4, SAME_INPUT, RELU_SQUARED)

    @triton.jit
    def _expert_bank_gemv(X, XScale, Routes, Weights, Scales, Y,
                          N: tl.constexpr, K4: tl.constexpr, TILE_N: tl.constexpr, TILE_K4: tl.constexpr,
                          SAME_INPUT: tl.constexpr, RELU_SQUARED: tl.constexpr):
        slot = tl.program_id(1)
        expert = tl.load(Routes + slot)
        weight = tl.load(Weights + expert).to(tl.pointer_type(tl.int8))
        scale = tl.load(Scales + expert).to(tl.pointer_type(tl.float32))
        _gemv_lanes(X, XScale, weight, scale, Y, slot, N, K4, TILE_N, TILE_K4, SAME_INPUT, RELU_SQUARED)

    @triton.jit
    def _gemv_lanes(X, XScale, Weight, Scale, Y, slot, N: tl.constexpr, K4: tl.constexpr,
                    TILE_N: tl.constexpr, TILE_K4: tl.constexpr, SAME_INPUT: tl.constexpr, RELU_SQUARED: tl.constexpr):
        row = tl.program_id(0) * TILE_N + tl.arange(0, TILE_N)
        col = tl.arange(0, TILE_K4)
        weight = Weight.to(tl.pointer_type(tl.int32))
        input_slot = 0 if SAME_INPUT else slot
        x = tl.load(X.to(tl.pointer_type(tl.int32)) + input_slot * K4 + col, col < K4, other=0)
        w = tl.load(weight + row[:, None] * K4 + col[None, :],
                    (row[:, None] < N) & (col[None, :] < K4), other=0)
        products = tl.inline_asm_elementwise("dp4a.s32.s32 $0, $1, $2, 0;", "=r,r,r",
                                             [w, x[None, :] + tl.full((TILE_N, 1), 0, tl.int32)],
                                             dtype=tl.int32, is_pure=True, pack=1)
        accum = tl.sum(products, axis=1).to(tl.float32)
        result = accum * tl.load(XScale + input_slot) * tl.load(Scale + row, row < N, other=0)
        if RELU_SQUARED:
            # The unfused expert rounds the GEMV result to BF16 before its activation.
            positive = tl.maximum(result.to(tl.bfloat16).to(tl.float32), 0.0)
            result = positive * positive
        tl.store(Y + slot * N + row, result, row < N)

    @triton.jit
    def _mamba_step(X, B, C, Dt, A, D, State, Y, WIDTH: tl.constexpr, STATE: tl.constexpr,
                    GROUP_HEADS: tl.constexpr, ROWS: tl.constexpr):
        head = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, STATE)
        group = head // GROUP_HEADS
        x = tl.load(X + head * WIDTH + row).to(tl.float32)
        b = tl.load(B + group * STATE + col)
        c = tl.load(C + group * STATE + col)
        dt = tl.load(Dt + head)
        decay = tl.exp(dt * tl.load(A + head))
        address = State + (head * WIDTH + row[:, None]) * STATE + col[None, :]
        old = tl.load(address)
        state = old * decay + (x * dt)[:, None] * b[None, :]
        tl.store(address, state)
        y = tl.sum(state * c[None, :], axis=1) + x * tl.load(D + head)
        tl.store(Y + head * WIDTH + row, y)


def available():
    return triton is not None


def expert_gemv(codes, scales, weights, weight_scales, shared_input=False, relu_squared=False):
    out_features, in_features = weights[0].shape
    output = torch.empty((6, out_features), dtype=torch.bfloat16, device=codes.device)
    tile = 4
    _expert_gemv[(triton.cdiv(out_features, tile), 6)](
        codes, scales, *weights, *weight_scales, output,
        out_features, in_features // 4, tile, triton.next_power_of_2(in_features // 4),
        shared_input, relu_squared, num_warps=4,
    )
    return output


def expert_bank_gemv(codes, scales, routes, pointers, out_features, shared_input=False, relu_squared=False):
    output = torch.empty((6, out_features), dtype=torch.bfloat16, device=codes.device)
    tile = 4
    k4 = codes.shape[-1] // 4
    _expert_bank_gemv[(triton.cdiv(out_features, tile), 6)](
        codes, scales, routes, *pointers, output, out_features, k4, tile, triton.next_power_of_2(k4),
        shared_input, relu_squared, num_warps=4,
    )
    return output


def mamba_step(x, b, c, dt, a, d, state):
    heads, _, width = x.shape
    output = torch.empty((heads, 1, width), dtype=torch.float32, device=x.device)
    rows = 4
    _mamba_step[(heads, width // rows)](
        x, b, c, dt, a, d, state, output, width, state.shape[-1], heads // b.shape[0], rows,
        num_warps=4,
    )
    return output
