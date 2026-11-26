"""PTO SIMT unary math and min/max on f16 vectors wider than two lanes.

PTO's packed unary math and floating min/max micro-ops stop at two lanes,
while the AscendC backend accepts widths 2/4/8. Codegen must split wider
f16 vectors (produced by T.Parallel in T.SimtVF or by T.vectorized) into
contiguous x2 pairs instead of aborting.
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _source(func, target):
    with tilelang.tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _exp_f16_kernel(num_elements, threads):
    @T.prim_func
    def main(
        a: T.Tensor((num_elements,), T.float16),
        out: T.Tensor((num_elements,), T.float16),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((num_elements,), T.float16)
            out_ub = T.alloc_shared((num_elements,), T.float16)
            T.copy(a, a_ub)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(num_elements):
                    out_ub[i] = T.exp(a_ub[i])
            T.copy(out_ub, out)

    return main


def _min_max_f16_kernel(num_elements, threads):
    @T.prim_func
    def main(
        a: T.Tensor((num_elements,), T.float16),
        b: T.Tensor((num_elements,), T.float16),
        out_min: T.Tensor((num_elements,), T.float16),
        out_max: T.Tensor((num_elements,), T.float16),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((num_elements,), T.float16)
            b_ub = T.alloc_shared((num_elements,), T.float16)
            out_min_ub = T.alloc_shared((num_elements,), T.float16)
            out_max_ub = T.alloc_shared((num_elements,), T.float16)
            T.copy(a, a_ub)
            T.copy(b, b_ub)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(num_elements):
                    out_min_ub[i] = T.min(a_ub[i], b_ub[i])
                    out_max_ub[i] = T.max(a_ub[i], b_ub[i])
            T.copy(out_min_ub, out_min)
            T.copy(out_max_ub, out_max)

    return main


def _pto_target(lanes):
    target = {"kind": "ascend", "keys": ["pto", "ascend"]}
    if lanes == 8:
        # The default 64-bit vector budget caps f16 vectors at 4 lanes; the
        # 8-lane shape needs a wider budget to materialize.
        target["max_vector_bits"] = 128
    return target


_WIDE_LANES = [
    pytest.param(4, id="f16x4"),
    pytest.param(8, id="f16x8"),
]


@pytest.mark.pto
@pytest.mark.parametrize("lanes", _WIDE_LANES)
def test_pto_wide_f16_unary_math_splits_into_x2_pairs(lanes):
    num_elements = 128
    kernel = _exp_f16_kernel(num_elements, threads=num_elements // lanes)
    source = _source(kernel, _pto_target(lanes))
    assert re.search(
        rf"tl\.vectorize_unary_f16\(pto\.exp, pto\.load\([^\n]*contiguous={lanes}\)",
        source,
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", _WIDE_LANES)
def test_pto_wide_f16_min_max_splits_into_x2_pairs(lanes):
    num_elements = 128
    kernel = _min_max_f16_kernel(num_elements, threads=num_elements // lanes)
    source = _source(kernel, _pto_target(lanes))
    assert re.search(
        rf"tl\.vectorize_binary_f16\(pto\.minimum, pto\.load\([^\n]*contiguous={lanes}\)",
        source,
    )
    assert re.search(
        rf"tl\.vectorize_binary_f16\(pto\.maximum, pto\.load\([^\n]*contiguous={lanes}\)",
        source,
    )


@pytest.mark.pto
def test_pto_pair_f16_unary_math_stays_packed():
    # Two lanes fit the packed micro-ops directly and must not be rerouted.
    source = _source(_exp_f16_kernel(128, threads=64), _pto_target(2))
    assert "tl.vectorize_unary_f16" not in source
    assert re.search(r"pto\.exp\(", source)


_NUMERICS_PARAMS = [
    pytest.param(4, {"kind": "ascend"}, id="f16x4-ascend"),
    pytest.param(4, {"kind": "ascend", "keys": ["pto", "ascend"]}, id="f16x4-pto", marks=pytest.mark.pto),
    pytest.param(8, {"kind": "ascend", "max_vector_bits": 128}, id="f16x8-ascend"),
    pytest.param(
        8,
        {"kind": "ascend", "keys": ["pto", "ascend"], "max_vector_bits": 128},
        id="f16x8-pto",
        marks=pytest.mark.pto,
    ),
]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("lanes,target", _NUMERICS_PARAMS)
def test_wide_f16_unary_math_matches_torch(lanes, target):
    import torch

    num_elements = 128
    kernel = tilelang.compile(
        _exp_f16_kernel(num_elements, threads=num_elements // lanes),
        target=target,
        out_idx=-1,
    )
    a = torch.linspace(-2, 2, num_elements, dtype=torch.float32, device="npu").to(torch.float16)
    result = kernel(a)
    torch.npu.synchronize()
    expected = torch.exp(a)
    torch.testing.assert_close(result.float().cpu(), expected.float().cpu(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("lanes,target", _NUMERICS_PARAMS)
def test_wide_f16_min_max_matches_torch(lanes, target):
    import torch

    num_elements = 128
    kernel = tilelang.compile(
        _min_max_f16_kernel(num_elements, threads=num_elements // lanes),
        target=target,
        out_idx=[-2, -1],
    )
    a = torch.arange(num_elements, dtype=torch.float16, device="npu") / 8
    b = torch.arange(num_elements - 1, -1, -1, dtype=torch.float16, device="npu") / 8
    result_min, result_max = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result_min.cpu(), torch.minimum(a, b).cpu(), rtol=0, atol=0)
    torch.testing.assert_close(result_max.cpu(), torch.maximum(a, b).cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
