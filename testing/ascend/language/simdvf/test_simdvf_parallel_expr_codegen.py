"""Regression tests for T.Parallel expression lowering in Ascend SIMD_VF blocks.

Covers issue #4209: Max/Min/Cast inside SimdVF T.Parallel bodies used to hit
LOG(FATAL) (process abort) in AscendSimdVFLowerParallel while the same bodies
lowered fine through other paths. Select must be rejected with a Python
ValueError instead of aborting (the PTO path rejects it too).
"""

import pytest

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _source(func):
    with tilelang.tvm.target.Target("ascend"), tilelang.transform.PassContext(
        opt_level=3
    ):
        return tilelang.lower(func, target="ascend").kernel_source


def _parallel_max_min_kernel():
    @T.prim_func
    def main(
        a: T.Tensor((64,), T.float32),
        b: T.Tensor((64,), T.float32),
        out_max: T.Tensor((64,), T.float32),
        out_min: T.Tensor((64,), T.float32),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), T.float32)
            b_ub = T.alloc_shared((64,), T.float32)
            out_max_ub = T.alloc_shared((64,), T.float32)
            out_min_ub = T.alloc_shared((64,), T.float32)
            T.copy(a, a_ub)
            T.copy(b, b_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    out_max_ub[i] = T.max(a_ub[i], b_ub[i])
                    out_min_ub[i] = T.min(a_ub[i], b_ub[i])
            T.copy(out_max_ub, out_max)
            T.copy(out_min_ub, out_min)

    return main


def _parallel_inner_cast_kernel():
    # f32 body whose stored value passes through an f16 cast: the frontend
    # wraps it as Cast(f32, Cast(f16, load)), the issue's original shape.
    @T.prim_func
    def main(a: T.Tensor((64,), T.float32), b: T.Tensor((64,), T.float32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), T.float32)
            b_ub = T.alloc_shared((64,), T.float32)
            T.copy(a, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    b_ub[i] = T.cast(T.cast(a_ub[i], T.float16), T.float32)
            T.copy(b_ub, b)

    return main


def _parallel_narrowing_cast_kernel():
    # f32 source stored into an f16 buffer: the cast is real (f32 -> f16) and
    # the load must be emitted at the source dtype, then converted with vcvt.
    # 128 elements: an f16 vector body requires the VReg extent (128) to be
    # divisible by the f16 VReg size.
    @T.prim_func
    def main(a: T.Tensor((128,), T.float32), b: T.Tensor((128,), T.float16)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128,), T.float32)
            b_ub = T.alloc_shared((128,), T.float16)
            T.copy(a, a_ub)
            with T.SimdVF():
                for i in T.Parallel(128):
                    b_ub[i] = T.cast(a_ub[i], T.float16)
            T.copy(b_ub, b)

    return main


def _parallel_select_kernel():
    @T.prim_func
    def main(a: T.Tensor((64,), T.float32), b: T.Tensor((64,), T.float32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), T.float32)
            b_ub = T.alloc_shared((64,), T.float32)
            T.copy(a, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    b_ub[i] = T.Select(a_ub[i] > 0.0, 1.0, 0.0)
            T.copy(b_ub, b)

    return main


def test_simdvf_parallel_max_min():
    source = _source(_parallel_max_min_kernel())

    assert "simd_inst::vmax(" in source
    assert "simd_inst::vmin(" in source


def test_simdvf_parallel_inner_cast_chain():
    source = _source(_parallel_inner_cast_kernel())

    # Cast(f32, Cast(f16, load)) lowers to vlds<float> + vcvt<half> +
    # vcvt<float> (PART_EVEN round trips preserve values).
    assert "simd_inst::vlds_norm<float>" in source
    assert "simd_inst::vcvt<half>(" in source
    assert "simd_inst::vcvt<float>(" in source


def test_simdvf_parallel_narrowing_cast():
    source = _source(_parallel_narrowing_cast_kernel())

    # The f32 buffer is loaded at its own dtype and converted once.
    assert "simd_inst::vlds_norm<float>" in source
    assert "simd_inst::vcvt<half>(" in source
    assert "simd_inst::vsts_norm" in source


def test_simdvf_parallel_select_raises_value_error():
    # Select is not lowerable on this path (nor on the PTO path); it must
    # raise a catchable ValueError instead of LOG(FATAL)-aborting.
    with pytest.raises(ValueError, match=r"unsupported expr.*Select"):
        _source(_parallel_select_kernel())
