"""T.ascend_* sync primitives are rejected inside T.SimdVF() at trace time.

The underlying C primitives (asc_lock/asc_unlock/asc_sync*/asc_threadfence)
are plain __aicore__ functions, so a simd_vf function cannot call them.
Without this guard the kernel only dies in bisheng with an unrelated
"no matching function for call to 'asc_lock'" error that never names the
Python API the user called (#4395).
"""

import pytest
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language.frame import inside_simdvf

N = 256


def _trace_get_buf_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_get_buf("PIPE_M", 0)


def _trace_rls_buf_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_rls_buf("PIPE_M", 0)


def _trace_pipe_barrier_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_pipe_barrier("PIPE_ALL")


def _trace_set_flag_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_set_flag("MTE2_MTE1", 0)


def _trace_wait_flag_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_wait_flag("MTE2_MTE1", 0)


def _trace_threadfence_rejected_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_threadfence()


@pytest.mark.parametrize(
    "trace,message",
    [
        (_trace_get_buf_rejected_inside_simdvf, "T.ascend_get_buf cannot be used inside T.SimdVF"),
        (_trace_rls_buf_rejected_inside_simdvf, "T.ascend_rls_buf cannot be used inside T.SimdVF"),
        (_trace_pipe_barrier_rejected_inside_simdvf, "T.ascend_pipe_barrier cannot be used inside T.SimdVF"),
        (_trace_set_flag_rejected_inside_simdvf, "T.ascend_set_flag cannot be used inside T.SimdVF"),
        (_trace_wait_flag_rejected_inside_simdvf, "T.ascend_wait_flag cannot be used inside T.SimdVF"),
        (_trace_threadfence_rejected_inside_simdvf, "T.ascend_threadfence cannot be used inside T.SimdVF"),
    ],
    ids=["get_buf", "rls_buf", "pipe_barrier", "set_flag", "wait_flag", "threadfence"],
)
def test_sync_primitives_are_rejected_inside_simdvf(trace, message):
    with pytest.raises(ValueError, match=message):
        trace()
    # The SimdVF frame must be unwound even when tracing raised.
    assert not inside_simdvf()


def test_get_buf_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_get_buf("PIPE_M", 0)


def test_rls_buf_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_rls_buf("PIPE_M", 0)


def test_pipe_barrier_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_pipe_barrier("PIPE_ALL")


def test_set_flag_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_set_flag("MTE2_MTE1", 0)


def test_wait_flag_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_wait_flag("MTE2_MTE1", 0)


def test_threadfence_traces_fine_outside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            T.ascend_threadfence()


def test_cross_core_set_flag_still_traces_inside_simdvf():
    # Cross-core sync is refused later, by the schedule-unit validator with
    # its own message; the new placement guard must not fire for it.

    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 0)

    assert not inside_simdvf()


def test_sync_inter_arrive_still_traces_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_sync_inter_arrive("PIPE_MTE3", 0)

    assert not inside_simdvf()


def test_sync_inter_wait_still_traces_inside_simdvf():
    @T.prim_func
    def main(X: T.Tensor((N,), "float32")):  # pylint: disable=unused-argument
        with T.Kernel(1):
            with T.SimdVF():
                T.ascend_sync_inter_wait("PIPE_MTE3", 0)

    assert not inside_simdvf()
