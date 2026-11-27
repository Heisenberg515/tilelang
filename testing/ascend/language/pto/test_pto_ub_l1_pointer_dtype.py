"""PTO UB->L1 emitters must take the pointer element type from the buffer.

The frontends emit local UB/L1 pointers as ``tl.access_ptr(BufferLoad(...))``
before ``LowerAccessPtr`` rewrites them into annotated
``tirx.tvm_access_ptr(type_annotation(...), ...)`` calls, and the standard
pipeline always annotates them before codegen. The pointer-form contract is
pinned here by feeding the pre-lowering ``tl.access_ptr`` form (and, for the
error contract, an untyped ``tvm_access_ptr``) directly to the PTO device
codegen: no device execution is involved.

``tl.ascend_copy_ubuf_to_cbuf``, ``tl.ascend_nd2nz_post_copy``, and
``tl.ascend_nd2nz_scatter`` used to compute their local pointers with a
hardcoded bfloat16 fallback dtype (#4312), so a copy into a float32 L1 buffer
cast the pointer to bf16 and scaled the byte offset with 2-byte elements.
The element dtype must instead be recovered from the buffer the pointer
refers to, as the GM->L1 emitter does, and a pointer with no recoverable
dtype must fail loudly instead of compiling as bfloat16.
"""

import pytest
import torch  # noqa: F401  (torch/torch_npu must import before tilelang/ptodsl)
import torch_npu  # noqa: F401

from tilelang import tvm
from tilelang.ascend.target import normalize_pto_target

tirx = tvm.tirx
_Op = tvm.ir.Op.get
_BUILD = tvm.ffi.get_global_func("target.build.tilelang_pto_without_compile")
_INTERNAL_ERROR = tvm.error.InternalError

_SIZE = 256


def _source_of(body):
    target = normalize_pto_target("pto")
    func = tirx.PrimFunc([], body).with_attr("global_symbol", "main")
    with target:
        return _BUILD(tvm.IRModule({"main": func}), target).inspect_source()


def _ub(name):
    # MergeUBAllocations hands the codegen a shared.dyn arena whose base is a
    # zero byte address; mirror that scope so the UB source pointer prints.
    return tirx.decl_buffer((_SIZE,), "float32", name=name, scope="shared.dyn")


def _l1(name):
    return tirx.decl_buffer((_SIZE,), "float32", name=name, scope="shared.l1")


def _i32(value):
    return tirx.IntImm("int32", value)


def _tl_access_ptr(buffer, offset):
    """The frontend pointer form (copy.cc / insert_nd2nz.cc) pre-lowering."""
    return tirx.Call(
        "handle",
        _Op("tl.access_ptr"),
        [tirx.BufferLoad(buffer, [_i32(offset)]), _i32(_SIZE - offset), _i32(2)],
    )


def _alloc_body(buffers, copy):
    return tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in buffers], tirx.Evaluate(copy)])


def _lines(source, needle):
    return [line.strip() for line in source.splitlines() if needle in line]


def test_copy_ubuf_to_cbuf_pointer_element_type():
    """The L1 destination pointer must carry the buffer's element type and a
    float32-sized byte offset, not the old bfloat16 default."""
    ub = _ub("ub")
    l1 = _l1("l1")
    copy = tirx.Call(
        "handle",
        _Op("tl.ascend_copy_ubuf_to_cbuf"),
        [
            _tl_access_ptr(l1, 16),  # dst: L1, 16 float32 elements in
            _tl_access_ptr(ub, 0),   # src: UB arena base
            _i32(0),                 # sub_blockid
            _i32(4),                 # burst_num
            _i32(8),                 # burst_len (float32 elements)
            _i32(0),                 # src_gap
            _i32(0),                 # dst_gap
        ],
    )
    source = _source_of(_alloc_body([l1, ub], copy))
    line = _lines(source, "pto.mte_ub_l1(")
    assert len(line) == 1, source
    assert 'pto.castptr(pto.const(64, dtype=pto.int64), pto.ptr(pto.f32, "mat"))' in line[0], line[0]
    # The UB source side keeps its own buffer dtype as well.
    assert 'pto.ptr(pto.f32, "ub")' in line[0], line[0]


def test_copy_ubuf_to_cbuf_untyped_pointer_fails_loudly():
    """A tvm_access_ptr without a recoverable element dtype must be rejected
    instead of silently compiling as bfloat16."""
    ub = _ub("ub")
    l1 = _l1("l1")
    untyped = tirx.tvm_access_ptr(_i32(0), l1.data, _i32(16), _i32(8), _i32(2))
    copy = tirx.Call(
        "handle",
        _Op("tl.ascend_copy_ubuf_to_cbuf"),
        [untyped, _tl_access_ptr(ub, 0), _i32(0), _i32(4), _i32(8), _i32(0), _i32(0)],
    )
    body = _alloc_body([l1, ub], copy)
    with pytest.raises(_INTERNAL_ERROR) as excinfo:
        _source_of(body)
    assert "element dtype annotation" in str(excinfo.value)


def test_nd2nz_post_copy_pointer_element_type():
    ub = _ub("ub")
    l1 = _l1("l1")
    post_copy = tirx.Call(
        "handle",
        _Op("tl.ascend_nd2nz_post_copy"),
        [
            _tl_access_ptr(l1, 8),
            _tl_access_ptr(ub, 16),
            _i32(16),  # rows
            _i32(32),  # cols
            _i32(32),  # src_pitch
            _i32(32),  # dst_pitch
            tirx.StringImm("float"),
        ],
    )
    source = _source_of(_alloc_body([l1, ub], post_copy))
    line = _lines(source, "pto.mte_ub_l1(")
    assert len(line) == 1, source
    assert 'pto.castptr(pto.const(64, dtype=pto.int64), pto.ptr(pto.f32, "ub"))' in line[0], line[0]
    assert 'pto.castptr(pto.const(32, dtype=pto.int64), pto.ptr(pto.f32, "mat"))' in line[0], line[0]


def test_nd2nz_scatter_pointer_element_type():
    ub = _ub("ub")
    nz = _ub("nz")
    scatter = tirx.Call(
        "handle",
        _Op("tl.ascend_nd2nz_scatter"),
        [
            _tl_access_ptr(ub, 8),
            _tl_access_ptr(nz, 0),
            _i32(16),  # rows
            _i32(32),  # cols
            _i32(16),  # dst_pitch
            tirx.StringImm("float"),
            tirx.StringImm("float"),
        ],
    )
    source = _source_of(_alloc_body([ub, nz], scatter))
    vlds = _lines(source, "pto.vlds(")
    vsstb = _lines(source, "pto.vsstb(")
    assert len(vlds) == 16 and len(vsstb) == 16, source
    # Every scatter endpoint keeps the float32 pointer element type; with the
    # old bfloat16 fallback every castptr would name pto.bf16 instead.
    assert 'pto.addptr(pto.castptr(pto.const(32, dtype=pto.int64), pto.ptr(pto.f32, "ub")), 0)' in vlds[0], vlds[0]
    assert 'pto.addptr(pto.castptr(pto.const(0, dtype=pto.int64), pto.ptr(pto.f32, "ub")), 0)' in vsstb[0], vsstb[0]
    assert "pto.ptr(pto.bf16" not in source, source


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
