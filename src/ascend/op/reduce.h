#ifndef TVM_TL_ASCEND_OP_REDUCE_H_
#define TVM_TL_ASCEND_OP_REDUCE_H_

#include "support/check.h"

#include <tvm/runtime/data_type.h>

namespace tvm {
namespace tl {
namespace ascend {

inline void CheckAllReduceWidth(int reducing_threads, int scale,
                                const char *op_name) {
  ICHECK_GT(reducing_threads, 0)
      << op_name << ": AllReduce threads must be positive, got "
      << reducing_threads;
  ICHECK_GT(scale, 0) << op_name << ": AllReduce scale must be positive, got "
                      << scale;
  ICHECK_EQ(reducing_threads % scale, 0)
      << op_name << ": AllReduce threads (" << reducing_threads
      << ") must be divisible by scale (" << scale << ")";
}

/*!
 * \brief Whether the warp-internal register path of tl::AscendAllReduce can
 * reduce values of this dtype without a scratch buffer.
 *
 * The register path (asc_shfl_xor butterfly / asc_reduce_* hardware reduce)
 * is only specialized for these dtypes; every other dtype falls through to
 * the ub_reduce fallback, which writes through the __ubuf__ scratch buffer.
 */
inline bool AllReduceWarpPathScratchFree(const DataType &dtype) {
  return dtype == DataType::Float(32) || dtype == DataType::Int(32) ||
         dtype == DataType::UInt(32) || dtype == DataType::Int(64) ||
         dtype == DataType::UInt(64);
}

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif
