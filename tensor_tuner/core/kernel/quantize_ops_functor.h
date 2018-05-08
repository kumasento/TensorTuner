/*!
 * \brief Functors for quantization operations.
 * \author Ruizhe Zhao <vincentzhaorz@gmail.com>
 */
#ifndef TENSOR_TUNER_QUANTIZE_OPS_FUNCTOR_H
#define TENSOR_TUNER_QUANTIZE_OPS_FUNCTOR_H

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

template <typename T>
using ConstFlat = typename tensorflow::TTypes<T>::ConstFlat;
template <typename T>
using Flat = typename tensorflow::TTypes<T>::Flat;

namespace tensor_tuner {

template <typename Device, typename T>
struct FixedPointQuantFunctor {
  void operator()(const Device &d, ConstFlat<float> inputs, const float scale,
                  const float min, const float max, Flat<T> outputs) {
    // round-to-nearest
    auto rounded = inputs * scale + 0.5f;
    auto clipped = rounded.cwiseMax(min).cwiseMin(max);
    outputs.device(d) = clipped.cast<T>();
  }
};

}  // namespace tensor_tuner

#endif