/*!
 * \brief This file implements various quantization operations.
 * \author Ruizhe Zhao <vincentzhaorz@gmail.com>
 *
 * This file is inspired by:
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/fake_quant_ops.cc
 */

#define EIGEN_USE_THREADS

#include "tensor_tuner/core/kernel/quantize_ops_functor.h"

#include <type_traits>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;
using namespace tensorflow::errors;

namespace tensor_tuner {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
int GetNumBits(int num_fb, int num_ib, bool signed_) {
  return num_fb + num_ib + static_cast<int>(signed_);
}

template <typename T>
bool IsTypeValid() {
  return std::is_same<T, int16>::value;
}

template <typename T>
bool IsNumBitsValid(int num_bits) {
  // we only supports the case that number of bits equals 16.
  if (std::is_same<T, int16>::value || std::is_same<T, uint16>::value)
    return num_bits == 16;
  else
    throw InvalidArgument("Type is not supported.");
}

template <typename T>
bool IsSignedValid(bool signed_) {
  if (std::is_same<T, int16>::value)
    return signed_ == true;
  else if (std::is_same<T, uint16>::value)
    return signed_ == false;
  else
    throw InvalidArgument("Type is not supported.");
}
}  // namespace

/*!
 * \brief Quantize a floating-point value into the fixed-point encoding.
 * \author Ruizhe Zhao <vincentzhaorz@gmail.com>
 *
 * This function implements **static** fixed-point quantization.
 * For different input tensors, we use the same fixed-point configuration
 * that is statically defined.
 */
template <typename Device, typename T>
class FixedPointQuantOp : public OpKernel {
 public:
  explicit FixedPointQuantOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_fb", &num_fb_));
    OP_REQUIRES_OK(context, context->GetAttr("num_ib", &num_ib_));
    OP_REQUIRES_OK(context, context->GetAttr("signed", &signed_));
    OP_REQUIRES(context, IsTypeValid<T>(),
                InvalidArgument("Template type T is not supported."));

    int num_bits = GetNumBits(num_fb_, num_ib_, signed_);
    OP_REQUIRES(context, IsNumBitsValid<T>(num_bits),
                InvalidArgument("Number of bits should equal 16."));
    OP_REQUIRES(context, IsSignedValid<T>(signed_),
                InvalidArgument("Signed value mismatch."));

    scale_ = static_cast<float>(1 << num_fb_);
    if (signed_) {
      min_ = -static_cast<float>(1 << (num_bits - 1)) / scale_;
      max_ = (static_cast<float>(1 << (num_bits - 1)) - 1) / scale_;
    } else {
      min_ = 0.0f;
      max_ = (static_cast<float>(1 << num_bits) - 1) / scale_;
    }
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);

    Tensor *output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    FixedPointQuantFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(), scale_, min_,
            max_, output->flat<T>());
  }

 private:
  int num_fb_;  /*!< Number of fraction bits */
  int num_ib_;  /*!< Number of integer bits */
  bool signed_; /*!< Signed or unsigned */
  float min_;   /*!< Minimum representable value */
  float max_;   /*!< Maximum representable value */
  float scale_; /*!< Scale to quantize */
};

#define SINGLE_ARG(...) __VA_ARGS__
#define REGISTER_KERNEL(NAME, TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<TYPE>("quant_type"), \
      SINGLE_ARG(FixedPointQuantOp<CPUDevice, TYPE>));

REGISTER_KERNEL("FixedPointQuant", int16)

#undef REGISTER_KERNEL
#undef SINGLE_ARG

}  // namespace tensor_tuner