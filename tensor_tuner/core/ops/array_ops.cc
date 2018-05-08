/*!
 * \brief This file registers operations defined in this project.
 * \author Ruizhe Zhao <vincentzhaorz@gmail.com>
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("FixedPointQuant")
    .Attr("quant_type: {int16}")
    .Attr("num_fb: int")
    .Attr("num_ib: int")
    .Attr("signed: bool")
    .Input("input: float32")
    .Output("result: quant_type")
    .SetShapeFn(shape_inference::UnchangedShape);