#ifndef TINY_NN_H
#define TINY_NN_H

/*
 * tiny_nn: small, dependency-free float32 inference primitives.
 *
 * Tensor layout for Conv2D and MaxPool2D: NCHW
 *   [batch][channels][height][width]
 *
 * Linear weights use [out_features][in_features].
 * Conv2D weights use [out_channels][in_channels][kernel_h][kernel_w].
 *
 * Padding is zero-padding. Functions return 0 on success and -1 on invalid
 * arguments or incompatible dimensions. Caller owns all memory.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise activations. Operate in place on n values. */
void nn_relu(float *x, size_t n);
void nn_sigmoid(float *x, size_t n);
void nn_tanh(float *x, size_t n);

/* Stable softmax over one vector of length n, in place. */
int nn_softmax(float *x, size_t n);

/* y[out_features] = W[out_features, in_features] * x + bias.
   bias may be NULL, in which case no bias is added. */
int nn_linear(
    const float *x,
    size_t in_features,
    const float *weights,
    const float *bias,
    float *y,
    size_t out_features
);

/* 2-D convolution, NCHW layout, cross-correlation convention (as in PyTorch).
   Input:   [batch, in_channels, in_h, in_w]
   Weights: [out_channels, in_channels, kernel_h, kernel_w]
   Bias:    [out_channels], or NULL
   Output:  [batch, out_channels, out_h, out_w]
   out_h = (in_h + 2*padding_h - kernel_h)/stride_h + 1
   out_w = (in_w + 2*padding_w - kernel_w)/stride_w + 1
   Padding is symmetric zero-padding. */
int nn_conv2d(
    const float *input,
    size_t batch,
    size_t in_channels,
    size_t in_h,
    size_t in_w,
    const float *weights,
    size_t out_channels,
    size_t kernel_h,
    size_t kernel_w,
    const float *bias,
    size_t stride_h,
    size_t stride_w,
    size_t padding_h,
    size_t padding_w,
    float *output,
    size_t output_capacity
);

/* Max pooling, NCHW layout.
   Output dimensions use the same formula as conv2d.
   Padding locations are treated as negative infinity (so padding cannot win
   a max-pooling comparison). */
int nn_maxpool2d(
    const float *input,
    size_t batch,
    size_t channels,
    size_t in_h,
    size_t in_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t padding_h,
    size_t padding_w,
    float *output,
    size_t output_capacity
);

/* Helpers for calculating output dimensions. Return 0 on success. */
int nn_conv2d_output_shape(
    size_t in_h, size_t in_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w,
    size_t *out_h, size_t *out_w
);

#ifdef __cplusplus
}
#endif

#endif
