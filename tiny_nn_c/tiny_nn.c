#include "tiny_nn.h"

#include <math.h>
#include <stdint.h>

static int valid_ptr(const void *p) { return p != NULL; }

void nn_relu(float *x, size_t n)
{
    if (!x) return;
    for (size_t i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void nn_sigmoid(float *x, size_t n)
{
    if (!x) return;
    /* Branching avoids exp overflow for large-magnitude inputs. */
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (v >= 0.0f) {
            x[i] = 1.0f / (1.0f + expf(-v));
        } else {
            const float e = expf(v);
            x[i] = e / (1.0f + e);
        }
    }
}

void nn_tanh(float *x, size_t n)
{
    if (!x) return;
    for (size_t i = 0; i < n; ++i)
        x[i] = tanhf(x[i]);
}

int nn_softmax(float *x, size_t n)
{
    if (!valid_ptr(x) || n == 0) return -1;

    float maxv = x[0];
    for (size_t i = 1; i < n; ++i)
        if (x[i] > maxv) maxv = x[i];

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += (double)x[i];
    }
    if (!(sum > 0.0) || !isfinite(sum)) return -1;

    for (size_t i = 0; i < n; ++i)
        x[i] = (float)((double)x[i] / sum);

    return 0;
}

int nn_linear(
    const float *x, size_t in_features,
    const float *weights, const float *bias,
    float *y, size_t out_features)
{
    if (!x || !weights || !y || in_features == 0 || out_features == 0)
        return -1;

    for (size_t o = 0; o < out_features; ++o) {
        float sum = bias ? bias[o] : 0.0f;
        const float *wrow = weights + o * in_features;
        for (size_t i = 0; i < in_features; ++i)
            sum += wrow[i] * x[i];
        y[o] = sum;
    }
    return 0;
}

int nn_conv2d_output_shape(
    size_t in_h, size_t in_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w,
    size_t *out_h, size_t *out_w)
{
    if (!out_h || !out_w || in_h == 0 || in_w == 0 ||
        kernel_h == 0 || kernel_w == 0 ||
        stride_h == 0 || stride_w == 0)
        return -1;

    /* Guard additions/multiplications against size_t overflow. */
    if (padding_h > (SIZE_MAX - in_h) / 2 ||
        padding_w > (SIZE_MAX - in_w) / 2)
        return -1;

    const size_t padded_h = in_h + 2 * padding_h;
    const size_t padded_w = in_w + 2 * padding_w;
    if (padded_h < kernel_h || padded_w < kernel_w)
        return -1;

    *out_h = (padded_h - kernel_h) / stride_h + 1;
    *out_w = (padded_w - kernel_w) / stride_w + 1;
    return 0;
}

int nn_conv2d(
    const float *input,
    size_t batch, size_t in_channels, size_t in_h, size_t in_w,
    const float *weights, size_t out_channels,
    size_t kernel_h, size_t kernel_w, const float *bias,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w,
    float *output, size_t output_capacity)
{
    if (!input || !weights || !output ||
        batch == 0 || in_channels == 0 || out_channels == 0)
        return -1;

    size_t out_h, out_w;
    if (nn_conv2d_output_shape(in_h, in_w, kernel_h, kernel_w,
                              stride_h, stride_w, padding_h, padding_w,
                              &out_h, &out_w) != 0)
        return -1;

    if (batch > SIZE_MAX / out_channels ||
        batch * out_channels > SIZE_MAX / out_h ||
        batch * out_channels * out_h > SIZE_MAX / out_w)
        return -1;
    const size_t needed = batch * out_channels * out_h * out_w;
    if (output_capacity < needed) return -1;

    for (size_t n = 0; n < batch; ++n) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            const ptrdiff_t ih =
                                (ptrdiff_t)(oh * stride_h + kh) -
                                (ptrdiff_t)padding_h;
                            if (ih < 0 || (size_t)ih >= in_h) continue;

                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                const ptrdiff_t iw =
                                    (ptrdiff_t)(ow * stride_w + kw) -
                                    (ptrdiff_t)padding_w;
                                if (iw < 0 || (size_t)iw >= in_w) continue;

                                const size_t in_idx =
                                    ((n * in_channels + ic) * in_h +
                                     (size_t)ih) * in_w + (size_t)iw;
                                const size_t w_idx =
                                    ((oc * in_channels + ic) * kernel_h +
                                     kh) * kernel_w + kw;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }

                    const size_t out_idx =
                        ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
    return 0;
}

int nn_maxpool2d(
    const float *input,
    size_t batch, size_t channels, size_t in_h, size_t in_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w,
    float *output, size_t output_capacity)
{
    if (!input || !output || batch == 0 || channels == 0)
        return -1;

    size_t out_h, out_w;
    if (nn_conv2d_output_shape(in_h, in_w, kernel_h, kernel_w,
                               stride_h, stride_w, padding_h, padding_w,
                               &out_h, &out_w) != 0)
        return -1;

    if (batch > SIZE_MAX / channels ||
        batch * channels > SIZE_MAX / out_h ||
        batch * channels * out_h > SIZE_MAX / out_w)
        return -1;
    const size_t needed = batch * channels * out_h * out_w;
    if (output_capacity < needed) return -1;

    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float maxv = -INFINITY;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        const ptrdiff_t ih =
                            (ptrdiff_t)(oh * stride_h + kh) -
                            (ptrdiff_t)padding_h;
                        if (ih < 0 || (size_t)ih >= in_h) continue;

                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            const ptrdiff_t iw =
                                (ptrdiff_t)(ow * stride_w + kw) -
                                (ptrdiff_t)padding_w;
                            if (iw < 0 || (size_t)iw >= in_w) continue;

                            const size_t idx =
                                ((n * channels + c) * in_h +
                                 (size_t)ih) * in_w + (size_t)iw;
                            if (input[idx] > maxv) maxv = input[idx];
                        }
                    }

                    const size_t out_idx =
                        ((n * channels + c) * out_h + oh) * out_w + ow;
                    output[out_idx] = maxv;
                }
            }
        }
    }
    return 0;
}
