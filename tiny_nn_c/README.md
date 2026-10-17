# tiny_nn_c

Small, dependency-free C99 float32 inference primitives:
- ReLU, sigmoid, tanh, stable softmax
- Linear (fully connected) layer
- Conv2D with multiple channels, stride, and symmetric zero padding
- MaxPool2D with stride and padding

## Files
- `tiny_nn.h`: public API and layout documentation
- `tiny_nn.c`: implementations
- `demo.c`: small examples of every primitive

## Build and run

```sh
cc -O2 -std=c99 -Wall -Wextra -pedantic tiny_nn.c demo.c -lm -o demo
./demo
```

## Tensor conventions

Conv2D and MaxPool2D use contiguous NCHW layout:
`[batch][channel][height][width]`.

Conv2D weights use `[out_channel][in_channel][kernel_h][kernel_w]`.
Linear weights use `[out_feature][in_feature]`.
Bias pointers may be `NULL` to omit bias.

Conv2D implements cross-correlation (the convention used by PyTorch `Conv2d`),
not a spatially flipped mathematical convolution. Padding is symmetric and
zero-filled. Output dimensions are:

`out_h = (in_h + 2*padding_h - kernel_h) / stride_h + 1`
`out_w = (in_w + 2*padding_w - kernel_w) / stride_w + 1`

The output buffer capacity is provided in elements; the function returns `-1`
if arguments, dimensions, or capacity are invalid.

## Deliberate limitations

This is a readable reference implementation, not an optimised inference engine.
It uses `float` and straightforward loops. It does not implement batch norm,
dilation, groups/depthwise convolution, quantisation, SIMD, or hardware-specific
acceleration. For small embedded models, benchmark it on the target processor
before relying on performance.

## PyTorch weight export

For `nn.Linear`, `weight` is already stored as `[out_features, in_features]`.
For `nn.Conv2d`, PyTorch's weight shape is `[out_channels, in_channels, kernel_h, kernel_w]`.
Both match this library's expected order. Export parameters as `float32` in
row-major order. Ensure input preprocessing and tensor layout match training.
