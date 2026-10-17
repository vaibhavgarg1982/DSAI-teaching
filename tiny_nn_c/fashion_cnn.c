/*
 * FashionCNN reference inference example using tiny_nn primitives.
 *
 * Architecture (input: one 28x28 grayscale image, NCHW):
 *   Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2)
 *   Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2)
 *   Flatten -> Linear(32*7*7,128) -> ReLU -> Linear(128,10)
 *
 * The arrays below are zero-filled placeholders. Replace them with the
 * trained PyTorch state_dict parameters exported as float32 C arrays.
 */

#include "tiny_nn.h"

#include <stdio.h>
#include <stddef.h>

#define IN_H 28
#define IN_W 28
#define CONV1_OUT 16
#define CONV2_OUT 32
#define POOL1_H 14
#define POOL1_W 14
#define POOL2_H 7
#define POOL2_W 7
#define FLAT_SIZE (32 * 7 * 7)
#define FC1_OUT 128
#define CLASS_COUNT 10

/* PyTorch Conv2d weight shape: [out_channels][in_channels][kh][kw] */
static const float conv1_weight[16][1][3][3] = {0};
static const float conv1_bias[16] = {0};

static const float conv2_weight[32][16][3][3] = {0};
static const float conv2_bias[32] = {0};

/* PyTorch Linear weight shape: [out_features][in_features] */
static const float fc1_weight[128][FLAT_SIZE] = {0};
static const float fc1_bias[128] = {0};

static const float fc2_weight[10][128] = {0};
static const float fc2_bias[10] = {0};

static int argmax(const float *x, size_t n)
{
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return (int)best;
}

/*
 * input: 784 row-major grayscale pixels, normalized exactly as in training.
 * logits: receives 10 output scores.
 * Returns 0 on success, -1 on an invalid primitive call.
 */
int fashion_cnn(const float input[IN_H * IN_W], float logits[CLASS_COUNT])
{
    /* Intermediate NCHW tensors. Batch size is 1. */
    float c1[CONV1_OUT * IN_H * IN_W];
    float p1[CONV1_OUT * POOL1_H * POOL1_W];
    float c2[CONV2_OUT * POOL1_H * POOL1_W];
    float p2[CONV2_OUT * POOL2_H * POOL2_W];
    float flat[FLAT_SIZE];
    float h1[FC1_OUT];

    /* Conv1: [1,1,28,28] -> [1,16,28,28] */
    if (nn_conv2d(input, 1, 1, IN_H, IN_W,
                  &conv1_weight[0][0][0][0], CONV1_OUT, 3, 3,
                  conv1_bias, 1, 1, 1, 1,
                  c1, CONV1_OUT * IN_H * IN_W) != 0)
        return -1;
    nn_relu(c1, CONV1_OUT * IN_H * IN_W);

    /* Pool1: [1,16,28,28] -> [1,16,14,14] */
    if (nn_maxpool2d(c1, 1, CONV1_OUT, IN_H, IN_W,
                     2, 2, 2, 2, 0, 0,
                     p1, CONV1_OUT * POOL1_H * POOL1_W) != 0)
        return -1;

    /* Conv2: [1,16,14,14] -> [1,32,14,14] */
    if (nn_conv2d(p1, 1, CONV1_OUT, POOL1_H, POOL1_W,
                  &conv2_weight[0][0][0][0], CONV2_OUT, 3, 3,
                  conv2_bias, 1, 1, 1, 1,
                  c2, CONV2_OUT * POOL1_H * POOL1_W) != 0)
        return -1;
    nn_relu(c2, CONV2_OUT * POOL1_H * POOL1_W);

    /* Pool2: [1,32,14,14] -> [1,32,7,7] */
    if (nn_maxpool2d(c2, 1, CONV2_OUT, POOL1_H, POOL1_W,
                     2, 2, 2, 2, 0, 0,
                     p2, CONV2_OUT * POOL2_H * POOL2_W) != 0)
        return -1;

    /* Flatten preserves NCHW memory order; copy to an explicit 1-D buffer. */
    for (size_t i = 0; i < FLAT_SIZE; ++i)
        flat[i] = p2[i];

    /* FC1: 1568 -> 128, then ReLU */
    if (nn_linear(flat, FLAT_SIZE, &fc1_weight[0][0], fc1_bias,
                  h1, FC1_OUT) != 0)
        return -1;
    nn_relu(h1, FC1_OUT);

    /* FC2: 128 -> 10 logits */
    if (nn_linear(h1, FC1_OUT, &fc2_weight[0][0], fc2_bias,
                  logits, CLASS_COUNT) != 0)
        return -1;

    return 0;
}

int main(void)
{
    /* Demo input: all-zero 28x28 image. Replace with a real normalized image. */
    float image[IN_H * IN_W] = {0};
    float logits[CLASS_COUNT];

    if (fashion_cnn(image, logits) != 0) {
        fprintf(stderr, "FashionCNN inference failed\n");
        return 1;
    }

    printf("FashionMNIST logits:\n");
    for (int i = 0; i < CLASS_COUNT; ++i)
        printf("  class %d: %.6f\n", i, logits[i]);

    printf("Predicted class index: %d\n", argmax(logits, CLASS_COUNT));
    return 0;
}
