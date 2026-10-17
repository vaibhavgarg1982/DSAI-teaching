#include "tiny_nn.h"

#include <stdio.h>

int main(void)
{
    /* Activations */
    float a[] = {-2.0f, 0.0f, 1.0f};
    nn_relu(a, 3);
    printf("ReLU:     %.3f %.3f %.3f\n", a[0], a[1], a[2]);

    float b[] = {-2.0f, 0.0f, 2.0f};
    nn_sigmoid(b, 3);
    printf("Sigmoid:  %.3f %.3f %.3f\n", b[0], b[1], b[2]);

    float c[] = {-2.0f, 0.0f, 2.0f};
    nn_tanh(c, 3);
    printf("Tanh:     %.3f %.3f %.3f\n", c[0], c[1], c[2]);

    float d[] = {1.0f, 2.0f, 3.0f};
    if (nn_softmax(d, 3) == 0)
        printf("Softmax:  %.3f %.3f %.3f\n", d[0], d[1], d[2]);

    /* Linear: 2 inputs -> 2 outputs.
       W = [[1, 2], [-1, 1]], bias = [0.5, 2]. */
    const float x[] = {3.0f, 4.0f};
    const float w[] = {1.0f, 2.0f, -1.0f, 1.0f};
    const float bias[] = {0.5f, 2.0f};
    float y[2];
    if (nn_linear(x, 2, w, bias, y, 2) == 0)
        printf("Linear:   %.3f %.3f\n", y[0], y[1]);

    /* Conv2D: one 3x3 input, one 2x2 filter, padding=1, stride=1.
       Output shape is 4x4. */
    const float image[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    const float filter[] = {
        1, 0,
        0, -1
    };
    float conv_out[16];
    if (nn_conv2d(image, 1, 1, 3, 3,
                  filter, 1, 2, 2, NULL,
                  1, 1, 1, 1,
                  conv_out, 16) == 0) {
        printf("Conv2D output (4x4):\n");
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col)
                printf("%7.2f", conv_out[r * 4 + col]);
            putchar('\n');
        }
    } else {
        fprintf(stderr, "Conv2D failed\n");
        return 1;
    }

    /* MaxPool2D: 1x1x4x4 input, 2x2 kernel, stride 2 -> 2x2. */
    const float pool_in[] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13,14, 15, 16
    };
    float pool_out[4];
    if (nn_maxpool2d(pool_in, 1, 1, 4, 4,
                     2, 2, 2, 2, 0, 0,
                     pool_out, 4) == 0) {
        printf("MaxPool:  %.1f %.1f %.1f %.1f\n",
               pool_out[0], pool_out[1], pool_out[2], pool_out[3]);
    } else {
        fprintf(stderr, "MaxPool2D failed\n");
        return 1;
    }

    return 0;
}
