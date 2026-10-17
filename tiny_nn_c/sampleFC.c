
#include <stdio.h>

#define INPUT_SIZE  784
#define HIDDEN1     256
#define HIDDEN2     128
#define OUTPUT_SIZE 10

/* Placeholder parameters.
   Replace these with trained weights and biases. */
static const float fc1_weight[HIDDEN1][INPUT_SIZE] = {0};
static const float fc1_bias[HIDDEN1] = {0};

static const float fc2_weight[HIDDEN2][HIDDEN1] = {0};
static const float fc2_bias[HIDDEN2] = {0};

static const float fc3_weight[OUTPUT_SIZE][HIDDEN2] = {0};
static const float fc3_bias[OUTPUT_SIZE] = {0};

/* Fully connected layer: y = Wx + b */
static void linear(
    const float *x,
    const float *weight,
    const float *bias,
    float *y,
    int in_features,
    int out_features
) {
    for (int j = 0; j < out_features; ++j) {
        float sum = bias[j];

        for (int i = 0; i < in_features; ++i) {
            sum += weight[j * in_features + i] * x[i];
        }

        y[j] = sum;
    }
}

/* ReLU activation */
static void relu(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f)
            x[i] = 0.0f;
    }
}

/* Forward pass through FashionMLP */
void fashion_mlp(
    const float input[INPUT_SIZE],
    float output[OUTPUT_SIZE]
) {
    float h1[HIDDEN1];
    float h2[HIDDEN2];

    linear(input, &fc1_weight[0][0], fc1_bias,
           h1, INPUT_SIZE, HIDDEN1);
    relu(h1, HIDDEN1);

    linear(h1, &fc2_weight[0][0], fc2_bias,
           h2, HIDDEN1, HIDDEN2);
    relu(h2, HIDDEN2);

    linear(h2, &fc3_weight[0][0], fc3_bias,
           output, HIDDEN2, OUTPUT_SIZE);
}

/* Index of the largest logit */
int predict_class(const float logits[OUTPUT_SIZE]) {
    int best = 0;

    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (logits[i] > logits[best])
            best = i;
    }

    return best;
}

int main(void) {
    /* Sample input: an all-black 28x28 image.
       Pixel values are assumed to be normalised to [0, 1]. */
    float image[INPUT_SIZE] = {0};

    float logits[OUTPUT_SIZE];

    /* Run inference */
    fashion_mlp(image, logits);

    /* Display the output logits */
    printf("Output logits:\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("Class %d: %.6f\n", i, logits[i]);
    }

    /* Get the predicted class */
    int predicted = predict_class(logits);
    printf("Predicted class index: %d\n", predicted);

    return 0;
}
