#include <cstdio>
#include <cstdlib>

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../stann.hpp"

#include "weights.h"
#include "weights_after.h"
#include "weights_after_batch.h"

#define TEST_PASSED 0
#define TEST_FAILED (-1)

#define SL_INPUTS 10
#define SL_OUTPUTS 8

struct SingleLayerParams {
    float weights[SL_INPUTS * SL_OUTPUTS];
    float biases[SL_OUTPUTS];
};

namespace SingleLayer {
void triplicate_params(SingleLayerParams &params,
                       SingleLayerParams &out_params1,
                       SingleLayerParams &out_params2) {
    for (int i = 0; i < SL_INPUTS * SL_OUTPUTS; i++) {
        #pragma HLS pipeline II = 3
        out_params1.weights[i] = params.weights[i];
        out_params2.weights[i] = params.weights[i];
    }

    for (int i = 0; i < SL_OUTPUTS; i++) {
        #pragma HLS pipeline II = 3
        out_params1.biases[i] = params.biases[i];
        out_params2.biases[i] = params.biases[i];
    }
}

template <int BATCH_SIZE>
void training(float *input, float *labels,
              SingleLayerParams &fw_params,
              SingleLayerParams &bw_params, SingleLayerParams &up_params,
              float learning_rate, int reps) {
    #pragma HLS Dataflow

    hls::stream<float> input_stream("input_stream");
    hls::stream<float> input_stream1("input_stream1");
    hls::stream<float> input_stream2("input_stream2");
    hls::stream<float> l1_out("l1_out");

    StreamUtil::tostream<SL_OUTPUTS>(input, input_stream, reps);
    StreamUtil::duplicate<SL_INPUTS * BATCH_SIZE>(input_stream, input_stream1,
                                                  input_stream2);
    DenseLayerStream::Float::forward<SL_INPUTS, SL_OUTPUTS, BATCH_SIZE, 1, 1, 1, 100>(
        input_stream1, fw_params.weights, fw_params.biases, l1_out, NONE,
        reps);

    hls::stream<float> label_stream("label_stream");
    hls::stream<float> l1_deltas("l1_deltas");

    StreamUtil::tostream<SL_OUTPUTS>(labels, label_stream, reps);
    Loss::MeanSquaredError_derivative_stream<8, BATCH_SIZE>(
        l1_out, label_stream, l1_deltas);

    DenseLayerStream::Float::update<SL_INPUTS, SL_OUTPUTS, BATCH_SIZE, float, 1, 1, 1>(
        l1_deltas, up_params.weights, up_params.biases, input_stream2,
        learning_rate);
}


template <int BATCH_SIZE>
void forward(float *input, SingleLayerParams &params, float *output, int reps) {
    #pragma HLS Dataflow

    hls::stream<float> input_stream("input_stream_inference");
    hls::stream<float> output_stream("output_stream_inference");
    StreamUtil::tostream<SL_INPUTS>(input, input_stream, reps);
    DenseLayerStream::Float::forward<SL_INPUTS,SL_OUTPUTS,BATCH_SIZE,1,1,1>(input_stream, params.weights, params.biases, output_stream, NONE, reps);
    StreamUtil::toarray<SL_OUTPUTS>(output_stream, output, reps);

}
}

template<int M, int N>
void print_mat(float *mat) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            printf("%f ", mat[m * N + n]);
        }
        printf("\n");
    }
}

template<int K, int M, int N>
void simple_matmul(float *a, float *b, float *c) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                c[k * N + n] += a[k * M + m] * b[m * N + n];
            }
        }
    }
}

int test_inference(SingleLayerParams &params) {
    float inputs[10] = {0,1,2,3,4,5,6,7,8,9};
    float outputs[8]          = {0,0,0,0,0,0,0,0};
    float expected_outputs[8] = {0,0,0,0,0,0,0,0};

    SingleLayer::forward<1>(inputs, params, outputs, 1);


    simple_matmul<SL_OUTPUTS, SL_INPUTS, 1>(params.weights, inputs, expected_outputs);
    for (int i = 0; i < SL_OUTPUTS; i++) {
        expected_outputs[i] += params.biases[i];
    }

    for (int i = 0; i < SL_OUTPUTS; i++) {
        if (abs(outputs[i] - expected_outputs[i]) > 0.0001) {
            print_mat<8, 1>(expected_outputs);
            print_mat<8, 1>(outputs);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_inference_batch(SingleLayerParams &params) {
    float inputs[10 * 4] = {
        0,0,0,0,
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4,
        5,5,5,5,
        6,6,6,6,
        7,7,7,7,
        8,8,8,8,
        9,9,9,9,
    };
    float outputs[8*4] = {
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    float expected_outputs[8*4] = {
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };

    SingleLayer::forward<4>(inputs, params, outputs, 4);

    simple_matmul<SL_OUTPUTS, SL_INPUTS, 4>(params.weights, inputs, expected_outputs);
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < SL_OUTPUTS; i++) {
            expected_outputs[i * 4 + j] += params.biases[i];
        }
    }

    for (int i = 0; i < SL_OUTPUTS * 4; i++) {
        if (abs(outputs[i] - expected_outputs[i]) > 0.0001) {
            print_mat<8, 4>(expected_outputs);
            print_mat<8, 4>(outputs);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_training() {
    SingleLayerParams params = {
        .weights = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases = {
            1,2,3,4,5,6,7,8
        }
    };
    print_mat<8,10>(params.weights);

    float inputs[10] = {0,1,2,3,4,5,6,7,8,9};
    float labels[8] = {1,2,3,4,5,6,7,8};
    float outputs[8] = {0,0,0,0,0,0,0,0};

    SingleLayerParams fw_params;
    SingleLayerParams bw_params;
    SingleLayer::triplicate_params(params, fw_params, bw_params);

    //SingleLayer::training<1>(inputs, labels, fw_params, bw_params, params, 0.01, 1);
    hls::stream<float> input_stream("input_stream");
    hls::stream<float> input_stream1("input_stream1");
    hls::stream<float> input_stream2("input_stream2");
    hls::stream<float> l1_out("l1_out");
    hls::stream<float> l1_out_copy("l1_out");

    StreamUtil::tostream<SL_INPUTS>(inputs, input_stream);
    StreamUtil::duplicate<SL_INPUTS>(input_stream, input_stream1, input_stream2);
    DenseLayerStream::Float::forward<SL_INPUTS, SL_OUTPUTS, 1, 1, 1, 1, 100>(
        input_stream1, fw_params.weights, fw_params.biases, l1_out, NONE, 1);

    hls::stream<float> label_stream("label_stream");
    hls::stream<float> l1_deltas("l1_deltas");

    StreamUtil::toarray<SL_OUTPUTS>(l1_out, outputs);
    float mseloss = 0;
    for (int i = 0; i < SL_OUTPUTS; i++) {
        mseloss += (outputs[i] - labels[i]) * (outputs[i] - labels[i]);
    }
    printf("mseloss: %f\n", mseloss/8);

    StreamUtil::tostream<SL_OUTPUTS>(outputs, l1_out_copy, 1);
    StreamUtil::tostream<SL_OUTPUTS>(labels, label_stream, 1);
    Loss::MeanSquaredError_derivative_stream<8, 1>(l1_out_copy, label_stream, l1_deltas);

    DenseLayerStream::Float::update<SL_INPUTS, SL_OUTPUTS, 1, float, 1, 1, 1>(
        l1_deltas, params.weights, params.biases, input_stream2, 0.01);

    printf("weights\n");
    print_mat<8,10>(params.weights);
    printf("biases\n");
    print_mat<8,1>(params.biases);

    return TEST_PASSED;
}

int test_training_batch() {
    SingleLayerParams params = {
        .weights = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases = {
            1,2,3,4,5,6,7,8
        }
    };
    print_mat<8,10>(params.weights);

    float inputs[10 * 4] = {
        0,0,0,0,
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4,
        5,5,5,5,
        6,6,6,6,
        7,7,7,7,
        8,8,8,8,
        9,9,9,9,
    };
    float labels[8 * 4] = {
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4,
        5,5,5,5,
        6,6,6,6,
        7,7,7,7,
        8,8,8,8,
        //1,2,3,4,5,6,7,8,
        //1,2,3,4,5,6,7,8,
        //1,2,3,4,5,6,7,8,
        //1,2,3,4,5,6,7,8,
    };
    float outputs[8 * 4] = {
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    float deltas[8 * 4];
    for (int i = 0; i < 8*4; i++) {
        deltas[i] = 0;
    }

    SingleLayerParams fw_params;
    SingleLayerParams bw_params;
    SingleLayer::triplicate_params(params, fw_params, bw_params);

    hls::stream<float> input_stream("input_stream");
    hls::stream<float> input_stream1("input_stream1");
    hls::stream<float> input_stream2("input_stream2");
    hls::stream<float> l1_out("l1_out");
    hls::stream<float> l1_out_copy("l1_out");

    StreamUtil::tostream<SL_INPUTS>(inputs, input_stream, 4);
    StreamUtil::duplicate<SL_INPUTS*4>(input_stream, input_stream1, input_stream2);
    DenseLayerStream::Float::forward<SL_INPUTS, SL_OUTPUTS, 4, 1, 1, 1, 100>(
        input_stream1, fw_params.weights, fw_params.biases, l1_out, NONE, 4);

    StreamUtil::toarray<8>(l1_out, outputs, 4);
    printf("outputs\n");
    print_mat<8,4>(outputs);
    StreamUtil::tostream<8>(outputs, l1_out_copy, 4);

    hls::stream<float> label_stream("label_stream");
    hls::stream<float> l1_deltas("l1_deltas");
    hls::stream<float> l1_deltas_copy("l1_deltas_copy");

    StreamUtil::tostream<SL_OUTPUTS>(labels, label_stream, 4);
    Loss::MeanSquaredError_derivative_stream<8, 4>(l1_out_copy, label_stream, l1_deltas);

    StreamUtil::toarray<8>(l1_deltas, deltas, 4);
    printf("deltas\n");
    print_mat<8,4>(deltas);
    StreamUtil::tostream<8>(deltas, l1_deltas_copy, 4);

    DenseLayerStream::Float::update<SL_INPUTS, SL_OUTPUTS, 4, float, 1, 1, 1>(
        l1_deltas_copy, params.weights, params.biases, input_stream2, 0.01);

    printf("weights\n");
    print_mat<8,10>(params.weights);
    printf("biases\n");
    print_mat<8,1>(params.biases);

    return TEST_PASSED;
}
int main () {
    SingleLayerParams params = {
        .weights = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases = {
            1,2,3,4,5,6,7,8
        }
    };

    if (test_inference(params) == TEST_PASSED) {
        printf("Inference test PASSED\n");
    } else {
        printf("Inference test FAILED\n");
    }

    if (test_inference_batch(params) == TEST_PASSED) {
        printf("Inference batch test PASSED\n");
    } else {
        printf("Inference batch test FAILED\n");
    }

    if (test_training() == TEST_PASSED) {
        printf("Training test PASSED\n");
    } else {
        printf("Training test FAILED\n");
    }


    if (test_training_batch() == TEST_PASSED) {
        printf("Training batch test PASSED\n");
    } else {
        printf("Training batch test FAILED\n");
    }

    return 0;
}
