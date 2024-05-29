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

#define TL_INPUTS 10
#define TL_HIDDEN 8
#define TL_OUTPUTS 6

struct TwoLayerParams {
    float weights_l1[TL_INPUTS * TL_HIDDEN];
    float biases_l1[TL_HIDDEN];
    float weights_l2[TL_HIDDEN * TL_OUTPUTS];
    float biases_l2[TL_OUTPUTS];
};

template<int M, int N>
void print_mat(float *mat) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            printf("%f ", mat[m * N + n]);
        }
        printf("\n");
    }
}

// template<int M, int N>
// void print_mat_t(float *mat) {
//     for (int m = 0; m < M; m++) {
//         for (int n = 0; n < N; n++) {
//             printf("%f ", mat[m * N + n]);
//         }
//         printf("\n");
//     }
// }

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

namespace TwoLayer {
    void triplicate_params(TwoLayerParams &params,
                                  TwoLayerParams &out_params1,
                                  TwoLayerParams &out_params2) {
      for (int i = 0; i < TL_INPUTS * TL_HIDDEN; i++) {
#pragma HLS pipeline II = 3
        out_params1.weights_l1[i] = params.weights_l1[i];
        out_params2.weights_l1[i] = params.weights_l1[i];
      }

      for (int i = 0; i < TL_HIDDEN; i++) {
#pragma HLS pipeline II = 3
        out_params1.biases_l1[i] = params.biases_l1[i];
        out_params2.biases_l1[i] = params.biases_l1[i];
      }

      for (int i = 0; i < TL_HIDDEN * TL_OUTPUTS; i++) {
#pragma HLS pipeline II = 3
        out_params1.weights_l2[i] = params.weights_l2[i];
        out_params2.weights_l2[i] = params.weights_l2[i];
      }

      for (int i = 0; i < TL_OUTPUTS; i++) {
#pragma HLS pipeline II = 3
        out_params1.biases_l2[i] = params.biases_l2[i];
        out_params2.biases_l2[i] = params.biases_l2[i];
      }
    }

template<int BATCH_SIZE>
    void forward(float *input, TwoLayerParams &params, float *output, int reps) {
    #pragma HLS Dataflow

        hls::stream<float> input_stream("input_stream_inference");
        hls::stream<float> output_stream("output_stream_inference");
        hls::stream<float> l1_out("l1_out");

        StreamUtil::tostream<TL_INPUTS>(input, input_stream, reps);
        DenseLayerStream::Float::forward<TL_INPUTS,TL_HIDDEN,BATCH_SIZE,1,1,1>(input_stream, params.weights_l1, params.biases_l1, l1_out, NONE, reps);
        DenseLayerStream::Float::forward<TL_HIDDEN,TL_OUTPUTS,BATCH_SIZE,1,1,1>(l1_out, params.weights_l2, params.biases_l2, output_stream, NONE, reps);
        StreamUtil::toarray<TL_OUTPUTS>(output_stream, output, reps);

    }
}

int test_inference(TwoLayerParams &params) {
    float inputs[TL_INPUTS] = {0,1,2,3,4,5,6,7,8,9};
    float outputs[TL_OUTPUTS]          = {0,0,0,0,0,0};
    float expected_outputs[TL_OUTPUTS] = {0,0,0,0,0,0};
    float hl_outputs[TL_HIDDEN] = {0,0,0,0,0,0,0,0};

    TwoLayer::forward<1>(inputs, params, outputs, 1);

    simple_matmul<TL_HIDDEN, TL_INPUTS, 1>(params.weights_l1, inputs, hl_outputs);
    for (int i = 0; i < TL_HIDDEN; i++) {
        hl_outputs[i] += params.biases_l1[i];
    }
    simple_matmul<TL_OUTPUTS, TL_HIDDEN, 1>(params.weights_l2, hl_outputs, expected_outputs);
    for (int i = 0; i < TL_OUTPUTS; i++) {
        expected_outputs[i] += params.biases_l2[i];
    }

    for (int i = 0; i < TL_OUTPUTS; i++) {
        if (abs(outputs[i] - expected_outputs[i]) > 0.0001) {
            print_mat<10, 1>(inputs);
            print_mat<8, 1>(hl_outputs);
            print_mat<6, 1>(expected_outputs);
            print_mat<6, 1>(outputs);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_inference_batch(TwoLayerParams &params) {
    float inputs[TL_INPUTS * 4] = {
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
    float outputs[TL_OUTPUTS*4] = {
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
    };
    float expected_outputs[TL_OUTPUTS*4] = {
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
    };
    float hl_outputs[TL_HIDDEN*4] = {
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };

    TwoLayer::forward<4>(inputs, params, outputs, 4);

    simple_matmul<TL_HIDDEN, TL_INPUTS, 4>(params.weights_l1, inputs, hl_outputs);
    for (int i = 0; i < TL_HIDDEN; i++) {
        for (int j = 0; j < 4; j++) {
            hl_outputs[i*4+j] += params.biases_l1[i];
        }
    }
    simple_matmul<TL_OUTPUTS, TL_HIDDEN, 4>(params.weights_l2, hl_outputs, expected_outputs);
    for (int i = 0; i < TL_OUTPUTS; i++) {
        for (int j = 0; j < 4; j++) {
            expected_outputs[i*4+j] += params.biases_l2[i];
        }
    }

    for (int i = 0; i < TL_OUTPUTS*4; i++) {
        if (abs(outputs[i] - expected_outputs[i]) > 0.0001) {
            print_mat<10, 1>(inputs);
            print_mat<8, 1>(hl_outputs);
            print_mat<6, 1>(expected_outputs);
            print_mat<6, 1>(outputs);
            return TEST_FAILED;
        }
    }
    return TEST_PASSED;

}

int test_training() {
    TwoLayerParams params = {
        .weights_l1 = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases_l1 = {
            1,2,3,4,5,6,7,8
        },

        .weights_l2 = {
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
        },

        .biases_l2 = {
            1,2,3,4,5,6
        }
    };
    for (int i = 0; i < 10 * 8; i++) {
        //params.weights_l1[i] = i;
        params.weights_l1[i] = i - 30;
    }
    for (int i = 0; i < 6 * 8; i++) {
        //params.weights_l2[i] = i;
        params.weights_l2[i] = i - 20;
    }
    TwoLayerParams fw_params;
    TwoLayerParams bw_params;
    TwoLayer::triplicate_params(params, fw_params, bw_params);
    printf("Weights L1\n");
    print_mat<8, 10>(params.weights_l1);
    printf("Weights L2\n");
    print_mat<6, 8>(params.weights_l2);

    float inputs[10] = {0,1,2,3,4,5,6,7,8,9};
    float labels[6] = {1,2,3,4,5,6};
    float outputs[6] = {0,0,0,0,0,0};

    hls::stream<float> input_stream("input_stream");
    hls::stream<float> input_stream1("input_stream1");
    hls::stream<float> input_stream2("input_stream2");
    hls::stream<float> l1_out("l1_out");
    hls::stream<float> l1_out_act("l1_out");
    hls::stream<float> l1_out_act_copy("l1_out");
    hls::stream<float> l2_out("l1_out");
    hls::stream<float> l2_out_copy("l1_out");
    hls::stream<float> label_stream("label_stream");
    hls::stream<float> l1_deltas("l1_deltas");
    hls::stream<float> l1_deltas_dbg("l1_deltas");
    hls::stream<float> l2_deltas("l2_deltas");
    hls::stream<float> l2_deltas_dbg("l2_deltas");
    hls::stream<float> l2_deltas_bw("l2_deltas");
    hls::stream<float> l2_deltas_up("l2_deltas");
    float l1_out_copy[TL_HIDDEN] = {0,0,0,0,0,0,0,0};

    StreamUtil::tostream<TL_INPUTS>(inputs, input_stream);
    StreamUtil::duplicate<TL_INPUTS>(input_stream, input_stream1, input_stream2);
    DenseLayerStream::Float::forward<TL_INPUTS,TL_HIDDEN,1,1,1,1>(input_stream1, params.weights_l1, params.biases_l1, l1_out, NONE, 1);
    ActivationLayer::Float::leaky_relu_stream<TL_HIDDEN, 1>(
        l1_out, l1_out_copy, l1_out_act, l1_out_act_copy, 1);
    DenseLayerStream::Float::forward<TL_HIDDEN,TL_OUTPUTS,1,1,1,1>(l1_out_act, params.weights_l2, params.biases_l2, l2_out, NONE, 1);
    StreamUtil::toarray<TL_OUTPUTS>(l2_out, outputs);

    print_mat<6, 1>(outputs);
    float mseloss = 0;
    for (int i = 0; i < TL_OUTPUTS; i++) {
        mseloss += (outputs[i] - labels[i]) * (outputs[i] - labels[i]);
    }
    printf("mseloss: %f\n", mseloss/TL_OUTPUTS);

    float l1_deltas_array[8];
    float l2_deltas_array[6];


    StreamUtil::tostream<TL_OUTPUTS>(outputs, l2_out_copy, 1);
    StreamUtil::tostream<TL_OUTPUTS>(labels, label_stream, 1);
    Loss::MeanSquaredError_derivative_stream<TL_OUTPUTS, 1>(l2_out_copy, label_stream, l2_deltas_dbg);

    StreamUtil::toarray<TL_OUTPUTS>(l2_deltas_dbg, l2_deltas_array);
    printf("L2 deltas: ");
    print_mat<6, 1>(l2_deltas_array);
    StreamUtil::tostream<TL_OUTPUTS>(l2_deltas_array, l2_deltas);

    StreamUtil::duplicate<TL_OUTPUTS>(l2_deltas, l2_deltas_bw, l2_deltas_up);

    DenseLayerStream::Float::backward<TL_INPUTS, TL_HIDDEN, TL_OUTPUTS, 1, 1, 1, 1, 100>(l1_out_copy, bw_params.weights_l2, l2_deltas_bw, l1_deltas_dbg, LEAKY_RELU, 1);

    StreamUtil::toarray<TL_HIDDEN>(l1_deltas_dbg, l1_deltas_array);
    printf("L1 deltas: ");
    print_mat<8, 1>(l1_deltas_array);
    StreamUtil::tostream<TL_HIDDEN>(l1_deltas_array, l1_deltas);

    DenseLayerStream::Float::update<TL_INPUTS, TL_HIDDEN, 1, float, 1, 1, 1>(
        l1_deltas, params.weights_l1, params.biases_l1, input_stream2, 0.01);

    DenseLayerStream::Float::update<TL_HIDDEN, TL_OUTPUTS, 1, float, 1, 1, 1>(
        l2_deltas_up, params.weights_l2, params.biases_l2, l1_out_act_copy, 0.01);

    printf("weights l1\n");
    print_mat<8,10>(params.weights_l1);
    printf("biases l1\n");
    print_mat<8,1>(params.biases_l1);
    printf("weights l2\n");
    print_mat<6,8>(params.weights_l2);
    printf("biases l2\n");
    print_mat<6,1>(params.biases_l2);

    return TEST_PASSED;
}

int test_training_batch() {
    TwoLayerParams params = {
        .weights_l1 = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases_l1 = {
            1,2,3,4,5,6,7,8
        },

        .weights_l2 = {
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
        },

        .biases_l2 = {
            1,2,3,4,5,6
        }
    };
    for (int i = 0; i < 10 * 8; i++) {
        //params.weights_l1[i] = i;
        params.weights_l1[i] = i - 30;
    }
    for (int i = 0; i < 6 * 8; i++) {
        //params.weights_l2[i] = i;
        params.weights_l2[i] = i - 20;
    }
    TwoLayerParams fw_params;
    TwoLayerParams bw_params;
    TwoLayer::triplicate_params(params, fw_params, bw_params);
    float inputs[10 * 4] = {
        0,1,2,3,4,5,6,7,8,9,
        0,1,2,3,4,5,6,7,8,9,
        0,1,2,3,4,5,6,7,8,9,
        0,1,2,3,4,5,6,7,8,9,
    };
    float labels[6 * 4] = {
        1,2,3,4,5,6,
        1,2,3,4,5,6,
        1,2,3,4,5,6,
        1,2,3,4,5,6,
    };
    float outputs[6 * 4] = {
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
        0,0,0,0,0,0,
    };
    float l1_out_copy[TL_HIDDEN*4] = {
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    hls::stream<float> input_stream("input_stream");
    hls::stream<float> input_stream1("input_stream1");
    hls::stream<float> input_stream2("input_stream2");
    hls::stream<float> l1_out("l1_out");
    hls::stream<float> l1_out_act("l1_out");
    hls::stream<float> l1_out_act_copy("l1_out");
    hls::stream<float> l2_out("l1_out");
    hls::stream<float> l2_out_copy("l1_out");
    hls::stream<float> label_stream("label_stream");
    hls::stream<float> l1_deltas("l1_deltas");
    hls::stream<float> l1_deltas_dbg("l1_deltas_dbg");
    hls::stream<float> l2_deltas("l2_deltas");
    hls::stream<float> l2_deltas_dbg("l2_deltas_dbg");
    hls::stream<float> l2_deltas_bw("l2_deltas_bw");
    hls::stream<float> l2_deltas_up("l2_deltas_up");
    float l1_deltas_array[8*4];
    float l2_deltas_array[6*4];

    StreamUtil::tostream<TL_INPUTS*4>(inputs, input_stream);
    StreamUtil::duplicate<TL_INPUTS*4>(input_stream, input_stream1, input_stream2);
    DenseLayerStream::Float::forward<TL_INPUTS,TL_HIDDEN,4,1,1,1>(input_stream1, params.weights_l1, params.biases_l1, l1_out, NONE, 4);
    ActivationLayer::Float::leaky_relu_stream<TL_HIDDEN, 4>(
        l1_out, l1_out_copy, l1_out_act, l1_out_act_copy, 4);
    DenseLayerStream::Float::forward<TL_HIDDEN,TL_OUTPUTS,4,1,1,1>(l1_out_act, params.weights_l2, params.biases_l2, l2_out, NONE, 4);
    StreamUtil::toarray<TL_OUTPUTS*4>(l2_out, outputs);
    //StreamUtil::toarray<TL_OUTPUTS>(l2_out, outputs, 4);

    printf("outputs\n");
    print_mat<4, 6>(outputs);

    //StreamUtil::tostream<TL_OUTPUTS>(outputs, l2_out_copy, 4);
    StreamUtil::tostream<TL_OUTPUTS*4>(outputs, l2_out_copy);
    //StreamUtil::tostream<TL_OUTPUTS>(labels, label_stream, 4);
    StreamUtil::tostream<TL_OUTPUTS*4>(labels, label_stream);
    Loss::MeanSquaredError_derivative_stream<TL_OUTPUTS, 4>(l2_out_copy, label_stream, l2_deltas_dbg);

    StreamUtil::toarray<TL_OUTPUTS>(l2_deltas_dbg, l2_deltas_array, 4);
    printf("L2 deltas:\n");
    print_mat<4, 6>(l2_deltas_array);
    StreamUtil::tostream<TL_OUTPUTS>(l2_deltas_array, l2_deltas, 4);

    StreamUtil::duplicate<TL_OUTPUTS*4>(l2_deltas, l2_deltas_bw, l2_deltas_up);

    DenseLayerStream::Float::backward<TL_INPUTS, TL_HIDDEN, TL_OUTPUTS, 4, 1, 1, 1, 100>(l1_out_copy, bw_params.weights_l2, l2_deltas_bw, l1_deltas_dbg, LEAKY_RELU, 4);

    StreamUtil::toarray<TL_HIDDEN>(l1_deltas_dbg, l1_deltas_array,4);
    printf("L1 deltas:\n");
    print_mat<4, 8>(l1_deltas_array);
    StreamUtil::tostream<TL_HIDDEN>(l1_deltas_array, l1_deltas,4);

    DenseLayerStream::Float::update<TL_INPUTS, TL_HIDDEN, 4, float, 1, 1, 1>(
        l1_deltas, params.weights_l1, params.biases_l1, input_stream2, 0.01);
    DenseLayerStream::Float::update<TL_HIDDEN, TL_OUTPUTS, 4, float, 1, 1, 1>(
        l2_deltas_up, params.weights_l2, params.biases_l2, l1_out_act_copy, 0.01);

    printf("weights l1\n");
    print_mat<8,10>(params.weights_l1);
    printf("biases l1\n");
    print_mat<8,1>(params.biases_l1);
    printf("weights l2\n");
    print_mat<6,8>(params.weights_l2);
    printf("biases l2\n");
    print_mat<6,1>(params.biases_l2);

    return TEST_PASSED;
}

int main () {
    TwoLayerParams params = {
        .weights_l1 = {
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
            0,1,2,3,4,5,6,7,8,9,
        },

        .biases_l1 = {
            1,2,3,4,5,6,7,8
        },

        .weights_l2 = {
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
            0,1,2,3,4,5,6,7,
        },

        .biases_l2 = {
            1,2,3,4,5,6
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
