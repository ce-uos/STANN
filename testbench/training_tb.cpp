#include <cstdio>
#include <cstdlib>

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../stann.hpp"

#include "weights.h"
#include "weights_after.h"
#include "weights_after_batch.h"
#include "weights_after_batch2.h"
#include "weights_after_batch100.h"

#define TEST_PASSED 0
#define TEST_FAILED (-1)

#define SINGLENET_HIDDEN_NEURONS 128

struct SingleNetParams {
    float weights_l1[SINGLENET_HIDDEN_NEURONS*6];
    float weights_l2[SINGLENET_HIDDEN_NEURONS*SINGLENET_HIDDEN_NEURONS];
    float weights_l3[8*SINGLENET_HIDDEN_NEURONS];
    float biases_l1[SINGLENET_HIDDEN_NEURONS];
    float biases_l2[SINGLENET_HIDDEN_NEURONS];
    float biases_l3[8];
};

void triplicate_params_single(SingleNetParams &params,
                              SingleNetParams &out_params1,
                              SingleNetParams &out_params2) {
  for (int i = 0; i < 6 * SINGLENET_HIDDEN_NEURONS; i++) {
#pragma HLS pipeline II = 3
    out_params1.weights_l1[i] = params.weights_l1[i];
    out_params2.weights_l1[i] = params.weights_l1[i];
  }
  for (int i = 0; i < SINGLENET_HIDDEN_NEURONS * SINGLENET_HIDDEN_NEURONS; i++) {
#pragma HLS pipeline II = 3
    out_params1.weights_l2[i] = params.weights_l2[i];
    out_params2.weights_l2[i] = params.weights_l2[i];
  }
  for (int i = 0; i < SINGLENET_HIDDEN_NEURONS * 8; i++) {
#pragma HLS pipeline II = 3
    out_params1.weights_l3[i] = params.weights_l3[i];
    out_params2.weights_l3[i] = params.weights_l3[i];
  }

  for (int i = 0; i < SINGLENET_HIDDEN_NEURONS; i++) {
#pragma HLS pipeline II = 3
    out_params1.biases_l1[i] = params.biases_l1[i];
    out_params2.biases_l1[i] = params.biases_l1[i];
  }
  for (int i = 0; i < SINGLENET_HIDDEN_NEURONS; i++) {
#pragma HLS pipeline II = 3
    out_params1.biases_l2[i] = params.biases_l2[i];
    out_params2.biases_l2[i] = params.biases_l2[i];
  }
  for (int i = 0; i < 8; i++) {
#pragma HLS pipeline II = 3
    out_params1.biases_l3[i] = params.biases_l3[i];
    out_params2.biases_l3[i] = params.biases_l3[i];
  }
}

namespace SingleNet {

template <int BATCH_SIZE>
void training_stream(float *input, float *labels,
                            SingleNetParams &fw_params,
                            SingleNetParams &bw_params,
                            SingleNetParams &up_params,
                            float learning_rate, int reps) {
#pragma HLS Dataflow
  hls::stream<float> input_stream("input_stream");
  hls::stream<float> input_stream1("input_stream1");
  hls::stream<float> input_stream2("input_stream2");
  hls::stream<float> output_stream("output_stream");

  hls::stream<float> l1_out("l1_out");
  hls::stream<float> l2_out("l2_out");
  hls::stream<float> l3_out("l3_out");

  float l1_out_copy[128 * BATCH_SIZE];
  float l2_out_copy[128 * BATCH_SIZE];
  float l3_out_copy[128 * BATCH_SIZE];

  hls::stream<float> l1_out_relu("l1_out_relu");
  hls::stream<float> l2_out_relu("l2_out_relu");
  hls::stream<float> l3_out_relu("l3_out_relu");

  hls::stream<float> l1_out_relu_copy("l1_out_relu_copy");
  hls::stream<float> l2_out_relu_copy("l2_out_relu_copy");

  StreamUtil::tostream<6>(input, input_stream, reps);
  StreamUtil::duplicate<6 * BATCH_SIZE>(input_stream, input_stream1,
                                        input_stream2);

  DenseLayerStream::Float::forward<6, 128, BATCH_SIZE, 1, 1, 1, 100>(
      input_stream1, fw_params.weights_l1, fw_params.biases_l1, l1_out, NONE,
      reps);
  ActivationLayer::Float::leaky_relu_stream<128, BATCH_SIZE>(l1_out, l1_out_copy, l1_out_relu, l1_out_relu_copy, reps);
  //ActivationLayer::Float::no_activation_stream<128, BATCH_SIZE>(l1_out, l1_out_copy, l1_out_relu, l1_out_relu_copy, reps);
  DenseLayerStream::Float::forward<128, 128, BATCH_SIZE, 1, 1, 1, 100>(
      l1_out_relu, fw_params.weights_l2, fw_params.biases_l2, l2_out, NONE,
      reps);
  ActivationLayer::Float::leaky_relu_stream<128, BATCH_SIZE>(l2_out, l2_out_copy, l2_out_relu, l2_out_relu_copy, reps);
  //ActivationLayer::Float::no_activation_stream<128, BATCH_SIZE>(l2_out, l2_out_copy, l2_out_relu, l2_out_relu_copy, reps);
  DenseLayerStream::Float::forward<128, 8, BATCH_SIZE, 1, 1, 1, 100>(
      l2_out_relu, fw_params.weights_l3, fw_params.biases_l3, l3_out, NONE,
      reps);

  hls::stream<float> l1_deltas("l1_deltas");
  hls::stream<float> l2_deltas("l2_deltas");
  hls::stream<float> l3_deltas("l3_deltas");

  hls::stream<float> l1_deltas_up("l1_deltas_up");
  hls::stream<float> l2_deltas_up("l2_deltas_up");
  hls::stream<float> l3_deltas_up("l3_deltas_up");

  hls::stream<float> l1_deltas_bw("l1_deltas_bw");
  hls::stream<float> l2_deltas_bw("l2_deltas_bw");
  hls::stream<float> l3_deltas_bw("l3_deltas_bw");

  hls::stream<float> label_stream("label_stream");

  hls::stream<float> l1_deltas_dbg("l1_deltas_dbg");
  hls::stream<float> l2_deltas_dbg("l2_deltas_dbg");
  hls::stream<float> l3_deltas_dbg("l3_deltas_dbg");
  float l1_deltas_array[BATCH_SIZE * 128];
  float l2_deltas_array[BATCH_SIZE * 128];
  float l3_deltas_array[BATCH_SIZE * 8];

  StreamUtil::tostream<8>(labels, label_stream, reps);

  Loss::MeanSquaredError_derivative_stream<8, BATCH_SIZE>(
      l3_out, label_stream, l3_deltas_dbg);

  StreamUtil::toarray<BATCH_SIZE*8>(l3_deltas_dbg, l3_deltas_array);
  // printf("L3 deltas:\n");
  // print_mat<8, 1>(l3_deltas_array);
  StreamUtil::tostream<BATCH_SIZE*8>(l3_deltas_array, l3_deltas);

  StreamUtil::duplicate<8 * BATCH_SIZE>(l3_deltas, l3_deltas_bw, l3_deltas_up);
  DenseLayerStream::Float::backward<128, 128, 8, BATCH_SIZE, 1, 1, 1, 100>(l2_out_copy, bw_params.weights_l3, l3_deltas_bw, l2_deltas_dbg, LEAKY_RELU, reps);
  //DenseLayerStream::Float::backward<128, 128, 8, BATCH_SIZE, 1, 1, 1, 100>(l2_out_copy, bw_params.weights_l3, l3_deltas_bw, l2_deltas, NONE, reps);

  StreamUtil::toarray<BATCH_SIZE*128>(l2_deltas_dbg, l2_deltas_array);
  // printf("L2 deltas:\n");
  // print_mat<128, 1>(l2_deltas_array);
  StreamUtil::tostream<BATCH_SIZE*128>(l2_deltas_array, l2_deltas);

  StreamUtil::duplicate<128 * BATCH_SIZE>(l2_deltas, l2_deltas_bw, l2_deltas_up);
  DenseLayerStream::Float::backward<6, 128, 128, BATCH_SIZE, 1, 1, 1, 100>(l1_out_copy, bw_params.weights_l2, l2_deltas_bw, l1_deltas_dbg, LEAKY_RELU, reps);
  //DenseLayerStream::Float::backward<6, 128, 128, BATCH_SIZE, 1, 1, 1, 100>(l1_out_copy, bw_params.weights_l2, l2_deltas_bw, l1_deltas_up, NONE, reps);
  StreamUtil::toarray<BATCH_SIZE*128>(l1_deltas_dbg, l1_deltas_array);
  // printf("L1 deltas:\n");
  // print_mat<128, 1>(l1_deltas_array);
  StreamUtil::tostream<BATCH_SIZE*128>(l1_deltas_array, l1_deltas_up);

  DenseLayerStream::Float::update<6, 128, BATCH_SIZE, float, 1, 1, 1>(
      l1_deltas_up, up_params.weights_l1, up_params.biases_l1, input_stream2,
      learning_rate);
  DenseLayerStream::Float::update<128, 128, BATCH_SIZE, float, 1, 1, 1>(
      l2_deltas_up, up_params.weights_l2, up_params.biases_l2, l1_out_relu_copy,
      learning_rate);
  DenseLayerStream::Float::update<128, 8, BATCH_SIZE, float, 1, 1, 1>(
      l3_deltas_up, up_params.weights_l3, up_params.biases_l3, l2_out_relu_copy,
      learning_rate);
}

template<int BATCH_SIZE>
void forward_stream(float *input, SingleNetParams &params,
                    float *output, int reps) {
#pragma HLS Dataflow

  hls::stream<float> input_stream("input_stream_inference");
  hls::stream<float> output_stream("output_stream_inference");

  hls::stream<float> l1_out("l1_out_inference");
  hls::stream<float> l2_out("l2_out_inference");
  hls::stream<float> l3_out("l3_out_inference");

  hls::stream<float> temp_stream("temp_stream");
  hls::stream<float> temp_stream2("temp_stream2");
  hls::stream<float> temp_stream3("temp_stream3");

  StreamUtil::tostream<6>(input, input_stream, reps);

  DenseLayerStream::Float::forward<6, 128, BATCH_SIZE, 1, 1, 1>(
      input_stream, params.weights_l1, params.biases_l1, l1_out, LEAKY_RELU,
      reps);
  DenseLayerStream::Float::forward<128, 128, BATCH_SIZE, 1, 1, 1>(
      l1_out, params.weights_l2, params.biases_l2, l2_out, LEAKY_RELU, reps);
  DenseLayerStream::Float::forward<128, 8, BATCH_SIZE, 1, 1, 1>(
      l2_out, params.weights_l3, params.biases_l3, output_stream, NONE, reps);

  StreamUtil::toarray<8>(output_stream, output, reps);
}

template<int BATCH_SIZE>
void forward_stream_split(float *input, SingleNetParams &params,
                    float *output, int reps) {

  hls::stream<float> input_stream("input_stream_inference");
  hls::stream<float> output_stream("output_stream_inference");

  hls::stream<float> l1_out("l1_out_inference");
  hls::stream<float> l2_out("l2_out_inference");
  hls::stream<float> l3_out("l3_out_inference");

  float l1_out_copy[128];
  float l2_out_copy[128];
  float l3_out_copy[128];

  hls::stream<float> l1_out_relu("l1_out_relu");
  hls::stream<float> l2_out_relu("l2_out_relu");
  hls::stream<float> l3_out_relu("l3_out_relu");

  hls::stream<float> l1_out_relu_copy("l1_out_relu_copy");
  hls::stream<float> l2_out_relu_copy("l2_out_relu_copy");
  hls::stream<float> l3_out_copy1("l3_out_copy1");
  hls::stream<float> l3_out_copy2("l3_out_copy2");

  StreamUtil::tostream<6>(input, input_stream, reps);

  DenseLayerStream::Float::forward<6, 128, BATCH_SIZE, 1, 1, 1, 100>(
      input_stream, params.weights_l1, params.biases_l1, l1_out, NONE,
      reps);
  ActivationLayer::Float::leaky_relu_stream<128, 1>(
      l1_out, l1_out_copy, l1_out_relu, l1_out_relu_copy, reps);
  DenseLayerStream::Float::forward<128, 128, BATCH_SIZE, 1, 1, 1, 100>(
      l1_out_relu, params.weights_l2, params.biases_l2, l2_out, NONE,
      reps);
  ActivationLayer::Float::leaky_relu_stream<128, 1>(
      l2_out, l2_out_copy, l2_out_relu, l2_out_relu_copy, reps);
  DenseLayerStream::Float::forward<128, 8, BATCH_SIZE, 1, 1, 1, 100>(
      l2_out_relu, params.weights_l3, params.biases_l3, l3_out, NONE,
      reps);

  StreamUtil::toarray<8>(l3_out, output, reps);

}

} // namespace SingleNet

int test_inference(SingleNetParams &params) {
    float input[6] = {
        1,2,3,4,5,6,
    };

    float output[8];
    float expected_output[8] = { 0.1020, -0.1010, -0.1131,  0.8706, -0.2712, -0.1622, -0.1986,  0.4110 };
    
    SingleNet::forward_stream<1>(input, params, output, 1);

    for (int i = 0; i < 8; i++) {
        if (fabs(output[i] - expected_output[i]) > 0.001) {
            print_mat<8,1>(output);
            print_mat<8,1>(expected_output);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_inference_split(SingleNetParams &params) {
    float input[6] = {
        1,2,3,4,5,6,
    };

    float output[8];
    float expected_output[8] = { 0.1020, -0.1010, -0.1131,  0.8706, -0.2712, -0.1622, -0.1986,  0.4110 };
    
    SingleNet::forward_stream_split<1>(input, params, output, 1);

    for (int i = 0; i < 8; i++) {
        if (fabs(output[i] - expected_output[i]) > 0.001) {
            print_mat<8,1>(output);
            print_mat<8,1>(expected_output);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_loss(SingleNetParams &params) {
    float input[6] = {
        1,2,3,4,5,6,
    };

    float output[8];
    float expected_output[8] = { 0.1020, -0.1010, -0.1131,  0.8706, -0.2712, -0.1622, -0.1986,  0.4110 };
    
    SingleNet::forward_stream<1>(input, params, output, 1);

    print_mat<8,1>(output);
    print_mat<8,1>(expected_output);

    for (int i = 0; i < 8; i++) {
        if (fabs(output[i] - expected_output[i]) > 0.001) {
            return TEST_FAILED;
        }
    }

    float labels[8] = {
        1,2,3,4,5,6,7,8,
    };

    float mseloss = 0;
    for (int i = 0; i < 8; i++) {
        mseloss += (output[i] - labels[i]) * (output[i] - labels[i]);
    }
    mseloss /= 8;

    float expected_loss = 24.9844;
    if (fabs(mseloss - expected_loss) > 0.001) {
        printf("mseloss %f, expected %f\n", mseloss, expected_loss);
        return TEST_FAILED;
    }

    hls::stream<float> label_stream;
    StreamUtil::tostream<8>(labels, label_stream, 1);

    hls::stream<float> output_stream;
    StreamUtil::tostream<8>(output, output_stream, 1);

    hls::stream<float> l3_delta_stream;

    Loss::MeanSquaredError_derivative_stream<8, 1>(
        output_stream, label_stream, l3_delta_stream);

    float l3_deltas[8];
    StreamUtil::toarray<8>(l3_delta_stream, l3_deltas);

    float expected_deltas[8] = { -0.2245, -0.5252, -0.7783, -0.7823, -1.3178, -1.5405, -1.7996, -1.8973  };

    for (int i = 0; i < 8; i++) {
        if (fabs(l3_deltas[i] - expected_deltas[i]) > 0.001) {
            print_mat<8,1>(l3_deltas);
            print_mat<8,1>(expected_deltas);
            return TEST_FAILED;
        }
    }

    print_mat<8,1>(l3_deltas);
    print_mat<8,1>(expected_deltas);

    return TEST_PASSED;
}

int test_loss_batch(SingleNetParams &params) {
    // input: IxB
    float input[6 * 32];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 32; j++) {
            input[i * 32 + j] = i + 1;
        }
    }

    float output[8 * 32];
    float expected_output_bs1[8] = { 0.1020, -0.1010, -0.1131,  0.8706, -0.2712, -0.1622, -0.1986,  0.4110 };
    float expected_output[8 * 32];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            expected_output[i * 32 + j] = expected_output_bs1[i];
        }
    }
    
    SingleNet::forward_stream<32>(input, params, output, 32);

    print_mat<8,1>(output);
    print_mat<8,1>(expected_output);
    for (int i = 0; i < 8 * 32; i++) {
        if (fabs(output[i] - expected_output[i]) > 0.0001) {
            return TEST_FAILED;
        }
    }

    float labels[8 * 32];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            labels[i * 32 + j] = i+1;
        }
    }

    float mseloss = 0;
    float losses[8 * 32];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            losses[i*32+j] = (output[i*32+j] - labels[i*32+j]) * (output[i*32+j] - labels[i*32+j]);
        }
    }
    //print_mat<8,32>(losses);
    for (int i = 0; i < 8; i++) {
        mseloss += losses[i*32];
    }
    mseloss /= 8;
    printf("mseloss %f\n", mseloss);

    float expected_loss = 24.9844;
    if (fabs(mseloss - expected_loss) > 0.001) {
        printf("mseloss %f, expected %f\n", mseloss, expected_loss);
        return TEST_FAILED;
    }

    hls::stream<float> label_stream("label stream");
    StreamUtil::tostream<8>(labels, label_stream, 32);

    hls::stream<float> output_stream("output stream");
    StreamUtil::tostream<8>(output, output_stream, 32);

    hls::stream<float> l3_delta_stream("delta stream");

    Loss::MeanSquaredError_derivative_stream<8, 32>(
        output_stream, label_stream, l3_delta_stream);

    float l3_deltas[8 * 32];
    StreamUtil::toarray<8>(l3_delta_stream, l3_deltas, 32);

    //print_mat<8, 32>(l3_deltas);

    float expected_deltas_bs1[8] = { -0.2245, -0.5252, -0.7783, -0.7823, -1.3178, -1.5405, -1.7996, -1.8973  };
    float expected_deltas[8*32];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            expected_deltas[j * 8 + i] = expected_deltas_bs1[i];
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            if (fabs(l3_deltas[i*32+j] - expected_deltas[j*8+i]) > 0.001) {
                printf("deltas\n");
                print_mat<8,1>(l3_deltas);
                printf("expected deltas\n");
                print_mat<8,1>(expected_deltas);
                return TEST_FAILED;
            }
        }
    }

    //print_mat<8,1>(l3_deltas);
    //print_mat<8,1>(expected_deltas);

    return TEST_PASSED;
}
int test_weights() {
    SingleNetParams params = {
        .weights_l1 = {L1W},
        .weights_l2 = {L2W},
        .weights_l3 = {L3W},
        .biases_l1 = {L1B},
        .biases_l2 = {L2B},
        .biases_l3 = {L3B},
    };
    float input[6] = {
        1,2,3,4,5,6,
    };
    float labels[8] = {
        1,2,3,4,5,6,7,8
    };

    SingleNetParams fw_params;
    SingleNetParams bw_params;
    triplicate_params_single(params, fw_params, bw_params);

    SingleNet::training_stream<1>(input, labels, fw_params, bw_params, params, 0.01, 1);

    SingleNetParams expected_params = {
        .weights_l1 = {L1WA},
        .weights_l2 = {L2WA},
        .weights_l3 = {L3WA},
        .biases_l1 = {L1BA},
        .biases_l2 = {L2BA},
        .biases_l3 = {L3BA},
    };

    for (int i = 0; i < 128*8; i++) {
        if (fabs(params.weights_l3[i] - expected_params.weights_l3[i]) > 0.001) {
            printf("Weights L3 don't match:\n");
            print_mat<10,1>(params.weights_l3);
            print_mat<10,1>(expected_params.weights_l3);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 128*128; i++) {
        if (fabs(params.weights_l2[i] - expected_params.weights_l2[i]) > 0.001) {
            printf("Weights L2 don't match:\n");
            print_mat<10,1>(params.weights_l2);
            print_mat<10,1>(expected_params.weights_l2);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (fabs(params.biases_l3[i] - expected_params.biases_l3[i]) > 0.001) {
            printf("Biases L3 don't match:\n");
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }

    print_mat<8,1>(params.weights_l3);
    print_mat<8,1>(expected_params.weights_l3);
    print_mat<8,1>(params.biases_l3);
    print_mat<8,1>(expected_params.biases_l3);
    return TEST_PASSED;
}

int test_weights_batch() {
    const int TEST_BATCH = 32;
    SingleNetParams params = {
        .weights_l1 = {L1W},
        .weights_l2 = {L2W},
        .weights_l3 = {L3W},
        .biases_l1 = {L1B},
        .biases_l2 = {L2B},
        .biases_l3 = {L3B},
    };
    float input[6*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 6; j++) {
            //input[i * 6 + j] = j + 1;
            input[j * TEST_BATCH + i] = j + 1;
        }
    }
    float labels[8*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 8; j++) {
            //labels[i * 8 + j] = j + 1;
            labels[j * TEST_BATCH + i] = j + 1;
        }
    }

    SingleNetParams fw_params;
    SingleNetParams bw_params;
    triplicate_params_single(params, fw_params, bw_params);

    SingleNet::training_stream<TEST_BATCH>(input, labels, fw_params, bw_params, params, 0.01, TEST_BATCH);

    SingleNetParams expected_params = {
        .weights_l1 = {L1WAB},
        .weights_l2 = {L2WAB},
        .weights_l3 = {L3WAB},
        .biases_l1 = {L1BAB},
        .biases_l2 = {L2BAB},
        .biases_l3 = {L3BAB},
    };

    //print_mat<8,1>(fw_params.biases_l3);
    //print_mat<8,1>(expected_params.biases_l3);
    //print_mat<8,1>(params.biases_l3);

    for (int i = 0; i < 128*8; i++) {
        if (fabs(params.weights_l3[i] - expected_params.weights_l3[i]) > 0.0001) {
            printf("Weights L3 don't match:\n");
            print_mat<10,1>(params.weights_l3);
            print_mat<10,1>(expected_params.weights_l3);
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }
    printf("Weights L3 match!\n");
    print_mat<10,1>(params.weights_l3);
    print_mat<10,1>(expected_params.weights_l3);
    for (int i = 0; i < 128*128; i++) {
        if (fabs(params.weights_l2[i] - expected_params.weights_l2[i]) > 0.0001) {
            printf("Weights L2 don't match:\n");
            print_mat<10,1>(params.weights_l2);
            print_mat<10,1>(expected_params.weights_l2);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (fabs(params.biases_l3[i] - expected_params.biases_l3[i]) > 0.0001) {
            printf("Biases don't match:\n");
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }

    print_mat<10,1>(params.weights_l3);
    print_mat<10,1>(params.weights_l2);
    print_mat<10,1>(params.weights_l1);

    return TEST_PASSED;
}

int test_weights_batch_twice() {
    const int TEST_BATCH = 32;
    SingleNetParams params = {
        .weights_l1 = {L1W},
        .weights_l2 = {L2W},
        .weights_l3 = {L3W},
        .biases_l1 = {L1B},
        .biases_l2 = {L2B},
        .biases_l3 = {L3B},
    };
    float input[6*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 6; j++) {
            //input[i * 6 + j] = j + 1;
            input[j * TEST_BATCH + i] = j + 1;
        }
    }
    float labels[8*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 8; j++) {
            //labels[i * 8 + j] = j + 1;
            labels[j * TEST_BATCH + i] = j + 1;
        }
    }

    for (int i = 0; i < 2; i++) {
        SingleNetParams fw_params;
        SingleNetParams bw_params;
        triplicate_params_single(params, fw_params, bw_params);

        SingleNet::training_stream<32>(input, labels, fw_params, bw_params, params, 0.01, 32);
    }

    SingleNetParams expected_params = {
        .weights_l1 = {L1WAB2},
        .weights_l2 = {L2WAB2},
        .weights_l3 = {L3WAB2},
        .biases_l1 = {L1BAB2},
        .biases_l2 = {L2BAB2},
        .biases_l3 = {L3BAB2},
    };

    print_mat<8,1>(expected_params.biases_l3);
    print_mat<8,1>(params.biases_l3);

    for (int i = 0; i < 128*8; i++) {
        if (fabs(params.weights_l3[i] - expected_params.weights_l3[i]) > 0.0001) {
            printf("Weights L3 don't match:\n");
            print_mat<10,1>(params.weights_l3);
            print_mat<10,1>(expected_params.weights_l3);
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 128*128; i++) {
        if (fabs(params.weights_l2[i] - expected_params.weights_l2[i]) > 0.0001) {
            printf("Weights L2 don't match:\n");
            print_mat<10,1>(params.weights_l2);
            print_mat<10,1>(expected_params.weights_l2);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (fabs(params.biases_l3[i] - expected_params.biases_l3[i]) > 0.0001) {
            printf("Biases don't match:\n");
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_weights_batch_hundred() {
    const int TEST_BATCH = 32;
    SingleNetParams params = {
        .weights_l1 = {L1W},
        .weights_l2 = {L2W},
        .weights_l3 = {L3W},
        .biases_l1 = {L1B},
        .biases_l2 = {L2B},
        .biases_l3 = {L3B},
    };
    float input[6*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 6; j++) {
            //input[i * 6 + j] = j + 1;
            input[j * TEST_BATCH + i] = j + 1;
        }
    }
    float labels[8*TEST_BATCH];
    for (int i = 0; i < TEST_BATCH; i++) {
        for (int j = 0; j < 8; j++) {
            //labels[i * 8 + j] = j + 1;
            labels[j * TEST_BATCH + i] = j + 1;
        }
    }

    for (int i = 0; i < 100; i++) {
        SingleNetParams fw_params;
        SingleNetParams bw_params;
        triplicate_params_single(params, fw_params, bw_params);

        SingleNet::training_stream<32>(input, labels, fw_params, bw_params, params, 0.01, 32);
    }

    SingleNetParams expected_params = {
        .weights_l1 = {L1WABX},
        .weights_l2 = {L2WABX},
        .weights_l3 = {L3WABX},
        .biases_l1 = {L1BABX},
        .biases_l2 = {L2BABX},
        .biases_l3 = {L3BABX},
    };

    print_mat<8,1>(expected_params.biases_l3);
    print_mat<8,1>(params.biases_l3);

    for (int i = 0; i < 128*8; i++) {
        if (fabs(params.weights_l3[i] - expected_params.weights_l3[i]) > 0.0001) {
            printf("Weights L3 don't match:\n");
            print_mat<10,1>(params.weights_l3);
            print_mat<10,1>(expected_params.weights_l3);
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 128*128; i++) {
        if (fabs(params.weights_l2[i] - expected_params.weights_l2[i]) > 0.0001) {
            printf("Weights L2 don't match:\n");
            print_mat<10,1>(params.weights_l2);
            print_mat<10,1>(expected_params.weights_l2);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (fabs(params.biases_l3[i] - expected_params.biases_l3[i]) > 0.0001) {
            printf("Biases don't match:\n");
            print_mat<8,1>(params.biases_l3);
            print_mat<8,1>(expected_params.biases_l3);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int main(int argc, const char *argv[])
{
    SingleNetParams params = {
        .weights_l1 = {L1W},
        .weights_l2 = {L2W},
        .weights_l3 = {L3W},
        .biases_l1 = {L1B},
        .biases_l2 = {L2B},
        .biases_l3 = {L3B},
    };

    printf("\nStart test inference...\n");
    if (test_inference(params) == TEST_PASSED) {
        printf("Inference test PASSED\n");
    } else {
        printf("Inference test FAILED\n");
    }

    printf("\nStart test inference split...\n");
    if (test_inference_split(params) == TEST_PASSED) {
        printf("Note: data left over in stream is intended.\n");
        printf("Inference split test PASSED\n");
    } else {
        printf("Inference split test FAILED\n");
    }

    printf("\nStart test mse loss...\n");
    if (test_loss(params) == TEST_PASSED) {
        printf("MSE loss test PASSED\n");
    } else {
        printf("MSE loss test FAILED\n");
    }

    printf("\nStart test mse loss batch...\n");
    if (test_loss_batch(params) == TEST_PASSED) {
        printf("MSE loss batch test PASSED\n");
    } else {
        printf("MSE loss batch test FAILED\n");
    }

    printf("\nStart test training...\n");
    if (test_weights() == TEST_PASSED) {
        printf("New weights test PASSED\n");
    } else {
        printf("New weights test FAILED\n");
    }

    printf("\nStart test training batch...\n");
    if (test_weights_batch() == TEST_PASSED) {
        printf("New weights batch test PASSED\n");
    } else {
        printf("New weights batch test FAILED\n");
    }

    printf("\nStart test training two batches...\n");
    if (test_weights_batch_twice() == TEST_PASSED) {
        printf("New weights batch twice test PASSED\n");
    } else {
        printf("New weights batch twice test FAILED\n");
    }

    // printf("\nStart test training one hundred batches...\n");
    // if (test_weights_batch_hundred() == TEST_PASSED) {
    //     printf("New weights batch hundred test PASSED\n");
    // } else {
    //     printf("New weights batch hundred test FAILED\n");
    // }
    return 0;
}
