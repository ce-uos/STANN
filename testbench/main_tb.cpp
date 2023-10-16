#define STANN_UNIT_TEST

#include <cstdio>
#include <cstdlib>

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "../stann_testable.hpp"

#define TEST_PASSED 0
#define TEST_FAILED (-1)

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

template<int M, int N>
void print_mat(float *mat) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            printf("%f ", mat[m * N + n]);
        }
        printf("\n");
    }
}

int test_matmul_square() {

    float a[16] = {
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
    };

    float b[16] = {
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
    };

    float out0[16];
    float out1[16];
    float out2[16];
    float baseline[16];

    for (int i = 0; i < 16; i++) {
        baseline[i] = 0;
        out0[i] = 0;
        out1[i] = 0;
        out2[i] = 0;
    }

    simple_matmul<4,4,4>(a, b, baseline);

    stann_sysarr_blockmatmul_4_4_4_1_1_1(a,b,out0);
    stann_sysarr_blockmatmul_4_4_4_1_1_1(a,b,out1);
    stann_sysarr_blockmatmul_4_4_4_1_1_1(a,b,out2);

    for (int i = 0; i < 16; i++) {
        if (baseline[i] != out0[i]) {
            printf("out0 != baseline\n");
            print_mat<4,4>(baseline);
            print_mat<4,4>(out0);
            return TEST_FAILED;
        }
        if (baseline[i] != out1[i]) {
            printf("out1 != baseline\n");
            print_mat<4,4>(baseline);
            print_mat<4,4>(out1);
            return TEST_FAILED;
        }
        if (baseline[i] != out2[i]) {
            printf("out2 != baseline\n");
            print_mat<4,4>(baseline);
            print_mat<4,4>(out2);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_matmul_nonsquare() {

    float a[64] = {
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
    };

    float b[64] = {
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
    };

    float b2[8] = {
        1,1,1,1,1,1,1,1,
    };

    float out0[64];
    float out1[64];
    float out2[64];
    float out3[8];
    float baseline[64];
    float baseline2[8];

    for (int i = 0; i < 64; i++) {
        baseline[i] = 0;
        out0[i] = 0;
        out1[i] = 0;
        out2[i] = 0;
    }

    for (int i = 0; i < 8; i++) {
        baseline2[i] = 0;
        out3[i] = 0;
    }

    simple_matmul<8,8,8>(a, b, baseline);
    simple_matmul<8,8,1>(a, b2, baseline2);

    stann_sysarr_blockmatmul_8_8_8_4_2_4(a,b,out0);
    stann_sysarr_blockmatmul_8_8_8_1_8_1(a,b,out1);
    stann_sysarr_blockmatmul_8_8_8_8_4_2(a,b,out2);
    stann_sysarr_blockmatmul_8_8_1_2_4_1(a,b2,out3);

    for (int i = 0; i < 64; i++) {
        if (baseline[i] != out0[i]) {
            printf("out0 != baseline\n");
            return TEST_FAILED;
        }
        if (baseline[i] != out1[i]) {
            printf("out1 != baseline\n");
            return TEST_FAILED;
        }
        if (baseline[i] != out2[i]) {
            printf("out2 != baseline\n");
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (baseline2[i] != out3[i]) {
            printf("out3 != baseline\n");
            print_mat<8,1>(baseline2);
            print_mat<8,1>(out3);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_matmul_stream() {

    float a[16] = {
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
    };

    float b[4] = {
        1,1,1,1,
    };

    float a2[64] = {
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
    };

    float b2[8] = {
        1,1,1,1,1,1,1,1,
    };


    float out0[4];
    float out1[4];
    float out2[8];
    float baseline[4];
    float baseline2[8];

    for (int i = 0; i < 4; i++) {
        out0[i] = 0;
        out1[i] = 0;
        baseline[i] = 0;
    }

    for (int i = 0; i < 8; i++) {
        out2[i] = 0;
        baseline2[i] = 0;
    }

    simple_matmul<4,4,1>(a, b, baseline);
    simple_matmul<8,8,1>(a2, b2, baseline2);

    stann_stream_blockmatmul_4_4_1_1(a,b,out0);
    stann_stream_blockmatmul_4_4_4_4(a,b,out1);
    stann_stream_blockmatmul_8_8_2_4(a2,b2,out2);

    for (int i = 0; i < 4; i++) {
        if (baseline[i] != out0[i]) {
            printf("out0 != baseline\n");
            print_mat<4,1>(baseline);
            print_mat<4,1>(out0);
            return TEST_FAILED;
        }
        if (baseline[i] != out1[i]) {
            printf("out1 != baseline\n");
            print_mat<4,1>(baseline);
            print_mat<4,1>(out1);
            return TEST_FAILED;
        }
    }
    for (int i = 0; i < 8; i++) {
        if (baseline2[i] != out2[i]) {
            printf("out2 != baseline\n");
            print_mat<8,1>(baseline);
            print_mat<8,1>(out1);
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_streamutils() {
    float data[32] = {
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
    };

    hls::stream<float> data_stream;
    hls::stream<float> data_stream2;

    float data_array[32];
    float data_array2[32];

    stann_streamutils_tostream_32(data, data_stream);
    stann_streamutils_toarray_32(data_stream, data_array);

    stann_streamutils_tostream_32_reps(data, data_stream2);
    stann_streamutils_toarray_32_reps(data_stream2, data_array2);

    for (int i = 0; i < 32; i++) {
        if (data[i] != data_array[i]) {
            return TEST_FAILED;
        }
    }

    for (int i = 0; i < 32; i++) {
        if (data[i] != data_array2[i]) {
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int test_denselayer_forward() {
    float input[8] = {
        1,2,3,4,5,6,7,8
    };

    float weights[64] = {
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
        1,2,3,4,5,6,7,8,
    };

    float biases[8] = {
        1,2,3,4,5,6,7,8
    };

    float outputs0[8];
    float outputs1[8];
    float outputs2[8];
    float baseline[8];

    for (int i = 0; i < 8; i++) {
        outputs0[i] = 0;
        outputs1[i] = 0;
        outputs2[i] = 0;
        baseline[i] = 0;
    }

    simple_matmul<8,8,1>(weights, input, baseline);
    for (int i = 0; i < 8; i++) {
        baseline[i] += biases[i];
    }

    stann_denselayer_forward_1_1(input, weights, biases, outputs0);
    stann_denselayer_forward_8_8(input, weights, biases, outputs1);
    stann_denselayer_forward_8_4(input, weights, biases, outputs2);

    for (int i = 0; i < 8; i++) {
        if (outputs0[i] != baseline[i]) {
            return TEST_FAILED;
        }
        if (outputs1[i] != baseline[i]) {
            return TEST_FAILED;
        }
        if (outputs2[i] != baseline[i]) {
            return TEST_FAILED;
        }
    }

    return TEST_PASSED;
}

int main(int argc, const char *argv[])
{
    if (test_matmul_square() == TEST_PASSED) {
        printf("Matmul test square PASSED\n");
    } else {
        printf("Matmul test square FAILED\n");
    }

    if (test_matmul_nonsquare() == TEST_PASSED) {
        printf("Matmul test nonsquare PASSED\n");
    } else {
        printf("Matmul test nonsquare FAILED\n");
    }

    if (test_matmul_stream() == TEST_PASSED) {
        printf("Matmul test stream PASSED\n");
    } else {
        printf("Matmul test stream FAILED\n");
    }

    if (test_streamutils() == TEST_PASSED) {
        printf("StreamUtil test PASSED\n");
    } else {
        printf("StreamUtil test FAILED\n");
    }

    if (test_denselayer_forward() == TEST_PASSED) {
        printf("DenseLayerStream Forward test PASSED\n");
    } else {
        printf("DenseLayerStream Forward test FAILED\n");
    }

    return 0;
}
