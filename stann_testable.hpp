#ifndef __STANN_TESTABLE__
#define __STANN_TESTABLE__



extern "C" {

void stann_sysarr_blockmatmul_4_4_4_1_1_1(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_4_4_4_2_2_2(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_4_4_4_4_4_4(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_8_8_8_4_2_4(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_8_8_8_1_8_1(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_8_8_8_8_4_2(float *a, float *b, float *c);

void stann_sysarr_blockmatmul_8_8_1_2_4_1(float *a, float *b, float *c);

void stann_stream_blockmatmul_4_4_1_1(float *a, float *b, float *c);

void stann_stream_blockmatmul_4_4_4_4(float *a, float *b, float *c);

void stann_stream_blockmatmul_8_8_2_4(float *a, float *b, float *c);

void stann_streamutils_tostream_32(float *input, hls::stream<float> &output);

void stann_streamutils_toarray_32(hls::stream<float> &input, float *output);

void stann_streamutils_tostream_32_reps(float *input, hls::stream<float> &output);

void stann_streamutils_toarray_32_reps(hls::stream<float> &input, float *output);

void stann_denselayer_forward_1_1(float *input, float *weights, float *biases, float *output);

void stann_denselayer_forward_8_8(float *input, float *weights, float *biases, float *output);

void stann_denselayer_forward_8_4(float *input, float *weights, float *biases, float *output);


}


#endif
