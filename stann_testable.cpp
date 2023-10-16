#include "stann.hpp"
#include "stann_testable.hpp"

extern "C" {

void stann_sysarr_blockmatmul_4_4_4_1_1_1(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<4,4,4,1,1,1>(a, b, c);
}

void stann_sysarr_blockmatmul_4_4_4_2_2_2(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<4,4,4,2,2,2>(a, b, c);
}

void stann_sysarr_blockmatmul_4_4_4_4_4_4(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<4,4,4,4,4,4>(a, b, c);
}

void stann_sysarr_blockmatmul_8_8_8_4_2_4(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<8,8,8,4,2,4>(a, b, c);
}

void stann_sysarr_blockmatmul_8_8_8_1_8_1(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<8,8,8,1,8,1>(a, b, c);
}

void stann_sysarr_blockmatmul_8_8_8_8_4_2(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<8,8,8,8,4,2>(a, b, c);
}

void stann_sysarr_blockmatmul_8_8_1_2_4_1(float *a, float *b, float *c) {
    MatrixUtil::SysArr::blockmatmul<8,8,1,2,4,1>(a, b, c);
}

void stann_stream_blockmatmul_4_4_1_1(float *a, float *b, float *c) {
    hls::stream<float> b_stream("b_stream0");
    hls::stream<float> c_stream("c_stream0");
    StreamUtil::tostream<4, float>(b, b_stream);
    MatrixUtilStream::blockmatmul<4,4,1,1,float>(a, b_stream, c_stream, 1);
    StreamUtil::toarray<4, float>(c_stream, c);
}

void stann_stream_blockmatmul_4_4_4_4(float *a, float *b, float *c) {
    hls::stream<float> b_stream("b_stream1");
    hls::stream<float> c_stream("c_stream1");
    StreamUtil::tostream<4, float>(b, b_stream);
    MatrixUtilStream::blockmatmul<4,4,4,4,float>(a, b_stream, c_stream, 1);
    StreamUtil::toarray<4, float>(c_stream, c);
}

void stann_stream_blockmatmul_8_8_2_4(float *a, float *b, float *c) {
    hls::stream<float> b_stream("b_stream2");
    hls::stream<float> c_stream("c_stream2");
    StreamUtil::tostream<8, float>(b, b_stream);
    MatrixUtilStream::blockmatmul<8,8,2,4,float>(a, b_stream, c_stream, 1);
    StreamUtil::toarray<8, float>(c_stream, c);
}

void stann_streamutils_tostream_32(float *input, hls::stream<float> &output) {
    StreamUtil::tostream<32, float>(input, output);
}

void stann_streamutils_toarray_32(hls::stream<float> &input, float *output) {
    StreamUtil::toarray<32, float>(input, output);
}

void stann_streamutils_tostream_32_reps(float *input, hls::stream<float> &output) {
    StreamUtil::tostream<8, float>(input, output, 4);
}

void stann_streamutils_toarray_32_reps(hls::stream<float> &input, float *output) {
    StreamUtil::toarray<8, float>(input, output, 4);
}

void stann_denselayer_forward_1_1(float *input, float *weights, float *biases, float *output) {
    hls::stream<float> in_stream("in_stream");
    hls::stream<float> out_stream("out_stream");
    StreamUtil::tostream<8, float>(input, in_stream);
    DenseLayerStream::Float::forward<8,8,1,1>(in_stream, weights, biases, out_stream, NONE, 1);
    StreamUtil::toarray<8, float>(out_stream, output);
}

void stann_denselayer_forward_8_8(float *input, float *weights, float *biases, float *output) {
    hls::stream<float> in_stream("in_stream");
    hls::stream<float> out_stream("out_stream");
    StreamUtil::tostream<8, float>(input, in_stream);
    DenseLayerStream::Float::forward<8,8,8,8>(in_stream, weights, biases, out_stream, NONE, 1);
    StreamUtil::toarray<8, float>(out_stream, output);
}

void stann_denselayer_forward_8_4(float *input, float *weights, float *biases, float *output) {
    hls::stream<float> in_stream("in_stream");
    hls::stream<float> out_stream("out_stream");
    StreamUtil::tostream<8, float>(input, in_stream);
    DenseLayerStream::Float::forward<8,8,8,4>(in_stream, weights, biases, out_stream, NONE, 1);
    StreamUtil::toarray<8, float>(out_stream, output);
}

}

