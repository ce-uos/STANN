#ifndef __STANN_CONV_IM2ROW_HPP__
#define __STANN_CONV_IM2ROW_HPP__

#include "stann.hpp"

/**
 * Namespace for conolutional layers.
 */
namespace ConvLayer {

/**
 * im2row implementation of convolutional layer.
 */
namespace im2row {

/**
 * Anonymous namespace for internal functions.
 */
namespace {

/**
 * Adding the bias values after the main convolution.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  OUTPUT_CHANNELS     number of output channels (equal to number of input channels)
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 *
 * @param[in]   input           input stream
 * @param{in]   biases          biases
 * @param[out]  output          output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int OUTPUT_CHANNELS, int KERNEL_SIZE>
void add_bias(hls::stream<float> &input, float *biases, hls::stream<float> &output, int reps) {
    const int KSUB = (KERNEL_SIZE / 2) * 2;
    for (int r = 0; r < reps; r++) {
        for (int m = 0; m < OUTPUT_CHANNELS; m++) {
            for (int i = 0; i < (INPUT_WIDTH-KSUB) * (INPUT_HEIGHT-KSUB); i++) {
                float val = input.read();
                val += biases[m];
                output.write(val);
            }
        }
    }

}

/**
 * Applying activation functions.
 *
 * @tparam  DIM     size of inputs and outputs
 *
 * @param[in]    input      input stream
 * @param[out]   output     output stream
 * @param[in]    act        activation constant
 * @param[in]    reps       number of repetitions
 */
template<int DIM>
void apply_activation_float(hls::stream<float> &input, hls::stream<float> &output, activation_t act, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < DIM; i++) {
            float val = input.read();
            float out_val = val;
            if (act == LEAKY_RELU) {
                out_val = Activation::leaky_relu_simple(val);
            } else if (act == RELU) {
                out_val = Activation::relu_simple(val);
            } else if (act == LIN_TANH) {
                out_val = Activation::lin_tanh_simple(val);
            }
            output.write(out_val);
        }
    }
}

/**
 * Applying activation functions. Quantized to uint8.
 *
 * @tparam  DIM     size of inputs and outputs
 *
 * @param[in]    input      input stream
 * @param[out]   output     output stream
 * @param[in]    act        activation constant
 * @param[in]    reps       number of repetitions
 */
template<int DIM>
void apply_activation_quantized(hls::stream<ap_uint<8>> &input, hls::stream<ap_uint<8>> &output, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < DIM; i++) {
            ap_uint<8> val = input.read();
            ap_uint<8> out_val = val;
            ap_uint<8> lower = 10;
            if (out_val < lower) {
                out_val = 0;
            }
            output.write(out_val);
        }
    }
}

/**
 * im2row preparation function for the input.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  FILTER_SIZE         size of kernel (NxN)
 *
 * @param[in]   input           input stream
 * @param[out]  output          output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL, int FILTER_SIZE, typename T>
void im2row(hls::stream<T> &input, hls::stream<T> &output, int reps) {
    const int OUTPUT_WI = INPUT_WI - FILTER_SIZE + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_SIZE + 1;

    T input_buffer[INPUT_WI*INPUT_HI*INPUT_CHANNEL];
    T output_buffer[FILTER_SIZE*FILTER_SIZE*INPUT_CHANNEL*OUTPUT_WI*OUTPUT_HI];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<INPUT_WI*INPUT_HI*INPUT_CHANNEL, T>(input, input_buffer, 1);

        for (int ic = 0; ic < INPUT_CHANNEL; ic++){
            for (int oh = 0; oh < OUTPUT_HI; oh++){
                for (int ow = 0; ow < OUTPUT_WI; ow++){
                    for (int fw = 0; fw < FILTER_SIZE; fw++){
                        for (int fh = 0; fh < FILTER_SIZE; fh++){
                        #pragma HLS pipeline II=3
                            output_buffer[ow +  oh * OUTPUT_WI + fw * OUTPUT_HI * OUTPUT_WI + fh*OUTPUT_HI * OUTPUT_WI*FILTER_SIZE + ic *OUTPUT_HI * OUTPUT_WI*FILTER_SIZE*FILTER_SIZE ]
                                = input_buffer[ow + oh + oh * OUTPUT_WI + fw + fh * INPUT_WI + ic * INPUT_WI * INPUT_HI ];

                        }
                    }
                }
            }
        }

        StreamUtil::tostream<FILTER_SIZE*FILTER_SIZE*INPUT_CHANNEL*OUTPUT_WI*OUTPUT_HI, T>(output_buffer, output, 1);
    }
}

/**
 * Internal base function for convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param{in]   kernel          kernel weights
 * @param[out]  output          output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_base(hls::stream<float> &input, float *kernel, hls::stream<float> &output, int reps) {
    const int OUTPUT_WI = INPUT_WIDTH - KERNEL_SIZE + 1;
    const int OUTPUT_HI = INPUT_HEIGHT - KERNEL_SIZE + 1;

    float input_buffer[KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS*OUTPUT_WI*OUTPUT_HI];
    float output_buffer[OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNELS];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS*OUTPUT_WI*OUTPUT_HI>(input, input_buffer, 1);

        Matrix::blockmatmul<OUTPUT_CHANNELS,KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS,OUTPUT_WI*OUTPUT_HI,PE1,PE2,PE3,float,80>(kernel, input_buffer, output_buffer);

        StreamUtil::tostream<OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNELS>(output_buffer, output, 1);

    }

}

/**
 * Internal base function for convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param{in]   kernel          kernel weights
 * @param{in]   biases          biases
 * @param{in]   m               constant for quantized computations
 * @param{in]   n               constant for quantized computations
 * @param{in]   z1              constant for quantized computations
 * @param{in]   z2              constant for quantized computations
 * @param{in]   z3              constant for quantized computations
 * @param[out]  output          output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_base_quantized(hls::stream<ap_uint<8>> &input, ap_uint<8> *kernel, hls::stream<ap_uint<8>> &output, ap_uint<32> *biases, ap_uint<32> m, ap_uint<32> n, ap_uint<8> z1, ap_uint<8> z2, ap_uint<8> z3, int reps) {
    const int OUTPUT_WI = INPUT_WIDTH - KERNEL_SIZE + 1;
    const int OUTPUT_HI = INPUT_HEIGHT - KERNEL_SIZE + 1;

    ap_uint<8> input_buffer[KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS*OUTPUT_WI*OUTPUT_HI];
    ap_uint<8> output_buffer[OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNELS];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS*OUTPUT_WI*OUTPUT_HI, ap_uint<8>>(input, input_buffer, 1);
        Matrix::blockmatmul_quantized<OUTPUT_CHANNELS,KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS,OUTPUT_WI*OUTPUT_HI,PE1,PE2,PE3,5>(kernel, input_buffer, output_buffer, biases, m, n, z1, z2, z3);

        StreamUtil::tostream<OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNELS, ap_uint<8>>(output_buffer, output, 1);

    }

}

}
namespace Float {

/**
 * Internal base function for convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param{in]   kernel          kernel weights
 * @param{in]   bias            biases
 * @param[out]  output          output stream
 * @param[in]   act             activation constant
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void forward(hls::stream<float> &input, float *kernel, float *bias, hls::stream<float> &output, activation_t act, int reps) {
#pragma HLS Dataflow

    hls::stream<float> im2row_stream;
    hls::stream<float> output_nobias;
    hls::stream<float> output_noact;

    im2row<INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, KERNEL_SIZE, float>(input, im2row_stream, reps);

    conv_base<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, PE1, PE2, PE3>(im2row_stream, kernel, output_nobias, reps);

    add_bias<INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_CHANNELS, KERNEL_SIZE>(output_nobias, bias, output_noact, reps);
    apply_activation_float<(INPUT_HEIGHT-KERNEL_SIZE+1)*(INPUT_WIDTH-KERNEL_SIZE+1)*OUTPUT_CHANNELS>(output_noact, output, act, reps);
}

} // namespace Float

namespace UInt8 {

/**
 * Internal base function for convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE1                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param{in]   kernel          kernel weights
 * @param{in]   bias            biases
 * @param[out]  output          output stream
 * @param[in]   act             activation constant
 * @param{in]   m               constant for quantized computations
 * @param{in]   n               constant for quantized computations
 * @param{in]   z1              constant for quantized computations
 * @param{in]   z2              constant for quantized computations
 * @param{in]   z3              constant for quantized computations
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void forward(hls::stream<ap_uint<8>> &input, ap_uint<8> *kernel, ap_uint<32> *bias, hls::stream<ap_uint<8>> &output, activation_t act, ap_uint<32> m, ap_uint<32> n, ap_uint<8> z1, ap_uint<8> z2, ap_uint<8> z3, int reps) {
#pragma HLS Dataflow

    hls::stream<ap_uint<8>> im2row_stream;
    hls::stream<ap_uint<8>> output_noact;

    im2row<INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, KERNEL_SIZE, ap_uint<8>>(input, im2row_stream, reps);

    conv_base_quantized<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, PE1, PE2, PE3>(im2row_stream, kernel, output_noact, bias, m, n, z1, z2, z3, reps);

    apply_activation_quantized<(INPUT_HEIGHT-KERNEL_SIZE+1)*(INPUT_WIDTH-KERNEL_SIZE+1)*OUTPUT_CHANNELS>(output_noact, output, reps);
}

} // namespace UInt8

} // namespace im2row
} // namespace ConvLayer


#endif
