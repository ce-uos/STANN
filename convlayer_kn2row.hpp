#ifndef __STANN_CONV_KN2ROW_HPP__
#define __STANN_CONV_KN2ROW_HPP__

#include "stann.hpp"

/**
 * Namespace for conolutional layers.
 */
namespace ConvLayer {

/**
 * kn2row implementation of convolutional layer.
 */
namespace kn2row {

/**
 * Anonymous namespace for internal functions.
 */
namespace {

/**
 * Converting the kernel weight array to a stream.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 *
 * @param[in]   kernel          kernel weight data
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE>
void conv_split_kernel(float *kernel, hls::stream<float> &output_stream, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE; k++) {
            for (int i = 0; i < INPUT_CHANNELS * OUTPUT_CHANNELS; i++) {
                float val = kernel[k * INPUT_CHANNELS * OUTPUT_CHANNELS + i];
                output_stream.write(val);
            }
        }
    }
}

/**
 * Compute the matrix multiplications for the convolution.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE2                 constant for paralellism
 * @tparam  PE3                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param[in]   kernel          kernel weight stream
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_mult(hls::stream<float> &input, hls::stream<float> &kernel, hls::stream<float> &output_stream, int reps) {
    float input_buffer[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    float kernel_buffer[INPUT_CHANNELS * OUTPUT_CHANNELS];
    float output_buffer[OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS>(input, input_buffer, 1);
        for (int k = 0; k < 9; k++) {
            StreamUtil::toarray<INPUT_CHANNELS*OUTPUT_CHANNELS>(kernel, kernel_buffer, 1);
            MatrixUtil::SysArr::blockmatmul<OUTPUT_CHANNELS, INPUT_CHANNELS, INPUT_WIDTH*INPUT_HEIGHT,PE1,PE2,PE3,float,80>(kernel_buffer, input_buffer, output_buffer);
            StreamUtil::tostream<OUTPUT_CHANNELS*INPUT_HEIGHT*INPUT_WIDTH>(output_buffer, output_stream);
        }
    }
}

/**
 * Shift add function to sum the intermediate results of the kn2row algorithm.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 *
 * @param[in]   input           input stream
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE>
void conv_shift_add(hls::stream<float> &input, hls::stream<float> &output_stream, int reps) {
    const int KSUB = (KERNEL_SIZE / 2) * 2;

    float input_buffer[OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    float output[OUTPUT_CHANNELS * (INPUT_HEIGHT-KSUB) * (INPUT_WIDTH-KSUB)];
    for (int r = 0; r < reps; r++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                StreamUtil::toarray<OUTPUT_CHANNELS*INPUT_HEIGHT*INPUT_WIDTH>(input, input_buffer, 1);
                for (int m = 0; m < OUTPUT_CHANNELS; m++) {
                    for (int w = 0; w < INPUT_WIDTH-KSUB; w++) {
                        for (int h = 0; h < INPUT_HEIGHT-KSUB; h++) {
                        #pragma HLS pipeline II=3
                            const float val = input_buffer[INPUT_WIDTH * INPUT_HEIGHT * m + INPUT_WIDTH * (h+ky) + (w+kx)];
                            output[(INPUT_WIDTH-KSUB) * (INPUT_HEIGHT-KSUB) * m + (INPUT_WIDTH-KSUB) * h + w] += val;
                        }
                    }
                }
            }
        }
        StreamUtil::tostream<OUTPUT_CHANNELS*(INPUT_WIDTH-KSUB)*(INPUT_HEIGHT-KSUB)>(output, output_stream, 1);
    }

}

/**
 * Internal base function for kn2row convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE2                 constant for paralellism
 * @tparam  PE3                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param[in]   kernel          kernel weight stream
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_base0(hls::stream<float> &input, float *kernel, hls::stream<float> &output_stream, int reps) {
#pragma HLS Dataflow
    hls::stream<float> kernel_stream;
    hls::stream<float> mult_stream;

    conv_split_kernel<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE>(kernel, kernel_stream, reps);
    conv_mult<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, PE1, PE2, PE3>(input, kernel_stream, mult_stream, reps);
    conv_shift_add<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE>(mult_stream, output_stream, reps);

}

/**
 * Internal base function for kn2row convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE2                 constant for paralellism
 * @tparam  PE3                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param[in]   kernel          kernel weight stream
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_base(hls::stream<float> &input, float *kernel, hls::stream<float> &output_stream, int reps) {
    const int KSUB = (KERNEL_SIZE / 2) * 2;

    float output_buffer[OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    float output[OUTPUT_CHANNELS * (INPUT_HEIGHT-KSUB) * (INPUT_WIDTH-KSUB)];
    float input_buffer[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];


    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS>(input, input_buffer, 1);
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                const int k = ky * KERNEL_SIZE + kx;
                MatrixUtil::SysArr::blockmatmul<OUTPUT_CHANNELS, INPUT_CHANNELS, INPUT_WIDTH*INPUT_HEIGHT,PE1,PE2,PE3,float,80>(&kernel[k * INPUT_CHANNELS * OUTPUT_CHANNELS], input_buffer, output_buffer);

                for (int m = 0; m < OUTPUT_CHANNELS; m++) {
                    for (int w = 0; w < INPUT_WIDTH-KSUB; w++) {
                        for (int h = 0; h < INPUT_HEIGHT-KSUB; h++) {
                        #pragma HLS pipeline II=3
                            const float val = output_buffer[INPUT_WIDTH * INPUT_HEIGHT * m + INPUT_WIDTH * (h+ky) + (w+kx)];
                            output[(INPUT_WIDTH-KSUB) * (INPUT_HEIGHT-KSUB) * m + (INPUT_WIDTH-KSUB) * h + w] += val;
                        }
                    }
                }


            }
        }
        StreamUtil::tostream<OUTPUT_CHANNELS*(INPUT_WIDTH-KSUB)*(INPUT_HEIGHT-KSUB)>(output, output_stream, 1);
    }

}

/**
 * Internal base function for kn2row convolution inference.
 *
 * @tparam  INPUT_HEIGHT        height of input image
 * @tparam  INPUT_WIDTH         width of input image
 * @tparam  INPUT_CHANNELS      number of input channels
 * @tparam  OUTPUT_CHANNELS     number of output channels
 * @tparam  KERNEL_SIZE         size of kernel (NxN)
 * @tparam  PE1                 constant for paralellism
 * @tparam  PE2                 constant for paralellism
 * @tparam  PE3                 constant for paralellism
 *
 * @param[in]   input           input stream
 * @param[in]   kernel          kernel weight stream
 * @param[out]  output_stream   output stream
 * @param[in]   reps            number of repetitions (similar to batch size)
 */
template<int INPUT_HEIGHT, int INPUT_WIDTH, int INPUT_CHANNELS, int OUTPUT_CHANNELS, int KERNEL_SIZE, int PE1, int PE2, int PE3>
void conv_base2(hls::stream<float> &input, float *kernel, hls::stream<float> &output_stream, int reps) {
    const int KSUB = (KERNEL_SIZE / 2) * 2;

    float output_buffer[OUTPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    float output[KERNEL_SIZE * KERNEL_SIZE * OUTPUT_CHANNELS * (INPUT_HEIGHT-KSUB) * (INPUT_WIDTH-KSUB)];
    float input_buffer[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];


    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS>(input, input_buffer, 1);
        MatrixUtil::SysArr::blockmatmul<KERNEL_SIZE * KERNEL_SIZE * OUTPUT_CHANNELS, INPUT_CHANNELS, INPUT_WIDTH*INPUT_HEIGHT,PE1,PE2,PE3,float,80>(kernel, input_buffer, output_buffer);

        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                for (int m = 0; m < OUTPUT_CHANNELS; m++) {
                    for (int w = 0; w < INPUT_WIDTH-KSUB; w++) {
                        for (int h = 0; h < INPUT_HEIGHT-KSUB; h++) {
                        #pragma HLS pipeline II=3
                            const float val = output_buffer[(ky * KERNEL_SIZE + kx) * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS + INPUT_WIDTH * INPUT_HEIGHT * m + INPUT_WIDTH * (h+ky) + (w+kx)];
                            output[(INPUT_WIDTH-KSUB) * (INPUT_HEIGHT-KSUB) * m + (INPUT_WIDTH-KSUB) * h + w] += val;
                        }
                    }
                }
            }
        }
        StreamUtil::tostream<OUTPUT_CHANNELS*(INPUT_WIDTH-KSUB)*(INPUT_HEIGHT-KSUB)>(output, output_stream, 1);
    }

}

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

} // anonymous namespace

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
    const int KSUB = (KERNEL_SIZE / 2) * 2;

    hls::stream<float> output_nobias;
    hls::stream<float> output_noact;

    conv_base<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, PE1, PE2, PE3>(input, kernel, output_nobias, reps);

    add_bias<INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_CHANNELS, KERNEL_SIZE>(output_nobias, bias, output_noact, reps);
    apply_activation_float<(INPUT_HEIGHT-KSUB)*(INPUT_WIDTH-KSUB)*OUTPUT_CHANNELS>(output_noact, output, act, reps);
}

} // namespace float


} // namespace kn2row
} // namespace ConvLayer



#endif
