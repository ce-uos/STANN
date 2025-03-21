#ifndef __STANN_CONV1D_KN2ROW_HPP__
#define __STANN_CONV1D_KN2ROW_HPP__

#include "stann.hpp"

/**
 * Namespace for convolutional layers.
 */
namespace ConvLayer1d {

namespace kn2row{

template<int OUTPUT_WIDTH, int OUTPUT_CHANNELS, typename T>
void add_bias(hls::stream<T> &input, T *biases, hls::stream<T> &output, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int m = 0; m < OUTPUT_CHANNELS; m++) {
            for (int i = 0; i < OUTPUT_WIDTH; i++) {
                T val = input.read();
                val += biases[m];
                output.write(val);
            }
        }
    }
}

template<int DIM, typename T>
void apply_activation_float(hls::stream<T> &input, hls::stream<T> &output, activation_t act, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < DIM; i++) {
            T val = input.read();
            T out_val = val;
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


template<int INPUT_CHANNELS, int INPUT_SIZE, int KERNEL_SIZE, int STRIDE, int OUTPUT_CHANNELS, int OUTPUT_SIZE, typename T>
void conv1d_kn2row_fast(hls::stream<T> &input, T *kernel, hls::stream<T> &output, int reps){

	T input_buffer[INPUT_CHANNELS * INPUT_SIZE];
	T output_buffer[OUTPUT_CHANNELS * OUTPUT_SIZE];

    StreamUtil::toarray<INPUT_SIZE>(input, input_buffer, 1);

    T row_matrix[INPUT_CHANNELS * KERNEL_SIZE];
    T kernel_matrix[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE];

        for (int out_channel = 0; out_channel < OUTPUT_CHANNELS; out_channel++) {
            for (int in_channel = 0; in_channel < INPUT_CHANNELS; in_channel++) {
                for (int k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS UNROLL
                    int kernel_idx = (out_channel * INPUT_CHANNELS * KERNEL_SIZE) + (in_channel * KERNEL_SIZE) + k;
                    kernel_matrix[kernel_idx] = kernel[out_channel * INPUT_CHANNELS * KERNEL_SIZE + in_channel * KERNEL_SIZE + k];
                }
            }
        }

        for (int out_channel = 0; out_channel < OUTPUT_CHANNELS; out_channel++) {
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                T sum = 0.0;
                for (int in_channel = 0; in_channel < INPUT_CHANNELS; in_channel++) {
    #pragma HLS UNROLL
                    for (int k = 0; k < KERNEL_SIZE; k++) {
	#pragma HLS UNROLL
                        int input_idx = (in_channel * INPUT_SIZE) + (i * STRIDE + k);
                        if (input_idx < INPUT_SIZE) {
                            row_matrix[in_channel * KERNEL_SIZE + k] = input_buffer[input_idx];
                        }
                    }
                }

                for (int j = 0; j < INPUT_CHANNELS * KERNEL_SIZE; j++) {
    #pragma HLS UNROLL
                    sum += row_matrix[j] * kernel_matrix[out_channel * INPUT_CHANNELS * KERNEL_SIZE + j];
                }
                output_buffer[out_channel * OUTPUT_SIZE + i] = sum;
            }
        }

    StreamUtil::tostream<OUTPUT_CHANNELS * OUTPUT_SIZE>(output_buffer, output, 1);
}




template<int INPUT_CHANNELS, int INPUT_SIZE, int KERNEL_SIZE, int STRIDE, int OUTPUT_CHANNELS, int OUTPUT_SIZE, int PE_NUM, typename T>
void conv1d_kn2row_base(hls::stream<T> &input, T *kernel, hls::stream<T> &output, int reps) {
    T input_buffer[INPUT_CHANNELS * INPUT_SIZE];
    T output_buffer[OUTPUT_CHANNELS * OUTPUT_SIZE];
    T kernel_matrix[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE];

    StreamUtil::toarray<INPUT_CHANNELS * INPUT_SIZE>(input, input_buffer, 1);

    for (int out_channel = 0; out_channel < OUTPUT_CHANNELS; out_channel++) {
        for (int in_channel = 0; in_channel < INPUT_CHANNELS; in_channel++) {
            for (int k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS UNROLL
                int kernel_idx = (out_channel * INPUT_CHANNELS * KERNEL_SIZE) + (in_channel * KERNEL_SIZE) + k;
                kernel_matrix[kernel_idx] = kernel[out_channel * INPUT_CHANNELS * KERNEL_SIZE + in_channel * KERNEL_SIZE + k];
            }
        }
    }

    for (int out_channel = 0; out_channel < OUTPUT_CHANNELS; out_channel++) {
        T sum[OUTPUT_SIZE] = {0};
        for (int in_channel = 0; in_channel < INPUT_CHANNELS; in_channel++) {
            for (int k = 0; k < KERNEL_SIZE; k++) {
                int kernel_idx = out_channel * INPUT_CHANNELS * KERNEL_SIZE + in_channel * KERNEL_SIZE + k;
                T kernel_val = kernel_matrix[kernel_idx];

                for (int i = 0; i < OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=PE_NUM
                    int input_idx = in_channel * INPUT_SIZE + i * STRIDE + k;
                    if (input_idx < INPUT_CHANNELS * INPUT_SIZE) {
                        sum[i] += input_buffer[input_idx] * kernel_val;
                    }
                }
            }
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
#pragma HLS UNROLL
            output_buffer[out_channel * OUTPUT_SIZE + i] = sum[i];
        }
    }

    StreamUtil::tostream<OUTPUT_CHANNELS * OUTPUT_SIZE>(output_buffer, output, 1);
}


namespace Float{

	template<int INPUT_CHANNELS, int INPUT_SIZE, int KERNEL_SIZE, int STRIDE, int OUTPUT_CHANNELS, int OUTPUT_SIZE, int PE_NUM, typename T>
	void forward(hls::stream<T> &input, T *kernel, T *biases, hls::stream<T> &output, activation_t act, int reps){
#pragma HLS Dataflow

		hls::stream<T> output_nobias;
		hls::stream<T> output_noact;

		conv1d_kn2row_base<INPUT_CHANNELS, INPUT_SIZE, KERNEL_SIZE, STRIDE, OUTPUT_CHANNELS, OUTPUT_SIZE, PE_NUM, T>(input, kernel, output_nobias, reps);
		add_bias<OUTPUT_SIZE, OUTPUT_CHANNELS, T>(output_nobias, biases, output_noact, reps);
		apply_activation_float<OUTPUT_SIZE * OUTPUT_CHANNELS, T>(output_noact, output, act, reps);
	}
}

}

}

#endif
