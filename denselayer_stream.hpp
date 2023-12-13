#ifndef __STANN_HLS_DENSE_STREAM_HPP__
#define __STANN_HLS_DENSE_STREAM_HPP__

#include "activations.hpp"
#include "matmul.hpp"
#include "stann.hpp"
#include "streamutils.hpp"

namespace MatrixStream = MatrixUtilStream;
namespace Matrix = MatrixUtil::SysArr;

/**
 * Namespace for stream-based version of dense layers.
 */
namespace DenseLayerStream {

/**
 * Anonymous namespace for internal functions.
 */
namespace {

/**
 * This function adds the bias values to the results of the dense layer.
 *
 * @tparam  DIM     output dimension of the layer
 * @tparam  T       type of the input values and biases
 *
 * @param[in]   input   results of dense layer without bias
 * @param[in]   biases  the biases
 * @param[out]  output  results of dense layer with bias
 * @param[in]   reps    number of repetitions (similar to batch size)
 */
template<int DIM, typename T>
void add_bias(hls::stream<T> &input, T *biases, hls::stream<T> &output, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < DIM; i++) {
            T val = input.read();
            val += biases[i];
            output.write(val);
        }
    }
}


/**
 * This function applies the activation function to the output of the layer.
 *
 * @tparam  DIM     output dimension of the layer
 *
 * @param[in]   input   results of dense layer without activation
 * @param[out]  output  results of dense layer with bias
 * @param[in]   act     activation constant
 * @param[in]   reps    number of repetitions (similar to batch size)
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
 * This function applies the activation function to the output of the layer.
 * Quantized to 8 bit.
 *
 * @tparam  DIM     output dimension of the layer
 *
 * @param[in]   input   results of dense layer without activation
 * @param[out]  output  results of dense layer with bias
 * @param[in]   act     activation constant
 * @param[in]   reps    number of repetitions (similar to batch size)
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
 * This function applies the derivative of the activation function for back
 * propagation.
 *
 * @tparam  DIM         output dimension of the layer
 * @tparam  BATCH_SIZE  training batch size
 *
 * @param[in]   delta_in    deltas without activation applied
 * @param[in]   this_output output of this layer
 * @param[out]  delta_out   deltas with activation applied
 * @param[in]   act         activation constant
 */
template<int DIM, int BATCH_SIZE>
void apply_activation_derivative_float(float *delta_in, float *this_output, hls::stream<float> &delta_out, activation_t act) {
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
        #pragma HLS pipeline II=10
            float tmp = delta_in[j * DIM + i];
            //float tmp = delta_in[i * BATCH_SIZE + j];
            if (act == LEAKY_RELU) {
                tmp *= Activation::leaky_relu_simple_derivative(this_output[i * BATCH_SIZE + j]);
                //tmp *= this_output[i * BATCH_SIZE + j];
            }
            delta_out.write(tmp);
        }
    }
}

template<int DIM, int BATCH_SIZE>
void apply_activation_derivative_float_inplace(float *delta_in, float *this_output, activation_t act) {
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
        #pragma HLS pipeline II=10
            if (act == LEAKY_RELU) {
                delta_in[j * DIM + i] *= Activation::leaky_relu_simple_derivative(this_output[i * BATCH_SIZE + j]);
                //delta_in[j * DIM + i] *= Activation::leaky_relu_simple_derivative(this_output[j * DIM + i]);
            }
        }
    }
}

} // Anonymous namespace

/**
 * Namespace for floating point implementation of dense layer.
 */
namespace Float {

/** 
 * Stream-based inference for the dense layer.
 *
 * @tparam INPUT_DIM    input size of this layer
 * @tparam OUTPUT_DIM   output size of this layer
 * @tparam PE1          constant for parallelism
 * @tparam PE2          constant for parallelism
 * @tparam PE3          constant for parallelism
 * @tparam PII          pipelining constant for the HLS
 * 
 * @param[in]   input       inputs for the layer
 * @param[in]   weights     weights of the layer
 * @param[in]   biases      biases of the layer
 * @param[out]  output      outputs computed by this function
 * @param[in]   act         constant to choose activation
 * @param[in]   reps        number of repetitions
 */
template<int INPUT_DIM, int OUTPUT_DIM, int PE1 = 1, int PE2 = 1, int PE3 = 1, int PII = 80>
void forward(hls::stream<float> &input, float *weights, float *biases, hls::stream<float> &output, activation_t act, int reps) {
#pragma HLS Dataflow

    hls::stream<float> output_nobias;
    hls::stream<float> output_noact;

    MatrixStream::blockmatmul<OUTPUT_DIM, INPUT_DIM, PE1, PE2, float, PII>(weights, input, output_nobias, reps);

    add_bias<OUTPUT_DIM, float>(output_nobias, biases, output_noact, reps);

    apply_activation_float<OUTPUT_DIM>(output_noact, output, act, reps);


}


/** 
 * Backpropagation for the dense layer.
 *
 * @tparam INPUT_DIM        input size of this layer
 * @tparam OUTPUT_DIM       output size of this layer
 * @tparam NEXT_LAYER_DIM   output size of the next layer
 * @tparam BATCH_SIZE       training batch size
 * @tparam PE1              constant for parallelism
 * @tparam PE2              constant for parallelism
 * @tparam PE3              constant for parallelism
 * @tparam PII              pipelining constant for the HLS
 * 
 * @param[in]   this_output     output values of the inference of this layer
 * @param[in]   next_weights    weights of the next layer
 * @param[in]   delta_next      partial errors of the next layer
 * @param[out]  delta           partial errors of this layer (computed by this function)
 * @param[in]   derivative      constant to select activation
 * @param[in]   reps            number of repetitions
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NEXT_LAYER_DIM, int BATCH_SIZE, int PE1 = 1, int PE2 = 1, int PE3 = 1, int PII = 80>
void backward(float *this_output, float *next_weights, hls::stream<float> &delta_next, hls::stream<float> &delta, activation_t derivative, int reps) {

    float delta_noact[BATCH_SIZE * OUTPUT_DIM];

    float delta_buffer[BATCH_SIZE * NEXT_LAYER_DIM];
    //StreamUtil::toarray<NEXT_LAYER_DIM*BATCH_SIZE>(delta_next, delta_buffer); 
    StreamUtil::toarray<NEXT_LAYER_DIM>(delta_next, delta_buffer, BATCH_SIZE); 

    Matrix::blockmatmul<BATCH_SIZE, NEXT_LAYER_DIM, OUTPUT_DIM, PE1, PE2, PE3, float, PII>(delta_buffer, next_weights, delta_noact);

    //apply_activation_derivative_float<OUTPUT_DIM, BATCH_SIZE>(delta_noact, this_output, delta, derivative);
    apply_activation_derivative_float_inplace<OUTPUT_DIM, BATCH_SIZE>(delta_noact, this_output, derivative);
    //StreamUtil::tostream<OUTPUT_DIM * BATCH_SIZE>(delta_noact, delta);
    StreamUtil::tostream<OUTPUT_DIM>(delta_noact, delta, BATCH_SIZE);
}

/** 
 * Weight update for the dense layer.
 * 
 * @tparam INPUT_DIM        input size of this layer
 * @tparam OUTPUT_DIM       output size of this layer
 * @tparam BATCH_SIZE       training batch size
 * @tparam T                type of the values and weights
 * @tparam PE1              constant for parallelism
 * @tparam PE2              constant for parallelism
 * @tparam PE3              constant for parallelism
 * @tparam PII              pipelining constant for the HLS
 *
 * @param[in]       deltas          partial errors for this layer
 * @param[in,out]   weights         weights of this layer (will be updated)
 * @param[in,out]   biases          biases of this layer (will be updated)
 * @param[in]       this_input      inputs to this layer
 * @param[in]       learning_rate   learning rate for training
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE, typename T, int PE1, int PE2, int PE3, int PII=100>
void update(hls::stream<T> &deltas, T *weights, T *biases, hls::stream<T> &this_input, T learning_rate) {

    T gradients[INPUT_DIM * OUTPUT_DIM];

    T buffer[BATCH_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer type=complete

    T input_buffer[INPUT_DIM * BATCH_SIZE];
    StreamUtil::toarray<INPUT_DIM>(this_input, input_buffer, BATCH_SIZE);

    T delta_buffer[OUTPUT_DIM * BATCH_SIZE];
    StreamUtil::toarray<OUTPUT_DIM>(deltas, delta_buffer, BATCH_SIZE);
    //StreamUtil::toarray<OUTPUT_DIM * BATCH_SIZE>(deltas, delta_buffer);


    Matrix::blockmatmul<INPUT_DIM, BATCH_SIZE, OUTPUT_DIM, PE1, PE2, PE3, float, PII>(input_buffer, delta_buffer, gradients);

    //printf("grads:\n");
    //for (int n = 0; n < OUTPUT_DIM; n++) {
    //    for (int m = 0; m < INPUT_DIM; m++) {
    //        printf("%f ", gradients[m * OUTPUT_DIM + n]);
    //    }
    //    printf("\n");
    //}

    //printf("deltas:\n");
    //for (int m = 0; m < BATCH_SIZE; m++) {
    //    for (int n = 0; n < OUTPUT_DIM; n++) {
    //        printf("%f ", delta_buffer[m * OUTPUT_DIM + n]);
    //    }
    //    printf("\n");
    //}

    //printf("inputs: %f %f %f\n", input_buffer[0], input_buffer[1], input_buffer[2]);
    //printf("deltas: %f %f %f\n", delta_buffer[0], delta_buffer[1], delta_buffer[2]);
    //printf("gradients: %f %f %f\n", gradients[0]/BATCH_SIZE, gradients[1]/BATCH_SIZE, gradients[2]/BATCH_SIZE);

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS PIPELINE II=3
            //weights[j * INPUT_DIM + i] -= learning_rate * (gradients[i * OUTPUT_DIM + j] / BATCH_SIZE * 2);
            weights[j * INPUT_DIM + i] -= learning_rate * (gradients[i * OUTPUT_DIM + j]) / BATCH_SIZE;
        }
    }

    for (int y = 0; y < OUTPUT_DIM; y++) {
    #pragma HLS pipeline II=20
        for (int b = 0; b < BATCH_SIZE; b++) {
        #pragma HLS unroll
            // TODO transpose?
            // delta buffer is BxO, so OUTPUT_DIM columns
            buffer[b] = learning_rate * delta_buffer[b * OUTPUT_DIM + y];
            //buffer[b] = learning_rate * delta_buffer[y * BATCH_SIZE + b];
        }
        for (int b = 0; b < BATCH_SIZE; b++) {
            biases[y] -= buffer[b] / BATCH_SIZE;
        }
    }
} 

} // namespace Float

/**
 *  Namespace for 8-bit implementation of the dense layer.
 */
namespace UInt8 {

/** 
 * Stream-based inference for the dense layer.
 *
 * @tparam INPUT_DIM    input size of this layer
 * @tparam OUTPUT_DIM   output size of this layer
 * @tparam PE1          constant for parallelism
 * @tparam PE2          constant for parallelism
 * @tparam PE3          constant for parallelism
 * @tparam PII          pipelining constant for the HLS
 * 
 * @param[in]   input       inputs for the layer
 * @param[in]   weights     weights of the layer
 * @param[in]   biases      biases of the layer
 * @param[out]  output      outputs computed by this function
 * @param[in]   act         constant to choose activation
 * @param[in]   m           constant for quantized computations
 * @param[in]   n           constant for quantized computations
 * @param[in]   z1          constant for quantized computations
 * @param[in]   z2          constant for quantized computations
 * @param[in]   z3          constant for quantized computations
 * @param[in]   reps        number of repetitions
 */
template<int INPUT_DIM, int OUTPUT_DIM, int PE1 = 1, int PE2 = 1, int PE3 = 1, int PII = 80>
void forward(hls::stream<ap_uint<8>> &input, ap_uint<8> *weights, ap_uint<32> *biases, hls::stream<ap_uint<8>> &output, activation_t act, ap_uint<32> m, ap_uint<32> n, ap_uint<8> z1, ap_uint<8> z2, ap_uint<8> z3, int reps) {
#pragma HLS Dataflow

    hls::stream<ap_uint<8>> output_noact;

    MatrixStream::blockmatmul_quantized<OUTPUT_DIM, INPUT_DIM, PE1, PE2, PII>(weights, input, output_noact, biases, m,n,z1,z2,z3,reps);

    apply_activation_quantized<OUTPUT_DIM>(output_noact, output, reps);


}

} // namespace UInt8

} // namespace DenseLayerStream


#endif
