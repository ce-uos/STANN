#ifndef __STANN_HLS_ACTIVATION_HPP__
#define __STANN_HLS_ACTIVATION_HPP__

#include "stann.hpp"

/**
 *  This namespace contains implementations of floating point activation functions and their derivatives.
 */
namespace Activation {

/**
 * Linearized approximation of hyperbolic tangent activation.
 *
 * @param   input   input data
 *
 * @return  approximation of tanh of input
 */
float lin_tanh_simple(float input) {
    if (input < -1) {
        return -1;
    } else if (input > 1) {
        return 1;
    } else {
        return input;
    }
}

/**
 * Derivative of the linearized approximation of hyperbolic tangent activation.
 *
 * @param   input   input data
 *
 * @return  derivatife of the approximation of tanh of the input
 */
float lin_tanh_simple_derivative(float input) {
    if (input < -1) {
        return 0;
    } else if (input > 1) {
        return 0;
    } else {
        return 1;
    }
}

/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
float leaky_relu_simple(float input) {
    return input <= 0.0f ? input * 0.0625 : input;
}

/**
 * ReLU activation.
 *
 * @param   input   input data
 *
 * @return  ReLU activation of input
 */
float relu_simple(float input) {
    return input <= 0.0f ? 0 : input;
}

/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
float leaky_relu_simple_derivative(float input) {
    return input <= 0 ? 0.0625 : 1;
}

float error_act(float input){
    return (input /( sqrt(1 + (input * input))));
}

/**
 * Softmax activation.
 *
 * @param   input   input data
 *
 * @return  softmax of input
 */
template<int DIM, int BATCH_SIZE, typename T>
void softmax(T* input, T* output){
    T partequation = 0.0;
    T max = input[0];
    for (int i = 1; i < DIM; i++) {
        if(max < input[i]){
            max = input[i];
        }
    }

    for (int i = 0; i < DIM; i++) {
        input[i] -= max;
    }

    for (int i = 0; i < DIM; i++) {
    #pragma HLS PIPELINE II=3
        partequation += std::exp(input[i]);
    }

    for (int i = 0; i < DIM; i++) {
        output[i] = (exp(input[i]))/partequation;
    }

}

/**
 * Sigmoid activation.
 *
 * @param   input   input data
 *
 * @return  Sigmoid of input
 */
float sigmoid_act(float input){
    return (1 / (1 + exp(-input)));
}

/**
 * tanh activation.
 *
 * @param   input   input data
 *
 * @return  tanh of input
 */
float tanh_act(float input){
    return tanh(input);
}

/**
 * Softmax activation.
 *
 * @param   input   input data
 *
 * @return  Softmax of input
 */
float softmax_act(float input1, float input2){
    return exp(input1) / input2;
}

} // namespace Activation

/**
 *  This namespace contains implementations of floating point activation functions and their derivatives.
 */
namespace ActivationHalf {

/**
 * Linearized approximation of hyperbolic tangent activation.
 *
 * @param   input   input data
 *
 * @return  approximation of tanh of input
 */
half lin_tanh_simple(half input) {
    if (input < -1) {
        return -1;
    } else if (input > 1) {
        return 1;
    } else {
        return input;
    }
}

/**
 * Derivative of the linearized approximation of hyperbolic tangent activation.
 *
 * @param   input   input data
 *
 * @return  derivatife of the approximation of tanh of the input
 */
half lin_tanh_simple_derivative(half input) {
    if (input < -1) {
        return 0;
    } else if (input > 1) {
        return 0;
    } else {
        return 1;
    }
}

/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
half leaky_relu_simple(half input) {
    return input <= (half)0.0 ? (half)(input * 0.0625) : input;
}

/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
half leaky_relu_simple_derivative(half input) {
    return input <= (half)0 ? (half)0.0625 : (half)1.0;
}


} // namespace ActivationHalf


/**
 *  This namespace contains implementations of fixed point activation functions and their derivatives.
 */
namespace ActivationFixed {
/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
fixed_t leaky_relu_simple(fixed_t input) {
    return input <= (fixed_t)0.0f ? input >> 4 : input;
}

/**
 * Leaky ReLU activation.
 *
 * @param   input   input data
 *
 * @return  leaky ReLU of input
 */
fixed_t leaky_relu_simple_derivative(fixed_t input) {
    return input <= 0 ? 0.0625 : 1;
}

/**
 * Sigmoid activation.
 *
 * @param   input   input data
 *
 * @return  Sigmoid of input
 */

} // namespace ActivationFixed

/**
 * This namespace contains funcrion that can be used as an activation layer. 
 */
namespace ActivationLayer {

namespace Float {

/**
 * Leaky ReLU activation layer working in-place.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 * @tparam BATCH_SIZE   batch size used in this layer
 *
 * @param[in,out]   data    data array the layer is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_inplace(float *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        data[i] = Activation::leaky_relu_simple(data[i]);
    }
}

/**
 * Derivative of Leaky ReLU activation layer working in-place.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 * @tparam BATCH_SIZE   batch size used in this layer
 *
 * @param[in,out]   data    data array the layer is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative_inplace(float *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        data[i] = Activation::leaky_relu_simple_derivative(data[i]);
    }
}

/**
 * Leaky ReLU activation layer.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 * @tparam BATCH_SIZE   batch size used in this layer
 *
 * @param[in]    input    input data for the activation layer
 * @param[out]   output   output data of the activation layer
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu(float *input, float *output) {
#pragma HLS inline
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        output[i] = Activation::leaky_relu_simple(input[i]);
    }
}

/**
 * Leaky ReLU activation layer.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 *
 * @param[in]    input    input stream for the activation layer
 * @param[out]   output   output stream of the activation layer
 * @param[in]    reps     number of repetitions the layer is doing (similar to batch size)
 */
template<int OUTDIM>
void leaky_relu_stream(hls::stream<float> &input, hls::stream<float> &output, int reps) {
#pragma HLS inline
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < OUTDIM; j++) {
            //   #pragma HLS pipeline II=10
            float tmp = input.read();
            tmp = Activation::leaky_relu_simple(tmp);
            output.write(tmp);
        }
    }
}

/**
 * Leaky ReLU activation layer. This implementation makes copies of the input
 * and output stream.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 *
 * @param[in]    input         input stream for the activation layer
 * @param[in]    input_copy    copy of input stream for the activation layer
 * @param[out]   output        output stream of the activation layer
 * @param[out]   output_copy   copy of output stream of the activation layer
 * @param[in]    reps          number of repetitions the layer is doing (similar to batch size)
 */
template<int OUTDIM>
void leaky_relu_stream(hls::stream<float> &input, hls::stream<float> &input_copy, hls::stream<float> &output, hls::stream<float> &output_copy, int reps) {
#pragma HLS inline
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < OUTDIM; j++) {
            //   #pragma HLS pipeline II=10
            float tmp = input.read();
            input_copy.write(tmp);
            tmp = Activation::leaky_relu_simple(tmp);
            output.write(tmp);
            output_copy.write(tmp);
        }
    }
}

/**
 * Leaky ReLU activation layer. This implementation makes copies of the input stream.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 *
 * @param[in]    input         input stream for the activation layer
 * @param[in]    input_copy    copy of input stream for the activation layer
 * @param[out]   output        output stream of the activation layer
 * @param[in]    reps          number of repetitions the layer is doing (similar to batch size)
 */
template<int OUTDIM>
void leaky_relu_stream(hls::stream<float> &input, hls::stream<float> &input_copy, hls::stream<float> &output, int reps) {
#pragma HLS inline
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < OUTDIM; j++) {
            //   #pragma HLS pipeline II=10
            float tmp = input.read();
            input_copy.write(tmp);
            tmp = Activation::leaky_relu_simple(tmp);
            output.write(tmp);
        }
    }
}

/**
 * Leaky ReLU activation layer. This implementation makes copies of the input
 * and output stream.
 *
 * @tparam OUTDIM       output size of this layer (same as input size)
 * @tparam BATCH_SIZE   batch size used in this layer
 *
 * @param[in]    input         input stream for the activation layer
 * @param[in]    input_copy    copy of input stream for the activation layer
 * @param[out]   output        output stream of the activation layer
 * @param[in]    reps          number of repetitions the layer is doing (similar to batch size)
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_stream(hls::stream<float> &input, float *input_copy, hls::stream<float> &output, hls::stream<float> &output_copy, int reps) {
#pragma HLS inline
    for (int i = 0; i < reps; i++) {
        for (int j = 0; j < OUTDIM; j++) {
            //   #pragma HLS pipeline II=10
            float tmp = input.read();
            input_copy[j * BATCH_SIZE + i] = tmp;
            tmp = Activation::leaky_relu_simple(tmp);
            output.write(tmp);
            output_copy.write(tmp);
        }
    }
}

/**
 * Derivative of leaky ReLU activation.
 *
 * @tparam  OUTDIM      output size of the layer
 * @tparam  BATCH_SIZE  batch size of the layer
 *
 * @param[in]   input   input array for the layer
 * @param[out]  output  output array of the layer
 *
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative(float *input, float *output) {
#pragma HLS inline
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        output[i] = Activation::leaky_relu_simple_derivative(input[i]);
    }
}

} // namespace ActivationLayer

/**
 * This namespace constains activation functions for the half datatype (16 bit
 * float).
 */
namespace Half {

/**
 * Leaky ReLU activation working in-place.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in,out]   data    data array the activation function is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_inplace(half *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        data[i] = ActivationHalf::leaky_relu_simple(data[i]);
    }
}

/**
 * Derivative of leaky ReLU activation working in-place.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in,out]   data    data array the activation function is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative_inplace(half *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        data[i] = ActivationHalf::leaky_relu_simple_derivative(data[i]);
    }
}

/**
 * Leaky ReLU activation.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in]   input    input array of the activation function
 * @param[out]  output   output array of the activation function
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu(half *input, half *output) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        output[i] = ActivationHalf::leaky_relu_simple(input[i]);
    }
}

/**
 * Deriative of leaky ReLU activation.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in]   input    input array of the activation function
 * @param[out]  output   output array of the activation function
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative(half *input, half *output) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        output[i] = ActivationHalf::leaky_relu_simple_derivative(input[i]);
    }
}

} // namespace Half

namespace Fixed {

/**
 * Leaky ReLU activation working in-place.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in,out]   data    data array the activation function is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_inplace(fixed_t *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        data[i] = ActivationFixed::leaky_relu_simple(data[i]);
    }
}

/**
 * Derivative of leaky ReLU activation working in-place.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in,out]   data    data array the activation function is working on
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative_inplace(fixed_t *data) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        data[i] = ActivationFixed::leaky_relu_simple_derivative(data[i]);
    }
}

/**
 * Leaky ReLU activation.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in]   input    input array of the activation function
 * @param[out]  output   output array of the activation function
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu(fixed_t *input, fixed_t *output) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        output[i] = ActivationFixed::leaky_relu_simple(input[i]);
    }
}

/**
 * Deriative of leaky ReLU activation.
 *
 * @tparam  OUTDIM      output size of the activation function
 * @tparam  BATCH_SIZE  batch size of the activation function
 *
 * @param[in]   input    input array of the activation function
 * @param[out]  output   output array of the activation function
 */
template<int OUTDIM, int BATCH_SIZE>
void leaky_relu_derivative(fixed_t *input, fixed_t *output) {
    for (int i = 0; i < OUTDIM * BATCH_SIZE; i++) {
        output[i] = ActivationFixed::leaky_relu_simple_derivative(input[i]);
    }
}

} // namespace Fixed



} // namespace ActivationLayer

#endif
