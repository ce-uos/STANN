#ifndef __STANN_HLS_LOSS_HPP__
#define __STANN_HLS_LOSS_HPP__

#include "stann.hpp"

/**
 * This namespace contains implementations of loss functions and their derivatives.
 */
namespace Loss {

/**
 * Summed squared error loss.
 *
 * @tparam  OUTPUT_DIM  output dimension of neural network
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   output  neural nework predictions
 * @param   target  data labels
 *
 * @return  the summed squared error loss
 */
template<int OUTPUT_DIM, typename T = DEFAULT_DATATYPE>
T SummedSquaredError(T *output, T *target) {
    T sse = 0;
    for (int i = 0; i < OUTPUT_DIM; i++) {
        sse += (output[i] - target[i]) * (output[i] - target[i]);
    }
    return sse;
}

/**
 * Derivative of summed squared error loss.
 *
 * @tparam  OUTPUT_DIM  output dimension of neural network
 * @tparam  BATCH_SIZE  number of data points
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   output  neural nework predictions
 * @param   target  data labels
 * @param   derr    derivative of the loss for each data point
 *
 */
template<int OUTPUT_DIM, int BATCH_SIZE = 1, typename T = DEFAULT_DATATYPE>
void SummedSquaredError_derivative(T *output, T *target, T *derr) {
    for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
        derr[i] = (output[i] - target[i]);
    }
}

/**
 * Derivative of mean squared error loss.
 *
 * @tparam  OUTPUT_DIM  output dimension of neural network
 * @tparam  BATCH_SIZE  number of data points
 * @tparam  T           datatype to work with (should usually be float or fixed_t)
 *
 * @param   output  neural nework predictions
 * @param   target  data labels
 * @param   derr    derivative of the loss for each data point
 *
 */
template<int OUTPUT_DIM, int BATCH_SIZE = 1, typename T = DEFAULT_DATATYPE>
void MeanSquaredError_derivative(T *output, T *target, T *derr) {

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            derr[j * OUTPUT_DIM + i] = (output[i * BATCH_SIZE + j] - target[j * OUTPUT_DIM + i]);// / BATCH_SIZE;
        }
    }
}

/**
 * Derivative of mean squared error loss.
 *
 * @tparam  OUTPUT_DIM  output dimension of neural network
 * @tparam  BATCH_SIZE  number of data points
 * @tparam  T           datatype to work with (should usually be float or fixed_t)
 *
 * @param   output  neural nework predictions
 * @param   target  data labels
 * @param   derr    derivative of the loss for each data point
 *
 */
template<int OUTPUT_DIM, int BATCH_SIZE = 1, typename T = DEFAULT_DATATYPE>
void MeanSquaredError_derivative_stream(hls::stream<T> &output, hls::stream<T> &target, hls::stream<T> &derr) {

    T output_buffer[OUTPUT_DIM * BATCH_SIZE];
    T target_buffer[OUTPUT_DIM * BATCH_SIZE];
    T derr_buffer[OUTPUT_DIM * BATCH_SIZE];

    StreamUtil::toarray<OUTPUT_DIM * BATCH_SIZE>(output, output_buffer);
    StreamUtil::toarray<OUTPUT_DIM * BATCH_SIZE>(target, target_buffer);

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            derr_buffer[j * OUTPUT_DIM + i] = -(target_buffer[i * BATCH_SIZE + j] - output_buffer[j * OUTPUT_DIM + i]) / BATCH_SIZE;
        }
    }

    StreamUtil::tostream<OUTPUT_DIM * BATCH_SIZE>(derr_buffer, derr);
}

/**
 * Derivative of summed squared error loss.
 * Note: same as above but for streamed implementation
 *
 * @tparam  OUTPUT_DIM  output dimension of neural network
 * @tparam  BATCH_SIZE  number of data points
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   output  neural nework predictions
 * @param   target  data labels
 * @param   derr    derivative of the loss for each data point
 *
 */
template<int OUTPUT_DIM, int BATCH_SIZE = 1, typename T = DEFAULT_DATATYPE>
void SummedSquaredError_derivative_stream(hls::stream<T> &output, hls::stream<T> &target, hls::stream<T> &derr) {
sse_derivative : for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
                     T s_out = output.read();
                     T s_target = target.read();
                     T s_derr = s_out - s_target;
                     derr.write(s_derr);
                 }
}

}; // namespace Loss

#endif
