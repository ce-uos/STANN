#ifndef __STANN_HLS_DENSE_HPP__
#define __STANN_HLS_DENSE_HPP__

#include "stann.hpp"

namespace Matrix = MatrixUtil::SysArr;

/**
 * This namespace contains all functions for dense neural network layers (fully connected layers).
 */
namespace DenseLayer {

/**
 * Anonymous namespace for internally used functions.
 */
namespace {


/** 
 * Old version of the weight update function. Slower than current version, and
 * less scalable. Still here for debugging purposes.
 *
 * @tparam  INPUT_DIM   input size of this layer
 * @tparam  OUTPUT_DIM  output size of this layer
 * @tparam  BATCH_SIZE  training batch size
 * @tparam  T           data type (currently not used)
 * @tparam  PII         pipelining constant for the HLS (currently not used)
 * 
 * @param[in]       deltas          partial errors for this layer
 * @param[in,out]   weights         weights of this layer
 * @param[in]       this_input      input to this layer
 * @param[in]       learning_rate   learning rate for the weight update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, typename T, int PII = 5>
void update_weights_slow(T *deltas, T *weights, T *this_input, T learning_rate) {
#pragma HLS inline

    float gradients[INPUT_DIM * OUTPUT_DIM];

    for (int i = 0; i < INPUT_DIM * OUTPUT_DIM; i++) {
        gradients[i] = 0;
    }

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
            #pragma HLS pipeline II=10
                gradients[j * INPUT_DIM + i] += this_input[i * BATCH_SIZE + b] * deltas[b * OUTPUT_DIM + j];
            }
        }
    }

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS pipeline II=15
            gradients[j * INPUT_DIM + i] = gradients[j * INPUT_DIM + i] / BATCH_SIZE * 2;
        }
    }


    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS PIPELINE II=5
            weights[j * INPUT_DIM + i] -= learning_rate * gradients[j * INPUT_DIM + i];
        }
    }


}

/** 
 * Weight update function for the dense layer.
 *
 * @tparam  INPUT_DIM   input size of this layer
 * @tparam  OUTPUT_DIM  output size of this layer
 * @tparam  BATCH_SIZE  training batch size
 * @tparam  T           data type (currently not used)
 * @tparam  PII         pipelining constant for the HLS (currently not used)
 * 
 * @param[in]       deltas          partial errors for this layer
 * @param[in,out]   weights         weights of this layer
 * @param[in]       this_input      input to this layer
 * @param[in]       learning_rate   learning rate for the weight update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, typename T, int PII = 5> 
void update_weights(T *deltas, T *weights, T *this_input, T learning_rate) {
#pragma HLS inline

#pragma HLS ARRAY_PARTITION variable=this_input type=cyclic factor=BATCH_SIZE
#pragma HLS ARRAY_PARTITION variable=deltas type=block factor=BATCH_SIZE

    float gradients[INPUT_DIM * OUTPUT_DIM];

    float buffer[BATCH_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer type=complete

    for (int i = 0; i < INPUT_DIM * OUTPUT_DIM; i++) {
        gradients[i] = 0;
    }

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS pipeline II=20
            for (int b = 0; b < BATCH_SIZE; b++) {
            #pragma HLS UNROLL
                buffer[b] = ((this_input[i * BATCH_SIZE + b] * deltas[b * OUTPUT_DIM + j]) / BATCH_SIZE * 2);
            }
            for (int b = 0; b < BATCH_SIZE; b++) {
                gradients[j * INPUT_DIM + i] += buffer[b];
            }
        }
    }

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS PIPELINE II=5
            weights[j * INPUT_DIM + i] -= learning_rate * gradients[j * INPUT_DIM + i];
        }
    }


}


/** 
 * Weight update function for the dense layer.
 *
 * @tparam  OUTPUT_DIM  output size of this layer
 * @tparam  BATCH_SIZE  training batch size
 * @tparam  T           data type of the weights and biases
 * @tparam  PII         pipelining constant for the HLS (currently not used)
 * 
 * @param[in]       deltas          partial errors for this layer
 * @param[in,out]   biases          bias values of this layer
 * @param[in]       learning_rate   learning rate for the weight update
 */
template<int OUTPUT_DIM, int BATCH_SIZE = 1, typename T, int PII = 5>
void update_biases(T *deltas, T *biases, T learning_rate) {
#pragma HLS inline

#pragma HLS ARRAY_PARTITION variable=deltas type=block factor=BATCH_SIZE

    //int y = 0;
    //int b = 0;

    T buffer[BATCH_SIZE];

    for (int y = 0; y < OUTPUT_DIM; y++) {
    #pragma HLS pipeline II=20
        for (int b = 0; b < BATCH_SIZE; b++) {
        #pragma HLS unroll
            buffer[b] = learning_rate * deltas[b * OUTPUT_DIM + y];
        }
        for (int b = 0; b < BATCH_SIZE; b++) {
            biases[y] -= buffer[b];
        }
    }

    //for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
    //#pragma HLS unroll factor=1
    //#pragma HLS pipeline II=5
    //    biases[y] -= learning_rate * deltas[b * OUTPUT_DIM + y];

    //    y++;
    //    if (y == OUTPUT_DIM) {
    //        b++;
    //        y = 0;
    //    }
    //}
}
} // Anonymous Namespace

/**
 * Namespace for floating-point functions for the dense layer.
 */
namespace Float {

/**
 *  Forward pass of the dense layer. (inference)
 *
 *  @tparam   INPUT_DIM   number of neurons of the previous layer
 *  @tparam   OUTPUT_DIM  number of neurons of this layer
 *  @tparam   BATCH_SIZE  batch size to work with
 *
 *  @param  input       input from previous layer
 *  @param  weights     weights of this layer
 *  @param  biases      biases of this layer
 *  @param  output      output produced by this layer
 *  @param  act         constant for the activation function
 *  @param  inst        instantiation parameter for HLS
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1>
void forward(float *input, float *weights, float *biases, float *output, activation_t act, int inst) {
#pragma HLS inline
#pragma HLS function_instantiate variable=inst

    Matrix::blockmatmul<OUTPUT_DIM, INPUT_DIM, BATCH_SIZE, PE1, PE2, PE3, float, 80>(weights, input, output);

    int m = 0;
    int n = 0;
    for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        output[n * BATCH_SIZE + m] += biases[n];

        m++;
        if (m >= BATCH_SIZE) {
            m = 0;
            n++;
        }

    }

    for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) {
    #pragma HLS pipeline II=10
    #pragma HLS unroll factor=1
        if (act == LEAKY_RELU) {
            output[i] = Activation::leaky_relu_simple(output[i]);
        } else if (act == LIN_TANH) {
            output[i] = Activation::lin_tanh_simple(output[i]);
        }
    }

}

/**
 *  Forward pass of the dense layer. (inference)
 *
 * @tparam  INPUT_DIM   number of neurons of the previous layer
 * @tparam  OUTPUT_DIM  number of neurons of this layer
 * @tparam  BATCH_SIZE  batch size to work with
 * @tparam  PE1         constant for paralellism
 * @tparam  PE2         constant for paralellism
 * @tparam  PE3         constant for paralellism
 *
 * @param  input       input from previous layer
 * @param  weights     weights of this layer
 * @param  biases      biases of this layer
 * @param  output      output produced by this layer
 * @param  act         constant for the activation function
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1>
void forward(float *input, float *weights, float *biases, float *output, activation_t act) {
#pragma HLS inline

    Matrix::blockmatmul<OUTPUT_DIM, INPUT_DIM, BATCH_SIZE, PE1, PE2, PE3, float, 15>(weights, input, output);

    int m = 0;
    int n = 0;
    for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=10
        output[n * BATCH_SIZE + m] += biases[n];

        m++;
        if (m >= BATCH_SIZE) {
            m = 0;
            n++;
        }

    }

    for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) {
    #pragma HLS pipeline II=10
    #pragma HLS unroll factor=1
        if (act == LEAKY_RELU) {
            output[i] = Activation::leaky_relu_simple(output[i]);
        } else if (act == LIN_TANH) {
            output[i] = Activation::lin_tanh_simple(output[i]);
        }
    }

}

/**
 * Implementation of backpropagation.
 *
 * @tparam  INPUT_DIM       number of neurons in the previous layer
 * @tparam  OUTPUT_DIM      number of neurons in this layer
 * @tparam  NEXT_LAYER_DIM  number of neurons in the following layer
 * @tparam  BATCH_SIZE      batch size to work with
 * @tparam  PE1         constant for paralellism
 * @tparam  PE2         constant for paralellism
 * @tparam  PE3         constant for paralellism
 *
 * @param   this_output   outputs produced by this layer
 * @param   next_weights  weights of the next layer
 * @param   delta_next    deltas computed for the next layer
 * @param   delta         deltas computed via backpropagation
 * @param   derivative    constant for the activation function that was applied at this layer
 *
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NEXT_LAYER_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1>
void backward(float *this_output, float *next_weights, float* delta_next, float *delta, activation_t derivative, int inst) {
#pragma HLS inline
#pragma HLS function_instantiate variable=inst

    Matrix::blockmatmul<BATCH_SIZE, NEXT_LAYER_DIM, OUTPUT_DIM, PE1, PE2, PE3, float, 15>(delta_next, next_weights, delta);

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
        #pragma HLS pipeline II=10
            if (derivative == LEAKY_RELU) {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i] * Activation::leaky_relu_simple_derivative(this_output[i * BATCH_SIZE + j]);
            } else {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i];
            }
        }
    }
}

/**
 * This function updates the weights and biases for a dense layer.
 *
 * @tparam  INPUT_DIM   number of neurons in previous layer
 * @tparam  OUTPUT_DIM  number of neurons in this layer
 * @tparam  BATCH_SIZE  batch size to work with
 *
 * @param   deltas         deltas computed via backpropagation
 * @param   weights        neural network weights (updated in-place)
 * @param   biases         neural network biases (updated in-place)
 * @param   this_input     input from previous layer
 * @param   learning_rate  the learning rate for the update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1>
void update_old(float *deltas, float *weights, float *biases, float *this_input, float learning_rate, int inst) {
#pragma HLS inline
#pragma HLS function_instantiate variable=inst
    update_weights<INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, float, 4, 5>(deltas, weights, this_input, learning_rate);
    update_biases<OUTPUT_DIM, BATCH_SIZE, float, 5>(deltas, biases, learning_rate);
}

/**
 * This function updates the weights and biases for a dense layer.
 *
 * @tparam  INPUT_DIM   number of neurons in previous layer
 * @tparam  OUTPUT_DIM  number of neurons in this layer
 * @tparam  BATCH_SIZE  batch size to work with
 *
 * @param   deltas         deltas computed via backpropagation
 * @param   weights        neural network weights (updated in-place)
 * @param   biases         neural network biases (updated in-place)
 * @param   this_input     input from previous layer
 * @param   learning_rate  the learning rate for the update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE, int PE1, int PE2, int PE3, int PII=80>
void update(float *deltas, float *weights, float *biases, float *this_input, float learning_rate, int inst) {
#pragma HLS inline
#pragma HLS function_instantiate variable=inst
    float gradients[INPUT_DIM * OUTPUT_DIM];

    float buffer[BATCH_SIZE];
    #pragma HLS ARRAY_PARTITION variable=buffer type=complete

    Matrix::blockmatmul<INPUT_DIM, BATCH_SIZE, OUTPUT_DIM, PE1, PE2, PE3, float, PII>(this_input, deltas, gradients);

    for (int i = 0; i < INPUT_DIM; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
        #pragma HLS PIPELINE II=3
            //weights[j * INPUT_DIM + i] -= gradients[j * INPUT_DIM + i];
            weights[j * INPUT_DIM + i] -= learning_rate * gradients[j * INPUT_DIM + i];
        }
    }

    for (int y = 0; y < OUTPUT_DIM; y++) {
    #pragma HLS pipeline II=20
        for (int b = 0; b < BATCH_SIZE; b++) {
        #pragma HLS unroll
            buffer[b] = learning_rate * deltas[b * OUTPUT_DIM + y];
        }
        for (int b = 0; b < BATCH_SIZE; b++) {
            biases[y] -= buffer[b];
        }
    }
}
} // namespace Float

namespace Half {

/**
 *  Forward pass of the dense layer. (inference)
 *
 *  @tparam   INPUT_DIM   number of neurons of the previous layer
 *  @tparam   OUTPUT_DIM  number of neurons of this layer
 *  @tparam   BATCH_SIZE  batch size to work with
 *
 *  @param  input       input from previous layer
 *  @param  weights     weights of this layer
 *  @param  biases      biases of this layer
 *  @param  output      output produced by this layer
 *  @param  act            constant for the activation function
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1>
void forward(half *input, half *weights, half *biases, half *output, activation_t act) {
#pragma HLS inline

    Matrix::blockmatmul<OUTPUT_DIM, INPUT_DIM, BATCH_SIZE, PE1, PE2, PE3, half, 3>(weights, input, output);

    int m = 0;
    int n = 0;
    for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=5
        output[n * BATCH_SIZE + m] += biases[n];

        m++;
        if (m >= BATCH_SIZE) {
            m = 0;
            n++;
        }

    }

    for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) {
    #pragma HLS pipeline II=10
    #pragma HLS unroll factor=1
        if (act == LEAKY_RELU) {
            output[i] = ActivationHalf::leaky_relu_simple(output[i]);
        } else if (act == LIN_TANH) {
            output[i] = ActivationHalf::lin_tanh_simple(output[i]);
        }
    }

}

/**
 * Implementation of backpropagation.
 *
 * @tparam  INPUT_DIM       number of neurons in the previous layer
 * @tparam  OUTPUT_DIM      number of neurons in this layer
 * @tparam  NEXT_LAYER_DIM  number of neurons in the following layer
 * @tparam  BATCH_SIZE      batch size to work with
 *
 * @param   this_output   outputs produced by this layer
 * @param   next_weights  weights of the next layer
 * @param   delta_next    deltas computed for the next layer
 * @param   delta         deltas computed via backpropagation
 * @param   derivative    constant for the activation function that was applied at this layer
 *
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NEXT_LAYER_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1, int PII = 5>
void backward(half *this_output, half *next_weights, half* delta_next, half *delta, activation_t derivative) {
#pragma HLS inline

    Matrix::blockmatmul<BATCH_SIZE, NEXT_LAYER_DIM, OUTPUT_DIM, PE1, PE2, PE3, half, 3>(delta_next, next_weights, delta);

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
        #pragma HLS pipeline II=PII
            if (derivative == LEAKY_RELU) {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i] * ActivationHalf::leaky_relu_simple_derivative(this_output[i * BATCH_SIZE + j]);
            } else {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i] * this_output[i * BATCH_SIZE + j];

            }
        }
    }
}

/**
 * This function updates the weights and biases for a dense layer.
 *
 * @tparam  INPUT_DIM   number of neurons in previous layer
 * @tparam  OUTPUT_DIM  number of neurons in this layer
 * @tparam  BATCH_SIZE  batch size to work with
 *
 * @param   deltas         deltas computed via backpropagation
 * @param   weights        neural network weights (updated in-place)
 * @param   biases         neural network biases (updated in-place)
 * @param   this_input     input from previous layer
 * @param   learning_rate  the learning rate for the update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1>
void update(half *deltas, half *weights, half *biases, half *this_input, half learning_rate) {
#pragma HLS inline
    update_weights<INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, half, 5>(deltas, weights, this_input, learning_rate);
    update_biases<OUTPUT_DIM, BATCH_SIZE, half, 5>(deltas, biases, learning_rate);
}
} // namespace Half

/**
 * Namespace for fixed-point functions for the dense layer.
 */
namespace Fixed {
/**
 *  Forward pass of the dense layer. (inference)
 *
 *  @tparam   INPUT_DIM   number of neurons of the previous layer
 *  @tparam   OUTPUT_DIM  number of neurons of this layer
 *  @tparam   BATCH_SIZE  batch size to work with
 *
 *  @param  input       input from previous layer
 *  @param  weights     weights of this layer
 *  @param  biases      biases of this layer
 *  @param  output      output produced by this layer
 *  @param  act            constant for the activation function
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1>
void forward(fixed_t *input, fixed_t *weights, fixed_t *biases, fixed_t *output, activation_t act) {
#pragma HLS inline

    Matrix::blockmatmul<OUTPUT_DIM, INPUT_DIM, BATCH_SIZE, PE1, PE2, PE3, fixed_t, 1>(weights, input, output);

    int m = 0;
    int n = 0;
    for (int i = 0; i < OUTPUT_DIM * BATCH_SIZE; i++) {
    #pragma HLS pipeline II=5
        output[n * BATCH_SIZE + m] += biases[n];

        m++;
        if (m >= BATCH_SIZE) {
            m = 0;
            n++;
        }

    }

    for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) {
    #pragma HLS pipeline II=1
    #pragma HLS unroll factor=1
        if (act == LEAKY_RELU) {
            output[i] = ActivationFixed::leaky_relu_simple(output[i]);
        }
    }

}

/**
 * Implementation of backpropagation.
 *
 * @tparam  INPUT_DIM       number of neurons in the previous layer
 * @tparam  OUTPUT_DIM      number of neurons in this layer
 * @tparam  NEXT_LAYER_DIM  number of neurons in the following layer
 * @tparam  BATCH_SIZE      batch size to work with
 *
 * @param   this_output   outputs produced by this layer
 * @param   next_weights  weights of the next layer
 * @param   delta_next    deltas computed for the next layer
 * @param   delta         deltas computed via backpropagation
 * @param   derivative    constant for the activation function that was applied at this layer
 *
 */
template<int INPUT_DIM, int OUTPUT_DIM, int NEXT_LAYER_DIM, int BATCH_SIZE = 1, int PE1 = 1, int PE2 = 1, int PE3 = 1, int PII = 1>
void backward(fixed_t *this_output, fixed_t *next_weights, fixed_t* delta_next, fixed_t *delta, activation_t derivative) {
#pragma HLS inline

    Matrix::blockmatmul<BATCH_SIZE, NEXT_LAYER_DIM, OUTPUT_DIM, PE1, PE2, PE3, fixed_t, 3>(delta_next, next_weights, delta);

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
        #pragma HLS pipeline II=PII
        #pragma HLS unroll factor=1
            if (derivative == LEAKY_RELU) {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i] * ActivationFixed::leaky_relu_simple_derivative(this_output[i * BATCH_SIZE + j]);
            } else {
                delta[j * OUTPUT_DIM + i] = delta[j * OUTPUT_DIM + i] * this_output[i * BATCH_SIZE + j];

            }
        }
    }
}


/**
 * This function updates the weights and biases for a dense layer.
 *
 * @tparam  INPUT_DIM   number of neurons in previous layer
 * @tparam  OUTPUT_DIM  number of neurons in this layer
 * @tparam  BATCH_SIZE  batch size to work with
 *
 * @param   deltas         deltas computed via backpropagation
 * @param   weights        neural network weights (updated in-place)
 * @param   biases         neural network biases (updated in-place)
 * @param   this_input     input from previous layer
 * @param   learning_rate  the learning rate for the update
 */
template<int INPUT_DIM, int OUTPUT_DIM, int BATCH_SIZE = 1>
void update(fixed_t *deltas, fixed_t *weights, fixed_t *biases, fixed_t *this_input, fixed_t learning_rate) {
#pragma HLS inline
    update_weights<INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, fixed_t,1>(deltas, weights, this_input, learning_rate);
    update_biases<OUTPUT_DIM, BATCH_SIZE, fixed_t,1>(deltas, biases, learning_rate);
}
} // namespace Fixed



}; // namespace DenseLayer

#endif
