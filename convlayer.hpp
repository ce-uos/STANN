#ifndef __STANN_HLS_CONV_HPP__
#define __STANN_HLS_CONV_HPP__

#include "stann.hpp"

/**
 * Namespace for convolutions.
 */
namespace ConvLayer {

/**
 * Namespace for direct convolutions.
 */
namespace Direct {

/**
 * Anonymous namespace for internal functions.
 */
namespace {

/**
 * Internal base function for 2D convolutional layer inference.
 *
 * @tparam  INPUT_WI         width of input image
 * @tparam  INPUT_HI         height of input image
 * @tparam  INPUT_CHANNEL    number channels of input image
 * @tparam  FILTER_WI        width of kernel
 * @tparam  FILTER_HI        height of kernel 
 * @tparam  OUTPUT_CHANNEL   number of kernels
 * @tparam  T                data type the function is working on
 *
 * @param[in]   input   input data
 * @param[in]   filter  filter weights of the conv layer
 * @param[out]  output  output data
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int OUTPUT_CHANNEL, typename T>
void forward_convolution_base(T *input,T *filter, T *output){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;
    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < OUTPUT_CHANNEL; fd++){
            for(int oh = 0; oh < OUTPUT_HI ; oh++){
                for(int ow = 0; ow < OUTPUT_WI ; ow++){
                    for(int ic = 0; ic < INPUT_CHANNEL; ic++){
                        for(int h = 0; h < FILTER_HI; h++){
                            for(int w = 0; w < FILTER_WI; w++){
                            #pragma HLS PIPELINE II=3
                                int INPUT_BATCH_SIZEInputOverhead  = ibs * INPUT_WI * INPUT_HI * INPUT_CHANNEL ;
                                int INPUT_BATCH_SIZEOutputOverhead = ibs * OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNEL;
                                int currentOutputElement     = oh * OUTPUT_WI + ow +  fd  * OUTPUT_HI * OUTPUT_WI + INPUT_BATCH_SIZEOutputOverhead;
                                int currentInputElement = ic  * INPUT_HI * INPUT_WI + ow + w + (oh + h) * INPUT_WI + INPUT_BATCH_SIZEInputOverhead;
                                int currentFilterElement =  (fd * INPUT_CHANNEL + ic) * FILTER_HI * FILTER_WI + w + h * FILTER_WI;

                                if (ic == 0 && h == 0 && w == 0) {
                                    output[currentOutputElement] = 0;
                                }
                                output[currentOutputElement] += input[currentInputElement] * filter[currentFilterElement];

                            }
                        }
                    }
                }
            }

        }
    }
}

/**
 * Internal base function for 2D convolutional layer backpropagation.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  OUTPUT_CHANNEL      number of kernels
 * @tparam  T                   data type the function is working on
 *
 * @param[in]   input   input data
 * @param[in]   filter  filter weights of the conv layer
 * @param[out]  output  output data
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int OUTPUT_CHANNEL, typename T>
void backward_convolution_base(T *input,T *filter, T *output){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;
    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < OUTPUT_CHANNEL; fd++){
            for(int oh = 0; oh < OUTPUT_HI ; oh++){
                for(int ow = 0; ow < OUTPUT_WI ; ow++){
                    for(int ic = 0; ic < INPUT_CHANNEL; ic++){
                        for(int h = 0; h < FILTER_HI; h++){
                            for(int w = 0; w < FILTER_WI; w++){
                            #pragma HLS PIPELINE II=3
                                int INPUT_BATCH_SIZEInputOverhead  = ibs * INPUT_WI * INPUT_HI * INPUT_CHANNEL ;
                                int INPUT_BATCH_SIZEOutputOverhead = ibs * OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNEL;
                                int currentOutputElement     = oh * OUTPUT_WI + ow +  fd  * OUTPUT_HI * OUTPUT_WI + INPUT_BATCH_SIZEOutputOverhead;
                                int currentInputElement = ic  * INPUT_HI * INPUT_WI + ow + w + (oh + h) * INPUT_WI + INPUT_BATCH_SIZEInputOverhead;
                                int currentFilterElement =  (fd * INPUT_CHANNEL + ic) * FILTER_HI * FILTER_WI + w + h * FILTER_WI;

                                if (oh == 0 && ow == 0 && ic == 0 && h == 0 && w == 0) {
                                    input[currentInputElement] = 0;
                                }

                                input[currentInputElement] += output[currentOutputElement] * filter[currentFilterElement];

                            }
                        }
                    }
                }
            }

        }
    }
}

/**
 * Internal base function for 2D convolutional layer weight update.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  OUTPUT_CHANNEL      number of kernels
 * @tparam  T                   data type the function is working on
 *
 * @param[in]       input           input data
 * @param[in,out]   filter          filter weights of the conv layer
 * @param[in]       output          output data of this layer
 * @param[in]       learning_rate   learning rate for the update
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int OUTPUT_CHANNEL, typename T>
void update_convolution_base(T *input,T *filter, T *output, T learning_rate){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;
    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < OUTPUT_CHANNEL; fd++){
            for(int oh = 0; oh < OUTPUT_HI ; oh++){
                for(int ow = 0; ow < OUTPUT_WI ; ow++){
                    for(int ic = 0; ic < INPUT_CHANNEL; ic++){
                        for(int h = 0; h < FILTER_HI; h++){
                            for(int w = 0; w < FILTER_WI; w++){
                            #pragma HLS PIPELINE II=3
                                int INPUT_BATCH_SIZEInputOverhead  = ibs * INPUT_WI * INPUT_HI * INPUT_CHANNEL ;
                                int INPUT_BATCH_SIZEOutputOverhead = ibs * OUTPUT_WI * OUTPUT_HI * OUTPUT_CHANNEL;
                                int currentOutputElement     = oh * OUTPUT_WI + ow +  fd  * OUTPUT_HI * OUTPUT_WI + INPUT_BATCH_SIZEOutputOverhead;
                                int currentInputElement = ic  * INPUT_HI * INPUT_WI + ow + w + (oh + h) * INPUT_WI + INPUT_BATCH_SIZEInputOverhead;
                                int currentFilterElement =  (fd * INPUT_CHANNEL + ic) * FILTER_HI * FILTER_WI + w + h * FILTER_WI;

                                // TODO: divide by batchsize?
                                filter[currentFilterElement] -=  learning_rate*(input[currentInputElement] * output[currentOutputElement]);

                            }
                        }
                    }
                }
            }

        }
    }
}

/**
 * Internal base function for 2D convolutional layer bias update.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  OUTPUT_CHANNEL      number of kernels
 * @tparam  T                   data type the function is working on
 *
 * @param[in,out]   biases          input data
 * @param[in]       this_output     output of this layer
 * @param[in]       learning_rate   learning rate for the update
 *
 * @note    TODO not working yet
 */
template<int INPUT_WI,int INPUT_HI, int INPUT_BATCH_SIZE,int FILTER_WI, int FILTER_HI,  int OUTPUT_CHANNEL, typename T>
void update_biases( T *biases, T *this_output,  T learning_rate) {
    int t = 0;

    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;
    for (int c = 0; c < INPUT_BATCH_SIZE; c++){
        int INPUT_BATCH_SIZEOutputOverhead  = c * OUTPUT_HI * OUTPUT_WI * OUTPUT_CHANNEL;
        for(int b = 0; b < OUTPUT_CHANNEL; b++){
            int outputbatchoversize = b * OUTPUT_WI * OUTPUT_HI;
            for(int i = 0; i < OUTPUT_HI; i++){
                for(int j = 0; j < OUTPUT_WI; j++){
                #pragma HLS PIPELINE II=5
                    biases[c * OUTPUT_CHANNEL + b] += this_output[i * OUTPUT_WI + j + outputbatchoversize + INPUT_BATCH_SIZEOutputOverhead];

                }
            }
        }
    }
}

} // anonymous namespace

namespace Float {


/**
 * Inference of direct convolution layer.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  FILTER_DEPTH        number of kernels
 *
 * @param[in]    input      input data
 * @param[in]    weights    filter weights
 * @param[in]    bias       biases of the layer
 * @param[out]   output     convolution output
 * @param[in]    act        activation constant 
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void forward(float *input, float *weights, float *bias, float *output, activation_t act){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;

    forward_convolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH,float>(input,weights,output);

    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < FILTER_DEPTH; fd++){
            for (int ow = 0; ow < OUTPUT_WI * OUTPUT_HI; ow++){
            #pragma HLS PIPELINE II=2
                output[ow + fd * OUTPUT_WI *OUTPUT_HI ] += bias[fd + FILTER_DEPTH * ibs];
            }
        }
    }

    if (act == LEAKY_RELU){
        ActivationLayer::Float::leaky_relu_inplace<OUTPUT_WI*OUTPUT_HI*FILTER_DEPTH, INPUT_BATCH_SIZE>(output);
    }
}

/**
 * Backpropagation of direct convolution layer.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  FILTER_DEPTH        number of kernels
 *
 * @param[in]    old_input      input data
 * @param[in]    filter         filter weights
 * @param[in]    bias           biases of the layer
 * @param[in]    delta          partial errors of next layer
 * @param[out]   newdelta       partial errors of this layer
 * @param[in]    act            activation constant 
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void backward(float *old_input,  float *filter, float *bias,float *delta, float *newdelta, activation_t act = NONE) {
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;

    backward_convolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL , INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI,  FILTER_DEPTH, float> (newdelta, filter,  delta);

    for (int j = 0; j < INPUT_BATCH_SIZE; j++) {
        for (int i = 0; i < OUTPUT_WI*OUTPUT_HI; i++) {
        #pragma HLS PIPELINE II=8
            if (act == LEAKY_RELU) {
                newdelta[j * OUTPUT_WI*OUTPUT_HI + i] = newdelta[j * OUTPUT_WI*OUTPUT_HI + i] * Activation::leaky_relu_simple_derivative(old_input[j * OUTPUT_WI*OUTPUT_HI + i]);
            } else {
                newdelta[j * OUTPUT_WI*OUTPUT_HI + i] = newdelta[j * OUTPUT_WI*OUTPUT_HI + i] * old_input[j * OUTPUT_WI*OUTPUT_HI + i];
            }

        }
    }
}

/**
 * Weight update of direct convolution layer.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  FILTER_DEPTH        number of kernels
 *
 * @param[in]       delta          partial errors for this layer
 * @param[in,out]   filter         filter weights
 * @param[in,out]   bias           biases of the layer
 * @param[in]       prev_input     input to this layer
 * @param[in]       learning_rate  learning rate for the update
 *
 * @note TODO not working yet
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void update(float *delta, float *filter, float *bias, float *prev_input, float learning_rate){
    update_convolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH, float>(prev_input, filter, delta, learning_rate);
    update_biases<INPUT_WI, INPUT_HI, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH, float>(bias, delta, learning_rate);
}


} // namespace Float

namespace Half {

/**
 * Inference of direct convolution layer.
 *
 * @tparam  INPUT_WI            width of input image
 * @tparam  INPUT_HI            height of input image
 * @tparam  INPUT_CHANNEL       number channels of input image
 * @tparam  INPUT_BATCH_SIZE    number channels of input image
 * @tparam  FILTER_WI           width of kernel
 * @tparam  FILTER_HI           height of kernel 
 * @tparam  FILTER_DEPTH        number of kernels
 *
 * @param[in]    input      input data
 * @param[in]    weights    filter weights
 * @param[in]    bias       biases of the layer
 * @param[out]   output     convolution output
 * @param[in]    act        activation constant 
 */
template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL , int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void forward(half *input, half *weights, half *bias, half *output, activation_t act){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;

    forward_convolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH, half>(input,weights,output);

    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < FILTER_DEPTH; fd++){
            for (int ow = 0; ow < OUTPUT_WI * OUTPUT_HI; ow++){
            #pragma HLS PIPELINE II=2
                output[ow + fd * OUTPUT_WI *OUTPUT_HI ] += bias[fd + FILTER_DEPTH * ibs];
            }
        }
    }

    if (act == LEAKY_RELU){
        ActivationLayer::Half::leaky_relu_inplace<OUTPUT_WI*OUTPUT_HI*FILTER_DEPTH, INPUT_BATCH_SIZE>(output);
    }
}

} // namespace Half

namespace Fixed {

} // namespace Fixed


} // namespace Direct
} // namespace ConvLayer


#endif
