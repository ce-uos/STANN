#ifndef __STANN_HLS_POOLING_HPP__
#define __STANN_HLS_POOLING_HPP__

#include "stann.hpp"

/**
 * Namespace for Pooling Layer implementations.
 */
namespace PoolingLayer {

/**
 * Anonymous namespace for internal functions.
 */
namespace {

/** 
 * Stream-based average pooling. 8-bit quantized version
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparan  KERNEL_SIZE     kernel size (width = height)
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 * @param[in]   reps    number of repetitions (similar to batch size)
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int KERNEL_SIZE>
void average_stream_quantized(hls::stream<ap_uint<8>> &input, hls::stream<ap_uint<8>> &output, int reps){
    const int OUTPUT_WI = INPUT_WI / KERNEL_SIZE;
    const int OUTPUT_HI = INPUT_HI / KERNEL_SIZE;

    ap_uint<8> output_buffer[INPUT_CHANNEL * OUTPUT_HI * OUTPUT_WI];
    ap_uint<8> input_buffer[INPUT_CHANNEL * INPUT_HI * INPUT_WI];
    ap_uint<32> acc;

    for (int b = 0; b < reps; b++){

        StreamUtil::toarray<INPUT_CHANNEL * INPUT_HI * INPUT_WI>(input, input_buffer, 1);

        for(int c = 0; c < INPUT_CHANNEL; c ++){
            for(int y = 0; y < OUTPUT_HI; y ++){
                for(int x = 0; x < OUTPUT_WI; x ++){
                    int currentOutputElement = x + y * OUTPUT_WI + c * OUTPUT_WI * OUTPUT_HI; 
                    int currentInputElement  = x  * KERNEL_SIZE + y * INPUT_WI * KERNEL_SIZE  + c * INPUT_WI  * INPUT_HI;

                    acc = 0;

                    for(int cy = 0; cy < KERNEL_SIZE; cy ++){
                        for(int cx = 0; cx < KERNEL_SIZE; cx ++){
                            output_buffer[currentOutputElement] += input_buffer[currentInputElement + cy * INPUT_WI + cx];
                        }
                    }
                    acc /= KERNEL_SIZE * KERNEL_SIZE;
                    acc = acc > ((ap_uint<32>)255) ? ((ap_uint<32>)255) : acc;
                    output_buffer[currentOutputElement] = static_cast<ap_uint<8>>(acc);
                }
            }
        }

        StreamUtil::tostream<INPUT_CHANNEL * OUTPUT_HI * OUTPUT_WI>(output_buffer, output, 1);
    }
}

/** 
 * Stream-based average pooling.
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparan  KERNEL_SIZE     kernel size (width = height)
 * @tparam  T               data type of the image
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 * @param[in]   reps    number of repetitions (similar to batch size)
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int KERNEL_SIZE, typename T>
void average_stream(hls::stream<T> &input, hls::stream<T> &output, int reps){
    const int OUTPUT_WI = INPUT_WI / KERNEL_SIZE;
    const int OUTPUT_HI = INPUT_HI / KERNEL_SIZE;

    T output_buffer[INPUT_CHANNEL * OUTPUT_HI * OUTPUT_WI];
    T input_buffer[INPUT_CHANNEL * INPUT_HI * INPUT_WI];

    for (int b = 0; b < reps; b++){

        StreamUtil::toarray<INPUT_CHANNEL * INPUT_HI * INPUT_WI>(input, input_buffer, 1);

        for(int c = 0; c < INPUT_CHANNEL; c ++){
            for(int y = 0; y < OUTPUT_HI; y ++){
                for(int x = 0; x < OUTPUT_WI; x ++){
                    int currentOutputElement = x + y * OUTPUT_WI + c * OUTPUT_WI * OUTPUT_HI; 
                    int currentInputElement  = x  * KERNEL_SIZE + y * INPUT_WI * KERNEL_SIZE  + c * INPUT_WI  * INPUT_HI;

                    output_buffer[currentOutputElement] = 0;

                    for(int cy = 0; cy < KERNEL_SIZE; cy ++){
                        for(int cx = 0; cx < KERNEL_SIZE; cx ++){
                            output_buffer[currentOutputElement] += input_buffer[currentInputElement + cy * INPUT_WI + cx];
                        }
                    }
                    output_buffer[currentOutputElement] /= KERNEL_SIZE * KERNEL_SIZE;
                }
            }
        }

        StreamUtil::tostream<INPUT_CHANNEL * OUTPUT_HI * OUTPUT_WI>(output_buffer, output, 1);
    }
}

/** 
 * Average pooling.
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * @tparam  T               data type of the image
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE, int KERNEL_WI, int KERNEL_HI, typename T>
void average_forward_base(T *input, T *output){
    const int OUTPUT_WI = INPUT_WI / KERNEL_WI;
    const int OUTPUT_HI = INPUT_HI / KERNEL_HI;

    for (int b = 0; b < BATCH_SIZE; b++){

        int BATCH_SIZEInputOverhead = b * INPUT_WI * INPUT_HI * INPUT_CHANNEL;
        int BATCH_SIZEOutputOverhead = b * OUTPUT_WI * OUTPUT_HI * INPUT_CHANNEL;

        for(int c = 0; c < INPUT_CHANNEL; c ++){
            for(int y = 0; y < OUTPUT_HI; y ++){
                for(int x = 0; x < OUTPUT_WI; x ++){
                    int currentOutputElement = x + y * OUTPUT_WI + c * OUTPUT_WI * OUTPUT_HI + BATCH_SIZEOutputOverhead;
                    int currentInputElement  = x  * KERNEL_WI + y * INPUT_WI * KERNEL_HI  + c * INPUT_WI  * INPUT_HI + BATCH_SIZEInputOverhead;

                    output[currentOutputElement] = 0;

                    for(int cy = 0; cy < KERNEL_HI; cy ++){
                        for(int cx = 0; cx < KERNEL_WI; cx ++){
                            output[currentOutputElement] += input[currentInputElement + cy * INPUT_WI + cx];
                        }
                    }
                    output[currentOutputElement] /= KERNEL_WI * KERNEL_HI;
                }
            }
        }
    }
}

/** 
 * Max pooling.
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * @tparam  T               data type of the image
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 */
template<int INPUT_WI,int INPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE,int KERNEL_WI, int KERNEL_HI, typename T>
void max_forward_base(T *input, T *output){
    const int OUTPUT_WI = INPUT_WI / KERNEL_WI;
    const int OUTPUT_HI = INPUT_HI / KERNEL_HI;

    for(int c = 0; c < INPUT_CHANNEL; c ++){
        for(int y = 0; y < OUTPUT_HI; y ++){
            for(int x = 0; x < OUTPUT_WI; x ++){
                int currentOutputElement = x + y * OUTPUT_WI + c * OUTPUT_WI * OUTPUT_HI;
                int currentInputElement  = x  * KERNEL_WI + y * INPUT_WI * KERNEL_HI  + c * INPUT_WI  * INPUT_HI;

                T currentMax = input[currentInputElement];

                for(int cy = 0; cy < KERNEL_HI; cy ++){
                    for(int cx = 0; cx < KERNEL_WI; cx ++){
                        if(currentMax < input[currentInputElement + cy * INPUT_WI + cx]){
                            currentMax = input[currentInputElement + cy * INPUT_WI + cx];
                        }
                    }
                }
                output[currentOutputElement] = currentMax;
            }
        }
    }
}

/** 
 * Average pooling backward pass.
 *
 * @tparam  OUTPUT_WI       width of output image
 * @tparam  OUTPUT_HI       height of output image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * @tparam  T               data type of the image
 * 
 * @param[in]   input   input image (smaller image)
 * @param[out]  output  output image (larger image, after reverse pooling)
 */
template<int OUTPUT_WI,int OUTPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE,int KERNEL_WI, int KERNEL_HI, typename T>
void average_backward_base(T *input,T *output) {
    int INPUT_WI = OUTPUT_WI / KERNEL_WI;
    int INPUT_HI = OUTPUT_HI / KERNEL_HI;

    for (int b = 0; b < BATCH_SIZE; b++){

        int BATCH_SIZEInputOverhead = b * INPUT_WI * INPUT_HI * INPUT_CHANNEL;
        int BATCH_SIZEOutputOverhead = b * OUTPUT_WI * OUTPUT_HI * INPUT_CHANNEL;

        for(int c = 0; c < INPUT_CHANNEL; c++){
            for(int j = 0; j < INPUT_HI; j++){
                for(int i = 0; i< INPUT_WI; i++){
                    int currentInputElement  = j * INPUT_WI  + i + c * INPUT_WI * INPUT_HI + BATCH_SIZEInputOverhead ;
                    int currentOutputElement = j * OUTPUT_WI * KERNEL_HI + i * KERNEL_WI + c * OUTPUT_WI * OUTPUT_HI  + BATCH_SIZEOutputOverhead;
                    for(int h = 0; h < KERNEL_HI; h++){
                        for(int k = 0; k< KERNEL_WI; k++){
                        #pragma HLS PIPELINE II=3
                            output[currentOutputElement + k + h * OUTPUT_WI] = input[currentInputElement];
                        }
                    }

                }
            }
        }
    }
}
}// Anonymous Namespace

namespace Float {

/** 
 * Max pooling.
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 */
template<int INPUT_WI,int INPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE,int KERNEL_WI, int KERNEL_HI>
void max_forward_base(float *input, float *output){
    max_forward_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, BATCH_SIZE, KERNEL_WI, KERNEL_HI,float>(input,output);
}

/** 
 * Average pooling.
 *
 * @tparam  INPUT_WI        width of input image
 * @tparam  INPUT_HI        height of input image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * 
 * @param[in]   input   input image
 * @param[out]  output  output image after average pooling
 */
template<int INPUT_WI,int INPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE,int KERNEL_WI, int KERNEL_HI>
void average_forward(float *input, float *output){
    average_forward_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, BATCH_SIZE, KERNEL_WI, KERNEL_HI,float>(input,output);
}

/** 
 * Average pooling backward pass.
 *
 * @tparam  OUTPUT_WI       width of output image
 * @tparam  OUTPUT_HI       height of output image
 * @tparam  INPUT_CHANNEL   number of channels of input image
 * @tparam  BATCH_SIZE      batch size
 * @tparan  KERNEL_WI       kernel width
 * @tparan  KERNEL_HI       kernel height
 * 
 * @param[in]   input   input image (smaller image)
 * @param[out]  output  output image (larger image, after reverse pooling)
 */
template<int OUTPUT_WI,int OUTPUT_HI, int INPUT_CHANNEL, int BATCH_SIZE,int KERNEL_WI, int KERNEL_HI>
void average_backward(float *input, float *output) {
    average_backward_base<OUTPUT_WI, OUTPUT_HI, INPUT_CHANNEL, BATCH_SIZE, KERNEL_WI, KERNEL_HI,float>(input,output);
}
}// namespace Float



}// namespace pooling

#endif
