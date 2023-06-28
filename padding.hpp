#ifndef __STANN_HLS_PAD_HPP__
#define __STANN_HLS_PAD_HPP__

#include "stann.hpp"

/**
 * Namespace for padding functions.
 */
namespace Padding{

/**
 * Function to add padding to input data.
 *
 * @tparam  INPUT_WI            input image width
 * @tparam  INPUT_HI            input image height
 * @tparam  INPUT_BATCH_SIZE    input batch size
 * @tparam  PAD                 number of padding pixel to add on each side of the image
 * @tparam  T                   data type of the image
 *
 * @param[in]    input      input image
 * @param[out]   output     padded output image
 * @param[in]    paddValue  value to add as padding
 */
template<int INPUT_WI, int INPUT_HI, int INPUT_BATCH_SIZE, int PAD, typename T>
void padding(const T *input, T *output, const T paddValue){
    int OUTPUT_WI = INPUT_WI + 2 * PAD;
    int OUTPUT_HI = INPUT_HI + 2 * PAD;


    //add zeros to input
    for (int c = 0; c < INPUT_BATCH_SIZE; c++){

        int currentFirstInput = c  * INPUT_WI * INPUT_HI ;
        int currentFirstOutput = c  * OUTPUT_WI * OUTPUT_HI;

        //initinal pads
        for(int d = 0; d < PAD; d++){
            for(int e = 0; e < OUTPUT_WI; e++){
                output[currentFirstOutput + e + OUTPUT_WI *d] = paddValue;
            }
        }
        currentFirstOutput += PAD * OUTPUT_WI;

        for(int i = 0; i < INPUT_HI; i++){

            for (int k = 0; k < PAD; k++){
                output[currentFirstOutput + k + i * OUTPUT_WI] = paddValue;
            }

            for(int j = 0; j < INPUT_WI; j++){
                output[currentFirstOutput + PAD + i * OUTPUT_WI + j]
                    = input[currentFirstInput+ i * INPUT_WI + j];
            }


            for (int k = 0; k < PAD; k++){
                output[currentFirstOutput + PAD + k + i * OUTPUT_WI + INPUT_WI] = paddValue;
            }
        }

        currentFirstOutput += INPUT_HI * OUTPUT_WI;
        for(int d = 0; d < PAD; d++){
            for(int e = 0; e < OUTPUT_WI; e++){
                output[currentFirstOutput + e + OUTPUT_WI *d] = paddValue;
            }
        }



    }
}

} // namespace Padding

#endif
