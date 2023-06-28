#ifndef __STANN_HLS_CONV_WINO_HPP__
#define __STANN_HLS_CONV_WINO_HPP__

#include "stann.hpp"

//
// TODO probably not working yet
//

namespace ConvLayer {
namespace Winograd {
namespace {

/**
 * Element-wise matrix multiplication.
 * computes c = a * b (element-wise)
 * a = MxN
 * b = MxN
 * c = MxN
 */
template<int M, int N, typename T>
void elem_matmul(T *a, T *b, T *c) {
    for (int i = 0; i < M*N; i++) {
    #pragma HLS PIPELINE II=3
        c[i] = a[i] * b[i];
    }
}

/**
 * Transposes matrix a with dimenstions KxM.
 * Transposed matrix will have dimentions MxK.
 */
template<int K, int M, typename T>
void transpose(T *a, T *a_transposed) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            a_transposed[m * K + k] = a[k * M + m];
        }
    }
}

/**
 * This function transforms a matrix "a" via matrix "c".
 * (see: change of basis for vector spaces)
 *
 * computes: t * a * t^T
 * a = KxK
 * t = TxK
 */
template<int K, int T, typename P, int PE1 = 1,int PE2 = 1,int PE3 = 1, int PII = 2>
void transform(P *a, P *t, P *c) {
    P t_transposed[K*T];
    #pragma HLS ARRAY_PARTITION variable=t_transposed factor=PE2 type=cyclic
    transpose<T,K>(t, t_transposed);

    P tmp[T*K];
    #pragma HLS ARRAY_PARTITION variable=tmp factor=PE1 type=cyclic
    clearArray<T*K>(tmp);

    clearArray<T*T>(c);

    MatrixUtil::SysArr::blockmatmul<T,K,K,PE1,PE2,PE3,P,PII>(t, a, tmp);
    MatrixUtil::SysArr::blockmatmul<T,K,T,PE1,PE2,PE3,P,PII>(tmp, t_transposed, c);
}

// F(2x2,3x3)
// kernel should be 3x3
// input should be 4x4
// so output should be 2x2
template<int FILTER_DIM,typename T, int PE1 = 1,int PE2 = 1,int PE3 = 1, int PII = 2>
void winograd(T *input, T *U, T *output) {
    if(FILTER_DIM == 3){

        T BT[16] = {
            1,0,-1,0,
            0,1,1,0,
            0,-1,1,0,
            0,1,0,-1
        };
        #pragma HLS ARRAY_PARTITION variable=BT type=complete
        T AT[8] = {
            1,1,1,0,
            0,1,-1,-1
        };
        #pragma HLS ARRAY_PARTITION variable=AT type=complete

        T V[16];
        #pragma HLS ARRAY_PARTITION variable=V type=complete
        T M[16];
        #pragma HLS ARRAY_PARTITION variable=M type=complete
        transform<4,4>(input, BT, V); // V = B^T d B

        elem_matmul<4,4>(U,V,M);

        transform<4,2>(M, AT, output); // Y = A^T M A
    }else if(FILTER_DIM == 5){

        T BT[8*8] = {
            1.0,   0.0,    -21.0/4.0,    0.0,    21.0/4.0,     0.0,    -1.0,  0.0,
            0.0,   1.0,      1.0,    -17.0/4.0,  -17.0/4.0,    1.0,    1.0,   0.0,
            0.0,   -1.0,     1.0,    17.0/4.0,   -17.0/4.0,   -1.0,    1.0,   0.0,
            0.0,  1.0/2.0,    1.0/4.0,   -5.0/2.0,   -5.0/4.0,     2.0,    1.0,   0.0,
            0.0,  -1.0/2.0,   1.0/4.0,    5.0/2.0,   -5.0/4.0,    -2.0,    1.0,   0.0,
            0.0,   2.0,      4.0,    -5.0/2.0,    -5.0,     1.0/2.0,   1.0,   0.0,
            0.0,   -2.0,     4.0,     5.0/2.0,    -5.0,    -1.0/2.0,   1.0,   0.0,
            0.0,   -1.0,     0.0,    21.0/4.0,     0.0,    -21.0/4.0,  0.0,   1.0,
        };
        #pragma HLS ARRAY_PARTITION variable=BT type=complete
        T AT[4*8] = {
            1.0,  1.0,  1.0,   1.0,  1.0,   8.0,  8.0,   0.0,
            0.0,  1.0,  -1.0,  2.0,  -2.0,  4.0,  -4.0,  0.0,
            0.0,  1.0,  1.0,   4.0,  4.0,   2.0,  2.0,   0.0,
            0.0,  1.0,  -1.0,  8.0,  -8.0,  1.0,  -1.0,  1.0
        };
        #pragma HLS ARRAY_PARTITION variable=AT type=complete
        T V[64];
        #pragma HLS ARRAY_PARTITION variable=AT type=complete
        T M[64];
        #pragma HLS ARRAY_PARTITION variable=M type=complete


        transform<8,8,T>(input, BT, V); // V = B^T d B

        elem_matmul<8,8,T>(U,V,M);

        transform<8,4,T>(M, AT, output); // Y = A^T M A


    }

}

template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL , int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int OUTPUT_CHANNEL,  typename T, int PE1 ,int PE2 ,int PE3 , int PII , int winoInputDim , int winobufferDim , int usize>
void winograd_base(T *input,T *filter, T *output, T* G){

#pragma HLS INLINE
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;


    T input_buffer[winoInputDim*winoInputDim];
    #pragma HLS ARRAY_PARTITION variable=input_buffer type=complete
    T winograd_buffer[winobufferDim*winobufferDim];
    #pragma HLS ARRAY_PARTITION variable=winograd_buffer type=complete
    T U[usize];
    #pragma HLS ARRAY_PARTITION variable=U type=complete

    for (int ibs = 0; ibs < INPUT_BATCH_SIZE; ibs++){
        for(int fd = 0; fd < OUTPUT_CHANNEL; fd++){
            for(int ic = 0; ic < INPUT_CHANNEL; ic++){

                transform<FILTER_WI,winoInputDim,T,PE1,PE2,PE3,PII>(filter +((ic + fd * INPUT_CHANNEL) * FILTER_HI *FILTER_WI), G, U); // U = G g G^T

                for (int ih = 0; ih < INPUT_HI - winoInputDim/2; ih += winoInputDim/2) {
                    for (int iw = 0; iw < INPUT_WI - winoInputDim/2; iw += winoInputDim/2) {


                        // copy 4x4 tile to buffer
                        for (int y = 0; y < winoInputDim; y++) {
                            for (int x = 0; x < winoInputDim; x++) {
                            #pragma HLS PIPELINE II=4
                                if(ih + y < INPUT_HI && iw + x < INPUT_WI ){
                                    input_buffer[y * winoInputDim + x] =
                                        input[(ih+y) * INPUT_WI + (iw+x) + ic * INPUT_WI * INPUT_HI + ibs * INPUT_CHANNEL * INPUT_WI * INPUT_HI];
                                }else{
                                    input_buffer[y * winoInputDim + x] = 0;
                                }
                            }
                        }
                        winograd<FILTER_WI,T,PE1,PE2,PE3,PII>(input_buffer, U, winograd_buffer);

                        // copy winograd results to output matrix
                        for (int y = 0; y < winobufferDim; y++) {
                        #pragma HLS PIPELINE II=80
                            // lower pipelines get time violations
                            for (int x = 0; x < winobufferDim; x++) {

                                if(iw + x < OUTPUT_WI && ih + y < OUTPUT_HI ){
                                    output[(ih+y) * (OUTPUT_WI) + (iw+x) + fd * OUTPUT_WI * OUTPUT_HI + ibs * OUTPUT_CHANNEL * OUTPUT_WI * OUTPUT_HI] += winograd_buffer[y * winobufferDim + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }


}


template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL , int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int OUTPUT_CHANNEL,  typename T, int PE1 = 1,int PE2 = 1,int PE3 = 1, int PII = 8>
void winograd_covolution_base(T *input,T *filter, T *output){


    if(FILTER_WI == 3 && FILTER_HI ==3){

        T G[12] = {
            1,0,0,
            0.5,0.5,0.5,
            0.5,-0.5,0.5,
            0,0,1
        };
        #pragma HLS ARRAY_PARTITION variable=G type=complete
        winograd_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, OUTPUT_CHANNEL,T , PE1 ,PE2 , PE3, PII,4, 2 , 16>
            (input,filter,output, G);

    }else if(FILTER_WI == 5 && FILTER_HI == 5){

        T G[8*5] = {
            1.0,      0.0,     0.0,      0.0,      0.0,
            -2.0/9.0,  -2.0/9.0,   -2.0/9.0,  -2.0/9.0,   -2.0/9.0,
            -2.0/9.0,   2.0/9.0,   -2.0/9.0,   2.0/9.0,   -2.0/9.0,
            1.0/90.0,  1.0/45.0,   2.0/45.0,  4.0/45.0,   8.0/45.0,
            1.0/90.0,  -1.0/45.0,  2.0/45.0,  -4.0/45.0,  8.0/45.0,
            4.0/45.0, 2.0/45.0,  1.0/45.0, 1.0/90.0,   1.0/180.0,
            4.0/45.0,  -2.0/45.0,  1.0/45.0,  -1.0/90.0,  1.0/180.0,
            0.0,      0.0,     0.0,      0.0,      1.0
        };
        #pragma HLS ARRAY_PARTITION variable=G type=complete
        winograd_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, OUTPUT_CHANNEL,T , PE1 ,PE2 , PE3, PII,8, 4 , 64>
            (input,filter,output, G);
    }



}

}

namespace Float {

template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL , int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void forward(float *input, float *weights, float *bias, float *output, activation_t act){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;

    winograd_covolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH,float>(input,filter,output);

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
    }else if(act == SIGMOID){
        Activation::sigmoid_act_inplace<OUTPUT_WI*OUTPUT_HI*FILTER_DEPTH,INPUT_BATCH_SIZE,float>(output);
    }

}

}

namespace Half {
template<int INPUT_WI,int INPUT_HI,int INPUT_CHANNEL , int INPUT_BATCH_SIZE, int FILTER_WI, int FILTER_HI, int FILTER_DEPTH>
void forward(half *input, half *weights, half *bias, half *output, activation_t act){
    const int OUTPUT_WI = INPUT_WI - FILTER_WI + 1;
    const int OUTPUT_HI = INPUT_HI - FILTER_HI + 1;

    winograd_covolution_base<INPUT_WI, INPUT_HI, INPUT_CHANNEL, INPUT_BATCH_SIZE, FILTER_WI, FILTER_HI, FILTER_DEPTH,half>(input,filter,output);

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
    }else if(act == SIGMOID){
        Activation::sigmoid_act_inplace<OUTPUT_WI*OUTPUT_HI*FILTER_DEPTH,INPUT_BATCH_SIZE,half>(output);
    }

}

}

} // namespace Winograd

} // namespace ConvLayer


#endif
