#ifndef __STANN_HLS_MATMUL_HPP__
#define __STANN_HLS_MATMUL_HPP__

#include "stann.hpp"

/**
 * Namespace for matrix operations.
 */
namespace MatrixUtil {

/**
 * Namespace for basic implementations of matrix operations.
 */
namespace Basic {

/**
 * Basic matrix multiplication. Multiplies two matrices A and B.
 * "A" should have dimensions KxM
 * "B" should have dimensions MxN
 *
 * @tparam  K  rows of first matrix
 * @tparam  M  cols of first matrix, rows of second matrix
 * @tparam  N  cols of second matrix
 * @tparam  T  data type used for the matrices
 *
 * @param[in]   a   first input matrix (KxM)
 * @param[in]   b   second input matrix (MxN)
 * @param[out]  c   output matrix (KxN)
 */
template<int K, int M, int N, typename T = float>
void matmul(T *a, T *b, T *c) {
#pragma HLS inline
    for (int k = 0; k < K; k++) {
    #pragma HLS pipeline
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                c[k * N + n] += a[k * M + m] * b[m * N + n];
            }
        }
    }

}


/**
 * Block matrix multiplication. Multiplies two matrices A and B.
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  N     cols of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  BN    cols of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a   first input matrix (KxM)
 * @param[in]     b   second input matrix (MxN)
 * @param[out]    c   output matrix (KxN)
 */
template<int K, int M, int N, int BK, int BM, int BN, typename T, int PII = 1>
void blockmatmul(T *a, T *b, T*c) {
#pragma HLS inline

    int k = 0;
    int m = 0;
    int n = 0;

    T bufferA[BK * BM];
    #pragma HLS ARRAY_PARTITION variable=bufferA complete
    T bufferB[BM * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferB complete
    T bufferC[BK * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferC complete

    for (int i = 0; i < (K/BK) * (M/BM) * (N/BN); i++) {
    #pragma HLS pipeline II=PII

        for (int bk = 0; bk < BK; bk++) {
        #pragma HLS unroll
            for (int bm = 0; bm < BM; bm++) {
            #pragma HLS unroll
                bufferA[bk * BM + bm] = a[(k+bk) * M + (m+bm)];
            }
        }

        for (int bm = 0; bm < BM; bm++) {
        #pragma HLS unroll
            for (int bn = 0; bn < BN; bn++) {
            #pragma HLS unroll
                bufferB[bm * BN + bn] = b[(m+bm) * N + (n+bn)];
            }
        }

        if (m == 0) {
            for (int c = 0; c < BK * BN; c++) {
            #pragma HLS unroll
                bufferC[c] = 0;
            }
        }

        matmul<BK,BM,BN,T>(bufferA, bufferB, bufferC);

        m += BM;
        if (m >= M) {

            for (int bk = 0; bk < BK; bk++) {
            #pragma HLS unroll
                for (int bn = 0; bn < BN; bn++) {
                #pragma HLS unroll
                    c[(k+bk) * N + (n+bn)] = bufferC[bk * BN + bn];
                }
            }


            m = 0;
            n += BN;
            if (n >= N) {
                k += BK;
                n = 0;
            }
        }
    }

}

} // namespace basic

/**
 * Namespace for systolic array based implementations of matrix operations.
 */
namespace SysArr {

/**
 * Systolic array based matrix multiplication.
 * Systolic array will have size MxN.
 *
 * @tparam  K  rows of first matrix
 * @tparam  M  cols of first matrix, rows of second matrix
 * @tparam  N  cols of second matrix
 * @tparam  T  data type used for the matrices
 *
 * @param[in]   input              first input matrix (KxM)
 * @param[in]   stationary_input   second input matrix (MxN)
 * @param[out]  output             output matrix (KxN)
 */
template<int K, int M, int N, typename T = float>
void matmul(T *input, T *stationary_input, T *output) {
#pragma HLS inline

    // stationary_input = MxN = number of PEs
    // input = KxM
    // output = KxN

    T pe_state[M*N];
    #pragma HLS ARRAY_PARTITION variable=pe_state complete
    T stationary_buffer[M*N];
    #pragma HLS ARRAY_PARTITION variable=stationary_buffer complete
    T input_buffer_left[M*N];
    #pragma HLS ARRAY_PARTITION variable=input_buffer_left complete
    T input_buffer_top[M*N];
    #pragma HLS ARRAY_PARTITION variable=input_buffer_top complete

    // initialization
    for(int i = 0; i < M*N; i++) {
    #pragma HLS pipeline II=1
        stationary_buffer[i] = stationary_input[i];
        pe_state[i] = 0;
        input_buffer_left[i] = 0;
        input_buffer_top[i] = 0;
    }

    //for (int t = K*2-2 + M + N; t > 0; t--) {
    for (int t = M + N; t > 0; t--) {
    #pragma HLS unroll factor=1

        // move input_buffer_left to the right
        for (int m = 0; m < M; m++) {
            for (int n = N-1; n > 0; n--) {
                input_buffer_left[m * N + n] = input_buffer_left[m * N + n-1];
            }
        }

        // read from input, assuming the input array is ordered for systolic computation
        for (int m = 0; m < M; m++) {
            int input_idx = m * (K*2) + (t-N);
            if (input_idx >= 0 && input_idx < M * (K*2)) {
                input_buffer_left[m * N + 0] = input[m * (K*2) + (t-N)];
            } else {
                input_buffer_left[m * N + 0] = 0;
            }
        }

        // fill input_buffer_top
        for (int m = 1; m < M; m++) {
            for (int n = 0; n < N; n++) {
                input_buffer_top[m * N + n] = pe_state[(m-1) * N + n];
            }
        }

        // pe computation
        for (int m = 0; m < M; m++) {
        #pragma HLS unroll
            for (int n = 0; n < N; n++) {
            #pragma HLS unroll
                //T tmp1 = stationary_buffer[m * N + n] * input_buffer_left[m * N + n];
                //#pragma HLS bind_op variable=tmp1 op=fmul impl=fulldsp
                //T tmp2 = tmp1 + input_buffer_top[m * N + n];
                //#pragma HLS bind_op variable=tmp2 op=fadd impl=fulldsp
                //pe_state[m * N + n] = tmp2;
                pe_state[m * N + n] = stationary_buffer[m * N + n] * input_buffer_left[m * N + n] + input_buffer_top[m * N + n];
            }
        }

        // move outputs down
        for (int k = K*2-1+N-1; k > 0; k--) {
            for (int n = 0; n < N; n++) {
                output[k * N + n] = output[(k-1) * N + n];
            }
        }

        // store new outpupt
        for (int n = 0; n < N; n++) {
            output[n] = pe_state[(M-1) * N + n];
        }

    }


}

/**
 * Block matrix multiplication based on systolic array. Multiplies two matrices A and B.
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  N     cols of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  BN    cols of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a   first input matrix (KxM)
 * @param[in]     b   second input matrix (MxN)
 * @param[out]    c   output matrix (KxN)
 */
template<int K, int M, int N, int BK, int BM, int BN, typename T, int PII = 1>
void blockmatmul(T *a, T *b, T*c) {
#pragma HLS inline off

    int k = 0;
    int m = 0;
    int n = 0;

    T bufferA[BM * (BK*2)];
    #pragma HLS ARRAY_PARTITION variable=bufferA complete
    T bufferB[BM * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferB complete
    T bufferC_systolic[(BK*2+BN-1) * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferC_systolic complete
    T bufferC[BK * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferC complete

    for (int i = 0; i < BM * (BK*2); i++) {
    #pragma HLS unroll
        bufferA[i] = 0;
    }

    for (int i = 0; i < (BK*2+BN-1)*BN; i++) {
    #pragma HLS unroll
        bufferC_systolic[i] = 0;
    }

    for (int i = 0; i < (K/BK) * (M/BM) * (N/BN); i++) {
    #pragma HLS pipeline II=PII

        // systolic input
        for (int bk = 0; bk < BK; bk++) {
        #pragma HLS unroll
            for (int bm = 0; bm < BM; bm++) {
            #pragma HLS unroll
                //#pragma HLS pipeline II=5
                bufferA[bm * (BK*2) + (bk+(BM-1-bm))] = a[(k+bk) * M + (m+bm)];
            }
        }

        // stationary buffer
        for (int bm = 0; bm < BM; bm++) {
        #pragma HLS unroll
            for (int bn = 0; bn < BN; bn++) {
            #pragma HLS unroll
                //#pragma HLS pipeline II=5
                bufferB[bm * BN + bn] = b[(m+bm) * N + (n+bn)];
            }
        }

        if (m == 0) {
            for (int c = 0; c < (BK*2-1+BN) * BN; c++) {
            #pragma HLS unroll
                bufferC_systolic[c] = 0;
            }

            for (int c = 0; c < BK * BN; c++) {
            #pragma HLS unroll
                bufferC[c] = 0;
            }
        }

        matmul<BK,BM,BN,T>(bufferA, bufferB, bufferC_systolic);

        for (int bk = 0; bk < BK; bk++) {
        #pragma HLS unroll
            for (int bn = 0; bn < BN; bn++) {
            #pragma HLS unroll
                bufferC[bn * BK + bk] += bufferC_systolic[(bk+(BN-1-bn)) * BN + bn];
            }
        }

        m += BM;
        if (m >= M) {

            for (int bk = 0; bk < BK; bk++) {
            #pragma HLS unroll
                for (int bn = 0; bn < BN; bn++) {
                #pragma HLS unroll
                    //#pragma HLS pipeline II=5
                    c[(k+bk) * N + (n+bn)] = bufferC[bn * BK + bk];
                }
            }

            m = 0;
            n += BN;
            if (n >= N) {
                k += BK;
                n = 0;
            }
        }
    }

}

/**
 * Systolic array based quantized matrix multiplication.
 * Systolic array will have size MxN.
 *
 * @tparam  K  rows of first matrix
 * @tparam  M  cols of first matrix, rows of second matrix
 * @tparam  N  cols of second matrix
 *
 * @param[in]   input              first input matrix (KxM)
 * @param[in]   stationary_input   second input matrix (MxN)
 * @param[out]  output             output matrix (KxN)
 * @param[out]  z1                 constant for quantized computation
 * @param[out]  z2                 constant for quantized computation
 */
template<int K, int M, int N>
void matmul_quantized(ap_uint<8> *input, ap_uint<8> *stationary_input, ap_uint<32> *output, ap_uint<8> z1, ap_uint<8> z2) {
#pragma HLS inline 

    // stationary_input = MxN = number of PEs
    // input = KxM
    // output = KxN

    ap_int<32> pe_state[M*N];
    #pragma HLS ARRAY_PARTITION variable=pe_state complete
    ap_uint<8> stationary_buffer[M*N];
    #pragma HLS ARRAY_PARTITION variable=stationary_buffer complete
    ap_uint<8> input_buffer_left[M*N];
    #pragma HLS ARRAY_PARTITION variable=input_buffer_left complete
    ap_int<32> input_buffer_top[M*N];
    #pragma HLS ARRAY_PARTITION variable=input_buffer_top complete

    // initialization
    for(int i = 0; i < M*N; i++) {
    #pragma HLS pipeline II=1
        stationary_buffer[i] = stationary_input[i] - z2;
        pe_state[i] = 0;
        input_buffer_left[i] = 0;
        input_buffer_top[i] = 0;
    }

    //for (int t = K*2-2 + M + N; t > 0; t--) {
    for (int t = M + N; t > 0; t--) {
    #pragma HLS unroll factor=1

        // move input_buffer_left to the right
        for (int m = 0; m < M; m++) {
            for (int n = N-1; n > 0; n--) {
                input_buffer_left[m * N + n] = input_buffer_left[m * N + n-1];
            }
        }

        // read from input, assuming the input array is ordered for systolic computation
        for (int m = 0; m < M; m++) {
            int input_idx = m * (K*2) + (t-N);
            if (input_idx >= 0 && input_idx < M * (K*2)) {
                input_buffer_left[m * N + 0] = input[m * (K*2) + (t-N)] - z1;
            } else {
                input_buffer_left[m * N + 0] = 0;
            }
        }

        // fill input_buffer_top
        for (int m = 1; m < M; m++) {
            for (int n = 0; n < N; n++) {
                input_buffer_top[m * N + n] = pe_state[(m-1) * N + n];
            }
        }

        // pe computation
        for (int m = 0; m < M; m++) {
        #pragma HLS unroll
            for (int n = 0; n < N; n++) {
            #pragma HLS unroll
                //T tmp1 = stationary_buffer[m * N + n] * input_buffer_left[m * N + n];
                //#pragma HLS bind_op variable=tmp1 op=fmul impl=fulldsp
                //T tmp2 = tmp1 + input_buffer_top[m * N + n];
                //#pragma HLS bind_op variable=tmp2 op=fadd impl=fulldsp
                //pe_state[m * N + n] = tmp2;
                ap_uint<16> tmp = ((ap_uint<8>)stationary_buffer[m * N + n]) * ((ap_uint<8>)input_buffer_left[m * N + n]);
                pe_state[m * N + n] = tmp + input_buffer_top[m * N + n];
            }
        }

        // move outputs down
        for (int k = K*2-1+N-1; k > 0; k--) {
            for (int n = 0; n < N; n++) {
                output[k * N + n] = output[(k-1) * N + n];
            }
        }

        // store new outpupt
        for (int n = 0; n < N; n++) {
            output[n] = pe_state[(M-1) * N + n];
        }

    }

}

/**
 * Quantized block matrix multiplication based on systolic array. Multiplies two matrices A and B.
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  N     cols of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  BN    cols of sub-matrices of second matrix
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a      first input matrix (KxM)
 * @param[in]     b      second input matrix (MxN)
 * @param[out]    c      output matrix (KxN)
 * @param[out]    bias   bias vector to be added after matrix multiplication
 * @param[out]    mscale constant for quantized computation
 * @param[out]    nscale constant for quantized computation
 * @param[out]    z1     constant for quantized computation
 * @param[out]    z2     constant for quantized computation
 * @param[out]    z3     constant for quantized computation
 */
template<int K, int M, int N, int BK, int BM, int BN, int PII = 1>
void blockmatmul_quantized(ap_uint<8> *a, ap_uint<8> *b, ap_uint<8>*c, ap_uint<32> *bias, ap_uint<32> mscale, ap_uint<32> nscale, ap_uint<8> z1, ap_uint<8> z2, ap_uint<8> z3) {
#pragma HLS inline

    int k = 0;
    int m = 0;
    int n = 0;

    ap_uint<8> bufferA[BM * (BK*2)];
    #pragma HLS ARRAY_PARTITION variable=bufferA complete
    ap_uint<8> bufferB[BM * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferB complete
    ap_uint<32> bufferC_systolic[(BK*2+BN-1) * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferC_systolic complete
    ap_uint<32> bufferC[BK * BN];
    #pragma HLS ARRAY_PARTITION variable=bufferC complete

    for (int i = 0; i < BM * (BK*2); i++) {
    #pragma HLS unroll
        bufferA[i] = 0;
    }

    for (int i = 0; i < (BK*2+BN-1)*BN; i++) {
    #pragma HLS unroll
        bufferC_systolic[i] = 0;
    }

    for (int i = 0; i < (K/BK) * (M/BM) * (N/BN); i++) {
    #pragma HLS pipeline II=PII

        // systolic input
        for (int bk = 0; bk < BK; bk++) {
        #pragma HLS unroll
            for (int bm = 0; bm < BM; bm++) {
            #pragma HLS unroll
                //#pragma HLS pipeline II=5
                bufferA[bm * (BK*2) + (bk+(BM-1-bm))] = a[(k+bk) * M + (m+bm)];
            }
        }

        // stationary buffer
        for (int bm = 0; bm < BM; bm++) {
        #pragma HLS unroll
            for (int bn = 0; bn < BN; bn++) {
            #pragma HLS unroll
                //#pragma HLS pipeline II=5
                bufferB[bm * BN + bn] = b[(m+bm) * N + (n+bn)];
            }
        }

        if (m == 0) {
            for (int c = 0; c < (BK*2-1+BN) * BN; c++) {
            #pragma HLS unroll
                bufferC_systolic[c] = 0;
            }

            for (int c = 0; c < BK * BN; c++) {
            #pragma HLS unroll
                bufferC[c] = 0;
            }
        }

        matmul_quantized<BK,BM,BN>(bufferA, bufferB, bufferC_systolic, z1, z2);

        for (int bk = 0; bk < BK; bk++) {
        #pragma HLS unroll
            for (int bn = 0; bn < BN; bn++) {
            #pragma HLS unroll
                bufferC[bn * BK + bk] += bufferC_systolic[(bk+(BN-1-bn)) * BN + bn];
            }
        }

        m += BM;
        if (m >= M) {

            for (int bk = 0; bk < BK; bk++) {
            #pragma HLS unroll
                for (int bn = 0; bn < BN; bn++) {
                #pragma HLS unroll
                    //#pragma HLS pipeline II=5
                    ap_uint<32> tmp = bufferC[bn * BK + bk] + bias[k+bk];
                    tmp = (tmp >> nscale) * mscale - z3; 
                    tmp = tmp > ((ap_uint<32>)255) ? ((ap_uint<32>)255) : tmp;
                    c[(k+bk) * N + (n+bn)] = static_cast<ap_uint<8>>(tmp);
                }
            }

            m = 0;
            n += BN;
            if (n >= N) {
                k += BK;
                n = 0;
            }
        }
    }

}

} // namespace SysArr

}


/**
 * Namespace for matrix multiplication using streams.
 */
namespace MatrixUtilStream {

/**
 * Block matrix multiplication with stream as second input and output
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  N     cols of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  BN    cols of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a     first input matrix (KxM)
 * @param[in]     b     second input matrix (MxN)
 * @param[out]    c     output matrix (KxN)
 * @param[out]    reps  number of repetitions (similar to batch size)
 */
template<int K, int M, int N, int BK, int BM, int BN, typename T, int PII = 1>
void blockmatmul_full(T *a, hls::stream<T> &b, hls::stream<T> &c, int reps) {
    T b_buffer[M];
    T c_buffer[K];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<M * N>(b, b_buffer, 1);
        MatrixUtil::SysArr::blockmatmul<K,M,N,BK,BM,BN,T,PII>(a, b_buffer, c_buffer);
        StreamUtil::tostream<K>(c_buffer, c, 1);
    }
}

/**
 * Block matrix multiplication with stream as second input and output
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a     first input matrix (KxM)
 * @param[in]     b     second input matrix (MxN)
 * @param[out]    c     output matrix (KxN)
 * @param[out]    reps  number of repetitions (similar to batch size)
 */
template<int K, int M, int BK, int BM, typename T, int PII = 1>
void blockmatmul(T *a, hls::stream<T> &b, hls::stream<T> &c, int reps) {
    T b_buffer[M];
    T c_buffer[K];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<M>(b, b_buffer, 1);
        MatrixUtil::SysArr::blockmatmul<K,M,1,BK,BM,1,T,PII>(a, b_buffer, c_buffer);
        StreamUtil::tostream<K>(c_buffer, c, 1);
    }
}

/**
 * Quantized block matrix multiplication with stream as second input and output
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a     first input matrix (KxM)
 * @param[in]     b     second input matrix (MxN)
 * @param[out]    c     output matrix (KxN)
 * @param[out]    m     constant for quantized computation
 * @param[out]    n     constant for quantized computation
 * @param[out]    z1    constant for quantized computation
 * @param[out]    z2    constant for quantized computation
 * @param[out]    z3    constant for quantized computation
 * @param[out]    reps  number of repetitions (similar to batch size)
 */
template<int K, int M, int BK, int BM, int PII = 1>
void blockmatmul_quantized(ap_uint<8> *a, hls::stream<ap_uint<8>> &b, hls::stream<ap_uint<8>> &c, ap_uint<32> *bias, ap_uint<32> m, ap_uint<32> n, ap_uint<8> z1, ap_uint<8> z2, ap_uint<8> z3, int reps) {
    ap_uint<8> b_buffer[M];
    ap_uint<8> c_buffer[K];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<M, ap_uint<8>>(b, b_buffer, 1);
        MatrixUtil::SysArr::blockmatmul_quantized<K,M,1,BK,BM,1,PII>(a, b_buffer, c_buffer,bias,m,n,z1,z2,z3);
        StreamUtil::tostream<K, ap_uint<8>>(c_buffer, c, 1);
    }
}

/**
 * Block matrix multiplication with stream as first input and output
 * "A" should have dimensions KxM.
 * "B" should have dimensions MxN.
 * "A" will be sliced into sub-matrices of size BKxBM.
 * "B" will be sliced into sub-matrices of size BMxNB.
 * Systolic array will have size BMxBN.
 *
 * @tparam  K     rows of first matrix
 * @tparam  M     cols of first matrix, rows of second matrix
 * @tparam  BK    rows of sub-matrices of first matrix
 * @tparam  BM    cols of sub-matrices of first matrix, rows of sub-matrices of second matrix
 * @tparam  T     data type used for the matrices
 * @tparam  PII   pipelining constant for HLS
 *
 * @param[in]     a     first input matrix (KxM)
 * @param[in]     b     second input matrix (MxN)
 * @param[out]    c     output matrix (KxN)
 * @param[out]    reps  number of repetitions (similar to batch size)
 */
template<int K, int M, int BK, int BM, typename T, int PII = 1>
void blockmatmul(hls::stream<T> &a, T *b, hls::stream<T> &c, int reps) {
    T a_buffer[K*M];
    T c_buffer[K];

    for (int r = 0; r < reps; r++) {
        StreamUtil::toarray<K*M>(a, a_buffer, 1);
        MatrixUtil::SysArr::blockmatmul<K,M,1,BK,BM,1,T,PII>(a_buffer, b, c_buffer);
        StreamUtil::tostream<K>(c_buffer, c, 1);
    }
}

} // namespace MatrixUtilStream

#endif
