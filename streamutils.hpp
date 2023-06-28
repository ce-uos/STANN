#ifndef __STANN_HLS_STREAM_HPP__
#define __STANN_HLS_STREAM_HPP__

#include <stdio.h>

#include "stann.hpp"

/**
 * This namespace contains some useful utility functions to handle streams.
 */
namespace StreamUtil {

template<int INPUT_DIM>
void print_stream_float(hls::stream<float> &input, hls::stream<float> &output, int reps) {
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            float tmp = input.read();
            printf("%f ", tmp);
            output.write(tmp);
        }
    }
}

/**
 * Takes an array and writes each element to a stream.
 *
 * @tparam  INPUT_DIM   dimension of input array
 * @tparam  T           datatype to work with (should usually be float or fixed_t)
 *
 * @param[in]   input       input array
 * @param[out]  output      output stream
 */
template<int INPUT_DIM, typename T = DEFAULT_DATATYPE>
void tostream(T *input, hls::stream<T> &output) {
#pragma HLS inline
    for (int i = 0; i < INPUT_DIM; i++) {
        output.write(input[i]);
    }
}

/**
 * Takes an array and writes each element to a stream. Multiple repetitions.
 *
 * @tparam  INPUT_DIM   dimension of input array
 * @tparam  T           datatype to work with (should usually be float or fixed_t)
 *
 * @param[in]   input       input array
 * @param[out]  output      output stream
 * @param[in]   reps        number of repetitions
 */
template<int INPUT_DIM, typename T = DEFAULT_DATATYPE>
void tostream(T *input, hls::stream<T> &output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            //output.write(input[r * INPUT_DIM + i]);
            output.write(input[i * reps + r]);
        }
    }
}

/**
 * Takes an array and writes each element to a stream. 
 * Converts between two data types while doing this.
 * Multiple repetitions.
 *
 * @tparam  INPUT_DIM   dimension of input array
 * @tparam  T_IN        input data type
 * @tparam  T_OUT       output data type
 *
 * @param[in]   input       input array
 * @param[out]  output      output stream
 * @param[in]   reps        number of repetitions
 */
template<int INPUT_DIM, typename T_IN, typename T_OUT >
void tostream_convert(T_IN *input, hls::stream<T_OUT> &output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            T_OUT tmp = *reinterpret_cast<T_OUT*>(&input[i * reps + r]);
            output.write(tmp);
        }
    }
}

/**
 * Takes a stream, copies the content into an array, and also forwards the
 * content into a next stream. Mostly useful for debugging.
 *
 * @tparam  INPUT_DIM   dimension of input array
 * @tparam  T           datatype to work with (should usually be float or fixed_t)
 *
 * @param[in]   input       input stream
 * @param[out]  copy        copy array
 * @param[out]  output      output stream
 * @param[in]   reps        number of repetitions
 */
template<int INPUT_DIM, typename T>
void getcopy(hls::stream<T> &input, T *copy, hls::stream<T> &output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            T tmp = input.read();
            copy[r * INPUT_DIM + i] = tmp;
            output.write(tmp);
        }
    }
}

/**
 * Takes data from a stream and writes to an array.
 *
 * @tparam  INPUT_DIM   amount of data to read from the stream
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 * 
 * @param   input       input stream
 * @param   output      output array
 */
template<int INPUT_DIM, typename T = DEFAULT_DATATYPE>
void toarray(hls::stream<T> &input, T *output) {
#pragma HLS inline
    for (int i = 0; i < INPUT_DIM; i++) {
        T data = input.read();
        output[i] = data;
    }
}

/**
 * Takes data from a stream and writes to an array. Multiple repetitions.
 *
 * @tparam  INPUT_DIM   amount of data to read from the stream
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 * 
 * @param   input       input stream
 * @param   output      output array
 */
template<int INPUT_DIM, typename T = DEFAULT_DATATYPE>
void toarray(hls::stream<T> &input, T *output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            T data = input.read();
            output[i * reps + r] = data;
            //output[r * INPUT_DIM + i] = data;
        }
    }
}

/**
 * Takes data from a stream and writes to an array. Multiple repetitions.
 * Stores the data in transposed order.
 *
 * @tparam  INPUT_DIM   amount of data to read from the stream
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 * 
 * @param   input       input stream
 * @param   output      output array
 */
template<int INPUT_DIM, typename T = DEFAULT_DATATYPE>
void toarray_transposed(hls::stream<T> &input, T *output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            T data = input.read();
            output[r * INPUT_DIM + i] = data;
        }
    }
}

/**
 * Takes data from a stream and writes to an array. Multiple repetitions.
 * Converts between two data types while doing this.
 *
 * @tparam  INPUT_DIM   amount of data to read from the stream
 * @tparam  T_IN        input data type
 * @tparam  T_OUT       output data type
 * 
 * @param   input       input stream
 * @param   output      output array
 */
template<int INPUT_DIM, typename T_IN, typename T_OUT>
void toarray_convert(hls::stream<T_IN> &input, T_OUT *output, int reps) {
#pragma HLS inline
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < INPUT_DIM; i++) {
            T_IN data = input.read();
            output[i] = *reinterpret_cast<T_OUT*>(&data);
        }
    }
}

/**
 * Takes a batch of data and converts it to a stream.
 *
 * @tparam  INPUT_DIM   input dimension
 * @tparam  BATCH_SIZE  number of inputs
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   input       input array
 * @param   output      output stream
 */
template<int INPUT_DIM, int BATCH_SIZE, typename T = DEFAULT_DATATYPE>
void batchtostream(T *input, hls::stream<T> &output) {
#pragma HLS inline
    for (int i = 0; i < INPUT_DIM; i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            output.write(input[b * INPUT_DIM + i]);
        }
    }
}

/**
 * Takes data from a source stream and writes it to two different streams.
 *
 * @tparam  SIZE   how much data to read from the source
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   in_stream    input stream
 * @param   out_stream1  first output stream
 * @param   out_stream2  second output stream
 */
template<int SIZE, typename T = DEFAULT_DATATYPE>
void duplicate(hls::stream<T> &in_stream, hls::stream<T> &out_stream1, hls::stream<T> &out_stream2) {
#pragma HLS inline
    for (int i = 0; i < SIZE; i++) {
        T d = in_stream.read();
        out_stream1.write(d);
        out_stream2.write(d);
    }
}

/**
 * Takes data from a source stream and writes it to three different streams.
 *
 * @tparam  SIZE   how much data to read from the source
 * @tparam     T            datatype to work with (should usually be float or fixed_t)
 *
 * @param   in_stream    input stream
 * @param   out_stream1  first output stream
 * @param   out_stream2  second output stream
 * @param   out_stream3  third output stream
 */
template<int SIZE, typename T = DEFAULT_DATATYPE>
void triplicate(hls::stream<T> &in_stream, hls::stream<T> &out_stream1, hls::stream<T> &out_stream2, hls::stream<T> &out_stream3) {
#pragma HLS inline
    for (int i = 0; i < SIZE; i++) {
        T d = in_stream.read();
        out_stream1.write(d);
        out_stream2.write(d);
        out_stream3.write(d);
    }
}
};

#endif
