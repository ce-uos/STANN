#ifndef __STANN_HLS_UTILS_HPP__
#define __STANN_HLS_UTILS_HPP__

#include "stann.hpp"

/**
 * Total number of bits in fixed point values.
 */
#define FIXED_TOTAL_SIZE 32

/**
 * Number of bits in the integer part of fixed point values.
 */
#define FIXED_INT_SIZE 12

/**
 * Fixed point datatype.
 */
typedef ap_fixed<FIXED_TOTAL_SIZE,FIXED_INT_SIZE,AP_RND> fixed_t;

/**
 * Data type used as default value for template parameters of functions
 * that can be used as fixed of floating point.
 */
#define DEFAULT_DATATYPE float


/**
 * Constants for activation functions.
 */
typedef enum {
    NONE = 1,
    LIN_TANH = 2,
    LEAKY_RELU = 3,
    SIGMOID = 4,
    TANH = 5,
    SOFTMAX = 6,
    RELU = 7
} activation_t;

#endif
