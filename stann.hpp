#ifndef __NNLIB_HLS_HPP__
#define __NNLIB_HLS_HPP__

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include "utils.hpp"
#include "streamutils.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "matmul.hpp"
#include "padding.hpp"

#include "denselayer.hpp"
#include "denselayer_stream.hpp"

#include "convlayer.hpp"
#include "convlayer_kn2row.hpp"
#include "convlayer_im2row.hpp"
#include "poolinglayers.hpp"

#endif
