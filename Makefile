# Build the TensorTuner C++ library
# 
# Usage:
# 	make

# TensorFlow flags
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Source files
SRCDIR=tensor_tuner
SRCS=$(shell find ${SRCDIR} -type f -name '*.cc' ! -name '*_test.cc')
INCDIR=. # we need only to include the current directory
TARGET=tensor_tuner
TARGET_LIBNAME=lib${TARGET}.so

# FLAGS
CFLAGS=-std=c++11 -shared -fPIC -O2
CFLAGS+=${TF_CFLAGS} -I${INCDIR}
LFLAGS=${TF_LFLAGS}

all:
	g++ ${CFLAGS} ${SRCS} -o ${TARGET_LIBNAME} ${LFLAGS}
