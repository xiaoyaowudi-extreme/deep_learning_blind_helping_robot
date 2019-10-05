#!/bin/bash
CC = g++ 
CPPFLAGS = -DNDEBUG -D_FILE_OFFSET_BITS=64 
CXXFLAGS = -fPIC -shared -O2 -finline-functions -fomit-frame-pointer -pipe -fvisibility=hidden -march=corei7-avx -mavx2 -mavx -msse4.1 -msse4.2 -mfma -m64 -maccumulate-outgoing-args -m128bit-long-double -fprefetch-loop-arrays 
LDFLAGS = -s -Wl,--enable-new-dtags 
INCLUDE = -I/usr/local/include/tf -I/usr/local/include/tf/tensorflow/contrib/makefile/downloads/eigen -I/usr/local/include/tf/tensorflow/contrib/makefile/downloads -I/usr/local/include/tf/tensorflow/contrib/makefile/downloads/absl  -L/usr/local/include/tf/tensorflow -ltensorflow_cc -ltensorflow_framework `pkg-config --cflags --libs protobuf` `pkg-config --cflags --libs opencv4` 
#INCLUDE = `pkg-config --cflags --libs opencv4` 
files = run/main.cxx 
execute = run/classify_server
refrence_files = run/extractSkeleton.hpp
all: $(refrence_files)
	$(CC)$(CPPFLAGS)$(CXXFLAGS)$(LDFLAGS)$(files)$(INCLUDE) -o $(execute)