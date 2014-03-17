# Compilers
NVCC   = nvcc
CXX    = g++
LINK   = $(NVCC)

# Flags
COMMONFLAGS = -O2
NVCCFLAGS   = --compiler-options -fno-strict-aliasing -arch sm_21
CXXFLAGS    = -fno-strict-aliasing
CFLAGS      = -g -fno-strict-aliasing

# Paths
INCLUDES    = -I:/usr/lib/nvidia-cuda-toolkit/include -Iconfig
LIBS        = -L/usr/lib/nvidia-cuda-toolkit/lib
BIN         = bin/Release
SOLINT  = src/Solaris.Integrator.Cuda
SOLARIS     = src/Solaris.NBody.Cuda
SOLTEST     = src/Solaris.NBody.Cuda.Test


#Build rules
all : integrator

integrator : euler.o
	$(LINK) $(LIBS) -o $(BIN)/integrator.o $(BIN)/euler.o

#CudaNBody : CudaNBody.o euler.o integrator.o nbody.o nbody_exception.o ode.o options.o rungekutta.o rungekuttanystrom.o util.o
#	$(LINK) $(LIBS) -o $(BIN)/CudaNBody $(BIN)/CudaNBody.o $(BIN)/euler.o $(BIN)/integrator.o $(BIN)/nbody.o $(BIN)/nbody_exception.o $(BIN)/ode.o $(BIN)/options.o $(BIN)/rungekutta.o $(BIN)/rungekuttanystrom.o $(BIN)/util.o

#CudaNBody.o : $(CUDANBODY)/main.cpp
#	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/CudaNBody.o -c $< 

euler.o : $(SOLINT)/euler.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/euler.o -c $<

clean:
	rm -f $(BIN)
