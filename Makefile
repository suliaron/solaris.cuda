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
SOLINT  	= src/Solaris.Integrator.Cuda
SOLARIS     = src/Solaris.NBody.Cuda
SOLTEST     = src/Solaris.NBody.Cuda.Test

INCLUDES    = -I:/usr/lib/nvidia-cuda-toolkit/include -Iconfig -I$(SOLINT)
LIBS        = -L/usr/lib/nvidia-cuda-toolkit/lib
BIN         = bin


#Build rules
all : integrator solaris | $(BIN)

integrator : $(BIN)/euler.o $(BIN)/integrator.o $(BIN)/integrator_exception.o $(BIN)/midpoint.o $(BIN)/ode.o $(BIN)/rk4.o $(BIN)/rkn76.o $(BIN)/rungekutta.o $(BIN)/rungekuttanystrom.o $(BIN)/util.o | $(BIN)
#	$(LINK) $(LIBS) -o $(BIN)/integrator.o $(BIN)/euler.o

solaris : integrator $(BIN)/gas_disk.o $(BIN)/nbody.o $(BIN)/nbody_exception.o $(BIN)/number_of_bodies.o $(BIN)/options.o $(BIN)/pp_disk.o

#CudaNBody : CudaNBody.o euler.o integrator.o nbody.o nbody_exception.o ode.o options.o rungekutta.o rungekuttanystrom.o util.o
#	$(LINK) $(LIBS) -o $(BIN)/CudaNBody $(BIN)/CudaNBody.o $(BIN)/euler.o $(BIN)/integrator.o $(BIN)/nbody.o $(BIN)/nbody_exception.o $(BIN)/ode.o $(BIN)/options.o $(BIN)/rungekutta.o $(BIN)/rungekuttanystrom.o $(BIN)/util.o

#CudaNBody.o : $(CUDANBODY)/main.cpp
#	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/CudaNBody.o -c $< 

$(BIN):
	mkdir $(BIN)

#integrator object files
	
$(BIN)/euler.o : $(SOLINT)/euler.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/euler.o -c $<
	
$(BIN)/integrator.o : $(SOLINT)/integrator.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/integrator.o -c $<
	
$(BIN)/integrator_exception.o : $(SOLINT)/integrator_exception.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/integrator_exception.o -c $<
	
$(BIN)/midpoint.o : $(SOLINT)/midpoint.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/midpoint.o -c $<
	
$(BIN)/ode.o : $(SOLINT)/ode.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/ode.o -c $<
	
$(BIN)/rk4.o : $(SOLINT)/rk4.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/rk4.o -c $<
	
$(BIN)/rkn76.o : $(SOLINT)/rkn76.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/rkn76.o -c $<
	
$(BIN)/rungekutta.o : $(SOLINT)/rungekutta.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/rungekutta.o -c $<
	
$(BIN)/rungekuttanystrom.o : $(SOLINT)/rungekuttanystrom.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/rungekuttanystrom.o -c $<
	
$(BIN)/util.o : $(SOLINT)/util.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/util.o -c $<
	
#nbody object files

$(BIN)/gas_disk.o : $(SOLARIS)/gas_disk.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/gas_disk.o -c $<
	
$(BIN)/nbody.o : $(SOLARIS)/nbody.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/nbody.o -c $<
	
$(BIN)/nbody_exception.o : $(SOLARIS)/nbody_exception.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/nbody_exception.o -c $<
	
$(BIN)/number_of_bodies.o : $(SOLARIS)/number_of_bodies.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/number_of_bodies.o -c $<
	
$(BIN)/options.o : $(SOLARIS)/options.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/options.o -c $<
	
$(BIN)/pp_disk.o : $(SOLARIS)/pp_disk.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(BIN)/pp_disk.o -c $<
	
clean:
	rm -f -R $(BIN)
