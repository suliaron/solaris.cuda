# Compilers
NVCC   = nvcc
LINK   = $(NVCC)

# Flags
COMMONFLAGS = -O2
NVCCFLAGS   = $(COMMONFLAGS) --compiler-options -fno-strict-aliasing -arch sm_21

# Paths
SOLINT  	= src/Solaris.Integrator.Cuda
SOLARIS     = src/Solaris.NBody.Cuda
SOLTEST     = src/Solaris.NBody.Cuda.Test

CUDA        = /usr/lib/nvidia-cuda-toolkit
INCLUDES    = -I$(CUDA)/include -Iconfig -I$(SOLINT) -I$(SOLARIS)
LIBS        = -L$(CUDA)/lib
BIN         = bin


#Build rules
all : $(BIN)/integrator.a $(BIN)/solaris.a $(BIN)/soltest | $(BIN)

$(BIN)/integrator.a : $(BIN)/euler.o $(BIN)/integrator.o $(BIN)/integrator_exception.o $(BIN)/midpoint.o $(BIN)/ode.o $(BIN)/rk4.o $(BIN)/rkn76.o $(BIN)/rungekutta.o $(BIN)/rungekuttanystrom.o $(BIN)/util.o | $(BIN)
	ar cr $@ $?

$(BIN)/solaris.a : $(BIN)/integrator.a $(BIN)/gas_disk.o $(BIN)/nbody.o $(BIN)/nbody_exception.o $(BIN)/number_of_bodies.o $(BIN)/options.o $(BIN)/pp_disk.o
	ar cr $@ $?

$(BIN)/soltest : $(BIN)/integrator.a $(BIN)/solaris.a $(BIN)/main.o
	$(LINK) $(LIBS) -o $@ $?

$(BIN):
	mkdir $(BIN)

#integrator object files
	
$(BIN)/%.o : $(SOLINT)/%.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
$(BIN)/%.o : $(SOLINT)/%.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
#nbody object files

$(BIN)/%.o : $(SOLARIS)/%.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
$(BIN)/%.o : $(SOLARIS)/%.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
#test

$(BIN)/%.o : $(SOLTEST)/%.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
$(BIN)/%.o : $(SOLTEST)/%.cpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
	
clean:
	rm -f -R $(BIN)
