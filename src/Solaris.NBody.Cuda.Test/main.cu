// includes system 
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

// includes CUDA
#include "cuda.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// includes project
#include "config.h"
#include "file_util.h"
#include "nbody.h"
#include "nbody_exception.h"
#include "ode.h"
#include "pp_disk.h"
#include "options.h"
#include "tools.h"

using namespace std;

__constant__ var_t d_cst_common[THRESHOLD_N];

///////////////////////////////

/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

// Shared Utilities (QA Testing)

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}

inline bool IsAppBuiltAs64()
{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int device_query(int argc, const char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }
#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
			   deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#ifdef _WIN32
        sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
        sprintf(cTemp, ", Device%d = ", dev);
#endif
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += cTemp;
        sProfileString += deviceProp.name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

	printf("Result = PASS\n");

    // finish
    return (EXIT_SUCCESS);
}

void print_step_stat(pp_disk *ppd, options *opt, integrator* intgr, std::ostream& log_f)
{
	char time_stamp[20];
	get_time_stamp(time_stamp);

	ttt_t t = ppd->get_currt();
	ttt_t avg_dt = (t - opt->start_time)/(var_t)intgr->get_n_step();
	log_f << time_stamp << ' ';
	log_f << intgr->get_n_failed_step() << " step(s) failed out of " << intgr->get_n_step() << " steps until " << t << " [day] average dt: " << setprecision(10) << setw(16) << avg_dt << " [d]\t";
	log_f << setprecision(5) << setw(6) << (t/opt->stop_time)*100.0 << " % done" << endl;

	log_f.flush();
}

int main(int argc, const char** argv)
{
	cout << "Solaris.NBody.Cuda.Test main.cu started" << endl;
	device_query(argc, argv);

	time_t start = time(NULL);

	// Integrate the pp_disk ode
	try {
		options opt(argc, argv);

		pp_disk* ppd		= opt.create_pp_disk();
		integrator* intgr	= opt.create_integrator(ppd);

		ttt_t currt			= ppd->get_currt();
		ttt_t ps			= 0;
		ttt_t dt			= 0;

		string path = combine_path(opt.printoutDir, "position.txt");
		ostream* pos_f = new ofstream(path.c_str(), ios::out);
		path = combine_path(opt.printoutDir, "event.txt");
		ostream* event_f = new ofstream(path.c_str(), ios::out);
		path = combine_path(opt.printoutDir, "log.txt");
		ostream* log_f = new ofstream(path.c_str(), ios::out);

		// Save initial conditions to the output file
		ppd->print_positions(*pos_f);
		while (currt <= opt.stop_time)
		{
			if (fabs(ps) >= opt.output_interval)
			{
				ps = 0.0;
				ppd->copy_to_host();
				ppd->print_positions(*pos_f);
				if (opt.verbose && currt != opt.start_time)
				{
					print_step_stat(ppd, &opt, intgr, *log_f);
					cout << "t: " << setw(15) << currt << ", dt: " << setw(15) << dt << " [d]" << endl;
				}
			}
			dt = intgr->step();
			ps += fabs(dt);
			currt = ppd->get_currt();
		}
		// Save final conditions to the output file
		ppd->print_positions(*pos_f);
		if (opt.verbose)
		{
			print_step_stat(ppd, &opt, intgr, *log_f);
			cout << "t: " << setw(15) << currt << ", dt: " << setw(15) << dt << " [d]" << endl;
		}
	} /* try */
	catch (nbody_exception& ex)
	{
		cerr << "Error: " << ex.what() << endl;
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return 0;
}

#if 0
// -nBodies 1 1 0 10000 0 100000 0 -i RKF78 -a 1.0e-10 -t 1000 -dt 10.0 -p 10 10 10 -o C:\Work\Solaris.Cuda.TestRuns\2MStar_5MJupiter_Disc65-270_01\GPU -f C:\Work\Solaris.Cuda.TestRuns\2MStar_5MJupiter_Disc65-270_01\GPU\nBodies_1_1_0_10000_0_100000_0.txt
int main(int argc, const char** argv)
{
	cout << "Solaris.NBody.Cuda.Test main.cu started" << endl;
	device_query(argc, argv);

	time_t start = time(NULL);

	// Integrate the pp_disk ode
	try {
		options opt(argc, argv);

		pp_disk* ppd		= opt.create_pp_disk();
		integrator* intgr	= opt.create_integrator(ppd);

		ttt_t pp			= 0;
		ttt_t ps			= 0;
		ttt_t dt			= 0;

		ostream* positionsf = 0;
		ostream* orbelemf	= 0;
		//ostream* collisionsf= 0;
		int pcount			= 0;
		int ccount			= 0;

		if (!opt.printoutToFile) {
			positionsf = &cout;
			orbelemf   = &cout;
			//collisionsf = &cerr;
		}
		else {
			//collisionsf = new ofstream(combine_path(opt.printoutDir, "col.txt").c_str());
			//positionsf = new ofstream(get_printout_file(opt, pcount++).c_str());
			string filename = get_filename_without_ext(opt.filename) + '.' + intgr->get_name() + '.' + (opt.gasDisk == 0 ? "" : "gas.CONSTANT.");
			string filenameWithExt = filename + get_extension(opt.filename);
			string path = combine_path(opt.printoutDir, filenameWithExt);
			//char *c_path = new char[path.length() + 1];
			//strcpy(c_path, path.c_str());
			positionsf = new ofstream(path.c_str(), ios::app);
			//filenameWithExt = filename + "oe." + get_extension(opt.filename);
			//orbelemf = new ofstream(combine_path(opt.printoutDir, filenameWithExt), std::ios::app);
		}

		while (ppd->t < opt.stop_time) {

			if (opt.printout) {
				if (pp >= opt.printoutPeriod) {
					pp = 0;
				}

				// Start of a print-out period, create new file if necessary
				if (pp == 0 && intgr->get_n_step() > 0) {
					var_t avg_dt = (ppd->t - opt.start_time) / intgr->get_n_step();
					cout << intgr->get_n_failed_step() << " step(s) failed out of " << intgr->get_n_step() << " steps until " << ppd->t << " [day]\naverage dt: " << setprecision(10) << setw(16) << avg_dt << " [d]" << endl;
					cerr << setprecision(5) << setw(6) << ((ppd->t - opt.start_time)/opt.stop_time*100) << " %" << endl;
				}
				//var_t avg_dt = (ppd->t - opt.start_time) / intgr->get_n_step();
				//cout << intgr->get_n_failed_step() << " step(s) failed out of " << intgr->get_n_step() << " steps until " << ppd->t << " [day]\naverage dt: " << setprecision(10) << setw(16) << avg_dt << " [d]" << endl;
				//cerr << setprecision(5) << setw(6) << ((ppd->t - opt.start_time)/opt.stop_time*100) << " %" << endl;

				if (0 <= pp && pp <= opt.printoutLength) {
					if (ps >= opt.printoutStep) {
						ps = 0;
					}

					if (ps == 0) {
						// Print out positions
						ppd->copy_to_host();
						ppd->print_positions(*positionsf);
						//pp_disk::h_orbelem_t orbelem = ppd->calculate_orbelem(0);
						//ppd->print_orbelem(*orbelemf);
					}
				}
			}
			dt = intgr->step();
			cerr << "t: " << setw(15) << (ppd->t) << ", dt: " << setw(15) << dt << " [d]" << endl;

			pp += dt;
			ps += dt;
		}

		delete ppd;
		delete intgr;
		delete positionsf;
		delete orbelemf;
	} /* try */
	catch (nbody_exception& ex)
	{
		cerr << "Error: " << ex.what() << endl;
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return 0;
}
#endif
