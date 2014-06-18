#pragma once

#include <cstdlib>

#include "config.h"
#include "integrator.h"
#include "nbody.h"
#include "ode.h"
#include "pp_disk.h"

class gas_disk;
class number_of_bodies;

using namespace std;

typedef enum frame_center
		{
			FRAME_CENTER_BARY,
			FRAME_CENTER_ASTRO
		} frame_center_t;

class options
{
public:
	typedef enum integrator_type
			{ 
				INTEGRATOR_EULER,
				INTEGRATOR_RUNGEKUTTA2,
				INTEGRATOR_OPT_RUNGEKUTTA2,
				INTEGRATOR_RUNGEKUTTA4,
				INTEGRATOR_OPT_RUNGEKUTTA4,
				INTEGRATOR_RUNGEKUTTAFEHLBERG78,
				INTEGRATOR_RUNGEKUTTANYSTROM,
				INTEGRATOR_OPT_RUNGEKUTTANYSTROM
			} integrator_type_t;


public:
	int		n;						// Number of bodies
	//ttt_t	start_time;				// Start time
	//ttt_t	stop_time;				// Stop time
	ttt_t	dt;						// Initial time step
	var_t	buffer_radius;			// collision buffer
	bool_t	printout;				// Printout enabled
	bool_t	printoutToFile;			// Printout to file
	ttt_t	printoutPeriod;			// Printout period
	ttt_t	printoutStep;			// Printout step size	
	ttt_t	printoutLength;			// Printout length
	string	printoutDir;			// Printout directory
	string	filename;				// Input file name

	number_of_bodies	*nBodies;
	gas_disk			*gasDisk;

	//integrator_type_t inttype;		// Integrator type
	bool_t adaptive;				// Adaptive step size
	//var_t tolerance;				// Tolerance
	bool_t file;					// Input file supplied
	int filen;						// Number of entries in input file
	bool_t random;					// Generate random data
	
	//! holds the path of the file containing the parameters of the simulation
	string	parameters_path;
	//! holds the path of the file containing the parameters of the nebula
	string	gasDisk_path;
	//! holds the path of the file containing the data of the bodies
	string	bodylist_path;

	//! holds a copy of the file containing the parameters of the simulation
	string	parameters_str;
	//! holds a copy of the file containing the parameters of the nebula
	string	gasDisk_str;

	//! name of the simulation
	string	sim_name;				
	//! description of the simulation
	string	sim_desc;
	//! the center of the reference frame (bary or astro centric frame)
	frame_center_t fr_cntr;
	//! type of the integrator
	integrator_type_t inttype;
	//! tolerance/eps/accuracy of the simulation
	var_t tolerance;
	//! start time of the simulation [day]
	ttt_t	start_time;
	//! length of the simulation [day]
	ttt_t	sim_length;
	//! stop time of the simulation [day] (= start_time + sim_length)
	ttt_t	stop_time;
	//! interval between two succesive output epoch [day]
	ttt_t	output_interval;
	//! the ejection distance: beyond this limit the body is removed from the simulation [AU]
	var_t	ejection_dst;
	//! the hit centrum distance: inside this limit the body is considered to have hitted the central body and removed from the simulation [AU]
	var_t	hit_centrum_dst;
	//! two bodies collide when their mutual distance is smaller than the sum of their radii multiplied by this number. Real physical collision corresponds to the value of 1.0.
	var_t	collision_factor;


public:
	options(int argc, const char** argv);
	~options();

	static void print_usage();

	ode*		create_ode();
	nbody*		create_nbody();
	pp_disk*	create_pp_disk();
	integrator* create_integrator(ode* f);

private:
	void create_default_options();
	void parse_options(int argc, const char** argv);
	void parse_params(void (options::*setter)(string& key, string& value, bool verbose));
	void parse_parameters();
	void parse_gasdisk();
	void load(string& path, string& result);
	void load(string& path);

	void set_parameters_param(string& key, string& value, bool verbose);
	void set_gasdisk_param(string& key, string& value, bool verbose);

	void initial_condition(nbody* nb);
};
