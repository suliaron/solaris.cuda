#include <algorithm>
#include <iostream>
#include <fstream>

#include <math.h>

#include "constants.h"
#include "file_util.h"
#include "gas_disk.h"
#include "options.h"
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "tokenizer.h"
#include "tools.h"

#include "euler.h"
#include "midpoint.h"
#include "rk4.h"
#include "rkn76.h"
#include "rungekutta.h"
#include "rungekuttanystrom.h"
#include "rkf7.h"

void set_parameters_param(string& key, string& value, void* data, bool verbose)
{
	static char n_call = 0;

	options* opt = (options*)data;

	n_call++;
	trim(key);
	trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name") {
		opt->sim_name = value;
    } 
    else if (key == "description") {
		opt->sim_desc = value;
    }
    else if (key == "frame_center") {
		transform(value.begin(), value.end(), value.begin(), ::tolower);
		if (     value == "bary") {
			opt->fr_cntr = FRAME_CENTER_BARY;
		}
		else if (value == "astro") {
			opt->fr_cntr = FRAME_CENTER_ASTRO;
		}
		else {
			throw nbody_exception("Invalid frame center type: " + value);
		}
    }
    else if (key == "integrator") {
		transform(value.begin(), value.end(), value.begin(), ::tolower);
		if (value == "e" || value == "euler") {
			opt->inttype = options::INTEGRATOR_EULER;
		}
		else if (value == "rk2" || value == "rungekutta2")	{
			opt->inttype = options::INTEGRATOR_RUNGEKUTTA2;
		}
		else if (value == "ork2" || value == "optimizedrungekutta2")	{
			opt->inttype = options::INTEGRATOR_OPT_RUNGEKUTTA2;
		}
		else if (value == "rk4" || value == "rungekutta4")	{
			opt->inttype = options::INTEGRATOR_RUNGEKUTTA4;
		}
		else if (value == "rkf78" || value == "rungekuttafehlberg78")	{
			opt->inttype = options::INTEGRATOR_RUNGEKUTTAFEHLBERG78;
		}			
		else if (value == "ork4" || value == "optimizedrungekutta4")	{
			opt->inttype = options::INTEGRATOR_OPT_RUNGEKUTTA4;
		}
		else if (value == "rkn" || value == "rungekuttanystrom") {
			opt->inttype = options::INTEGRATOR_RUNGEKUTTANYSTROM;
		}
		else if (value == "orkn" || value == "optimizedrungekuttanystrom") {
			opt->inttype = options::INTEGRATOR_OPT_RUNGEKUTTANYSTROM;
		}
		else {
			throw nbody_exception("Invalid integrator type: " + value);
		}
	}
    else if (key == "tolerance") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->adaptive = true;
		opt->tolerance = atof(value.c_str());
	}
    else if (key == "start_time") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->start_time = atof(value.c_str()) * Constants::YearToDay;
	}
    else if (key == "length") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->sim_length = atof(value.c_str()) * Constants::YearToDay;
	}
    else if (key == "output_interval") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->output_interval = atof(value.c_str()) * Constants::YearToDay;
	}
    else if (key == "ejection") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->ejection_dst = atof(value.c_str());
	}
    else if (key == "hit_centrum") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->hit_centrum_dst = atof(value.c_str());
	}
    else if (key == "collision_factor") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		opt->collision_factor = atof(value.c_str());
	}
	else {
		throw nbody_exception("Invalid parameter :" + key + ".");
	}

	if (verbose) {
		if (n_call == 1) {
			cout << "The following common parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << endl;
	}
}

void set_gasdisk_param(string& key, string& value, void* data, bool verbose)
{
	static char n_call = 0;

	gas_disk* gasDisk = (gas_disk*)data;

	n_call++;
	trim(key);
	trim(value);
	transform(key.begin(), key.end(), key.begin(), ::tolower);

	if (     key == "name") {
		gasDisk->name = value;
    } 
    else if (key == "description") {
		gasDisk->desc = value;
    }

	else if (key == "mean_molecular_weight" || key == "mmw") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->mean_molecular_weight = atof(value.c_str());
	}
	else if (key == "particle_diameter" || key == "diameter") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->particle_diameter = atof(value.c_str());
	}

	else if (key == "alpha") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->alpha = atof(value.c_str());
	}

	else if (key == "time_dependence") {
		if (     value == "constant" || value == "const") {
			gasDisk->gas_decrease = GAS_DENSITY_CONSTANT;
		}
		else if (value == "linear" || value == "lin") {
			gasDisk->gas_decrease = GAS_DENSITY_DECREASE_LINEAR;
		}
		else if (value == "exponential" || value == "exp") {
			gasDisk->gas_decrease = GAS_DENSITY_DECREASE_EXPONENTIAL;
		}
		else {
			throw nbody_exception("Invalid value at: " + key);
		}
	}

	else if (key == "t0") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->t0 = atof(value.c_str());
	}
	else if (key == "t1") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->t1 = atof(value.c_str());
	}
	else if (key == "e_folding_time") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->e_folding_time = atof(value.c_str());
	}

	else if (key == "eta_c") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->eta.x = atof(value.c_str());
	}
    else if (key == "eta_p") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->eta.y = atof(value.c_str());
	}

    else if (key == "rho_c") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->rho.x = atof(value.c_str());
	}
    else if (key == "rho_p") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->rho.y = atof(value.c_str());
	}

    else if (key == "sch_c") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->sch.x = atof(value.c_str());
	}
    else if (key == "sch_p") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->sch.y = atof(value.c_str());
	}

    else if (key == "tau_c") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->tau.x = atof(value.c_str());
	}
    else if (key == "tau_p") {
		if (!is_number(value)) {
			throw nbody_exception("Invalid number at: " + key);
		}
		gasDisk->tau.y = atof(value.c_str());
	}

	else {
		throw nbody_exception("Invalid parameter :" + key + ".");
	}

	if (verbose) {
		if (n_call == 1) {
			cout << "The following gas disk parameters are setted:" << endl;
		}
		cout << "\t'" << key << "' was assigned to '" << value << "'" << std::endl;
	}
}


options::options(int argc, const char** argv)
{
	create_default_options();
	parse_options(argc, argv);

	if (parameters_path.length() > 0) {
		load(parameters_path, parameters_str);
		parse_params(parameters_str, (void*)this, set_parameters_param);
	}
	if (gasDisk_path.length() > 0) {
		load(gasDisk_path, gasDisk_str);
		// If default gas disk was created, delete it
		delete[] gasDisk;
		gasDisk = new gas_disk;
		parse_params(gasDisk_str, (void*)this->gasDisk, set_gasdisk_param);
	}
	stop_time = start_time + sim_length;
	
	// TODO set __constant__ var_t d_cst_common[THRESHOLD_N];

}

options::~options() 
{
}

void options::create_default_options()
{
	verbose			= false;
	n				= 256;
	nBodies			= 0;
	inttype			= INTEGRATOR_EULER;
	adaptive		= false;
	tolerance		= 0;
	start_time		= 0;
	stop_time		= 1000;
	dt				= (var_t)0.1;
	buffer_radius	= 3;
	printout		= false;
	printoutPeriod	= 100;
	printoutStep	= 1;
	printoutLength	= 10;
	printoutToFile	= false;
	printoutDir		= "";
	file			= false;
	filen			= 0;
	filename		= "";
	random			= true;
	gasDisk			= 0;
}

void options::print_usage()
{
	cout << "Usage: CudaNBody <parameterlis>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -ip <path>         : the file containig the parameters of the simulation"  << endl;
	cout << "     -igd <path>        : the file containig the parameters of the gas disk"  << endl;
	cout << "     -ibl <path>        : the file containig the bodylist with their initial conditions"  << endl;
	cout << "     -n <number>        : Number of self-interacting bodies" << endl;
	cout << "     -nBodies <nStar> <nGP> <nRP> <nPP> <nSP> <nPl> <nTP> : Number of bodies" << endl;
	cout << "     -i <type>          : Integrator type" << endl;
	cout << "                          E : Euler" << endl;
	cout << "                          oRK2 : optimized 2nd order Runge-Kutta" << endl;
	cout << "                          oRK4 : optimized 4th order Runge-Kutta" << endl;
	cout << "                          RKF78: 7(8)th order Runge-Kutta-Fehlberg method" << endl;
	cout << "                          oRKN : optimized 7(6) Runge-Kutta-Nystrom" << endl;
	cout << "                          RK2  : 2nd order Runge-Kutta" << endl;
	cout << "                          RK4  : 4th order Runge-Kutta" << endl;
	cout << "                          RKN  : Runge-Kutta-Nystrom" << endl;
	cout << "     -gasDefault   : Embed the planets into a gas disk with default values" << endl;
	cout << "     -a <number>   : Use adaptive time step with <number> as tolerance" << endl;
	cout << "     -t0 <number>  : Start time " << endl;
	cout << "     -t <number>   : Stop time " << endl;
	cout << "     -dt <number>  : Initial time step" << endl;
	cout << "     -b <number>   : Buffer factor for collisions" << endl;
	cout << "     -p <period> <length> <stepsize>" << endl;
	cout << "                   : Print-out enabled with given parameters" << endl;
	cout << "     -o <directory>: Output directory" << endl;
	cout << "     -f <filename> : Input file, number of entries" << endl;
	cout << "     -r            : Generate random data" << endl;
}

void options::parse_options(int argc, const char** argv)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		if (     p == "-ip") {
			i++;
			parameters_path = argv[i];
		}
		else if (p == "-igd") {
			i++;
			gasDisk_path = argv[i];
		}
		else if (p == "-ibl") {
			i++;
			bodylist_path = argv[i];
		}
		else if (p == "--verbose" || p == "-v") {
			verbose = true;
		}


		// Number of bodies
		else if (p == "-n") {
			i++;
			n = atoi(argv[i]);
			if (2 > n) {
				throw nbody_exception("Number of bodies must exceed 2.");
			}
		}

		else if (p == "-nBodies") {
			i++;
			int	ns		= atoi(argv[i++]);
			int	ngp		= atoi(argv[i++]);
			int	nrp		= atoi(argv[i++]);
			int	npp		= atoi(argv[i++]);
			int	nspl	= atoi(argv[i++]);
			int	npl		= atoi(argv[i++]);
			int	ntp		= atoi(argv[i]);
			this->nBodies = new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
		}
		// Initialize a gas_disk object with default values
		else if (p == "-gasDefault") {
			delete gasDisk;
			gasDisk = new gas_disk;
		}
		// Integrator type
		else if (p == "-i") {
			i++;
			p = argv[i];
			if (p == "E") {
				inttype = INTEGRATOR_EULER;
			}
			else if (p == "RK2")	{
				inttype = INTEGRATOR_RUNGEKUTTA2;
			}
			else if (p == "oRK2")	{
				inttype = INTEGRATOR_OPT_RUNGEKUTTA2;
			}
			else if (p == "RK4")	{
				inttype = INTEGRATOR_RUNGEKUTTA4;
			}
			else if (p == "RKF78")	{
				inttype = INTEGRATOR_RUNGEKUTTAFEHLBERG78;
			}			
			else if (p == "oRK4")	{
				inttype = INTEGRATOR_OPT_RUNGEKUTTA4;
			}
			else if (p == "RKN") {
				inttype = INTEGRATOR_RUNGEKUTTANYSTROM;
			}
			else if (p == "oRKN") {
				inttype = INTEGRATOR_OPT_RUNGEKUTTANYSTROM;
			}
			else {
				throw nbody_exception("Invalid integrator type.");
			}
		}
		// Adaptive method
		else if (p == "-a")	{
			adaptive = true;
			i++;
			tolerance = (var_t)atof(argv[i]);
		}
		// Time start
		else if (p == "-t0")	{
			i++;
			start_time = (var_t)atof(argv[i]) * Constants::YearToDay;
		}
		// Time end
		else if (p == "-t")	{
			i++;
			stop_time = (var_t)atof(argv[i]) * Constants::YearToDay;
		}
		// Time step
		else if (p == "-dt") {
			i++;
			dt = (var_t)atof(argv[i]);
		}
		// Radius buffer factor
		else if (p == "-b") {
			i++;
			buffer_radius = (var_t)atof(argv[i]);
		}
		// Print out period
		else if (p == "-p")	{
			printout = true;
			i++;
			printoutPeriod = (var_t)atof(argv[i]) * Constants::YearToDay;
			i++;
			printoutLength = (var_t)atof(argv[i]) * Constants::YearToDay;
			i++;
			printoutStep = (var_t)atof(argv[i]) * Constants::YearToDay;
		}
		// Print-out location
		else if (p == "-o")	{
			i++;
			printoutDir = argv[i];
			printoutToFile = true;
		}
		// Input file
		else if (p == "-f")	{
			i++;
			filename = argv[i];
			file = true;
		}
		else if (p == "-r")	{
			random = true;
		}
		else {
			throw nbody_exception("Invalid switch on command-line.");
		}
		i++;
	}
}

void options::parse_params(string& input, void *data, void (*setter)(string& key, string& value, void* data, bool verbose))
{
	// instantiate Tokenizer classes
	Tokenizer fileTokenizer;
	Tokenizer lineTokenizer;
	string line;

	fileTokenizer.set(input, "\n");
	while ((line = fileTokenizer.next()) != "") {
		lineTokenizer.set(line, "=");
		string token;
		int tokenCounter = 1;

		string key; 
		string value;
		while ((token = lineTokenizer.next()) != "" && tokenCounter <= 2) {

			if (tokenCounter == 1)
				key = token;
			else if (tokenCounter == 2)
				value = token;

			tokenCounter++;
		}
		if (tokenCounter > 2) {
			setter(key, value, data, verbose);
		}
		else {
			throw nbody_exception("Invalid key/value pair: " + line + ".");
		}
	}
}

void options::load(string& path, string& result)
{
	std::ifstream file(path);
	if (file) {
		string str;
		while (std::getline(file, str))
		{
			// ignore zero length lines
			if (str.length() == 0)
				continue;
			// ignore comment lines
			if (str[0] == '#')
				continue;
			// delete comment after the value
			trim_right(str, '#');
			result += str;
			result.push_back('\n');
		} 	
	}
	else {
		throw nbody_exception("The file '" + path + "' could not opened!\r\n");
	}
	file.close();
}

void options::get_number_of_bodies(string& path)
{
	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		ns = ngp = nrp = npp = nspl = npl = ntp = 0;
		input >> ns;
		input >> ngp;
		input >> nrp;
		input >> npp;
		input >> nspl;
		input >> npl;
		input >> ntp;
		this->nBodies = new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
	}
	else 
	{
		throw nbody_exception("Cannot open " + path + ".");
	}
	input.close();
}

void options::initial_condition(nbody* nb)
{
	vec_t*	coor = (vec_t*)nb->h_y[0].data();
	vec_t*	velo = (vec_t*)nb->h_y[1].data();
	nbody::param_t* param = (nbody::param_t*)nb->h_p.data();

	int i = 0;
	// Set the initial conditions
	{
		// Star
		param[i].mass = 1.0;		// M_sun
		param[i].radius = 0.0014;	// AU

		coor[i].x = 0.0;			// AU
		coor[i].y = 0.0;
		coor[i].z = 0.0;
		coor[i].w = 0.0;

		velo[i].x = 0.0;			// AU / day
		velo[i].y = 0.0;
		velo[i].z = 0.0;
		velo[i].w = 0.0;

		// Planet 0, 1, 2, ..., n
		for (i = 1; i < this->n; i++) {
			param[i].mass = 1.0e-7;			// M_sun
			param[i].radius = 0.000014;		// AU

			coor[i].x = 1.0 + (i-1)*0.1;	// AU
			coor[i].y = 0.0;
			coor[i].z = 0.0;
			coor[i].w = 0.0;

			// Compute the circular velocity for planet 0
			var_t	r = sqrt( SQR(coor[i].x - coor[0].x) + SQR(coor[i].y - coor[0].y) + SQR(coor[i].z - coor[0].z) );
			var_t	v = K * sqrt(param[0].mass / r);

			velo[i].x = 0.0;			// AU / day
			velo[i].y = v;
			velo[i].z = 0.0;
			velo[i].w = 0.0;
		}
	}

	// Transform the variables to the barycentric system
	{
		// Compute the total mass of the system
		var_t totalMass = 0.0;
		for (int j = 0; j < this->n; j++ ) {
			totalMass += param[j].mass;
		}

		// Position and velocity of the system's barycenter
		vec_t R0 = {0.0, 0.0, 0.0, 0.0};
		vec_t V0 = {0.0, 0.0, 0.0, 0.0};

		// Compute the position and velocity of the barycenter of the system
		for (int j = 0; j < this->n; j++ ) {
			R0.x += param[j].mass * coor[j].x;
			R0.y += param[j].mass * coor[j].y;
			R0.z += param[j].mass * coor[j].z;

			V0.x += param[j].mass * velo[j].x;
			V0.y += param[j].mass * velo[j].y;
			V0.z += param[j].mass * velo[j].z;
		}
		R0.x /= totalMass;
		R0.y /= totalMass;
		R0.z /= totalMass;

		V0.x /= totalMass;
		V0.y /= totalMass;
		V0.z /= totalMass;

		// Transform the bodies coordinates and velocities
		for (int j = 0; j < this->n; j++ ) {
			coor[j].x -= R0.x;
			coor[j].y -= R0.y;
			coor[j].z -= R0.z;

			velo[j].x -= V0.x;
			velo[j].y -= V0.y;
			velo[j].z -= V0.z;
		}
	}
}

//ode* options::create_ode()
//{
//	nbody* nb = new nbody(n, start_time);
//
//	nb->t = start_time;
//	
//	if (file) {
//		nb->load(filename, n);
//	}
//
//	nb->copy_to_device();
//
//	return nb;
//}
//
//nbody*	options::create_nbody()
//{
//	nbody*	nb = new nbody(n, start_time);
//
//	nb->t = start_time;
//
//	if (file) {
//		nb->load(filename, n);
//	}
//	else {
//		initial_condition(nb);
//	}
//	nb->copy_to_device();
//
//	return nb;
//}

pp_disk*	options::create_pp_disk()
{
	pp_disk *ppd = 0;

	if (this->bodylist_path.length() > 0)
	{
		// set the nBodies field using the data in the bodylist_path
		get_number_of_bodies(bodylist_path);
		ppd = new pp_disk(nBodies, (gasDisk == 0 ? false : true), start_time);
		ppd->load(bodylist_path);
	}
	else
	{
		ppd = new pp_disk(nBodies, (gasDisk == 0 ? false : true), start_time);
		if (file) {
			ppd->load(filename, nBodies->total);
		}
		else {
			throw nbody_exception("file is missing!");
		}
	}
	if (gasDisk != 0)
	{
		gasDisk->m_star = ppd->get_mass_of_star();
		gasDisk->calculate();

		ppd->h_gasDisk = gasDisk;
		// Copies gas disk parameters and variables to the cuda device from the host
		cudaMalloc((void**)&(ppd->d_gasDisk), sizeof(gas_disk));
		cudaMemcpy(ppd->d_gasDisk, ppd->h_gasDisk, sizeof(gas_disk), cudaMemcpyHostToDevice );
	}

	ppd->t = start_time;
	ppd->transform_to_bc();
	ppd->copy_to_device();

	return ppd;
}

integrator* options::create_integrator(ode* f)
{
	integrator* intgr;

	switch (inttype)
	{
	case INTEGRATOR_EULER:
		intgr = new euler(*f, dt);
		break;
	case INTEGRATOR_RUNGEKUTTA2:
		intgr = new rungekutta<2>(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_OPT_RUNGEKUTTA2:
		intgr = new midpoint(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTA4:
		intgr = new rungekutta<4>(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_OPT_RUNGEKUTTA4:
		intgr = new rk4(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTAFEHLBERG78:
		intgr = new rkf7(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTANYSTROM:
		intgr = new rungekuttanystrom<9>(*f, dt, adaptive, tolerance);
		break;
	case INTEGRATOR_OPT_RUNGEKUTTANYSTROM:
		intgr = new rkn76(*f, dt, adaptive, tolerance);
		break;
	default:
		throw nbody_exception("Requested integrator is not implemented.");
	}

	return intgr;
}
