// includes, system 
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

// includes, project
#include "constants.h"
#include "file_util.h"
#include "number_of_bodies.h"
#include "pp_disk.h"

#define MASS_STAR		1.0			// M_sol
#define MASS_JUPITER	1.0e-3		// M_sol
#define RAD_STAR		0.005		// AU

#define MASS_SUN		1.9891E+30	// kg
#define MASS_FACTOR		5e-7		// M_sol
#define MASS_MU			log(4.0)
#define MASS_S			0.3
#define MASS_MIN		1.0e-20		// M_sol
#define MASS_MAX		1.0e-19		// M_sol

#define DIST_MIN		4.5			// AU
#define DIST_MAX		15			// AU

//#define DENSITY			3000.0		// kg m-3
#define AU				149.6e9		// m

// It must be enclosed in parentheses in order to give correct results in
// the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)			((x)*(x))
#define	CUBE(x)			((x)*(x)*(x))

using namespace std;

static string body_type_names[] = {"star", "giant", "rocky", "proto", "superpl", "pl", "testp"}; 

typedef enum output_version
		{
			OUTPUT_VERSION_FIRST,
			OUTPUT_VERSION_SECOND
		} output_version_t;

typedef struct distribution
		{
			var2_t	limits;
			var_t	(*pdf)(var_t);
		} distribution_t;

typedef struct oe_dist
		{
			distribution_t	item[6];
		} oe_dist_t;

typedef struct phys_prop_dist
		{
			distribution_t	item[4];
		} phys_prop_dist_t;

typedef struct body_disk
		{
			vector<string>		names;
			int_t				nBody[BODY_TYPE_N];
			oe_dist_t			oe_d[BODY_TYPE_N];
			phys_prop_dist_t	pp_d[BODY_TYPE_N];
			migration_type_t	*mig_type;
			var_t				*stop_at;
		} body_disk_t;


// Draw a number from a given distribution
var_t generate_random(var_t xmin, var_t xmax, var_t p(var_t))
{
	var_t x;
	var_t y;

	do
	{
		x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		y = (var_t)rand() / RAND_MAX;
	}
	while (y > p(x));

	return x;
}

var_t pdf_mass_lognormal(var_t x)
{
	return 1.0 / sqrt(2 * PI) / MASS_S * exp(-pow(log(x) - MASS_MU, 2) / 2 / MASS_S / MASS_MU);
}

var_t pdf_distance_squared(var_t d)
{
	return d * d / DIST_MAX / DIST_MAX;
}

var_t pdf_distance_exp(var_t d)
{
	return exp(-d) * d * d;
}

var_t pdf_const(var_t x)
{
	return 1;
}

#define FOUR_PI_OVER_THREE	4.1887902047863909846168578443727
var_t calculate_radius(var_t m, var_t density)
{
	return pow(1.0/FOUR_PI_OVER_THREE * m/density, 1.0/3.0);
}

var_t calculate_density(var_t m, var_t R)
{
	return m / (FOUR_PI_OVER_THREE * CUBE(R));
}

var_t caclulate_mass(var_t R, var_t density)
{
	return FOUR_PI_OVER_THREE * CUBE(R) * density;
}
#undef FOUR_PI_OVER_THREE

int_t	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E)
{
	if (ecc == 0.0 || mean == 0.0 || mean == PI) {
        *E = mean;
		return 0;
    }
    *E = mean + ecc * (sin(mean)) / (1.0 - sin(mean + ecc) + sin(mean));
    var_t E1 = 0.0;
    var_t error;
    int_t step = 0;
    do {
        E1 = *E - (*E - ecc * sin(*E) - mean) / (1.0 - ecc * cos(*E));
        error = fabs(E1 - *E);
        *E = E1;
    } while (error > eps && step++ <= 15);
	if (step > 15 ) {
		return 1;
	}

	return 0;
}

int_t	calculate_phase(var_t mu, const pp_disk::orbelem_t* oe, vec_t* rVec, vec_t* vVec)
{
    var_t ecc = oe->ecc;
	var_t E = 0.0;
	if (kepler_equation_solver(ecc, oe->mean, 1.0e-14, &E) == 1) {
		return 1;
	}
    var_t v = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E / 2.0));

    var_t p = oe->sma * (1.0 - SQR(ecc));
    var_t r = p / (1.0 + ecc * cos(v));
    var_t kszi = r * cos(v);
    var_t eta = r * sin(v);
    var_t vKszi = -sqrt(mu / p) * sin(v);
    var_t vEta = sqrt(mu / p) * (ecc + cos(v));

    var_t cw = cos(oe->peri);
    var_t sw = sin(oe->peri);
    var_t cO = cos(oe->node);
    var_t sO = sin(oe->node);
    var_t ci = cos(oe->inc);
    var_t si = sin(oe->inc);

    vec_t P;
	P.x = cw * cO - sw * sO * ci;
	P.y = cw * sO + sw * cO * ci;
	P.z = sw * si;
    vec_t Q;
	Q.x = -sw * cO - cw * sO * ci;
	Q.y = -sw * sO + cw * cO * ci;
	Q.z = cw * si;

	rVec->x = kszi * P.x + eta * Q.x;
	rVec->y = kszi * P.y + eta * Q.y;
	rVec->z = kszi * P.z + eta * Q.z;

	vVec->x = vKszi * P.x + vEta * Q.x;
	vVec->y = vKszi * P.y + vEta * Q.y;
	vVec->z = vKszi * P.z + vEta * Q.z;

	return 0;
}


void print_body_record(ofstream &output, string name, var_t epoch, pp_disk::param_t *param, vec_t *r, vec_t *v, output_version_t o_version)
{
	static char sep = ' ';

	int type = static_cast<body_type_t>(param->body_type);

	switch (o_version)
	{
	case OUTPUT_VERSION_FIRST:
		output << param->id << sep << epoch << sep;
		output << param->mass << sep << param->radius << sep << param->density << sep << param->cd << sep;
		output << param->migType << sep << param->migStopAt << sep;
		output << r->x << sep << r->y << sep << r->z << sep;
		output << v->x << sep << v->y << sep << v->z << sep;
		output << endl;
		break;
	case OUTPUT_VERSION_SECOND:
		output << param->id << sep
			   << name << sep 
			   << param->body_type << sep 
			   << param->epoch << sep;
		output << param->mass << sep << param->radius << sep << param->density << sep << param->cd << sep;
		output << param->migType << sep << param->migStopAt << sep;
		output << r->x << sep << r->y << sep << r->z << sep;
		output << v->x << sep << v->y << sep << v->z << sep;
		output << endl;
		break;
	default:
		cerr << "Invalid output version!" << endl;
		exit(1);
	}
}

void set(distribution_t& d, var_t x, var_t (*pdf)(var_t))
{
	d.limits.x = d.limits.y = x;
	d.pdf = pdf;
}

void set(distribution_t& d, var_t x, var_t y, var_t (*pdf)(var_t))
{
	d.limits.x = x;
	d.limits.y = y;
	d.pdf = pdf;
}

string create_name(int i, int type)
{
	ostringstream convert;	// stream used for the conversion
	string i_str;			// string which will contain the number
	string name;

	convert << i;			// insert the textual representation of 'i' in the characters in the stream
	i_str = convert.str();  // set 'i_str' to the contents of the stream
	name = body_type_names[type] + i_str;

	return name;
}

void set_default(body_disk &bd)
{
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		bd.nBody[body_type] = 0;
		for (int i = 0; i < 6; i++) 
		{
			set(bd.oe_d[body_type].item[i], 0.0, 0.0, pdf_const);
		}
		for (int i = 0; i < 4; i++) 
		{
			set(bd.pp_d[body_type].item[i], 0.0, 0.0, pdf_const);
		}
	}
}

int_t	calculate_number_of_bodies(body_disk &bd)
{
	int_t result = 0;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		result += bd.nBody[body_type];
	}
	return result;
}

void generate_oe(oe_dist_t oe_d, pp_disk::orbelem_t& oe)
{
	oe.sma  = generate_random(oe_d.item[SMA].limits.x,  oe_d.item[SMA].limits.y,  oe_d.item[SMA].pdf);
	oe.ecc  = generate_random(oe_d.item[ECC].limits.x,  oe_d.item[ECC].limits.y,  oe_d.item[ECC].pdf);
	oe.inc  = generate_random(oe_d.item[INC].limits.x,  oe_d.item[INC].limits.y,  oe_d.item[INC].pdf);
	oe.peri = generate_random(oe_d.item[PERI].limits.x, oe_d.item[PERI].limits.y, oe_d.item[PERI].pdf);
	oe.node = generate_random(oe_d.item[NODE].limits.x, oe_d.item[NODE].limits.y, oe_d.item[NODE].pdf);
	oe.mean = generate_random(oe_d.item[MEAN].limits.x, oe_d.item[MEAN].limits.y, oe_d.item[MEAN].pdf);
}

void generate_pp(phys_prop_dist_t pp_d, pp_disk::param_t& param)
{
	param.mass = generate_random(pp_d.item[MASS].limits.x, pp_d.item[MASS].limits.y, pp_d.item[MASS].pdf);

	if (	 pp_d.item[DENSITY].limits.x == 0.0 && pp_d.item[DENSITY].limits.y == 0.0 &&
			 pp_d.item[RADIUS].limits.x == 0.0 && pp_d.item[RADIUS].limits.y == 0.0 )
	{
		param.radius = 0.0;
		param.density = 0.0;
	}
	else if (pp_d.item[DENSITY].limits.x == 0.0 && pp_d.item[DENSITY].limits.y == 0.0 &&
			 pp_d.item[RADIUS].limits.x > 0.0 && pp_d.item[RADIUS].limits.y > 0.0 )
	{
		param.radius = generate_random(pp_d.item[RADIUS].limits.x, pp_d.item[RADIUS].limits.y, pp_d.item[RADIUS].pdf);
		param.density = calculate_density(param.mass, param.radius);
	}
	else if (pp_d.item[DENSITY].limits.x > 0.0 && pp_d.item[DENSITY].limits.y > 0.0 &&
			 pp_d.item[RADIUS].limits.x == 0.0 && pp_d.item[RADIUS].limits.y == 0.0 )
	{
		param.density = generate_random(pp_d.item[DENSITY].limits.x, pp_d.item[DENSITY].limits.y, pp_d.item[DENSITY].pdf);
		param.radius = calculate_radius(param.mass, param.density);
	}
	else
	{
		param.radius = generate_random(pp_d.item[RADIUS].limits.x, pp_d.item[RADIUS].limits.y, pp_d.item[RADIUS].pdf);
		param.density = generate_random(pp_d.item[DENSITY].limits.x, pp_d.item[DENSITY].limits.y, pp_d.item[DENSITY].pdf);
	}

	param.cd = generate_random(pp_d.item[DRAG_COEFF].limits.x, pp_d.item[DRAG_COEFF].limits.y, pp_d.item[DRAG_COEFF].pdf);
}

int generate_pp_disk(string path, body_disk_t& body_disk, output_version_t o_version)
{
	static char sep = ' ';

	ofstream	output(path, ios_base::out);
	if (output)
	{
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
		{
			output << body_disk.nBody[body_type] << sep;
		}
		output << endl;

		var_t	t = 0.0;
		vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
		vec_t	vVec = {0.0, 0.0, 0.0, 0.0};

		pp_disk::param_t	param0;
		pp_disk::param_t	param;
		pp_disk::orbelem_t	oe;

		int_t bodyId = 0;
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
		{
			srand ((unsigned int)time(0));
			for (int i = 0; i < body_disk.nBody[body_type]; i++, bodyId++)
			{
				if (body_type == BODY_TYPE_STAR)
				{
					param0.id = bodyId;
					param0.body_type = BODY_TYPE_STAR;
					param0.epoch = 0.0;

					generate_pp(body_disk.pp_d[body_type], param0);
					param0.migType = body_disk.mig_type[bodyId];
					param0.migStopAt = body_disk.stop_at[bodyId];
					print_body_record(output, body_disk.names[bodyId], t, &param0, &rVec, &vVec, o_version);
				} /* if */
				else 
				{
					param.id = bodyId;
					param.body_type = static_cast<body_type_t>(body_type);
					param.epoch = 0.0;

					generate_oe(body_disk.oe_d[body_type], oe);
					generate_pp(body_disk.pp_d[body_type], param);
					param.migType = body_disk.mig_type[bodyId];
					param.migStopAt = body_disk.stop_at[bodyId];

					var_t mu = K2*(param0.mass + param.mass);
					int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
					if (ret_code == 1) {
						cerr << "Could not calculate the phase." << endl;
						return ret_code;
					}

					print_body_record(output, body_disk.names[bodyId], t, &param, &rVec, &vVec, o_version);
				} /* else */
			} /* for */
		} /* for */
		output.flush();
		output.close();
	}
	else
	{
		cerr << "Cannot open " << path << ".";
		exit(0);
	}

	return 0;
}

void set_parameters_of_Dvorak_disk(body_disk_t& disk)
{
	const var_t mCeres		= 9.43e20 /* kg */ * Constants::KilogramToSolar;
	const var_t mMoon		= 9.43e20 /* kg */ * Constants::KilogramToSolar;
	const var_t rhoBasalt	= 2.7 /* g/cm^3 */ * Constants::GramPerCm3ToSolarPerAu3;

	set_default(disk);

	disk.nBody[BODY_TYPE_STAR] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET] = 1000;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

	int	body_id = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	set(disk.pp_d[type].item[MASS], 1.0, pdf_const);
	set(disk.pp_d[type].item[RADIUS], 1.0 * Constants::SolarRadiusToAu, pdf_const);
	set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);
	disk.mig_type[body_id] = MIGRATION_TYPE_NO;
	disk.stop_at[body_id] = 0.0;

	body_id++;

	type = BODY_TYPE_GIANTPLANET;

	type = BODY_TYPE_ROCKYPLANET;

	type = BODY_TYPE_PROTOPLANET;
	{
		set(disk.oe_d[type].item[SMA], 0.9, 2.5, pdf_const);
		set(disk.oe_d[type].item[ECC], 0.0, 0.3, pdf_const);
		set(disk.oe_d[type].item[INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[PERI], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);
		set(disk.oe_d[type].item[NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[MEAN], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);

		set(disk.pp_d[type].item[MASS], mCeres, mMoon/10.0, pdf_mass_lognormal);
		set(disk.pp_d[type].item[DENSITY], rhoBasalt, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

		for (int i = 0; i < disk.nBody[type]; i++, body_id++) 
		{
			disk.names.push_back(create_name(i, type));
			disk.mig_type[body_id] = MIGRATION_TYPE_NO;
			disk.stop_at[body_id] = 0.0;
		}
	}

	type = BODY_TYPE_SUPERPLANETESIMAL;

	type = BODY_TYPE_PLANETESIMAL;

	type = BODY_TYPE_TESTPARTICLE;
}

int parse_options(int argc, const char **argv, number_of_bodies **nBodies, string &outDir)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		if (p == "-nBodies") {
			i++;
			int	star				= atoi(argv[i++]);
			int	giant_planet		= atoi(argv[i++]);
			int	rocky_planet		= atoi(argv[i++]);
			int	proto_planet		= atoi(argv[i++]);
			int	super_planetesimal	= atoi(argv[i++]);
			int	planetesimal		= atoi(argv[i++]);
			int	test_particle		= atoi(argv[i]);
			*nBodies = new number_of_bodies(star, giant_planet, rocky_planet, proto_planet, super_planetesimal, planetesimal, test_particle);
		}
		else if (p == "-o") {
			i++;
			outDir = argv[i];
		}
		else {
			cerr << "Invalid switch on command-line.";
			return 1;
		}
		i++;
	}

	return 0;
}

int main(int argc, const char **argv)
{
	body_disk_t disk;
	string outDir;
	string filename;

	{
		set_parameters_of_Dvorak_disk(disk);
		outDir = "C:\\Work\\Projects\\solaris.cuda\\TestRun\\InputTest";
		filename = "Dvorak_disk.txt";
	}
	{
	}
	generate_pp_disk(combine_path(outDir, filename), disk, OUTPUT_VERSION_SECOND);

	return 0;

	ostringstream convert;	// stream used for the conversion
	string i_str;			// string which will contain the number
	string name;

	body_disk_t test_disk;
	set_default(test_disk);

	test_disk.nBody[BODY_TYPE_STAR] = 1;
	test_disk.nBody[BODY_TYPE_GIANTPLANET] = 2;
	test_disk.nBody[BODY_TYPE_ROCKYPLANET] = 0;
	test_disk.nBody[BODY_TYPE_PROTOPLANET] = 0;
	test_disk.nBody[BODY_TYPE_SUPERPLANETESIMAL] = 0;
	test_disk.nBody[BODY_TYPE_PLANETESIMAL] = 0;
	test_disk.nBody[BODY_TYPE_TESTPARTICLE] = 0;

	int_t nBodies = calculate_number_of_bodies(test_disk);
	test_disk.mig_type = new migration_type_t[nBodies];
	test_disk.stop_at = new var_t[nBodies];

	int	index_of_body = 0;

	test_disk.names.push_back("star");
	set(test_disk.pp_d[BODY_TYPE_STAR].item[MASS], 1.0, pdf_const);
	set(test_disk.pp_d[BODY_TYPE_STAR].item[RADIUS], 1.0 * Constants::SolarRadiusToAu, pdf_const);
	set(test_disk.pp_d[BODY_TYPE_STAR].item[DRAG_COEFF], 0.0, pdf_const);
	test_disk.mig_type[index_of_body] = MIGRATION_TYPE_NO;
	test_disk.stop_at[index_of_body] = 0.0;

	index_of_body++;
	for (int i = 0; i < test_disk.nBody[BODY_TYPE_GIANTPLANET]; i++, index_of_body++)
	{
		convert << i;			// insert the textual representation of 'i' in the characters in the stream
		i_str = convert.str();  // set 'i_str' to the contents of the stream
		name = body_type_names[BODY_TYPE_GIANTPLANET] + i_str;		
		test_disk.names.push_back(name);
		convert.str("");

		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[SMA], 1.0, 4.0, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ECC], 0.0, 0.2, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[INC], 0.0, 30.0 * Constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[PERI], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[NODE], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[MEAN], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);

		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[MASS], 1.0 * Constants::JupiterToSolar, 2.0 * Constants::JupiterToSolar, pdf_mass_lognormal);
		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[DENSITY], 0.7 * Constants::GramPerCm3ToSolarPerAu3, 1.2 * Constants::GramPerCm3ToSolarPerAu3, pdf_mass_lognormal);
		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[DRAG_COEFF], 0.0, pdf_const);
		test_disk.mig_type[index_of_body] = MIGRATION_TYPE_TYPE_II;
		test_disk.stop_at[index_of_body] = 0.4;
	}

	outDir = "C:\\Work\\Projects\\solaris.cuda\\TestRun\\InputTest";
	generate_pp_disk(combine_path(outDir, "test_disk.txt"), test_disk, OUTPUT_VERSION_SECOND);

	delete[] test_disk.mig_type;
	delete[] test_disk.stop_at;

	return 0;
}
