// includes, system 
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

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

typedef struct oe_range
		{
			distribution_t	oe[6];
		} oe_range_t;

typedef struct phys_prop_range
		{
			distribution_t	pp[4];
		} phys_prop_range_t;

typedef struct body_disk
		{
			int_t				nBody[BODY_TYPE_N];
			oe_range_t			oe_r[BODY_TYPE_N];
			phys_prop_range_t	pp_r[BODY_TYPE_N];
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

//var_t calculate_radius(var_t m)
//{
//	var_t V = m * MASS_SUN / DENSITY;	// m3
//	V /= AU * AU * AU;		// AU3
//	
//	return pow(3.0 / 4.0 / PI * V, 1.0 / 3.0);
//}

#define FOUR_PI_OVER_THREE	4.1887902047863909846168578443727
var_t calculate_radius(var_t m, var_t density)
{
	return pow(1.0/FOUR_PI_OVER_THREE * m/density ,1.0/3.0);
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


void print_body_record(std::ofstream &output, string name, var_t epoch, pp_disk::param_t *param, vec_t *r, vec_t *v, output_version_t o_version)
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
		output << param->id << sep << name << sep << type << sep << epoch << sep;
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

//int generate_Dvorak_disk(string path, var2_t disk, number_of_bodies *nBodies)
//{
//	var_t t = 0.0;
//	int_t bodyId = 0;
//
//	var_t mCeres		= 9.43e20;	// kg
//	var_t mMoon			= 9.43e20;	// kg
//	var_t rhoBasalt		= 2.7;		// g / cm^3
//	
//	param_t param0;
//	param_t param;
//	vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
//	vec_t	vVec = {0.0, 0.0, 0.0, 0.0};
//
//	std::ofstream	output;
//	output.open(path, std::ios_base::app);
//
//	// Output central mass
//	for (int i = 0; i < nBodies->star; i++, bodyId++)
//	{
//		param0.id = bodyId;
//		param0.mass = 1.0;
//		param0.radius = Constants::SolarRadiusToAu;
//		param0.density = calculate_density(param0.mass, param0.radius);
//		param0.cd = 0.0;
//		param0.migType = NO;
//		param0.migStopAt = 0.0;
//		print_body_record(output, bodyId, "", t, &param0, &rVec, &vVec, FIRST_VERSION);
//	}
//
//	srand ((unsigned int)time(0));
//	orbelem oe;
//	// Output proto planets
//	for (int i = 0; i < nBodies->proto_planet; i++, bodyId++)
//	{
//		oe.sma = generate_random(disk.x, disk.y, pdf_const);
//		oe.ecc = generate_random(0.0, 0.3, pdf_const);
//		oe.inc = 0.0;
//		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
//		oe.node = 0.0;
//		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);
//
//		param.id = bodyId;
//		param.mass = generate_random(mCeres, mMoon / 10.0, pdf_const) * Constants::KilogramToSolar;
//		param.density = rhoBasalt * Constants::GramPerCm3ToSolarPerAu3;
//		param.radius = calculate_radius(param.mass, param.density);
//		param.cd = 0.0;
//		param.migType = NO;
//		param.migStopAt = 0.0;
//
//		var_t mu = K2*(param0.mass + param.mass);
//		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
//		if (ret_code == 1) {
//			cerr << "Could not calculate the phase." << endl;
//			return ret_code;
//		}
//		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
//	}
//
//	output.flush();
//	output.close();
//
//	return 0;
//}


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

void set_default(body_disk &bd)
{
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		bd.nBody[body_type] = 0;
		for (int i = 0; i < 6; i++) 
		{
			set(bd.oe_r[body_type].oe[i], 0.0, 0.0, pdf_const);
		}
		for (int i = 0; i < 4; i++) 
		{
			set(bd.pp_r[body_type].pp[i], 0.0, 0.0, pdf_const);
		}
	}
}

void generate_oe(oe_range_t oe_r, pp_disk::orbelem_t& oe)
{
	oe.sma  = generate_random(oe_r.oe[SMA].limits.x,  oe_r.oe[SMA].limits.y,  oe_r.oe[SMA].pdf);
	oe.ecc  = generate_random(oe_r.oe[ECC].limits.x,  oe_r.oe[ECC].limits.y,  oe_r.oe[ECC].pdf);
	oe.inc  = generate_random(oe_r.oe[INC].limits.x,  oe_r.oe[INC].limits.y,  oe_r.oe[INC].pdf);
	oe.peri = generate_random(oe_r.oe[PERI].limits.x, oe_r.oe[PERI].limits.y, oe_r.oe[PERI].pdf);
	oe.node = generate_random(oe_r.oe[NODE].limits.x, oe_r.oe[NODE].limits.y, oe_r.oe[NODE].pdf);
	oe.mean = generate_random(oe_r.oe[MEAN].limits.x, oe_r.oe[MEAN].limits.y, oe_r.oe[MEAN].pdf);
}

void generate_pp(phys_prop_range_t pp_r, pp_disk::param_t& param)
{
	param.mass = generate_random(pp_r.pp[MASS].limits.x, pp_r.pp[MASS].limits.y, pp_r.pp[MASS].pdf);

	if (	 pp_r.pp[DENSITY].limits.x == 0.0 && pp_r.pp[DENSITY].limits.y == 0.0 &&
			 pp_r.pp[RADIUS].limits.x == 0.0 && pp_r.pp[RADIUS].limits.y == 0.0 )
	{
		param.radius = 0.0;
		param.density = 0.0;
	}
	else if (pp_r.pp[DENSITY].limits.x == 0.0 && pp_r.pp[DENSITY].limits.y == 0.0 &&
			 pp_r.pp[RADIUS].limits.x > 0.0 && pp_r.pp[RADIUS].limits.y > 0.0 )
	{
		param.radius = generate_random(pp_r.pp[RADIUS].limits.x, pp_r.pp[RADIUS].limits.y, pp_r.pp[RADIUS].pdf);
		param.density = calculate_density(param.mass, param.radius);
	}
	else if (pp_r.pp[DENSITY].limits.x > 0.0 && pp_r.pp[DENSITY].limits.y > 0.0 &&
			 pp_r.pp[RADIUS].limits.x == 0.0 && pp_r.pp[RADIUS].limits.y == 0.0 )
	{
		param.density = generate_random(pp_r.pp[DENSITY].limits.x, pp_r.pp[DENSITY].limits.y, pp_r.pp[DENSITY].pdf);
		param.radius = calculate_radius(param.mass, param.density);
	}
	else
	{
		param.radius = generate_random(pp_r.pp[RADIUS].limits.x, pp_r.pp[RADIUS].limits.y, pp_r.pp[RADIUS].pdf);
		param.density = generate_random(pp_r.pp[DENSITY].limits.x, pp_r.pp[DENSITY].limits.y, pp_r.pp[DENSITY].pdf);
	}

	param.cd = generate_random(pp_r.pp[DRAG_COEFF].limits.x, pp_r.pp[DRAG_COEFF].limits.y, pp_r.pp[DRAG_COEFF].pdf);
}

int generate_pp_disk(string path, body_disk_t& body_disk, output_version_t o_version)
{
	ostringstream convert;	// stream used for the conversion
	var_t t = 0.0;

	vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
	vec_t	vVec = {0.0, 0.0, 0.0, 0.0};

	ofstream	output;
	output.open(path, std::ios_base::app);

	pp_disk::param_t	param0;
	pp_disk::param_t	param;
	pp_disk::orbelem_t	oe;

	int_t bodyId = 0;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		srand ((unsigned int)time(0));
		string i_str;	// string which will contain the number
		string name;
		for (int i = 0; i < body_disk.nBody[body_type]; i++, bodyId++)
		{
			if (body_type == BODY_TYPE_STAR)
			{
				param0.body_type = BODY_TYPE_STAR;
				param0.id = bodyId;

				generate_pp(body_disk.pp_r[body_type], param0);
				param0.migType = MIGRATION_TYPE_NO;
				param0.migStopAt = 0.0;

				convert << i;			// insert the textual representation of 'i' in the characters in the stream
				i_str = convert.str();  // set 'i_str' to the contents of the stream
				name = body_type_names[body_type] + i_str;

				print_body_record(output, name, t, &param0, &rVec, &vVec, o_version);
			} /* if */
			else 
			{
				param.id = bodyId;
				param.body_type = static_cast<body_type_t>(body_type);

				generate_oe(body_disk.oe_r[body_type], oe);
				generate_pp(body_disk.pp_r[body_type], param);

				convert << i;			// insert the textual representation of 'i' in the characters in the stream
				i_str = convert.str();  // set 'i_str' to the contents of the stream
				name = body_type_names[body_type] + i_str;

				if (body_type == BODY_TYPE_ROCKYPLANET || 
					body_type == BODY_TYPE_PROTOPLANET)
				{
					param.migType = MIGRATION_TYPE_TYPE_I;
					param.migStopAt = 0.0;
				}
				else if (body_type == BODY_TYPE_GIANTPLANET)
				{
					param.migType = MIGRATION_TYPE_TYPE_II;
					param.migStopAt = 0.0;
				}
				else 
				{
					param.migType = MIGRATION_TYPE_NO;
					param.migStopAt = 0.0;
				}

				var_t mu = K2*(param0.mass + param.mass);
				int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
				if (ret_code == 1) {
					cerr << "Could not calculate the phase." << endl;
					return ret_code;
				}

				print_body_record(output, name, t, &param, &rVec, &vVec, o_version);
			} /* else */
			convert.str("");
		} /* for */
	} /* for */
	output.flush();
	output.close();

	return 0;
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

string create_number_of_bodies_str(const number_of_bodies *nBodies)
{
	ostringstream converter;   // stream used for the conversion
	
	converter << nBodies->star << '_' 
		<< nBodies->giant_planet << '_' 
		<< nBodies->rocky_planet << '_' 
		<< nBodies->proto_planet << '_' 
		<< nBodies->super_planetesimal << '_' 
		<< nBodies->planetesimal << '_' 
		<< nBodies->test_particle;

	return converter.str();
}

int main(int argc, const char **argv)
{
	//number_of_bodies *nBodies = 0;

	//string outDir;
	//int retCode = parse_options(argc, argv, &nBodies, outDir);
	//if (0 != retCode) {
	//	exit(retCode);
	//}
	//string nbstr = create_number_of_bodies_str(nBodies);

	//var2_t disk = {5.0, 6.0};	// AU
	//retCode = generate_pp_disk(combine_path(outDir, ("nBodies_" + nbstr + ".txt")), disk, nBodies);

	//var2_t disk = {65, 270};
	//retCode = generate_Rezso_disk(combine_path(outDir, ("nBodies_" + nbstr + ".txt")), disk, nBodies);

	// Generate Dvorak disk
	//var2_t disk = {0.9, 2.5};	// AU
	//retCode = generate_Dvorak_disk(combine_path(outDir, ("DvorakDisk01_" + nbstr + ".txt")), disk, nBodies);

	body_disk_t general;
	set_default(general);

	general.nBody[BODY_TYPE_STAR] = 1;
	general.nBody[BODY_TYPE_GIANTPLANET] = 2;
	general.nBody[BODY_TYPE_ROCKYPLANET] = 0;
	general.nBody[BODY_TYPE_PROTOPLANET] = 0;
	general.nBody[BODY_TYPE_SUPERPLANETESIMAL] = 0;
	general.nBody[BODY_TYPE_PLANETESIMAL] = 0;
	general.nBody[BODY_TYPE_TESTPARTICLE] = 0;

	set(general.pp_r[BODY_TYPE_STAR].pp[MASS], 1.0, pdf_const);
	set(general.pp_r[BODY_TYPE_STAR].pp[RADIUS], 1.0 * Constants::SolarRadiusToAu, pdf_const);
	set(general.pp_r[BODY_TYPE_STAR].pp[DRAG_COEFF], 0.0, pdf_const);

	for (int i = 0; i < general.nBody[BODY_TYPE_GIANTPLANET]; i++)
	{
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[SMA], 1.0, 4.0, pdf_const);
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[ECC], 0.0, 0.2, pdf_const);
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[INC], 0.0, 30.0 * Constants::DegreeToRadian, pdf_const);
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[PERI], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[NODE], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);
		set(general.oe_r[BODY_TYPE_GIANTPLANET].oe[MEAN], 0.0, 360.0 * Constants::DegreeToRadian, pdf_const);

		set(general.pp_r[BODY_TYPE_GIANTPLANET].pp[MASS], 1.0 * Constants::JupiterToSolar, 2.0 * Constants::JupiterToSolar, pdf_mass_lognormal);
		set(general.pp_r[BODY_TYPE_GIANTPLANET].pp[DENSITY], 0.7 * Constants::GramPerCm3ToSolarPerAu3, 1.2 * Constants::GramPerCm3ToSolarPerAu3, pdf_mass_lognormal);
		set(general.pp_r[BODY_TYPE_GIANTPLANET].pp[DRAG_COEFF], 0.0, pdf_const);
	}

	string outDir = "C:\\Work\\Projects\\solaris.cuda\\TestRun\\InputTest";
	generate_pp_disk(combine_path(outDir, "general.txt"), general, OUTPUT_VERSION_SECOND);

	return 0;
}
