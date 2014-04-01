#include "twobody.h"
#include "util.h"

// Calculate acceleration caused by one particle on another
__inline__ __device__ 
	vec_t calculate_accel_pair(const vec_t c1, const vec_t c2, var_t m, vec_t a)
{
	vec_t d;
	
	d.x = c1.x - c2.x;
	d.y = c1.y - c2.y;
	d.z = c1.z - c2.z;

	d.w = d.x * d.x + d.y * d.y + d.z * d.z;
	if (d.w > 0)
	{
		d.w = d.w * d.w * d.w;
		d.w = - 6.67384e-11 * m / sqrt(d.w);

		a.x += d.x * d.w;
		a.y += d.y * d.w;
		a.z += d.z * d.w;
	}

	return a;
}

// Calculate and sum up accelerations
__global__
	void calculate_accel_kernel(const var_t* p, const vec_t* c, vec_t* a)
{
	// Index of this particle
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	a[0].x = a[0].y = a[0].z = a[0].w = 0;
	a[1].x = a[1].y = a[1].z = a[1].w = 0;

	a[1] = calculate_accel_pair(c[1], c[0], p[0], a[1]);
}

twobody::twobody()
	: ode(2, 0.0)
{
	allocate_vectors();
	init_values();
}

void twobody::calculate_dy(int i, int r, ttt_t t, const d_var_t& p, const std::vector<d_var_t>& y, d_var_t& dy)
{
	if (i == 0)
	{
		copy_vec(dy, y[1]);		// velocities
	}
	else if (i == 1)
	{
		dim3 blocks(1, 1, 1);
		dim3 threads(2);

		vec_t* c = (vec_t*)y[0].data().get();

		calculate_accel_kernel<<<blocks, threads>>>(p.data().get(), c, (vec_t*)dy.data().get());
	}
}

void twobody::allocate_vectors()
{
	// Parameters
	h_p.resize(2);

	// Aliases to coordinates and velocities
	h_y[0].resize(8);
	h_y[1].resize(8);
}

void twobody::init_values()
{
	// Earth
	
	h_p[0] = 5.9736e24;	

	h_y[0][0] = 0;
	h_y[0][1] = 0;
	h_y[0][2] = 0;
	h_y[0][3] = 0;

	h_y[1][0] = 0;
	h_y[1][1] = 0;
	h_y[1][2] = 0;
	h_y[1][3] = 0;
	
	// Moon

	h_p[1] = 7.349e22;

	h_y[0][4] = 4.055e8;
	h_y[0][5] = 0;
	h_y[0][6] = 0;
	h_y[0][7] = 0;

	h_y[1][4] = 0;
	h_y[1][5] = 0.964e3;
	h_y[1][6] = 0;
	h_y[1][7] = 0;
}