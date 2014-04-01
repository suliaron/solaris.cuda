#include "config.h"
#include "rkorder.h"

// --- Runge-Kutta 4(5)

#define	LAMBDA	1.0/10.0

ttt_t rk45::c[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0};
var_t rk45::a[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
var_t rk45::bh[] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0, 0.0};
var_t rk45::b[] =  {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 -LAMBDA, LAMBDA};

int rk45::order = 4;
std::string rk45::name = "RungeKutta4(5)";