#include <iostream>

#include "config.h"
#include "rungekutta.h"

#include "twobody.h"

using namespace std;

int RK4test()
{
	twobody ode = twobody();
	ode.copy_to_device();

	rungekutta<rk45> rk45test = rungekutta<rk45>(ode, 3600, false, 0.01);

	cout << rk45test.get_name() << endl;

	for (int i = 0; i < 1000; i ++)
	{
		rk45test.step();

		ode.copy_to_host();

		cout << ode.h_y[0][4] << '\t' << ode.h_y[0][5] << endl;
	}

	return 0;
}

int main(int argc, char* argv[])
{
	RK4test();

	return 0;
}