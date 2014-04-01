#pragma once

#include "ode.h"

class twobody : public ode
{
public:
	twobody();
	virtual void calculate_dy(int i, int r, ttt_t t, const d_var_t& p, const std::vector<d_var_t>& y, d_var_t& dy);

private:
	void allocate_vectors();
	void init_values();
};