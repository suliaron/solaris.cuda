#pragma once

#include "config.h"

class rkorder
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

	static int order;
	static std::string name;
};

class rk45 : public rkorder
{
};