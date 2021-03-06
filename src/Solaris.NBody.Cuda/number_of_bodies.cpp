#include "number_of_bodies.h"

number_of_bodies::number_of_bodies(int star, int giant_planet, int rocky_planet, int proto_planet, int super_planetesimal, int planetesimal, int test_particle) : 
		star(star), 
		giant_planet(giant_planet), 
		rocky_planet(rocky_planet), 
		proto_planet(proto_planet), 
		super_planetesimal(super_planetesimal), 
		planetesimal(planetesimal), 
		test_particle(test_particle) 
{
	total = star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal + test_particle;
	total_rounded_up = 0;
}

int	number_of_bodies::n_total()
{
	return star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal + test_particle;
}

int	number_of_bodies::n_massive()
{
	return star + giant_planet + rocky_planet + proto_planet + super_planetesimal + planetesimal;
}

int	number_of_bodies::n_self_interacting() 
{
	return star + giant_planet + rocky_planet + proto_planet;
}

int	number_of_bodies::n_gas_drag()
{
	return super_planetesimal + planetesimal;
}

int	number_of_bodies::n_migrate_typeI()
{
	return rocky_planet + proto_planet;
}

int	number_of_bodies::n_migrate_typeII()
{
	return giant_planet;
}

interaction_bound number_of_bodies::get_self_interacting()
{
	sink.x		= 0;
	sink.y		= n_self_interacting();
	source.x	= 0;
	source.y	= n_massive();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_nonself_interacting()
{
	sink.x			= n_self_interacting();
	sink.y			= n_massive();
	source.x		= 0;
	source.y		= n_self_interacting();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_non_interacting()
{
	sink.x			= n_massive();
	sink.y			= total;
	source.x		= 0;
	source.y		= n_massive();
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_gasdrag() {
	sink.x			= n_self_interacting();
	sink.y			= n_massive();
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_migrate_typeI() {
	sink.x			= star + giant_planet;
	sink.y			= n_massive();
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
}

interaction_bound number_of_bodies::get_bodies_migrate_typeII() {
	sink.x			= star;
	sink.y			= star + giant_planet;
	source.x		= 0;
	source.y		= 0;
	interaction_bound iBound(sink, source);

	return iBound;
}
