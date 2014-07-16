#include <cctype>
#include <ctime>

#include "tools.h"

bool is_number(const string& str)
{
   for (size_t i = 0; i < str.length(); i++) {
	   if (std::isdigit(str[i]) || str[i] == 'e' || str[i] == 'E' || str[i] == '.' || str[i] == '-' || str[i] == '+')
           continue;
	   else
		   return false;
   }
   return true;
}

/// Removes all trailing white-space characters from the current std::string object.
void trim_right(string& str)
{
	// trim trailing spaces
	size_t endpos = str.find_last_not_of(" \t");
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos+1 );
	}
}

/// Removes all trailing characters after the first # character
void trim_right(string& str, char c)
{
	// trim trailing spaces

	size_t endpos = str.find(c);
	if (string::npos != endpos ) {
		str = str.substr( 0, endpos);
	}
}

/// Removes all leading white-space characters from the current std::string object.
void trim_left(string& str)
{
	// trim leading spaces
	size_t startpos = str.find_first_not_of(" \t");
	if (string::npos != startpos ) {
		str = str.substr( startpos );
	}
}

/// Removes all leading and trailing white-space characters from the current std::string object.
void trim(string& str)
{
	trim_right(str);
	trim_left(str);
}

void get_time_stamp(char *time_stamp)
{
	time_t now = time(0);
	strftime(time_stamp, 20, "%Y-%m-%d %H:%M:%S", localtime(&now));
}