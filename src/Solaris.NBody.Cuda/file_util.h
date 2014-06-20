#pragma once
#include <string>

using namespace std;

string combine_path(string dir, string filename);
string get_filename(const string& path);
string get_filename_without_ext(const string& path);
string get_directory(const string& path);
string get_extension(const string& path);
