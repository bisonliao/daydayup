#ifndef __PERLC_H_INCLUDED__
#define __PERLC_H_INCLUDED__

#include <string>
#include <deque>
using std::deque;
using std::string;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace lnb{

void perlc_end();
void perlc_substitute(string & s, const string& pattern, const string & to, const string & flags);
void perlc_translate(string & s, const string& pattern, const string & to);
void perlc_match(const string & s, const string& pattern, bool & bMatched, 
		deque<string> & lstBackTrace, const string & flags);
};

#endif
