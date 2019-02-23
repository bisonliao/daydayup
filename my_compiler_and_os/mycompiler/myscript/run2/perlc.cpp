#include <EXTERN.h> 
#include <perl.h> 

#include "perlc.h"

#include <deque>
using std::deque;

static PerlInterpreter * getPerlInterpreter()
{
	static PerlInterpreter *my_perl = NULL; 
	if (NULL == my_perl)
	{
		char *embedding[] = { "", "-e", "0" }; 
		
		PERL_SYS_INIT3(NULL, NULL, NULL);
		my_perl = perl_alloc(); 
		perl_construct( my_perl ); 
		
		perl_parse(my_perl, NULL, 3, embedding, NULL); 
		perl_run(my_perl); 
	}
	return my_perl;
}

void lnb::perlc_end()
{
	PerlInterpreter *my_perl = getPerlInterpreter();
	if (my_perl != NULL)
	{
	perl_destruct(my_perl);
	perl_free(my_perl);
	PERL_SYS_TERM();
	}
}

void lnb::perlc_substitute(string & s, const string& pattern, const string & to, const string & flags)
{
	STRLEN n_a; 

	PerlInterpreter *my_perl = getPerlInterpreter();
	if (NULL == my_perl)
	{
		fprintf(stderr, "getPerlInterpreter() failed!\n");
		return;
	}
	
	string local_s, local_pattern, local_to;

	local_s = s;
	local_pattern = pattern;
	local_to = to;


	string str = "$a = \"" + local_s + "\"; $a =~ s/" + local_pattern + "/" + local_to + "/" + flags + ";";
//	printf("[%s]\n", str.c_str());
	eval_pv(str.c_str(), TRUE); 
	s = SvPV(get_sv("a", FALSE), n_a); 
	
}
void lnb::perlc_translate(string & s, const string& pattern, const string & to)
{
	STRLEN n_a; 

	PerlInterpreter *my_perl = getPerlInterpreter();
	if (NULL == my_perl)
	{
		fprintf(stderr, "getPerlInterpreter() failed!\n");
		return;
	}
	
	string local_s, local_pattern, local_to;

	local_s = s;
	local_pattern = pattern;
	local_to = to;


	string str = "$a = \"" + local_s + "\"; $a =~ tr/" + local_pattern + "/" + local_to + "/;";
	eval_pv(str.c_str(), TRUE); 
	s = SvPV(get_sv("a", FALSE), n_a); 
}
/*
 * if (s =~m/pattern/flags)
 * {
 *    bMatched = true;
 * }
 * else
 * {
 *	 bMatched = false;
 * }
 * lstBackTrace.push_back($0, $1, $2, $3...);
 * 
 */
void lnb::perlc_match(const string & s, const string& pattern, bool & bMatched, 
			deque<string> & lstBackTrace, const string & flags)
{
	STRLEN n_a; 

	PerlInterpreter *my_perl = getPerlInterpreter();
	if (NULL == my_perl)
	{
		fprintf(stderr, "getPerlInterpreter() failed!\n");
		return;
	}
	
	string perlstr = "$a = '" + s + "';\n";
	perlstr.append("if ($a =~m/" + pattern + "/" + flags + ") {$matched = 1;} else {$matched = 0;}\n");
	perlstr.append("$count = @-;\n--$count;\n");
	//perlstr.append("print '>>>',$count, \"\\n\";\n");
	perlstr.append("$array[0] = '';\n");
	perlstr.append("$array[1] = '';\n");
	perlstr.append("for ($i = 1; $i <= $count; ++$i) { $array[ $i-1 ] = substr($a, @-[$i], @+[$i] - @-[$i]);}\n");
	//perlstr.append("print '>>>',$array[0], \"\\n\";\n");
	//perlstr.append("print '>>>',$array[1], \"\\n\";\n");
	perlstr.append("$mmm = substr($a, @-[0], @+[0] - @-[0]);\n");

	//printf("%s\n\n", perlstr.c_str());
	eval_pv(perlstr.c_str(), TRUE); 

    if ( SvIV(get_sv("matched", FALSE)) ) 
	{
		bMatched = true;
	}
	else
	{
		bMatched = false;
	}

	lstBackTrace.clear();
	string sBackTrace = SvPV(get_sv("mmm", FALSE), n_a); 
	lstBackTrace.push_back(sBackTrace);

	AV * submatchlist = get_av("array", FALSE);
	int submatchlistlen = SvIV(get_sv("count", FALSE));
//	printf("<<<%d\n", submatchlistlen);

	int i;
	for (i = 0; i < submatchlistlen; i++)
	{
//		printf("match: %s\n", SvPV(*av_fetch(submatchlist, i, FALSE),n_a));
    	string sSubMatch =  SvPV(*av_fetch(submatchlist, i, FALSE),n_a);
		lstBackTrace.push_back(sSubMatch);	
	}


	return;
}
