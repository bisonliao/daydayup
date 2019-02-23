#include "script.h"
using namespace lnb;

void CScript::InitFuncMap( map<string, FUNC_PTR> & mapFuncs)
{
		mapFuncs.insert(std::pair<string, FUNC_PTR>(string("lnb_print"), &CScript::InnerFuncs));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_toint", &CScript::InnerFuncs));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_tofloat", &CScript::InnerFuncs));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_tostr", &CScript::InnerFuncs));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_ltrim", &CScript::InnerFuncs));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_rtrim", &CScript::InnerFuncs));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fopen", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fclose", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_popen", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_pclose", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fread", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fwrite", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fseek", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_ftell", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fgets", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_fprintf", &CScript::filefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_printf", &CScript::filefunc));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_split", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_length", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_match", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_sprintf", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_find", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_substitute", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_translate", &CScript::stringfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_substr", &CScript::stringfunc));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_system", &CScript::ipcfunc));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_connect", &CScript::mysqlfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_close", &CScript::mysqlfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_query", &CScript::mysqlfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_fetchrow", &CScript::mysqlfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_getAffectedRowNum", &CScript::mysqlfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_mysql_getFieldTitle", &CScript::mysqlfunc));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_time", &CScript::datetimefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_str2time", &CScript::datetimefunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_localtime", &CScript::datetimefunc));

		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_open", &CScript::mapfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_close", &CScript::mapfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_find", &CScript::mapfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_insert", &CScript::mapfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_erase", &CScript::mapfunc));
		mapFuncs.insert(std::pair<string, FUNC_PTR>("lnb_map_size", &CScript::mapfunc));
}
