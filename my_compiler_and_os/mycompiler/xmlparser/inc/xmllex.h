#if !defined(__XMLLEX_H_INCLUDED__)
#define __XMLLEX_H_INCLUDED__

#include "AnsiString.h"

	#define XML_LEX_HEADER 	256
	#define XML_LEX_BEGIN	257
	#define XML_LEX_VALUE	258
	#define XML_LEX_END		259
	#define XML_LEX_EMPNODE 260

	class  yyval_type
	{
	public:
		AnsiString name;
		AnsiString property;
		AnsiString value;
		yyval_type(){};
		~yyval_type(){};
	};
	typedef yyval_type YYVAL_TYPE;

#endif
