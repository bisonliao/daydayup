#ifndef YYERRCODE
#define YYERRCODE 256
#endif

#define ID 257
#define CONST_STRING 258
#define CONST_INTEGER 259
#define CONST_FLOAT 260
#define FUNCTION 261
#define IF 262
#define THEN 263
#define ELSE 264
#define ENDIF 265
#define WHILE 266
#define DO 267
#define ENDWHILE 268
#define INTEGER 269
#define MEMBLOCK 270
#define STRING 271
#define FLOAT 272
#define RETURN 273
#define BEGIN_FLOW 274
#define END_FLOW 275
#define RUN 276
#define FOR 277
#define ENDFOR 278
#define CONTINUE 279
#define BREAK 280
#define REPEAT 281
#define UNTIL 282
#define SWITCH 283
#define ENDSWITCH 284
#define CASE 285
#define OR 286
#define AND 287
#define LT 288
#define LE 289
#define EQ 290
#define NE 291
#define GT 292
#define GE 293
#define UMINUS 294
#define NOT 295
typedef union{
	char const_string_val[QSTRING_BUF_MAX];
	int const_int_val;
	char const_float_val[100];
	char id_val[ID_MAX];
	int expr_type;
	char sign_val;
	Labels fake_val;
	int arg_count;
} YYSTYPE;
extern YYSTYPE yylval;
