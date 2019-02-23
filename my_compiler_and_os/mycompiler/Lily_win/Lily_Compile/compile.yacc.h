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
#define STRING 270
#define FLOAT 271
#define RETURN 272
#define BEGIN_FLOW 273
#define END_FLOW 274
#define RUN 275
#define FOR 276
#define ENDFOR 277
#define CONTINUE 278
#define BREAK 279
#define REPEAT 280
#define UNTIL 281
#define SWITCH 282
#define ENDSWITCH 283
#define CASE 284
#define OR 285
#define AND 286
#define LT 287
#define LE 288
#define EQ 289
#define NE 290
#define GT 291
#define GE 292
#define UMINUS 293
#define NOT 294
#define MEMBLOCK 295
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
