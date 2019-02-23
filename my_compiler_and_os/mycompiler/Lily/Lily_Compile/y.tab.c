#ifndef lint
static char const 
yyrcsid[] = "$FreeBSD: src/usr.bin/yacc/skeleton.c,v 1.28 2000/01/17 02:04:06 bde Exp $";
#endif
#include <stdlib.h>
#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define YYLEX yylex()
#define YYEMPTY -1
#define yyclearin (yychar=(YYEMPTY))
#define yyerrok (yyerrflag=0)
#define YYRECOVERING() (yyerrflag!=0)
static int yygrowstack();
#define YYPREFIX "yy"
#line 2 "Lily_compile.yacc"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "common.h"
#include "Token_Stack.h"
#include "FlowName_Stack.h"
#include "int_Stack.h"
#include "Labels_Stack.h"
#include "lily_fc.h"
#include <assert.h>
#include "tool.h"

int yylex();
void RemoveTokenFromTab();
void CheckIDAlreadyExist(const char* id_name);
void CheckIDValid(const char * id_name, Token &ttt);
void InsertToken(const char *id_name, const char* id_type);
unsigned int GetNewLabel();
int GetNewCSIndex();
int MergeObjectFile(const char *output_file);


extern  int yylineno;
extern char yyfilename[255];
const char* getyytext();

/*当前分析的流程块的名字*/
static char current_flow_name[ID_MAX];
/*符号表*/
static Token_Stack token_stack;
/*每个流程块的变量个数保存栈*/
static int_Stack varnum_stack;
/*声明变量时的类型*/
static char declare_type[20];
/*循环语句用到的跳转标签的保存栈，用于翻译break、continue*/
static Labels_Stack labels_stack;
/*暂存yyout,翻译for语句时用到*/
static FILE * for_fp = NULL;

/*保存流程块的名字，防止重复定义*/
static FlowName_Stack flowname_stack;

/*标识是否定义了main流程块*/
static int hasMain = 0;

/*常量字符串的写入文件*/
FILE * csout = NULL;

#define INSTRUCT_FILE "./lily.instruct"
#define CSTRING_FILE "./lily.cstring"
#define TMP_FILE     "./lily.tmp"



#line 59 "Lily_compile.yacc"
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
#line 84 "y.tab.c"
#define YYERRCODE 256
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
#define SWITCHEND 283
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
const short yylhs[] = {                                        -1,
    0,    0,   13,   11,   11,   12,   12,   14,   14,   14,
   16,   14,   14,   17,   14,   18,   19,   20,   14,   14,
   14,   14,   14,   14,   14,   14,   15,   22,   22,   22,
   23,   23,   21,   21,   24,    1,    1,    1,    1,    1,
    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    1,    3,    3,    2,    2,    2,    2,    2,   10,    9,
    9,    9,    4,    5,    6,    7,    8,
};
const short yylen[] = {                                         2,
    2,    1,    0,    7,    5,    2,    1,    4,    1,    2,
    0,    9,    6,    0,    7,    0,    0,    0,   19,    8,
    5,    2,    2,    2,    5,    1,    3,    1,    1,    1,
    3,    1,    2,    1,    5,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    2,    2,
    1,    1,    1,    1,    1,    1,    1,    1,    4,    3,
    1,    0,    0,    0,    0,    0,    0,
};
const short yydefred[] = {                                      0,
    0,    0,    2,    0,    1,    0,    0,    5,    0,    0,
   55,   56,   57,    0,   64,   28,   29,   30,    0,    0,
    0,    0,    0,   66,   67,   52,   53,    0,   26,    0,
    0,   51,    0,   58,    0,    7,    9,    0,    0,    0,
    0,    0,    0,   22,    0,    0,   23,   24,    0,    0,
   50,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   10,   49,    4,    6,   32,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   34,   48,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   38,   39,   27,    0,    8,   59,    0,    0,
    0,    0,    0,    0,    0,    0,   33,   31,    0,    0,
    0,   21,    0,    0,    0,   25,   11,   13,    0,   16,
    0,    0,    0,   15,   65,    0,    0,    0,    0,   20,
   12,    0,   17,    0,    0,    0,    0,   18,    0,    0,
    0,   19,
};
const short yydgoto[] = {                                       2,
   31,   32,   33,   74,   43,  129,   49,   50,   73,   34,
    3,   35,    9,   36,   37,  123,  101,  125,  134,  139,
   80,   38,   70,   81,
};
const short yysindex[] = {                                    -78,
 -234,  -78,    0,  -63,    0, -241, -236,    0,  117,  -39,
    0,    0,    0,  169,    0,    0,    0,    0,  -20,    1,
    2,  -19,  -15,    0,    0,    0,    0,  169,    0,  169,
  462,    0,  169,    0,  -14,    0,    0, -214,  169,  169,
    6,  471,  169,    0,  169, -210,    0,    0,  117, -235,
    0,  417,  169,  169,  169,  169,  169,  169,  169,  169,
  169,  169,  169,  169,    0,    0,    0,    0,    0,  -38,
  538,  471,  -37, -215,  471,  425,  -11,   12,   14, -249,
    0,    0,  582,  588,  -33,  -33,  -33,  -33,  -33,  -33,
  -31,  -31,    0,    0,    0, -201,    0,    0,  169,  117,
 -209,    3,  169,   19,  169,    4,    0,    0,  471,  -40,
  117,    0,  546,  169,  433,    0,    0,    0,   39,    0,
  441,  117,  117,    0,    0,    7,  117,   65,  169,    0,
    0,  564,    0, -197,    8,  169,  449,    0, -206,  117,
   91,    0,
};
const short yyrindex[] = {                                      0,
    0,    0,    0,    0,    0,    0,  143,    0,    0,  573,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  -24,
  389, -198,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  -17,    0,    0, -200,    0,    0,    0,    0,    0,
    0,    0,  -26,   53,  245,  479,  509,  517,  525,  555,
  397,  406,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,  -16,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0, -247,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,
};
const short yygindex[] = {                                      0,
  853,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   66,  -47,    0,  -27,    0,    0,    0,    0,    0,    0,
    0,    0,    0,  -10,
};
#define YYTABLESIZE 989
const short yytable[] = {                                      30,
   40,   78,   26,   98,   27,   96,   99,   68,   63,   61,
   63,   62,    1,   64,   41,   64,   62,   41,   29,   62,
   95,   39,    4,   61,   60,   30,   61,   60,   26,    6,
   27,    7,   41,  106,   79,   35,   35,    8,   44,   47,
   45,   46,   69,   48,   29,   40,   77,  100,   79,  103,
   68,   30,  110,  105,   26,  108,   27,  111,  114,  135,
  140,  112,  116,  119,   63,  130,   14,    5,  136,  107,
   29,    0,    0,    0,  127,  128,    0,    0,   30,    0,
    0,   26,   68,   27,    0,    0,    0,    0,    0,    0,
    0,   68,  141,   40,    0,    0,   40,   29,    0,   68,
   68,    0,    0,    0,   30,    0,    0,   26,    0,   27,
    0,   40,    0,   68,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   29,    0,    0,    0,    0,    0,    0,
   30,    0,    0,   26,    0,   27,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,   29,
    0,    0,    0,    0,    0,    0,   30,    0,    0,   26,
    0,   27,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   29,    0,    0,    0,    0,
    0,    0,    3,    0,    0,    3,    0,    3,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    3,    0,    0,    0,    0,    0,    0,   30,    0,
    0,   26,    0,   27,    0,    0,   10,   11,   12,   13,
    0,   14,    0,  117,  118,   15,    0,    0,   16,   17,
   18,   19,    0,    0,   20,   21,   41,   22,   23,   24,
   41,   25,   10,   11,   12,   13,    0,   14,    0,    0,
    0,   15,    0,   28,   16,   17,   18,   19,   41,   67,
   20,   21,    0,   22,   23,   24,    0,   25,   10,   11,
   12,   13,    0,   14,    0,    0,    0,   15,    0,   28,
   16,   17,   18,   19,    0,   42,   20,   21,   42,   22,
   23,   24,  104,   25,    0,   10,   11,   12,   13,    0,
   14,    0,    0,   42,   15,   28,  124,   16,   17,   18,
   19,    0,    0,   20,   21,   40,   22,   23,   24,   40,
   25,   10,   11,   12,   13,    0,   14,    0,    0,  131,
   15,    0,   28,   16,   17,   18,   19,   40,   40,   20,
   21,    0,   22,   23,   24,    0,   25,   10,   11,   12,
   13,    0,   14,    0,    0,    0,   15,    0,   28,   16,
   17,   18,   19,    0,    0,   20,   21,  142,   22,   23,
   24,    0,   25,   10,   11,   12,   13,    0,   14,    0,
    0,    0,   15,    0,   28,   16,   17,   18,   19,    0,
    0,   20,   21,    0,   22,   23,   24,    0,   25,    3,
    3,    3,    3,    0,    3,    0,    0,    0,    3,    0,
   28,    3,    3,    3,    3,    0,    0,    3,    3,    0,
    3,    3,    3,    0,    3,   41,   11,   12,   13,   54,
   54,   54,   54,   54,    0,   54,    3,   36,    0,   36,
   36,   36,    0,    0,    0,    0,   37,   54,   37,   37,
   37,    0,    0,    0,    0,   36,    0,   82,   63,   61,
    0,   62,   28,   64,   37,  102,   63,   61,    0,   62,
    0,   64,    0,  122,   63,   61,    0,   62,    0,   64,
    0,  126,   63,   61,    0,   62,    0,   64,    0,  138,
   63,   61,    0,   62,    0,   64,    0,    0,    0,    0,
    0,    0,    0,   63,   61,    0,   62,   42,   64,    0,
    0,   42,   63,   61,    0,   62,    0,   64,    0,   43,
   65,    0,   43,    0,    0,    0,    0,    0,    0,   42,
   42,   42,   42,   42,   42,   42,   42,   43,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,   45,
    0,    0,   45,    0,    0,    0,    0,   44,    0,    0,
   44,    0,    0,    0,    0,   46,    0,   45,   46,    0,
    0,    0,    0,    0,    0,   44,    0,    0,    0,   63,
   61,    0,   62,   46,   64,    0,    0,   63,   61,    0,
   62,    0,   64,    0,    0,   47,   97,    0,   47,    0,
    0,    0,    0,    0,  120,   63,   61,    0,   62,    0,
   64,    0,    0,   47,   54,   54,    0,   54,    0,   54,
    0,    0,  133,   63,   61,    0,   62,    0,   64,   63,
   61,   54,   62,    0,   64,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   54,    0,    0,    0,   54,    0,    0,    0,   36,
    0,    0,    0,   36,    0,    0,    0,    0,   37,    0,
    0,    0,   37,   54,   54,   54,   54,   54,   54,   54,
   54,   36,   36,   36,   36,   36,   36,   36,   36,    0,
   37,   37,   37,   37,   37,   37,   37,   37,    0,    0,
    0,   53,   54,   55,   56,   57,   58,   59,   60,   53,
   54,   55,   56,   57,   58,   59,   60,   53,   54,   55,
   56,   57,   58,   59,   60,   53,   54,   55,   56,   57,
   58,   59,   60,   53,   54,   55,   56,   57,   58,   59,
   60,   43,    0,    0,    0,   43,   53,   54,   55,   56,
   57,   58,   59,   60,    0,   53,   54,   55,   56,   57,
   58,   59,   60,   43,   43,   43,   43,   43,   43,   43,
   43,   45,    0,    0,    0,   45,    0,    0,    0,   44,
    0,    0,    0,   44,    0,    0,    0,   46,    0,    0,
    0,   46,    0,   45,   45,   45,   45,   45,   45,   45,
   45,   44,   44,   44,   44,   44,   44,   44,   44,   46,
   46,   46,   46,   46,   46,   46,   46,   47,    0,    0,
    0,   47,   53,   54,   55,   56,   57,   58,   59,   60,
   53,   54,   55,   56,   57,   58,   59,   60,    0,   47,
   47,   47,   47,   47,   47,   47,   47,    0,   53,   54,
   55,   56,   57,   58,   59,   60,    0,   54,   54,   54,
   54,   54,   54,   54,   54,    0,   42,   54,   55,   56,
   57,   58,   59,   60,   55,   56,   57,   58,   59,   60,
   51,    0,   52,    0,    0,   66,    0,    0,    0,    0,
    0,   71,   72,    0,    0,   75,    0,   76,    0,    0,
    0,    0,    0,    0,    0,   83,   84,   85,   86,   87,
   88,   89,   90,   91,   92,   93,   94,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  109,    0,    0,    0,  113,    0,  115,    0,    0,
    0,    0,    0,    0,    0,    0,  121,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,  132,    0,    0,    0,    0,    0,    0,  137,
};
const short yycheck[] = {                                      40,
   40,   49,   43,   41,   45,   44,   44,   35,   42,   43,
   42,   45,   91,   47,   41,   47,   41,   44,   59,   44,
   59,   61,  257,   41,   41,   40,   44,   44,   43,   93,
   45,  273,   59,  283,  284,  283,  284,  274,   59,   59,
   40,   40,  257,   59,   59,   40,  257,  263,  284,   61,
   78,   40,  100,   40,   43,  257,   45,  267,   40,  257,
  267,   59,   59,  111,  263,   59,  267,    2,   61,   80,
   59,   -1,   -1,   -1,  122,  123,   -1,   -1,   40,   -1,
   -1,   43,  110,   45,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  119,  140,   41,   -1,   -1,   44,   59,   -1,  127,
  128,   -1,   -1,   -1,   40,   -1,   -1,   43,   -1,   45,
   -1,   59,   -1,  141,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   59,   -1,   -1,   -1,   -1,   -1,   -1,
   40,   -1,   -1,   43,   -1,   45,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   59,
   -1,   -1,   -1,   -1,   -1,   -1,   40,   -1,   -1,   43,
   -1,   45,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   59,   -1,   -1,   -1,   -1,
   -1,   -1,   40,   -1,   -1,   43,   -1,   45,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   59,   -1,   -1,   -1,   -1,   -1,   -1,   40,   -1,
   -1,   43,   -1,   45,   -1,   -1,  257,  258,  259,  260,
   -1,  262,   -1,  264,  265,  266,   -1,   -1,  269,  270,
  271,  272,   -1,   -1,  275,  276,  263,  278,  279,  280,
  267,  282,  257,  258,  259,  260,   -1,  262,   -1,   -1,
   -1,  266,   -1,  294,  269,  270,  271,  272,  285,  274,
  275,  276,   -1,  278,  279,  280,   -1,  282,  257,  258,
  259,  260,   -1,  262,   -1,   -1,   -1,  266,   -1,  294,
  269,  270,  271,  272,   -1,   41,  275,  276,   44,  278,
  279,  280,  281,  282,   -1,  257,  258,  259,  260,   -1,
  262,   -1,   -1,   59,  266,  294,  268,  269,  270,  271,
  272,   -1,   -1,  275,  276,  263,  278,  279,  280,  267,
  282,  257,  258,  259,  260,   -1,  262,   -1,   -1,  265,
  266,   -1,  294,  269,  270,  271,  272,  285,  286,  275,
  276,   -1,  278,  279,  280,   -1,  282,  257,  258,  259,
  260,   -1,  262,   -1,   -1,   -1,  266,   -1,  294,  269,
  270,  271,  272,   -1,   -1,  275,  276,  277,  278,  279,
  280,   -1,  282,  257,  258,  259,  260,   -1,  262,   -1,
   -1,   -1,  266,   -1,  294,  269,  270,  271,  272,   -1,
   -1,  275,  276,   -1,  278,  279,  280,   -1,  282,  257,
  258,  259,  260,   -1,  262,   -1,   -1,   -1,  266,   -1,
  294,  269,  270,  271,  272,   -1,   -1,  275,  276,   -1,
  278,  279,  280,   -1,  282,  257,  258,  259,  260,   41,
   42,   43,   44,   45,   -1,   47,  294,   41,   -1,   43,
   44,   45,   -1,   -1,   -1,   -1,   41,   59,   43,   44,
   45,   -1,   -1,   -1,   -1,   59,   -1,   41,   42,   43,
   -1,   45,  294,   47,   59,   41,   42,   43,   -1,   45,
   -1,   47,   -1,   41,   42,   43,   -1,   45,   -1,   47,
   -1,   41,   42,   43,   -1,   45,   -1,   47,   -1,   41,
   42,   43,   -1,   45,   -1,   47,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   42,   43,   -1,   45,  263,   47,   -1,
   -1,  267,   42,   43,   -1,   45,   -1,   47,   -1,   41,
   59,   -1,   44,   -1,   -1,   -1,   -1,   -1,   -1,  285,
  286,  287,  288,  289,  290,  291,  292,   59,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   41,
   -1,   -1,   44,   -1,   -1,   -1,   -1,   41,   -1,   -1,
   44,   -1,   -1,   -1,   -1,   41,   -1,   59,   44,   -1,
   -1,   -1,   -1,   -1,   -1,   59,   -1,   -1,   -1,   42,
   43,   -1,   45,   59,   47,   -1,   -1,   42,   43,   -1,
   45,   -1,   47,   -1,   -1,   41,   59,   -1,   44,   -1,
   -1,   -1,   -1,   -1,   59,   42,   43,   -1,   45,   -1,
   47,   -1,   -1,   59,   42,   43,   -1,   45,   -1,   47,
   -1,   -1,   59,   42,   43,   -1,   45,   -1,   47,   42,
   43,   59,   45,   -1,   47,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,  263,
   -1,   -1,   -1,  267,   -1,   -1,   -1,   -1,  263,   -1,
   -1,   -1,  267,  285,  286,  287,  288,  289,  290,  291,
  292,  285,  286,  287,  288,  289,  290,  291,  292,   -1,
  285,  286,  287,  288,  289,  290,  291,  292,   -1,   -1,
   -1,  285,  286,  287,  288,  289,  290,  291,  292,  285,
  286,  287,  288,  289,  290,  291,  292,  285,  286,  287,
  288,  289,  290,  291,  292,  285,  286,  287,  288,  289,
  290,  291,  292,  285,  286,  287,  288,  289,  290,  291,
  292,  263,   -1,   -1,   -1,  267,  285,  286,  287,  288,
  289,  290,  291,  292,   -1,  285,  286,  287,  288,  289,
  290,  291,  292,  285,  286,  287,  288,  289,  290,  291,
  292,  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,  263,
   -1,   -1,   -1,  267,   -1,   -1,   -1,  263,   -1,   -1,
   -1,  267,   -1,  285,  286,  287,  288,  289,  290,  291,
  292,  285,  286,  287,  288,  289,  290,  291,  292,  285,
  286,  287,  288,  289,  290,  291,  292,  263,   -1,   -1,
   -1,  267,  285,  286,  287,  288,  289,  290,  291,  292,
  285,  286,  287,  288,  289,  290,  291,  292,   -1,  285,
  286,  287,  288,  289,  290,  291,  292,   -1,  285,  286,
  287,  288,  289,  290,  291,  292,   -1,  285,  286,  287,
  288,  289,  290,  291,  292,   -1,   14,  286,  287,  288,
  289,  290,  291,  292,  287,  288,  289,  290,  291,  292,
   28,   -1,   30,   -1,   -1,   33,   -1,   -1,   -1,   -1,
   -1,   39,   40,   -1,   -1,   43,   -1,   45,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   53,   54,   55,   56,   57,
   58,   59,   60,   61,   62,   63,   64,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   99,   -1,   -1,   -1,  103,   -1,  105,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  114,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  129,   -1,   -1,   -1,   -1,   -1,   -1,  136,
};
#define YYFINAL 2
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 294
#if YYDEBUG
const char * const yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'","'*'","'+'","','","'-'",0,"'/'",0,0,0,0,0,0,0,0,0,0,0,
"';'",0,"'='",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'['",0,
"']'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"ID","CONST_STRING","CONST_INTEGER","CONST_FLOAT","FUNCTION","IF",
"THEN","ELSE","ENDIF","WHILE","DO","ENDWHILE","INTEGER","STRING","FLOAT",
"RETURN","BEGIN_FLOW","END_FLOW","RUN","FOR","ENDFOR","CONTINUE","BREAK",
"REPEAT","UNTIL","SWITCH","SWITCHEND","CASE","OR","AND","LT","LE","EQ","NE",
"GT","GE","UMINUS","NOT",
};
const char * const yyrule[] = {
"$accept : flow_list",
"flow_list : flow_list flow",
"flow_list : flow",
"$$1 :",
"flow : '[' ID ']' BEGIN_FLOW $$1 statement_list END_FLOW",
"flow : '[' ID ']' BEGIN_FLOW END_FLOW",
"statement_list : statement_list statement",
"statement_list : statement",
"statement : ID '=' expr ';'",
"statement : declarations",
"statement : expr ';'",
"$$2 :",
"statement : IF expr fake1 THEN statement_list ELSE $$2 statement_list ENDIF",
"statement : IF expr fake1 THEN statement_list ENDIF",
"$$3 :",
"statement : WHILE fake2 expr $$3 DO statement_list ENDWHILE",
"$$4 :",
"$$5 :",
"$$6 :",
"statement : FOR '(' ID '=' expr ';' $$4 fake3 expr ';' $$5 ID '=' expr ')' $$6 DO statement_list ENDFOR",
"statement : REPEAT fake4 statement_list UNTIL '(' expr ')' ';'",
"statement : RUN '(' expr ')' ';'",
"statement : RETURN ';'",
"statement : CONTINUE ';'",
"statement : BREAK ';'",
"statement : SWITCH fake5 case_statement_list SWITCHEND ';'",
"statement : ';'",
"declarations : var_type id_list ';'",
"var_type : INTEGER",
"var_type : STRING",
"var_type : FLOAT",
"id_list : id_list ',' ID",
"id_list : ID",
"case_statement_list : case_statement_list case_statement",
"case_statement_list : case_statement",
"case_statement : CASE '(' expr ')' statement_list",
"expr : expr '+' expr",
"expr : expr '-' expr",
"expr : expr '*' expr",
"expr : expr '/' expr",
"expr : expr AND expr",
"expr : expr OR expr",
"expr : expr LT expr",
"expr : expr LE expr",
"expr : expr NE expr",
"expr : expr EQ expr",
"expr : expr GT expr",
"expr : expr GE expr",
"expr : '(' expr ')'",
"expr : sign expr",
"expr : NOT expr",
"expr : factor",
"sign : '+'",
"sign : '-'",
"factor : ID",
"factor : CONST_STRING",
"factor : CONST_INTEGER",
"factor : CONST_FLOAT",
"factor : function",
"function : ID '(' arg_list ')'",
"arg_list : arg_list ',' expr",
"arg_list : expr",
"arg_list :",
"fake1 :",
"fake2 :",
"fake3 :",
"fake4 :",
"fake5 :",
};
#endif
#if YYDEBUG
#include <stdio.h>
#endif
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 10000
#define YYMAXDEPTH 10000
#endif
#endif
#define YYINITSTACKSIZE 200
int yydebug;
int yynerrs;
int yyerrflag;
int yychar;
short *yyssp;
YYSTYPE *yyvsp;
YYSTYPE yyval;
YYSTYPE yylval;
short *yyss;
short *yysslim;
YYSTYPE *yyvs;
int yystacksize;
#line 631 "Lily_compile.yacc"
extern FILE * yyin;
extern FILE * yyout;
int yyparse();
int main(int argc, char** argv)
{
	/*读取参数*/
	int opt;
	char input_file[255];
	char output_file[255];
	memset(input_file, 0, sizeof(input_file));
	memset(output_file, 0, sizeof(output_file));
	/*
	*-o 输出文件名 只能有一个，如果有多个，以最后一个为准
	*-f 输入的文件名 只能有一个，如果有多个，以最后一个为准
	*/
	while (1)
	{
   		opt = getopt(argc, argv, "o:f:");
		if (opt == -1)
		{
			break;
		}
		switch (opt)
		{
			case 'o':
				strncpy(output_file, optarg, sizeof(output_file));
				break;
			case 'f':
				strncpy(input_file, optarg, sizeof(input_file));
				break;
			default:
				return -1;
				break;
		}
	}
	yyin = stdin;
	if (strlen(input_file) > 0)
	{
		if ((yyin = fopen(input_file, "rb")) == NULL)
		{
			fprintf(stderr, "打开文件%s失败!\n", input_file);
			return -1;
		}
	}
	/*打开指令输出文件*/
	if ( (yyout = fopen(INSTRUCT_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", INSTRUCT_FILE);
		return -1;
	}
	/*打开常量字符串输出文件*/
	if ( (csout = fopen(CSTRING_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", CSTRING_FILE);
		return -1;
	}
	/*读取函数的配置*/
	if (fcInit(getenv("FC")) < 0)
	{
		fprintf(stderr, "读取函数配置信息失败!\n");
		return -1;
	}
	/*编译*/
	yyparse();

	fclose(yyin);
	fclose(yyout);
	fclose(csout);
	yyin = NULL;
	yyout = NULL;
	csout = NULL;

	if (hasMain == 0)
	{
		fprintf(stderr, "主流程main没有定义!\n");
		return -1;
	}
	/*合并常量字符串文件和指令文件*/
	if (strlen(output_file) > 0)
	{
		MergeObjectFile(output_file);
	}
	else
	{
		MergeObjectFile(NULL);
	}

	return 0;
}
int GetNewCSIndex()
{
	static int index = 0;
	return index++;
}
int MergeObjectFile(const char *output_file)
{
	FILE * in, *out;

	out = stdout;
	if (output_file != NULL)
	{
		if ( (out = fopen(output_file, "wb")) == NULL)
		{
			fprintf(stderr, "打开文件%s失败!\n", output_file);
			return -1;
		}
	}
	if ( (in = fopen(CSTRING_FILE, "rb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", CSTRING_FILE);
		fclose(out);
		return -1;
	}
	char buf[100];
	int len;
	while ( (len = fread(buf, 1, sizeof(buf), in)) > 0)
	{
		fwrite(buf, 1, len, out);
	}
	fclose(in);
	fprintf(out, "%%%%\n");
	if ( (in = fopen(INSTRUCT_FILE, "rb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", INSTRUCT_FILE);
		fclose(out);
		return -1;
	}
	while ( (len = fread(buf, 1, sizeof(buf), in)) > 0)
	{
		fwrite(buf, 1, len, out);
	}
	fclose(in);
	fclose(out);
	remove(INSTRUCT_FILE);
	remove(CSTRING_FILE);
	return 0;
}
void yyerror(const char *msg)
{
	fprintf(stderr, "%s\tfilename=[%s]\tlineno=[%d]\n",
		msg,
		yyfilename,
		yylineno);
	fflush(stderr);
	exit(-1);
}
/*
*检查语句中的标识符是否已经声明过
*/
void CheckIDValid(const char * id_name, Token &ttt)
{
	/*首先检查该变量是否是全局的(预先定义的变量)*/	
	int gIdx;
	if (isGlobalID(id_name, &gIdx) == 1)
	{
		strcpy(ttt.tk_name, id_name);
		ttt.tk_type = TYPE_STRING;
		return;
	}


	int varnum;
	char tmp[256];
	int i;
	Token tk;
	varnum_stack.peek(varnum);
	token_stack.BeginPeekFrmTop();
	for (i = 0; i < varnum; i++)
	{
		token_stack.PeekNextFrmTop(tk);	
		if (strcmp(tk.tk_name, id_name) == 0)
		{
			ttt = tk;
			return;
		}
	}
	sprintf(tmp, "变量%s没有定义!", id_name);
	yyerror(tmp);
}
/*
*检查变量是否重复定义
*/
void CheckIDAlreadyExist(const char* id_name)
{
	/*首先检查该变量是否是全局的(预先定义的变量)*/	
	int gIdx;
	char tmp[256];
	if (isGlobalID(id_name, &gIdx) == 1)
	{
			sprintf(tmp, "变量%s是预先定义的全局变量，不能重复定义!", id_name);
			yyerror(tmp);
	}

	int varnum;
	int i;
	Token tk;
	varnum_stack.peek(varnum);
	token_stack.BeginPeekFrmTop();
	for (i = 0; i < varnum; i++)
	{
		token_stack.PeekNextFrmTop(tk);	
		if (strcmp(tk.tk_name, id_name) == 0)
		{
			sprintf(tmp, "变量%s重复定义!", id_name);
			yyerror(tmp);
		}
	}
	return;
}
/*
*符号表退栈
*/
void RemoveTokenFromTab()
{
	int varnum, i;
	Token tk;

	varnum_stack.pop(varnum);
	for (i = 0; i < varnum; i++)
	{
		token_stack.pop(tk);
	}
}
/*
*插入符号表
*/
void InsertToken(const char *id_name, const char* id_type)
{
	int type;
	if (strcmp(id_type, "INTEGER") == 0)
	{
		type = TYPE_INTEGER;
	}
	if (strcmp(id_type, "STRING") == 0)
	{
		type = TYPE_STRING;
	}
	if (strcmp(id_type, "FLOAT") == 0)
	{
		type = TYPE_FLOAT;
	}
	Token tk;
	strcpy(tk.tk_name, id_name);
	tk.tk_type = type;
	token_stack.push(tk);
	
	int varnum;
	varnum_stack.pop(varnum);
	varnum++;
	varnum_stack.push(varnum);
}
/*
*每调用一次，产生一个唯一的标签号
*/
unsigned int GetNewLabel()
{
	static int label_index = 0;
	return ++label_index;
}
#line 784 "y.tab.c"
/* allocate initial stack or double stack size, up to YYMAXDEPTH */
static int yygrowstack()
{
    int newsize, i;
    short *newss;
    YYSTYPE *newvs;

    if ((newsize = yystacksize) == 0)
        newsize = YYINITSTACKSIZE;
    else if (newsize >= YYMAXDEPTH)
        return -1;
    else if ((newsize *= 2) > YYMAXDEPTH)
        newsize = YYMAXDEPTH;
    i = yyssp - yyss;
    newss = yyss ? (short *)realloc(yyss, newsize * sizeof *newss) :
      (short *)malloc(newsize * sizeof *newss);
    if (newss == NULL)
        return -1;
    yyss = newss;
    yyssp = newss + i;
    newvs = yyvs ? (YYSTYPE *)realloc(yyvs, newsize * sizeof *newvs) :
      (YYSTYPE *)malloc(newsize * sizeof *newvs);
    if (newvs == NULL)
        return -1;
    yyvs = newvs;
    yyvsp = newvs + i;
    yystacksize = newsize;
    yysslim = yyss + newsize - 1;
    return 0;
}

#define YYABORT goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR goto yyerrlab

#ifndef YYPARSE_PARAM
#if defined(__cplusplus) || __STDC__
#define YYPARSE_PARAM_ARG void
#define YYPARSE_PARAM_DECL
#else	/* ! ANSI-C/C++ */
#define YYPARSE_PARAM_ARG
#define YYPARSE_PARAM_DECL
#endif	/* ANSI-C/C++ */
#else	/* YYPARSE_PARAM */
#ifndef YYPARSE_PARAM_TYPE
#define YYPARSE_PARAM_TYPE void *
#endif
#if defined(__cplusplus) || __STDC__
#define YYPARSE_PARAM_ARG YYPARSE_PARAM_TYPE YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#else	/* ! ANSI-C/C++ */
#define YYPARSE_PARAM_ARG YYPARSE_PARAM
#define YYPARSE_PARAM_DECL YYPARSE_PARAM_TYPE YYPARSE_PARAM;
#endif	/* ANSI-C/C++ */
#endif	/* ! YYPARSE_PARAM */

int
yyparse (YYPARSE_PARAM_ARG)
    YYPARSE_PARAM_DECL
{
    register int yym, yyn, yystate;
#if YYDEBUG
    register const char *yys;

    if ((yys = getenv("YYDEBUG")))
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = (-1);

    if (yyss == NULL && yygrowstack()) goto yyoverflow;
    yyssp = yyss;
    yyvsp = yyvs;
    *yyssp = yystate = 0;

yyloop:
    if ((yyn = yydefred[yystate])) goto yyreduce;
    if (yychar < 0)
    {
        if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, reading %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
    }
    if ((yyn = yysindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: state %d, shifting to state %d\n",
                    YYPREFIX, yystate, yytable[yyn]);
#endif
        if (yyssp >= yysslim && yygrowstack())
        {
            goto yyoverflow;
        }
        *++yyssp = yystate = yytable[yyn];
        *++yyvsp = yylval;
        yychar = (-1);
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if ((yyn = yyrindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag) goto yyinrecovery;
#if defined(lint) || defined(__GNUC__)
    goto yynewerror;
#endif
yynewerror:
    yyerror("syntax error");
#if defined(lint) || defined(__GNUC__)
    goto yyerrlab;
#endif
yyerrlab:
    ++yynerrs;
yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if ((yyn = yysindex[*yyssp]) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: state %d, error recovery shifting\
 to state %d\n", YYPREFIX, *yyssp, yytable[yyn]);
#endif
                if (yyssp >= yysslim && yygrowstack())
                {
                    goto yyoverflow;
                }
                *++yyssp = yystate = yytable[yyn];
                *++yyvsp = yylval;
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: error recovery discarding state %d\n",
                            YYPREFIX, *yyssp);
#endif
                if (yyssp <= yyss) goto yyabort;
                --yyssp;
                --yyvsp;
            }
        }
    }
    else
    {
        if (yychar == 0) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, error recovery discards token %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
        yychar = (-1);
        goto yyloop;
    }
yyreduce:
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: state %d, reducing by rule %d (%s)\n",
                YYPREFIX, yystate, yyn, yyrule[yyn]);
#endif
    yym = yylen[yyn];
    yyval = yyvsp[1-yym];
    switch (yyn)
    {
case 3:
#line 103 "Lily_compile.yacc"
{
			/*检查流程块重复定义*/
			FlowName fn;
			strcpy(fn.fn_name, yyvsp[-2].id_val);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "流程%s重复定义!", yyvsp[-2].id_val);
				yyerror(tmp);
			}
			flowname_stack.push(fn);
			/*局部变量个数*/
			varnum_stack.push(0);
		 	strcpy(current_flow_name, yyvsp[-2].id_val); 
			fprintf(yyout, "LABEL F_%s_BEGIN\n", yyvsp[-2].id_val);		   
			fprintf(yyout, "DEPTH\n");		   
			
			
		}
break;
case 4:
#line 123 "Lily_compile.yacc"
{
			fprintf(yyout, "LABEL F_%s_END\n", yyvsp[-5].id_val);		   
			fprintf(yyout, "_DEPTH\n");		   
			if (strcmp(yyvsp[-5].id_val, "main") == 0)
			{
				hasMain = 1;
				fprintf(yyout, "HALT\n");		   
			}
			else
			{
				fprintf(yyout, "RECALL\n");		   
			}
			
			/*符号表退栈*/
			RemoveTokenFromTab();
		}
break;
case 5:
#line 141 "Lily_compile.yacc"
{
			/*检查流程块重复定义*/
			FlowName fn;
			strcpy(fn.fn_name, yyvsp[-3].id_val);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "流程%s重复定义!", yyvsp[-3].id_val);
				yyerror(tmp);
			}
			flowname_stack.push(fn);

			fprintf(yyout, "LABEL F_%s_BEGIN\n", yyvsp[-3].id_val);		   
			fprintf(yyout, "LABEL F_%s_END\n", yyvsp[-3].id_val);		   

			if (strcmp(yyvsp[-3].id_val, "main") == 0)
			{
				hasMain = 1;
				fprintf(yyout, "HALT\n");		   
			}
			else
			{
				fprintf(yyout, "RECALL\n");		   
			}
		}
break;
case 8:
#line 173 "Lily_compile.yacc"
{
			Token tk;	/*保存变量的属性*/
			CheckIDValid(yyvsp[-3].id_val, tk);

			/*整数和浮点数可以相互赋值,字符串可以相互赋值*/
			if (tk.tk_type != yyvsp[-1].expr_type &&
			  (tk.tk_type == TYPE_STRING || yyvsp[-1].expr_type == TYPE_STRING))
			{
				yyerror("赋值操作符两边类型不符!");
			}
			fprintf(yyout, "SAV %s\n", yyvsp[-3].id_val);
			}
break;
case 9:
#line 185 "Lily_compile.yacc"
{
				if (labels_stack.isEmpty() != TRUE)
				{
					yyerror("不可在循环结构内声明变量!");
				}
			}
break;
case 10:
#line 191 "Lily_compile.yacc"
{fprintf(yyout, "CLR\n");}
break;
case 11:
#line 193 "Lily_compile.yacc"
{
		 	fprintf(yyout, "GOTO L_%d\n", yyvsp[-3].fake_val.label_goto);
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-3].fake_val.label_false);
		 }
break;
case 12:
#line 198 "Lily_compile.yacc"
{
			if (yyvsp[-7].expr_type != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-6].fake_val.label_goto);
		 }
break;
case 13:
#line 206 "Lily_compile.yacc"
{
			if (yyvsp[-4].expr_type != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-3].fake_val.label_false);
		  }
break;
case 14:
#line 215 "Lily_compile.yacc"
{
		 	labels_stack.push(yyvsp[-1].fake_val);
			fprintf(yyout, "GOTOFALSE L_%d\n", yyvsp[-1].fake_val.label_false);
		 }
break;
case 15:
#line 219 "Lily_compile.yacc"
{
			if (yyvsp[-4].expr_type != TYPE_INTEGER)
			{
			  yyerror("本行endwhile对应的while的条件表达式类型不为整型!");
			}
			fprintf(yyout, "GOTO L_%d\n", yyvsp[-5].fake_val.label_goto);
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-5].fake_val.label_false);
			Labels lbls;
			labels_stack.pop(lbls);
		 }
break;
case 16:
#line 235 "Lily_compile.yacc"
{
			fprintf(yyout, "SAV %s\n", yyvsp[-3].id_val);
		  }
break;
case 17:
#line 241 "Lily_compile.yacc"
{
		  	if (yyvsp[-1].expr_type != TYPE_INTEGER)
			{
				yyerror("for语句的条件表达式类型应该为整型!\n");
			}
			fprintf(yyout, "GOTOFALSE L_%d\n", yyvsp[-2].fake_val.label_false);
			/*把for(A;B;C)中的C部分翻译到临时文件中*/
			for_fp = yyout;
			fflush(yyout);
			if ( (yyout = fopen(TMP_FILE, "wb") ) == NULL)
			{
				yyerror("写打开临时文件./lily.tmp失败!");
			}
		  }
break;
case 18:
#line 259 "Lily_compile.yacc"
{
		  	/*关闭临时文件，恢复原来的yyout本来面目*/
			fprintf(yyout, "SAV %s\n", yyvsp[-3].id_val);
			fclose(yyout);
			yyout = for_fp;
			/*将标签压栈，便于break/continue使用*/
			labels_stack.push(yyvsp[-7].fake_val);
		  }
break;
case 19:
#line 270 "Lily_compile.yacc"
{
		  	fprintf(yyout, "LABEL L_%d\n", yyvsp[-11].fake_val.label_goto);
			/*把for(A;B;C)中的C部分从临时文件中读出插入到这里*/
			FILE *fp = NULL;
			char tmp_buf[512];
			int len;
			if ( (fp = fopen(TMP_FILE, "rb")) == NULL)
			{
				yyerror("读打开临时文件./lily.tmp失败!");
			}
			while ( (len = fread(tmp_buf, 1, sizeof(tmp_buf), fp)) > 0)
			{
				fwrite(tmp_buf, 1, len, yyout);
			}
			fclose(fp);
			remove(TMP_FILE);

			fprintf(yyout, "GOTO L_%d\n", yyvsp[-11].fake_val.label_true);
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-11].fake_val.label_false);
			Labels lbls;
			labels_stack.pop(lbls);
		  }
break;
case 20:
#line 297 "Lily_compile.yacc"
{
				fprintf(yyout, "GOTOFALSE L_%d\n", yyvsp[-6].fake_val.label_goto);
				fprintf(yyout, "LABEL L_%d\n", yyvsp[-6].fake_val.label_false);
				Labels lbls;
				labels_stack.pop(lbls);
			}
break;
case 21:
#line 304 "Lily_compile.yacc"
{
				if (yyvsp[-2].expr_type != TYPE_STRING)
				{
					yyerror("run函数的参数类型必须为string!");
				}
				unsigned int label = GetNewLabel();
				fprintf(yyout, "SAVCALL L_%d\n", label);
				fprintf(yyout, "JMP\n");
				fprintf(yyout, "LABEL L_%d\n", label);
			}
break;
case 22:
#line 314 "Lily_compile.yacc"
{fprintf(yyout, "GOTO F_%s_END\n", current_flow_name);}
break;
case 23:
#line 315 "Lily_compile.yacc"
{
				if (labels_stack.isEmpty())
				{
					yyerror("continue应该位于循环语句内!");
				}
				Labels lbls;
				labels_stack.peek(lbls);
				fprintf(yyout, "GOTO L_%d\n", lbls.label_goto);
			}
break;
case 24:
#line 324 "Lily_compile.yacc"
{
				if (labels_stack.isEmpty())
				{
					yyerror("break应该位于循环语句内!");
				}
				Labels lbls;
				labels_stack.peek(lbls);
				fprintf(yyout, "GOTO L_%d\n", lbls.label_false);
			}
break;
case 25:
#line 333 "Lily_compile.yacc"
{
				Labels lbls;
				labels_stack.pop(lbls);
				fprintf(yyout, "LABEL L_%d\n", lbls.label_false);
				}
break;
case 28:
#line 342 "Lily_compile.yacc"
{strcpy(declare_type, "INTEGER");}
break;
case 29:
#line 343 "Lily_compile.yacc"
{strcpy(declare_type, "STRING");}
break;
case 30:
#line 344 "Lily_compile.yacc"
{strcpy(declare_type, "FLOAT");}
break;
case 31:
#line 346 "Lily_compile.yacc"
{
			CheckIDAlreadyExist(yyvsp[0].id_val);
			fprintf(yyout, "VAR %s %s\n", yyvsp[0].id_val, declare_type);
			/*插入符号表*/
			InsertToken(yyvsp[0].id_val, declare_type);
			}
break;
case 32:
#line 352 "Lily_compile.yacc"
{
			CheckIDAlreadyExist(yyvsp[0].id_val);
			fprintf(yyout, "VAR %s %s\n", yyvsp[0].id_val, declare_type);
			/*插入符号表*/
			InsertToken(yyvsp[0].id_val, declare_type);
			}
break;
case 36:
#line 368 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING && yyvsp[0].expr_type == TYPE_STRING ||
			   yyvsp[-2].expr_type != TYPE_STRING && yyvsp[0].expr_type != TYPE_STRING)
			{
			}
			else
			{
				yyerror("加法操作对象类型不正确!");
			}
			fprintf(yyout, "ADD\n");
			if (yyvsp[-2].expr_type == TYPE_STRING)
			{
				yyval.expr_type = TYPE_STRING;
			}
			else if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
				yyval.expr_type = yyvsp[-2].expr_type;
			}
			else
			{
				yyval.expr_type = TYPE_FLOAT;
			}
		}
break;
case 37:
#line 391 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING || yyvsp[0].expr_type == TYPE_STRING)
			{
				yyerror("减法操作对象类型不正确!");
			}
			fprintf(yyout, "SUB\n");
			if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
				yyval.expr_type = yyvsp[-2].expr_type;
			}
			else
			{
				yyval.expr_type = TYPE_FLOAT;
			}
		}
break;
case 38:
#line 406 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING || yyvsp[0].expr_type == TYPE_STRING)
			{
				yyerror("乘法操作对象类型不正确!");
			}
			fprintf(yyout, "MUL\n");
			if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
				yyval.expr_type = yyvsp[-2].expr_type;
			}
			else
			{
				yyval.expr_type = TYPE_FLOAT;
			}
		}
break;
case 39:
#line 421 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING || yyvsp[0].expr_type == TYPE_STRING)
			{
				yyerror("除法操作对象类型不正确!");
			}
			fprintf(yyout, "DIV\n");
			if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
				yyval.expr_type = yyvsp[-2].expr_type;
			}
			else
			{
				yyval.expr_type = TYPE_FLOAT;
			}
		}
break;
case 40:
#line 436 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER || yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("逻辑与的操作对象类型不正确!");
			}
			fprintf(yyout, "AND\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 41:
#line 444 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER || yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("逻辑或的操作对象类型不正确!");
			}
			fprintf(yyout, "OR\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 42:
#line 452 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING || yyvsp[0].expr_type == TYPE_STRING)
			{
				yyerror("符号 < 两边操作数类型不正确!");
			}
			fprintf(yyout, "LT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 43:
#line 460 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING || yyvsp[0].expr_type == TYPE_STRING)
			{
				yyerror("符号 =< 两边操作数类型不正确!");
			}
			fprintf(yyout, "LE\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 44:
#line 468 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING && yyvsp[0].expr_type == TYPE_STRING ||
			    yyvsp[-2].expr_type != TYPE_STRING && yyvsp[0].expr_type != TYPE_STRING)
			{
			}
			else
			{
				yyerror("符号 != 两边操作数类型不正确!");
			}
			fprintf(yyout, "NE\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 45:
#line 480 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type == TYPE_STRING && yyvsp[0].expr_type == TYPE_STRING ||
			    yyvsp[-2].expr_type != TYPE_STRING && yyvsp[0].expr_type != TYPE_STRING)
			{
			}
			else
			{
				yyerror("符号 == 两边操作数类型不正确!");
			}
			fprintf(yyout, "EQ\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 46:
#line 492 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER && yyvsp[-2].expr_type != TYPE_FLOAT
			  || yyvsp[0].expr_type != TYPE_INTEGER && yyvsp[0].expr_type != TYPE_FLOAT)
			{
				yyerror("符号 > 两边操作数类型不正确!");
			}
			fprintf(yyout, "GT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 47:
#line 501 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER && yyvsp[-2].expr_type != TYPE_FLOAT
			  || yyvsp[0].expr_type != TYPE_INTEGER && yyvsp[0].expr_type != TYPE_FLOAT)
			{
				yyerror("符号 >= 两边操作数类型不正确!");
			}
			fprintf(yyout, "GE\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 48:
#line 510 "Lily_compile.yacc"
{yyval.expr_type = yyvsp[-1].expr_type;}
break;
case 49:
#line 511 "Lily_compile.yacc"
{
			if (yyvsp[0].expr_type != TYPE_INTEGER && yyvsp[0].expr_type != TYPE_FLOAT)
			{
				yyerror("正负号操作只能作用于整数或者浮点数类型!");
			}
			if (yyvsp[-1].sign_val == '-')
			{
				fprintf(yyout, "UMINUS\n");
			}
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 50:
#line 522 "Lily_compile.yacc"
{
			if (yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("非操作只能作用于整数类型!");
			}
			fprintf(yyout, "NOT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 51:
#line 530 "Lily_compile.yacc"
{yyval.expr_type = yyvsp[0].expr_type;}
break;
case 52:
#line 532 "Lily_compile.yacc"
{yyval.sign_val = '+';}
break;
case 53:
#line 533 "Lily_compile.yacc"
{yyval.sign_val = '-';}
break;
case 54:
#line 535 "Lily_compile.yacc"
{
		Token tk;
		CheckIDValid(yyvsp[0].id_val, tk);
		yyval.expr_type = tk.tk_type;
		fprintf(yyout, "PUSH %s\n", yyvsp[0].id_val);
		}
break;
case 55:
#line 541 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_STRING;	
			/*将字符串值和指令分别写到不同的文件中*/
			int cs_index = GetNewCSIndex();
			fprintf(csout, "%d %s\n", cs_index, yyvsp[0].const_string_val);
			fprintf(yyout, "PUSH %%%d\n", cs_index);
			}
break;
case 56:
#line 548 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_INTEGER;	
			fprintf(yyout, "PUSH #%d\n", yyvsp[0].const_int_val);
			}
break;
case 57:
#line 552 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_FLOAT;	
			fprintf(yyout, "PUSH #%s\n", yyvsp[0].const_float_val);
			}
break;
case 59:
#line 559 "Lily_compile.yacc"
{
				int argnum, number, type;/*函数的参数个数和编号,类型*/
				char tmp[100];
				if (fcGetFunc(yyvsp[-3].id_val, argnum, number, type) < 0)
				{
					sprintf(tmp, "函数%s没有定义!", yyvsp[-3].id_val);
					yyerror(tmp);
				}
				yyval.expr_type = type;
				/*检查参数个数*/
				if (argnum < 0)
				{
					if (yyvsp[-1].arg_count > abs(argnum))
					{
						sprintf(tmp, "函数%s参数个数不对!", yyvsp[-3].id_val);
						yyerror(tmp);
					}
				}
				else
				{
					if (yyvsp[-1].arg_count != argnum)
					{
						sprintf(tmp, "函数%s参数个数不对!", yyvsp[-3].id_val);
						yyerror(tmp);
					}
				}
				/*参数个数压栈*/
				fprintf(yyout, "PUSH #%d\n", yyvsp[-1].arg_count);
				fprintf(yyout, "CALL %d\n", number);
			}
break;
case 60:
#line 590 "Lily_compile.yacc"
{yyval.arg_count = yyvsp[-2].arg_count + 1;}
break;
case 61:
#line 591 "Lily_compile.yacc"
{yyval.arg_count = 1;}
break;
case 62:
#line 592 "Lily_compile.yacc"
{yyval.arg_count = 0;}
break;
case 63:
#line 595 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "GOTOFALSE L_%d\n", yyval.fake_val.label_false);
		}
break;
case 64:
#line 602 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_goto);
		}
break;
case 65:
#line 608 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_true);
		}
break;
case 66:
#line 615 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_goto);
		labels_stack.push(yyval.fake_val);
		}
break;
case 67:
#line 623 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		labels_stack.push(yyval.fake_val);
		}
break;
#line 1613 "y.tab.c"
    }
    yyssp -= yym;
    yystate = *yyssp;
    yyvsp -= yym;
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: after reduction, shifting from state 0 to\
 state %d\n", YYPREFIX, YYFINAL);
#endif
        yystate = YYFINAL;
        *++yyssp = YYFINAL;
        *++yyvsp = yyval;
        if (yychar < 0)
        {
            if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
            if (yydebug)
            {
                yys = 0;
                if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
                if (!yys) yys = "illegal-symbol";
                printf("%sdebug: state %d, reading %d (%s)\n",
                        YYPREFIX, YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == 0) goto yyaccept;
        goto yyloop;
    }
    if ((yyn = yygindex[yym]) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: after reduction, shifting from state %d \
to state %d\n", YYPREFIX, *yyssp, yystate);
#endif
    if (yyssp >= yysslim && yygrowstack())
    {
        goto yyoverflow;
    }
    *++yyssp = yystate;
    *++yyvsp = yyval;
    goto yyloop;
yyoverflow:
    yyerror("yacc stack overflow");
yyabort:
    return (1);
yyaccept:
    return (0);
}
