#ifndef lint
static char yysccsid[] = "@(#)yaccpar	1.9 (Berkeley) 02/21/93";
#endif
#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define yyclearin (yychar=(-1))
#define yyerrok (yyerrflag=0)
#define YYRECOVERING (yyerrflag!=0)
#define YYPREFIX "yy"
#line 2 "Lily_compile.yacc"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32_)
#include <unistd.h>
#endif
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
#line 79 "y.tab.c"
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
#define YYERRCODE 256
short yylhs[] = {                                        -1,
    0,    0,   14,   12,   12,   13,   13,   15,   15,   15,
   17,   15,   15,   18,   15,   19,   20,   21,   15,   15,
   15,   15,   15,   15,   15,   15,   16,   23,   23,   23,
   23,   24,   24,   22,   22,   25,    1,    1,    1,    1,
    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    1,    1,    1,    3,    3,    2,    2,    2,    2,    2,
   11,   10,   10,   10,    4,    5,    6,    7,    8,    9,
};
short yylen[] = {                                         2,
    2,    1,    0,    7,    5,    2,    1,    4,    1,    2,
    0,    9,    6,    0,    7,    0,    0,    0,   19,    8,
    5,    2,    2,    2,    5,    1,    3,    1,    1,    1,
    1,    3,    1,    2,    1,    5,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    2,    2,    1,    1,    1,    1,    1,    1,    1,    1,
    4,    3,    1,    0,    0,    0,    0,    0,    0,    0,
};
short yydefred[] = {                                      0,
    0,    0,    2,    0,    1,    0,    0,    5,    0,    0,
   57,   58,   59,    0,   66,   28,   29,   30,    0,    0,
    0,    0,    0,   68,   69,   54,   55,    0,   26,    0,
   31,    0,   53,    0,   60,    0,    7,    9,    0,    0,
    0,    0,    0,    0,   22,    0,    0,   23,   24,    0,
    0,   52,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   10,   51,    4,    6,
   33,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   35,   50,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   39,   40,   41,   27,    0,    8,
   61,    0,    0,    0,    0,    0,    0,    0,    0,   34,
   32,    0,    0,    0,   21,    0,    0,   70,   25,   11,
   13,    0,   16,    0,    0,    0,   15,   67,    0,    0,
    0,    0,   20,   12,    0,   17,    0,    0,    0,    0,
   18,    0,    0,    0,   19,
};
short yydgoto[] = {                                       2,
   32,   33,   34,   76,   44,  132,   50,   51,  125,   75,
   35,    3,   36,    9,   37,   38,  126,  104,  128,  137,
  142,   82,   39,   72,   83,
};
short yysindex[] = {                                    -77,
 -241,  -77,    0,  -73,    0, -236, -235,    0,  125,  -39,
    0,    0,    0,   40,    0,    0,    0,    0,  -18,    2,
    5,  -12,   -9,    0,    0,    0,    0,   40,    0,   40,
    0,  414,    0,   40,    0,  -13,    0,    0, -217,   40,
   40,    9,  504,   40,    0,   40, -206,    0,    0,  125,
 -232,    0,  423,   40,   40,   40,   40,   40,   40,   40,
   40,   40,   40,   40,   40,   40,    0,    0,    0,    0,
    0,  -38,  432,  504,  -33, -210,  504,  441,   -5,   14,
   40, -274,    0,    0,  513,  520,  -30,  -30,  -30,  -30,
  -30,  -30,  -19,  -19,    0,    0,    0,    0, -199,    0,
    0,   40,  125, -207,    3,   40,   21,  450,    4,    0,
    0,  504,  -40,  125,    0,  459,   40,    0,    0,    0,
    0,   44,    0,  468,  125,  125,    0,    0,    6,  125,
   71,   40,    0,    0,  477,    0, -193,    7,   40,  486,
    0, -200,  125,   98,    0,
};
short yyrindex[] = {                                      0,
    0,    0,    0,    0,    0,    0,  152,    0,    0,  495,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
  -10,  396, -194,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   -8,    0,    0, -197,    0,    0,    0,
    0,    0,    0,    0,  -15,  848,  358,  404,  593,  774,
  810,  818,  550,  558,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   -6,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0, -259,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
};
short yygindex[] = {                                      0,
  874,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   69,  -48,    0,  -32,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   -7,
};
#define YYTABLESIZE 1134
short yytable[] = {                                      30,
   41,   80,   26,   70,   27,   99,   66,  101,  109,   81,
  102,   64,   62,    1,   63,    4,   65,   66,   29,    6,
   98,   40,   64,   36,   36,   43,   30,   65,   43,   26,
   64,   27,   63,   64,   62,   63,    7,   62,    8,   71,
   45,   46,   43,   43,   47,   29,   48,   70,   41,   49,
   79,   81,  103,   30,  113,  106,   26,  111,   27,  114,
  117,  115,  119,  138,  133,  122,  143,  139,   65,   14,
    5,    0,   29,    0,  110,    0,  130,  131,    0,   30,
   70,    0,   26,   30,   27,    0,   26,    0,   27,   70,
    0,    0,    0,    0,  144,    0,    0,   70,   70,    0,
    0,    0,   29,    0,    0,    0,    0,    0,    0,    0,
   30,   70,    0,   26,    0,   27,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,   29,
    0,    0,    0,    0,    0,    0,    0,   30,    0,    0,
   26,    0,   27,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   29,    0,    0,    0,
    0,    0,    0,    0,   30,    0,    0,   26,    0,   27,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,   29,    0,    0,    0,    0,    0,    0,
    0,    3,    0,    0,    3,    0,    3,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    3,    0,    0,    0,    0,    0,   10,   11,   12,   13,
    0,   14,    0,  120,  121,   15,    0,    0,   16,   17,
   18,   19,    0,    0,   20,   21,    0,   22,   23,   24,
    0,   25,    0,   10,   11,   12,   13,   43,   14,    0,
    0,   43,   15,   28,   31,   16,   17,   18,   19,    0,
   69,   20,   21,    0,   22,   23,   24,    0,   25,   43,
   10,   11,   12,   13,    0,   14,    0,    0,    0,   15,
   28,   31,   16,   17,   18,   19,    0,    0,   20,   21,
    0,   22,   23,   24,  107,   25,   42,   11,   12,   13,
   10,   11,   12,   13,    0,   14,    0,   28,   31,   15,
    0,  127,   16,   17,   18,   19,    0,    0,   20,   21,
    0,   22,   23,   24,    0,   25,    0,   10,   11,   12,
   13,    0,   14,   28,    0,  134,   15,   28,   31,   16,
   17,   18,   19,    0,    0,   20,   21,    0,   22,   23,
   24,    0,   25,    0,   10,   11,   12,   13,    0,   14,
    0,    0,    0,   15,   28,   31,   16,   17,   18,   19,
    0,    0,   20,   21,  145,   22,   23,   24,    0,   25,
    0,   10,   11,   12,   13,    0,   14,    0,    0,    0,
   15,   28,   31,   16,   17,   18,   19,    0,   44,   20,
   21,   44,   22,   23,   24,    0,   25,    0,    3,    3,
    3,    3,    0,    3,    0,   44,   44,    3,   28,   31,
    3,    3,    3,    3,    0,    0,    3,    3,    0,    3,
    3,    3,   56,    3,    0,    0,   56,   56,   56,   56,
   56,    0,   56,    0,   45,    3,    3,   45,    0,    0,
   66,    0,    0,   56,   56,   64,   62,    0,   63,   66,
   65,   45,   45,   84,   64,   62,    0,   63,   66,   65,
    0,    0,   67,   64,   62,    0,   63,   66,   65,    0,
    0,  105,   64,   62,    0,   63,   66,   65,    0,    0,
  100,   64,   62,    0,   63,   66,   65,    0,    0,    0,
   64,   62,    0,   63,   66,   65,    0,  118,  129,   64,
   62,    0,   63,   66,   65,    0,    0,  123,   64,   62,
    0,   63,   66,   65,    0,    0,  141,   64,   62,    0,
   63,   56,   65,    0,    0,  136,   56,   56,    0,   56,
   66,   56,    0,    0,    0,   64,   62,    0,   63,   66,
   65,    0,    0,   56,   64,   62,   66,   63,    0,   65,
    0,   64,   62,    0,   63,    0,   65,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   37,    0,   37,   37,   37,    0,    0,    0,   38,    0,
   38,   38,   38,    0,    0,    0,    0,   37,   37,    0,
    0,    0,    0,    0,    0,   38,   38,    0,    0,    0,
   44,    0,    0,    0,   44,    0,    0,    0,    0,    0,
    0,    0,    0,   47,    0,    0,   47,    0,    0,    0,
    0,    0,   44,   44,   44,   44,   44,   44,   44,   44,
   47,   47,    0,    0,    0,    0,    0,    0,   56,    0,
    0,    0,   56,    0,    0,    0,   45,    0,    0,    0,
   45,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   56,   56,   56,   56,   56,   56,   56,   56,   45,   45,
   45,   45,   45,   45,   45,   45,    0,    0,   54,   55,
   56,   57,   58,   59,   60,   61,    0,   54,   55,   56,
   57,   58,   59,   60,   61,    0,   54,   55,   56,   57,
   58,   59,   60,   61,    0,   54,   55,   56,   57,   58,
   59,   60,   61,    0,   54,   55,   56,   57,   58,   59,
   60,   61,    0,   54,   55,   56,   57,   58,   59,   60,
   61,    0,   54,   55,   56,   57,   58,   59,   60,   61,
    0,   54,   55,   56,   57,   58,   59,   60,   61,    0,
   54,   55,   56,   57,   58,   59,   60,   61,    0,   56,
   56,   56,   56,   56,   56,   56,   56,    0,   54,   55,
   56,   57,   58,   59,   60,   61,    0,    0,   55,   56,
   57,   58,   59,   60,   61,    0,   56,   57,   58,   59,
   60,   61,   37,    0,   46,    0,   37,   46,    0,    0,
   38,    0,    0,    0,   38,    0,    0,    0,    0,    0,
    0,   46,   46,    0,   37,   37,   37,   37,   37,   37,
   37,   37,   38,   38,   38,   38,   38,   38,   38,   38,
   48,    0,    0,   48,    0,   47,    0,    0,   49,   47,
    0,   49,    0,    0,    0,    0,    0,   48,   48,    0,
    0,    0,    0,    0,    0,   49,   49,   47,   47,   47,
   47,   47,   47,   47,   47,    0,    0,   43,   42,    0,
    0,   42,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   52,    0,   53,    0,   42,   42,   68,    0,    0,
    0,    0,    0,   73,   74,    0,    0,   77,    0,   78,
    0,    0,    0,    0,    0,    0,    0,   85,   86,   87,
   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  108,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,  112,    0,    0,    0,  116,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
  124,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,  135,    0,    0,    0,    0,
    0,    0,  140,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   46,    0,    0,    0,
   46,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,   46,   46,
   46,   46,   46,   46,   46,   46,    0,    0,    0,    0,
    0,    0,   48,    0,    0,    0,   48,    0,    0,    0,
   49,    0,    0,    0,   49,    0,    0,    0,    0,    0,
    0,    0,    0,    0,   48,   48,   48,   48,   48,   48,
   48,   48,   49,   49,   49,   49,   49,   49,   49,   49,
   42,    0,    0,    0,   42,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   42,   42,
};
short yycheck[] = {                                      40,
   40,   50,   43,   36,   45,   44,   37,   41,  283,  284,
   44,   42,   43,   91,   45,  257,   47,   37,   59,   93,
   59,   61,   42,  283,  284,   41,   40,   47,   44,   43,
   41,   45,   41,   44,   41,   44,  273,   44,  274,  257,
   59,   40,   58,   59,   40,   59,   59,   80,   40,   59,
  257,  284,  263,   40,  103,   61,   43,  257,   45,  267,
   40,   59,   59,  257,   59,  114,  267,   61,  263,  267,
    2,   -1,   59,   -1,   82,   -1,  125,  126,   -1,   40,
  113,   -1,   43,   40,   45,   -1,   43,   -1,   45,  122,
   -1,   -1,   -1,   -1,  143,   -1,   -1,  130,  131,   -1,
   -1,   -1,   59,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   40,  144,   -1,   43,   -1,   45,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   59,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   40,   -1,   -1,
   43,   -1,   45,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   59,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   40,   -1,   -1,   43,   -1,   45,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   59,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   40,   -1,   -1,   43,   -1,   45,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   59,   -1,   -1,   -1,   -1,   -1,  257,  258,  259,  260,
   -1,  262,   -1,  264,  265,  266,   -1,   -1,  269,  270,
  271,  272,   -1,   -1,  275,  276,   -1,  278,  279,  280,
   -1,  282,   -1,  257,  258,  259,  260,  263,  262,   -1,
   -1,  267,  266,  294,  295,  269,  270,  271,  272,   -1,
  274,  275,  276,   -1,  278,  279,  280,   -1,  282,  285,
  257,  258,  259,  260,   -1,  262,   -1,   -1,   -1,  266,
  294,  295,  269,  270,  271,  272,   -1,   -1,  275,  276,
   -1,  278,  279,  280,  281,  282,  257,  258,  259,  260,
  257,  258,  259,  260,   -1,  262,   -1,  294,  295,  266,
   -1,  268,  269,  270,  271,  272,   -1,   -1,  275,  276,
   -1,  278,  279,  280,   -1,  282,   -1,  257,  258,  259,
  260,   -1,  262,  294,   -1,  265,  266,  294,  295,  269,
  270,  271,  272,   -1,   -1,  275,  276,   -1,  278,  279,
  280,   -1,  282,   -1,  257,  258,  259,  260,   -1,  262,
   -1,   -1,   -1,  266,  294,  295,  269,  270,  271,  272,
   -1,   -1,  275,  276,  277,  278,  279,  280,   -1,  282,
   -1,  257,  258,  259,  260,   -1,  262,   -1,   -1,   -1,
  266,  294,  295,  269,  270,  271,  272,   -1,   41,  275,
  276,   44,  278,  279,  280,   -1,  282,   -1,  257,  258,
  259,  260,   -1,  262,   -1,   58,   59,  266,  294,  295,
  269,  270,  271,  272,   -1,   -1,  275,  276,   -1,  278,
  279,  280,   37,  282,   -1,   -1,   41,   42,   43,   44,
   45,   -1,   47,   -1,   41,  294,  295,   44,   -1,   -1,
   37,   -1,   -1,   58,   59,   42,   43,   -1,   45,   37,
   47,   58,   59,   41,   42,   43,   -1,   45,   37,   47,
   -1,   -1,   59,   42,   43,   -1,   45,   37,   47,   -1,
   -1,   41,   42,   43,   -1,   45,   37,   47,   -1,   -1,
   59,   42,   43,   -1,   45,   37,   47,   -1,   -1,   -1,
   42,   43,   -1,   45,   37,   47,   -1,   58,   41,   42,
   43,   -1,   45,   37,   47,   -1,   -1,   59,   42,   43,
   -1,   45,   37,   47,   -1,   -1,   41,   42,   43,   -1,
   45,   37,   47,   -1,   -1,   59,   42,   43,   -1,   45,
   37,   47,   -1,   -1,   -1,   42,   43,   -1,   45,   37,
   47,   -1,   -1,   59,   42,   43,   37,   45,   -1,   47,
   -1,   42,   43,   -1,   45,   -1,   47,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   41,   -1,   43,   44,   45,   -1,   -1,   -1,   41,   -1,
   43,   44,   45,   -1,   -1,   -1,   -1,   58,   59,   -1,
   -1,   -1,   -1,   -1,   -1,   58,   59,   -1,   -1,   -1,
  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   41,   -1,   -1,   44,   -1,   -1,   -1,
   -1,   -1,  285,  286,  287,  288,  289,  290,  291,  292,
   58,   59,   -1,   -1,   -1,   -1,   -1,   -1,  263,   -1,
   -1,   -1,  267,   -1,   -1,   -1,  263,   -1,   -1,   -1,
  267,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  285,  286,  287,  288,  289,  290,  291,  292,  285,  286,
  287,  288,  289,  290,  291,  292,   -1,   -1,  285,  286,
  287,  288,  289,  290,  291,  292,   -1,  285,  286,  287,
  288,  289,  290,  291,  292,   -1,  285,  286,  287,  288,
  289,  290,  291,  292,   -1,  285,  286,  287,  288,  289,
  290,  291,  292,   -1,  285,  286,  287,  288,  289,  290,
  291,  292,   -1,  285,  286,  287,  288,  289,  290,  291,
  292,   -1,  285,  286,  287,  288,  289,  290,  291,  292,
   -1,  285,  286,  287,  288,  289,  290,  291,  292,   -1,
  285,  286,  287,  288,  289,  290,  291,  292,   -1,  285,
  286,  287,  288,  289,  290,  291,  292,   -1,  285,  286,
  287,  288,  289,  290,  291,  292,   -1,   -1,  286,  287,
  288,  289,  290,  291,  292,   -1,  287,  288,  289,  290,
  291,  292,  263,   -1,   41,   -1,  267,   44,   -1,   -1,
  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,   -1,   -1,
   -1,   58,   59,   -1,  285,  286,  287,  288,  289,  290,
  291,  292,  285,  286,  287,  288,  289,  290,  291,  292,
   41,   -1,   -1,   44,   -1,  263,   -1,   -1,   41,  267,
   -1,   44,   -1,   -1,   -1,   -1,   -1,   58,   59,   -1,
   -1,   -1,   -1,   -1,   -1,   58,   59,  285,  286,  287,
  288,  289,  290,  291,  292,   -1,   -1,   14,   41,   -1,
   -1,   44,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   28,   -1,   30,   -1,   58,   59,   34,   -1,   -1,
   -1,   -1,   -1,   40,   41,   -1,   -1,   44,   -1,   46,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   54,   55,   56,
   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   81,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  102,   -1,   -1,   -1,  106,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
  117,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,  132,   -1,   -1,   -1,   -1,
   -1,   -1,  139,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,  263,   -1,   -1,   -1,
  267,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  285,  286,
  287,  288,  289,  290,  291,  292,   -1,   -1,   -1,   -1,
   -1,   -1,  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,
  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,  285,  286,  287,  288,  289,  290,
  291,  292,  285,  286,  287,  288,  289,  290,  291,  292,
  263,   -1,   -1,   -1,  267,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  285,  286,
};
#define YYFINAL 2
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 295
#if YYDEBUG
char *yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,"'%'",0,0,"'('","')'","'*'","'+'","','","'-'",0,"'/'",0,0,0,0,0,0,0,0,0,0,
"':'","';'",0,"'='",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"'['",0,"']'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,"ID","CONST_STRING","CONST_INTEGER","CONST_FLOAT",
"FUNCTION","IF","THEN","ELSE","ENDIF","WHILE","DO","ENDWHILE","INTEGER",
"STRING","FLOAT","RETURN","BEGIN_FLOW","END_FLOW","RUN","FOR","ENDFOR",
"CONTINUE","BREAK","REPEAT","UNTIL","SWITCH","ENDSWITCH","CASE","OR","AND","LT",
"LE","EQ","NE","GT","GE","UMINUS","NOT","MEMBLOCK",
};
char *yyrule[] = {
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
"statement : SWITCH fake5 case_statement_list ENDSWITCH ';'",
"statement : ';'",
"declarations : var_type id_list ';'",
"var_type : INTEGER",
"var_type : STRING",
"var_type : FLOAT",
"var_type : MEMBLOCK",
"id_list : id_list ',' ID",
"id_list : ID",
"case_statement_list : case_statement_list case_statement",
"case_statement_list : case_statement",
"case_statement : CASE expr ':' fake6 statement_list",
"expr : expr '+' expr",
"expr : expr '-' expr",
"expr : expr '*' expr",
"expr : expr '/' expr",
"expr : expr '%' expr",
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
"fake6 :",
};
#endif
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 500
#define YYMAXDEPTH 500
#endif
#endif
int yydebug;
int yynerrs;
int yyerrflag;
int yychar;
short *yyssp;
YYSTYPE *yyvsp;
YYSTYPE yyval;
YYSTYPE yylval;
short yyss[YYSTACKSIZE];
YYSTYPE yyvs[YYSTACKSIZE];
#define yystacksize YYSTACKSIZE
#line 737 "Lily_compile.yacc"
extern FILE * yyin;
extern FILE * yyout;
int yyparse();

#undef SHOW_STEP
#if defined(_DEBUG)
	#define SHOW_STEP	printf("运行到[%s]文件的[%d]行\n", __FILE__, __LINE__);
#else
	#define SHOW_STEP 	;
#endif

int main(int argc, char** argv)
{
	/*读取参数*/
	int opt;
	char input_file[255];
	char output_file[255];
	
	SHOW_STEP
	
	memset(input_file, 0, sizeof(input_file));
	memset(output_file, 0, sizeof(output_file));
	/*
	*-o 输出文件名 只能有一个，如果有多个，以最后一个为准
	*-f 输入的文件名 只能有一个，如果有多个，以最后一个为准
	*/
	#if !defined(_WIN32_)
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
	#else
	if (argc < 3)
	{
		printf("usage:%s <input file> <output file>\n", argv[0]);
		return -1;	
	}
	strcpy(input_file, argv[1]);
	strcpy(output_file, argv[2]);
	#endif
	yyin = stdin;
	if (strlen(input_file) > 0)
	{
		if ((yyin = fopen(input_file, "rb")) == NULL)
		{
			fprintf(stderr, "打开文件%s失败!\n", input_file);
			return -1;
		}
	}
	SHOW_STEP
	/*打开指令输出文件*/
	if ( (yyout = fopen(INSTRUCT_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", INSTRUCT_FILE);
		return -1;
	}
	SHOW_STEP
	/*打开常量字符串输出文件*/
	if ( (csout = fopen(CSTRING_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "打开文件%s失败!\n", CSTRING_FILE);
		return -1;
	}
	SHOW_STEP
	/*读取函数的配置*/
	/*
	if (fcInit(getenv("FC")) < 0)
	*/
	if (fcInit(".\\fc.cfg") < 0)
	{
		fprintf(stderr, "读取函数配置信息.\\fc.cfg失败!\n");
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
		if ( (out = fopen(output_file, "wb+")) == NULL)
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
	//	ttt.tk_type = TYPE_STRING;
		ttt.tk_type = TYPE_MEMBLOCK;
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
	if (strcmp(id_type, "MEMBLOCK") == 0)
	{
		type = TYPE_MEMBLOCK;
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
#line 815 "y.tab.c"
#define YYABORT goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR goto yyerrlab
int
yyparse()
{
    register int yym, yyn, yystate;
#if YYDEBUG
    register char *yys;
    extern char *getenv();

    if (yys = getenv("YYDEBUG"))
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = (-1);

    yyssp = yyss;
    yyvsp = yyvs;
    *yyssp = yystate = 0;

yyloop:
    if (yyn = yydefred[yystate]) goto yyreduce;
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
        if (yyssp >= yyss + yystacksize - 1)
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
#ifdef lint
    goto yynewerror;
#endif
yynewerror:
    yyerror("syntax error");
#ifdef lint
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
                if (yyssp >= yyss + yystacksize - 1)
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
#line 104 "Lily_compile.yacc"
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
#line 124 "Lily_compile.yacc"
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
#line 142 "Lily_compile.yacc"
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
#line 174 "Lily_compile.yacc"
{
			Token tk;	/*保存变量的属性*/
			CheckIDValid(yyvsp[-3].id_val, tk);

			/*整数和浮点数可以相互赋值,字符串可以相互赋值,内存块可以相互赋值*/
			if (tk.tk_type == yyvsp[-1].expr_type)
			{
			}
			else if (tk.tk_type == TYPE_INTEGER && yyvsp[-1].expr_type == TYPE_FLOAT)
			{
			}
			else if (tk.tk_type == TYPE_FLOAT && yyvsp[-1].expr_type == TYPE_INTEGER)
			{
			}
			else
			{
				yyerror("赋值操作操作数类型不正确!");
			}
			fprintf(yyout, "SAV %s\n", yyvsp[-3].id_val);
			}
break;
case 9:
#line 194 "Lily_compile.yacc"
{
				if (labels_stack.isEmpty() != TRUE)
				{
					yyerror("不可在循环结构内声明变量!");
				}
			}
break;
case 10:
#line 200 "Lily_compile.yacc"
{fprintf(yyout, "CLR\n");}
break;
case 11:
#line 202 "Lily_compile.yacc"
{
		 	fprintf(yyout, "GOTO L_%d\n", yyvsp[-3].fake_val.label_goto);
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-3].fake_val.label_false);
		 }
break;
case 12:
#line 207 "Lily_compile.yacc"
{
			if (yyvsp[-7].expr_type != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-6].fake_val.label_goto);
		 }
break;
case 13:
#line 215 "Lily_compile.yacc"
{
			if (yyvsp[-4].expr_type != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", yyvsp[-3].fake_val.label_false);
		  }
break;
case 14:
#line 224 "Lily_compile.yacc"
{
		 	labels_stack.push(yyvsp[-1].fake_val);
			fprintf(yyout, "GOTOFALSE L_%d\n", yyvsp[-1].fake_val.label_false);
		 }
break;
case 15:
#line 228 "Lily_compile.yacc"
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
#line 244 "Lily_compile.yacc"
{
			fprintf(yyout, "SAV %s\n", yyvsp[-3].id_val);
		  }
break;
case 17:
#line 250 "Lily_compile.yacc"
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
#line 268 "Lily_compile.yacc"
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
#line 279 "Lily_compile.yacc"
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
#line 306 "Lily_compile.yacc"
{
				fprintf(yyout, "GOTOFALSE L_%d\n", yyvsp[-6].fake_val.label_goto);
				fprintf(yyout, "LABEL L_%d\n", yyvsp[-6].fake_val.label_false);
				Labels lbls;
				labels_stack.pop(lbls);
			}
break;
case 21:
#line 313 "Lily_compile.yacc"
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
#line 323 "Lily_compile.yacc"
{fprintf(yyout, "GOTO F_%s_END\n", current_flow_name);}
break;
case 23:
#line 324 "Lily_compile.yacc"
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
#line 333 "Lily_compile.yacc"
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
#line 342 "Lily_compile.yacc"
{
				Labels lbls;
				labels_stack.pop(lbls);
				fprintf(yyout, "LABEL L_%d\n", lbls.label_false);
				}
break;
case 28:
#line 351 "Lily_compile.yacc"
{strcpy(declare_type, "INTEGER");}
break;
case 29:
#line 352 "Lily_compile.yacc"
{strcpy(declare_type, "STRING");}
break;
case 30:
#line 353 "Lily_compile.yacc"
{strcpy(declare_type, "FLOAT");}
break;
case 31:
#line 354 "Lily_compile.yacc"
{strcpy(declare_type, "MEMBLOCK");}
break;
case 32:
#line 356 "Lily_compile.yacc"
{
			CheckIDAlreadyExist(yyvsp[0].id_val);
			fprintf(yyout, "VAR %s %s\n", yyvsp[0].id_val, declare_type);
			/*插入符号表*/
			InsertToken(yyvsp[0].id_val, declare_type);
			}
break;
case 33:
#line 362 "Lily_compile.yacc"
{
			CheckIDAlreadyExist(yyvsp[0].id_val);
			fprintf(yyout, "VAR %s %s\n", yyvsp[0].id_val, declare_type);
			/*插入符号表*/
			InsertToken(yyvsp[0].id_val, declare_type);
			}
break;
case 36:
#line 372 "Lily_compile.yacc"
{
						Labels lbls;
						labels_stack.peek(lbls);
						fprintf(yyout, "GOTO L_%d\n", lbls.label_false);
						fprintf(yyout, "LABEL L_%d\n", yyvsp[-1].fake_val.label_goto);
					}
break;
case 37:
#line 383 "Lily_compile.yacc"
{
			/*
			 *字符串和字符串可以相加，
			 *整数、浮点数可以相互相加
			 *内存块不可以相加
			 */
			if (yyvsp[-2].expr_type == TYPE_STRING && yyvsp[0].expr_type == TYPE_STRING)
			{
			}
			else if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
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
case 38:
#line 414 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能相减
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else 
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
case 39:
#line 436 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能相乘
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
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
case 40:
#line 458 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能相除
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
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
case 41:
#line 480 "Lily_compile.yacc"
{
			/*只有整数才能求余*/
			if (yyvsp[-2].expr_type != TYPE_INTEGER || yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("求余操作的对象类型不正确!");
			}
			fprintf(yyout, "MOD\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 42:
#line 489 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER || yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("逻辑与的操作对象类型不正确!");
			}
			fprintf(yyout, "AND\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 43:
#line 497 "Lily_compile.yacc"
{
			if (yyvsp[-2].expr_type != TYPE_INTEGER || yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("逻辑或的操作对象类型不正确!");
			}
			fprintf(yyout, "OR\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 44:
#line 505 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能比较大小
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 < 两边操作数类型不正确!");
			}
			fprintf(yyout, "LT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 45:
#line 520 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能比较大小
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 =< 两边操作数类型不正确!");
			}
			fprintf(yyout, "LE\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 46:
#line 535 "Lily_compile.yacc"
{
			/*
			*整数和浮点数能比较是否相等
			*两个字符串可以比较是否相等
			*两个内存块可以比较是否相等
			*/
			if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
			}
			else if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
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
case 47:
#line 555 "Lily_compile.yacc"
{
			/*
			*整数和浮点数能比较是否相等
			*两个字符串可以比较是否相等
			*两个内存块可以比较是否相等
			*/
			if (yyvsp[-2].expr_type == yyvsp[0].expr_type)
			{
			}
			else if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
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
case 48:
#line 579 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能比较大小
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 > 两边操作数类型不正确!");
			}
			fprintf(yyout, "GT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 49:
#line 594 "Lily_compile.yacc"
{
			/*
			*整数和浮点数才能比较大小
			*/
			if( (yyvsp[-2].expr_type == TYPE_INTEGER || yyvsp[-2].expr_type == TYPE_FLOAT) &&
				(yyvsp[0].expr_type == TYPE_INTEGER || yyvsp[0].expr_type == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 >= 两边操作数类型不正确!");
			}
			fprintf(yyout, "GE\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 50:
#line 609 "Lily_compile.yacc"
{yyval.expr_type = yyvsp[-1].expr_type;}
break;
case 51:
#line 610 "Lily_compile.yacc"
{
			if (yyvsp[0].expr_type != TYPE_INTEGER && yyvsp[0].expr_type != TYPE_FLOAT)
			{
				yyerror("正负号操作只能作用于整数或者浮点数类型!");
			}
			if (yyvsp[-1].sign_val == '-')
			{
				fprintf(yyout, "UMINUS\n");
			}
			yyval.expr_type = yyvsp[0].expr_type;
		}
break;
case 52:
#line 621 "Lily_compile.yacc"
{
			if (yyvsp[0].expr_type != TYPE_INTEGER)
			{
				yyerror("非操作只能作用于整数类型!");
			}
			fprintf(yyout, "NOT\n");
			yyval.expr_type = TYPE_INTEGER;
		}
break;
case 53:
#line 629 "Lily_compile.yacc"
{yyval.expr_type = yyvsp[0].expr_type;}
break;
case 54:
#line 631 "Lily_compile.yacc"
{yyval.sign_val = '+';}
break;
case 55:
#line 632 "Lily_compile.yacc"
{yyval.sign_val = '-';}
break;
case 56:
#line 634 "Lily_compile.yacc"
{
		Token tk;
		CheckIDValid(yyvsp[0].id_val, tk);
		yyval.expr_type = tk.tk_type;
		fprintf(yyout, "PUSH %s\n", yyvsp[0].id_val);
		}
break;
case 57:
#line 640 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_STRING;	
			/*将字符串值和指令分别写到不同的文件中*/
			int cs_index = GetNewCSIndex();
			fprintf(csout, "%d %s\n", cs_index, yyvsp[0].const_string_val);
			fprintf(yyout, "PUSH %%%d\n", cs_index);
			}
break;
case 58:
#line 647 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_INTEGER;	
			fprintf(yyout, "PUSH #%d\n", yyvsp[0].const_int_val);
			}
break;
case 59:
#line 651 "Lily_compile.yacc"
{
			yyval.expr_type = TYPE_FLOAT;	
			fprintf(yyout, "PUSH #%s\n", yyvsp[0].const_float_val);
			}
break;
case 61:
#line 658 "Lily_compile.yacc"
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
case 62:
#line 689 "Lily_compile.yacc"
{yyval.arg_count = yyvsp[-2].arg_count + 1;}
break;
case 63:
#line 690 "Lily_compile.yacc"
{yyval.arg_count = 1;}
break;
case 64:
#line 691 "Lily_compile.yacc"
{yyval.arg_count = 0;}
break;
case 65:
#line 694 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "GOTOFALSE L_%d\n", yyval.fake_val.label_false);
		}
break;
case 66:
#line 701 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_goto);
		}
break;
case 67:
#line 707 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_true);
		}
break;
case 68:
#line 714 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", yyval.fake_val.label_goto);
		labels_stack.push(yyval.fake_val);
		}
break;
case 69:
#line 722 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		labels_stack.push(yyval.fake_val);
		}
break;
case 70:
#line 729 "Lily_compile.yacc"
{
		yyval.fake_val.label_true = GetNewLabel();
		yyval.fake_val.label_false = GetNewLabel();
		yyval.fake_val.label_goto = GetNewLabel();
		fprintf(yyout, "GOTOFALSE L_%d\n", yyval.fake_val.label_goto);
		}
break;
#line 1707 "y.tab.c"
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
    if (yyssp >= yyss + yystacksize - 1)
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
