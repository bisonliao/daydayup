%{
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



%}

%union{
	char const_string_val[QSTRING_BUF_MAX];
	int const_int_val;
	char const_float_val[100];
	char id_val[ID_MAX];
	int expr_type;
	char sign_val;
	Labels fake_val;
	int arg_count;
}

%token<id_val> ID
%token<const_string_val> CONST_STRING
%token<const_int_val> CONST_INTEGER
%token<const_float_val> CONST_FLOAT
%token<id_val> FUNCTION
%token IF THEN ELSE ENDIF WHILE DO ENDWHILE INTEGER MEMBLOCK STRING FLOAT RETURN BEGIN_FLOW END_FLOW RUN FOR ENDFOR CONTINUE BREAK REPEAT UNTIL SWITCH ENDSWITCH CASE

%left '='
%left OR
%left AND 
%left LT LE EQ NE GT GE 
%left '+' '-'
%left '*' '/' '%'
%right UMINUS NOT

%type<expr_type>expr
%type<expr_type>factor
%type<sign_val>sign
%type<fake_val>fake1
%type<fake_val>fake2
%type<fake_val>fake3
%type<fake_val>fake4
%type<fake_val>fake5
%type<fake_val>fake6
%type<arg_count>arg_list
%type<expr_type>function

%%
/*
*流程
*/
flow_list	:flow_list flow
		|flow
		;
flow		: '[' ID ']' BEGIN_FLOW {
			/*检查流程块重复定义*/
			FlowName fn;
			strcpy(fn.fn_name, $2);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "流程%s重复定义!", $2);
				yyerror(tmp);
			}
			flowname_stack.push(fn);
			/*局部变量个数*/
			varnum_stack.push(0);
		 	strcpy(current_flow_name, $2); 
			fprintf(yyout, "LABEL F_%s_BEGIN\n", $2);		   
			fprintf(yyout, "DEPTH\n");		   
			
			
		} 
		statement_list 
		END_FLOW {
			fprintf(yyout, "LABEL F_%s_END\n", $2);		   
			fprintf(yyout, "_DEPTH\n");		   
			if (strcmp($2, "main") == 0)
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

		| '[' ID ']' BEGIN_FLOW  END_FLOW 
		{
			/*检查流程块重复定义*/
			FlowName fn;
			strcpy(fn.fn_name, $2);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "流程%s重复定义!", $2);
				yyerror(tmp);
			}
			flowname_stack.push(fn);

			fprintf(yyout, "LABEL F_%s_BEGIN\n", $2);		   
			fprintf(yyout, "LABEL F_%s_END\n", $2);		   

			if (strcmp($2, "main") == 0)
			{
				hasMain = 1;
				fprintf(yyout, "HALT\n");		   
			}
			else
			{
				fprintf(yyout, "RECALL\n");		   
			}
		}
		;
/*
*语句
*/
statement_list	:statement_list statement
		|statement	
		;
statement	:ID '=' expr ';' 	{
			Token tk;	/*保存变量的属性*/
			CheckIDValid($1, tk);

			/*整数和浮点数可以相互赋值,字符串可以相互赋值,内存块可以相互赋值*/
			if (tk.tk_type == $3)
			{
			}
			else if (tk.tk_type == TYPE_INTEGER && $3 == TYPE_FLOAT)
			{
			}
			else if (tk.tk_type == TYPE_FLOAT && $3 == TYPE_INTEGER)
			{
			}
			else
			{
				yyerror("赋值操作操作数类型不正确!");
			}
			fprintf(yyout, "SAV %s\n", $1);
			}
		|declarations	{
				if (labels_stack.isEmpty() != TRUE)
				{
					yyerror("不可在循环结构内声明变量!");
				}
			}
		|expr ';'	{fprintf(yyout, "CLR\n");}
		|IF expr fake1 THEN statement_list ELSE 
		 {
		 	fprintf(yyout, "GOTO L_%d\n", $3.label_goto);
			fprintf(yyout, "LABEL L_%d\n", $3.label_false);
		 }
		 statement_list 
		 ENDIF	{
			if ($2 != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", $3.label_goto);
		 }

		|IF expr fake1 THEN statement_list ENDIF	{
			if ($2 != TYPE_INTEGER)
			{
				yyerror("本行endif对应的if的条件表达式类型不为整型!");
			}
			fprintf(yyout, "LABEL L_%d\n", $3.label_false);
		  }

		|WHILE fake2 expr 
		 {
		 	labels_stack.push($2);
			fprintf(yyout, "GOTOFALSE L_%d\n", $2.label_false);
		 }
		 DO statement_list ENDWHILE {
			if ($3 != TYPE_INTEGER)
			{
			  yyerror("本行endwhile对应的while的条件表达式类型不为整型!");
			}
			fprintf(yyout, "GOTO L_%d\n", $2.label_goto);
			fprintf(yyout, "LABEL L_%d\n", $2.label_false);
			Labels lbls;
			labels_stack.pop(lbls);
		 }
		| FOR 
		  '(' 
		  ID 
		  '=' 
		  expr 
		  ';' 
		  {
			fprintf(yyout, "SAV %s\n", $3);
		  }
		  fake3 
		  expr 
		  ';' 
		  {
		  	if ($9 != TYPE_INTEGER)
			{
				yyerror("for语句的条件表达式类型应该为整型!\n");
			}
			fprintf(yyout, "GOTOFALSE L_%d\n", $8.label_false);
			/*把for(A;B;C)中的C部分翻译到临时文件中*/
			for_fp = yyout;
			fflush(yyout);
			if ( (yyout = fopen(TMP_FILE, "wb") ) == NULL)
			{
				yyerror("写打开临时文件./lily.tmp失败!");
			}
		  }
		  ID 
		  '=' 
		  expr 
		  ')' 
		  {
		  	/*关闭临时文件，恢复原来的yyout本来面目*/
			fprintf(yyout, "SAV %s\n", $12);
			fclose(yyout);
			yyout = for_fp;
			/*将标签压栈，便于break/continue使用*/
			labels_stack.push($8);
		  }
		  DO 
		  statement_list 
		  ENDFOR
		  {
		  	fprintf(yyout, "LABEL L_%d\n", $8.label_goto);
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

			fprintf(yyout, "GOTO L_%d\n", $8.label_true);
			fprintf(yyout, "LABEL L_%d\n", $8.label_false);
			Labels lbls;
			labels_stack.pop(lbls);
		  }

		|REPEAT  
		 fake4 
		 statement_list 
		 UNTIL 
		 '(' expr ')' ';' {
				fprintf(yyout, "GOTOFALSE L_%d\n", $2.label_goto);
				fprintf(yyout, "LABEL L_%d\n", $2.label_false);
				Labels lbls;
				labels_stack.pop(lbls);
			}

		|RUN '(' expr ')' ';' {
				if ($3 != TYPE_STRING)
				{
					yyerror("run函数的参数类型必须为string!");
				}
				unsigned int label = GetNewLabel();
				fprintf(yyout, "SAVCALL L_%d\n", label);
				fprintf(yyout, "JMP\n");
				fprintf(yyout, "LABEL L_%d\n", label);
			}
		|RETURN ';'	{fprintf(yyout, "GOTO F_%s_END\n", current_flow_name);}
		|CONTINUE ';'	{
				if (labels_stack.isEmpty())
				{
					yyerror("continue应该位于循环语句内!");
				}
				Labels lbls;
				labels_stack.peek(lbls);
				fprintf(yyout, "GOTO L_%d\n", lbls.label_goto);
			}
		|BREAK	';'	{
				if (labels_stack.isEmpty())
				{
					yyerror("break应该位于循环语句内!");
				}
				Labels lbls;
				labels_stack.peek(lbls);
				fprintf(yyout, "GOTO L_%d\n", lbls.label_false);
			}
		|SWITCH fake5 case_statement_list ENDSWITCH ';' {
				Labels lbls;
				labels_stack.pop(lbls);
				fprintf(yyout, "LABEL L_%d\n", lbls.label_false);
				}
		| ';'
		;
declarations	:var_type id_list ';'
		;
var_type	:INTEGER	{strcpy(declare_type, "INTEGER");}
		|STRING		{strcpy(declare_type, "STRING");}
		|FLOAT		{strcpy(declare_type, "FLOAT");}
		|MEMBLOCK	{strcpy(declare_type, "MEMBLOCK");}
		;
id_list		:id_list ',' ID {
			CheckIDAlreadyExist($3);
			fprintf(yyout, "VAR %s %s\n", $3, declare_type);
			/*插入符号表*/
			InsertToken($3, declare_type);
			}
		|ID	{
			CheckIDAlreadyExist($1);
			fprintf(yyout, "VAR %s %s\n", $1, declare_type);
			/*插入符号表*/
			InsertToken($1, declare_type);
			}
		;
case_statement_list	:case_statement_list case_statement
					|case_statement
					;
case_statement	:CASE  expr ':' fake6 statement_list {
						Labels lbls;
						labels_stack.peek(lbls);
						fprintf(yyout, "GOTO L_%d\n", lbls.label_false);
						fprintf(yyout, "LABEL L_%d\n", $4.label_goto);
					}
				;

/*
*表达式
*/
expr	:expr '+' expr	{
			/*
			 *字符串和字符串可以相加，
			 *整数、浮点数可以相互相加
			 *内存块不可以相加
			 */
			if ($1 == TYPE_STRING && $3 == TYPE_STRING)
			{
			}
			else if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("加法操作对象类型不正确!");
			}
			fprintf(yyout, "ADD\n");
			if ($1 == TYPE_STRING)
			{
				$$ = TYPE_STRING;
			}
			else if ($1 == $3)
			{
				$$ = $1;
			}
			else
			{
				$$ = TYPE_FLOAT;
			}
		}
	|expr '-' expr	{
			/*
			*整数和浮点数才能相减
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else 
			{
				yyerror("减法操作对象类型不正确!");
			}
			fprintf(yyout, "SUB\n");
			if ($1 == $3)
			{
				$$ = $1;
			}
			else
			{
				$$ = TYPE_FLOAT;
			}
		}
	|expr '*' expr	{
			/*
			*整数和浮点数才能相乘
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("乘法操作对象类型不正确!");
			}
			fprintf(yyout, "MUL\n");
			if ($1 == $3)
			{
				$$ = $1;
			}
			else
			{
				$$ = TYPE_FLOAT;
			}
		}
	|expr '/' expr	{
			/*
			*整数和浮点数才能相除
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("除法操作对象类型不正确!");
			}
			fprintf(yyout, "DIV\n");
			if ($1 == $3)
			{
				$$ = $1;
			}
			else
			{
				$$ = TYPE_FLOAT;
			}
		}
	|expr '%' expr	{
			/*只有整数才能求余*/
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("求余操作的对象类型不正确!");
			}
			fprintf(yyout, "MOD\n");
			$$ = TYPE_INTEGER;
		}
	|expr AND expr	{
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("逻辑与的操作对象类型不正确!");
			}
			fprintf(yyout, "AND\n");
			$$ = TYPE_INTEGER;
		}
	|expr OR expr	{
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("逻辑或的操作对象类型不正确!");
			}
			fprintf(yyout, "OR\n");
			$$ = TYPE_INTEGER;
		}
	|expr LT expr	{
			/*
			*整数和浮点数才能比较大小
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 < 两边操作数类型不正确!");
			}
			fprintf(yyout, "LT\n");
			$$ = TYPE_INTEGER;
		}
	|expr LE expr	{
			/*
			*整数和浮点数才能比较大小
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 =< 两边操作数类型不正确!");
			}
			fprintf(yyout, "LE\n");
			$$ = TYPE_INTEGER;
		}
	|expr NE expr	{
			/*
			*整数和浮点数能比较是否相等
			*两个字符串可以比较是否相等
			*两个内存块可以比较是否相等
			*/
			if ($1 == $3)
			{
			}
			else if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 != 两边操作数类型不正确!");
			}
			fprintf(yyout, "NE\n");
			$$ = TYPE_INTEGER;
		}
	|expr EQ expr	{
			/*
			*整数和浮点数能比较是否相等
			*两个字符串可以比较是否相等
			*两个内存块可以比较是否相等
			*/
			if ($1 == $3)
			{
			}
			else if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			if ($1 == TYPE_STRING && $3 == TYPE_STRING ||
			    $1 != TYPE_STRING && $3 != TYPE_STRING)
			{
			}
			else
			{
				yyerror("符号 == 两边操作数类型不正确!");
			}
			fprintf(yyout, "EQ\n");
			$$ = TYPE_INTEGER;
		}
	|expr GT expr	{
			/*
			*整数和浮点数才能比较大小
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 > 两边操作数类型不正确!");
			}
			fprintf(yyout, "GT\n");
			$$ = TYPE_INTEGER;
		}
	|expr GE expr	{
			/*
			*整数和浮点数才能比较大小
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("符号 >= 两边操作数类型不正确!");
			}
			fprintf(yyout, "GE\n");
			$$ = TYPE_INTEGER;
		}
	|'(' expr ')'	{$$ = $2;}
	| sign expr %prec UMINUS	{
			if ($2 != TYPE_INTEGER && $2 != TYPE_FLOAT)
			{
				yyerror("正负号操作只能作用于整数或者浮点数类型!");
			}
			if ($1 == '-')
			{
				fprintf(yyout, "UMINUS\n");
			}
			$$ = $2;
		}
	| NOT expr	{
			if ($2 != TYPE_INTEGER)
			{
				yyerror("非操作只能作用于整数类型!");
			}
			fprintf(yyout, "NOT\n");
			$$ = TYPE_INTEGER;
		}
	|factor	{$$ = $1;}
	;
sign	: '+'	{$$ = '+';}
	| '-'	{$$ = '-';}
	;
factor 	: ID	{
		Token tk;
		CheckIDValid($1, tk);
		$$ = tk.tk_type;
		fprintf(yyout, "PUSH %s\n", $1);
		}
	| '&' ID	{
		Token tk;
		CheckIDValid($2, tk);
		$$ = TYPE_STRING;
		fprintf(yyout, "ADDR %s\n", $2);
	}
	| CONST_STRING	{
			$$ = TYPE_STRING;	
			/*将字符串值和指令分别写到不同的文件中*/
			int cs_index = GetNewCSIndex();
			fprintf(csout, "%d %s\n", cs_index, $1);
			fprintf(yyout, "PUSH %%%d\n", cs_index);
			}
	| CONST_INTEGER	{
			$$ = TYPE_INTEGER;	
			fprintf(yyout, "PUSH #%d\n", $1);
			}
	| CONST_FLOAT	{
			$$ = TYPE_FLOAT;	
			fprintf(yyout, "PUSH #%s\n", $1);
			}
	| function
	;
  /*函数*/
function	:ID '(' arg_list ')'	{
				int argnum, number, type;/*函数的参数个数和编号,类型*/
				char tmp[100];
				if (fcGetFunc($1, argnum, number, type) < 0)
				{
					sprintf(tmp, "函数%s没有定义!", $1);
					yyerror(tmp);
				}
				$$ = type;
				/*检查参数个数*/
				if (argnum < 0)
				{
					if ($3 > abs(argnum))
					{
						sprintf(tmp, "函数%s参数个数不对!", $1);
						yyerror(tmp);
					}
				}
				else
				{
					if ($3 != argnum)
					{
						sprintf(tmp, "函数%s参数个数不对!", $1);
						yyerror(tmp);
					}
				}
				/*参数个数压栈*/
				fprintf(yyout, "PUSH #%d\n", $3);
				fprintf(yyout, "CALL %d\n", number);
			}
		;
arg_list	:arg_list ',' expr	{$$ = $1 + 1;}
		|expr	{$$ = 1;}
		|	{$$ = 0;}
		;
	/*伪产生式*/
fake1	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		fprintf(yyout, "GOTOFALSE L_%d\n", $$.label_false);
		}
	;
fake2	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", $$.label_goto);
		}
fake3	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", $$.label_true);
		}
	;
fake4	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		fprintf(yyout, "LABEL L_%d\n", $$.label_goto);
		labels_stack.push($$);
		}
	;
fake5	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		labels_stack.push($$);
		}
	;
fake6	:	{
		$$.label_true = GetNewLabel();
		$$.label_false = GetNewLabel();
		$$.label_goto = GetNewLabel();
		fprintf(yyout, "GOTOFALSE L_%d\n", $$.label_goto);
		}
	;
%%
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
