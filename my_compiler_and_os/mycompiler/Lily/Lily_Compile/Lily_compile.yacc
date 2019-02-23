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

/*��ǰ���������̿������*/
static char current_flow_name[ID_MAX];
/*���ű�*/
static Token_Stack token_stack;
/*ÿ�����̿�ı�����������ջ*/
static int_Stack varnum_stack;
/*��������ʱ������*/
static char declare_type[20];
/*ѭ������õ�����ת��ǩ�ı���ջ�����ڷ���break��continue*/
static Labels_Stack labels_stack;
/*�ݴ�yyout,����for���ʱ�õ�*/
static FILE * for_fp = NULL;

/*�������̿�����֣���ֹ�ظ�����*/
static FlowName_Stack flowname_stack;

/*��ʶ�Ƿ�����main���̿�*/
static int hasMain = 0;

/*�����ַ�����д���ļ�*/
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
*����
*/
flow_list	:flow_list flow
		|flow
		;
flow		: '[' ID ']' BEGIN_FLOW {
			/*������̿��ظ�����*/
			FlowName fn;
			strcpy(fn.fn_name, $2);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "����%s�ظ�����!", $2);
				yyerror(tmp);
			}
			flowname_stack.push(fn);
			/*�ֲ���������*/
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
			
			/*���ű���ջ*/
			RemoveTokenFromTab();
		}

		| '[' ID ']' BEGIN_FLOW  END_FLOW 
		{
			/*������̿��ظ�����*/
			FlowName fn;
			strcpy(fn.fn_name, $2);
			if (flowname_stack.contain(fn))
			{
				char tmp[100];
				sprintf(tmp, "����%s�ظ�����!", $2);
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
*���
*/
statement_list	:statement_list statement
		|statement	
		;
statement	:ID '=' expr ';' 	{
			Token tk;	/*�������������*/
			CheckIDValid($1, tk);

			/*�����͸����������໥��ֵ,�ַ��������໥��ֵ,�ڴ������໥��ֵ*/
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
				yyerror("��ֵ�������������Ͳ���ȷ!");
			}
			fprintf(yyout, "SAV %s\n", $1);
			}
		|declarations	{
				if (labels_stack.isEmpty() != TRUE)
				{
					yyerror("������ѭ���ṹ����������!");
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
				yyerror("����endif��Ӧ��if���������ʽ���Ͳ�Ϊ����!");
			}
			fprintf(yyout, "LABEL L_%d\n", $3.label_goto);
		 }

		|IF expr fake1 THEN statement_list ENDIF	{
			if ($2 != TYPE_INTEGER)
			{
				yyerror("����endif��Ӧ��if���������ʽ���Ͳ�Ϊ����!");
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
			  yyerror("����endwhile��Ӧ��while���������ʽ���Ͳ�Ϊ����!");
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
				yyerror("for�����������ʽ����Ӧ��Ϊ����!\n");
			}
			fprintf(yyout, "GOTOFALSE L_%d\n", $8.label_false);
			/*��for(A;B;C)�е�C���ַ��뵽��ʱ�ļ���*/
			for_fp = yyout;
			fflush(yyout);
			if ( (yyout = fopen(TMP_FILE, "wb") ) == NULL)
			{
				yyerror("д����ʱ�ļ�./lily.tmpʧ��!");
			}
		  }
		  ID 
		  '=' 
		  expr 
		  ')' 
		  {
		  	/*�ر���ʱ�ļ����ָ�ԭ����yyout������Ŀ*/
			fprintf(yyout, "SAV %s\n", $12);
			fclose(yyout);
			yyout = for_fp;
			/*����ǩѹջ������break/continueʹ��*/
			labels_stack.push($8);
		  }
		  DO 
		  statement_list 
		  ENDFOR
		  {
		  	fprintf(yyout, "LABEL L_%d\n", $8.label_goto);
			/*��for(A;B;C)�е�C���ִ���ʱ�ļ��ж������뵽����*/
			FILE *fp = NULL;
			char tmp_buf[512];
			int len;
			if ( (fp = fopen(TMP_FILE, "rb")) == NULL)
			{
				yyerror("������ʱ�ļ�./lily.tmpʧ��!");
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
					yyerror("run�����Ĳ������ͱ���Ϊstring!");
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
					yyerror("continueӦ��λ��ѭ�������!");
				}
				Labels lbls;
				labels_stack.peek(lbls);
				fprintf(yyout, "GOTO L_%d\n", lbls.label_goto);
			}
		|BREAK	';'	{
				if (labels_stack.isEmpty())
				{
					yyerror("breakӦ��λ��ѭ�������!");
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
			/*������ű�*/
			InsertToken($3, declare_type);
			}
		|ID	{
			CheckIDAlreadyExist($1);
			fprintf(yyout, "VAR %s %s\n", $1, declare_type);
			/*������ű�*/
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
*���ʽ
*/
expr	:expr '+' expr	{
			/*
			 *�ַ������ַ���������ӣ�
			 *�����������������໥���
			 *�ڴ�鲻�������
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
				yyerror("�ӷ������������Ͳ���ȷ!");
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
			*�����͸������������
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else 
			{
				yyerror("���������������Ͳ���ȷ!");
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
			*�����͸������������
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("�˷������������Ͳ���ȷ!");
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
			*�����͸������������
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("���������������Ͳ���ȷ!");
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
			/*ֻ��������������*/
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("��������Ķ������Ͳ���ȷ!");
			}
			fprintf(yyout, "MOD\n");
			$$ = TYPE_INTEGER;
		}
	|expr AND expr	{
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("�߼���Ĳ����������Ͳ���ȷ!");
			}
			fprintf(yyout, "AND\n");
			$$ = TYPE_INTEGER;
		}
	|expr OR expr	{
			if ($1 != TYPE_INTEGER || $3 != TYPE_INTEGER)
			{
				yyerror("�߼���Ĳ����������Ͳ���ȷ!");
			}
			fprintf(yyout, "OR\n");
			$$ = TYPE_INTEGER;
		}
	|expr LT expr	{
			/*
			*�����͸��������ܱȽϴ�С
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("���� < ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "LT\n");
			$$ = TYPE_INTEGER;
		}
	|expr LE expr	{
			/*
			*�����͸��������ܱȽϴ�С
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("���� =< ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "LE\n");
			$$ = TYPE_INTEGER;
		}
	|expr NE expr	{
			/*
			*�����͸������ܱȽ��Ƿ����
			*�����ַ������ԱȽ��Ƿ����
			*�����ڴ����ԱȽ��Ƿ����
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
				yyerror("���� != ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "NE\n");
			$$ = TYPE_INTEGER;
		}
	|expr EQ expr	{
			/*
			*�����͸������ܱȽ��Ƿ����
			*�����ַ������ԱȽ��Ƿ����
			*�����ڴ����ԱȽ��Ƿ����
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
				yyerror("���� == ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "EQ\n");
			$$ = TYPE_INTEGER;
		}
	|expr GT expr	{
			/*
			*�����͸��������ܱȽϴ�С
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("���� > ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "GT\n");
			$$ = TYPE_INTEGER;
		}
	|expr GE expr	{
			/*
			*�����͸��������ܱȽϴ�С
			*/
			if( ($1 == TYPE_INTEGER || $1 == TYPE_FLOAT) &&
				($3 == TYPE_INTEGER || $3 == TYPE_FLOAT))
			{
			}
			else
			{
				yyerror("���� >= ���߲��������Ͳ���ȷ!");
			}
			fprintf(yyout, "GE\n");
			$$ = TYPE_INTEGER;
		}
	|'(' expr ')'	{$$ = $2;}
	| sign expr %prec UMINUS	{
			if ($2 != TYPE_INTEGER && $2 != TYPE_FLOAT)
			{
				yyerror("�����Ų���ֻ���������������߸���������!");
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
				yyerror("�ǲ���ֻ����������������!");
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
			/*���ַ���ֵ��ָ��ֱ�д����ͬ���ļ���*/
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
  /*����*/
function	:ID '(' arg_list ')'	{
				int argnum, number, type;/*�����Ĳ��������ͱ��,����*/
				char tmp[100];
				if (fcGetFunc($1, argnum, number, type) < 0)
				{
					sprintf(tmp, "����%sû�ж���!", $1);
					yyerror(tmp);
				}
				$$ = type;
				/*����������*/
				if (argnum < 0)
				{
					if ($3 > abs(argnum))
					{
						sprintf(tmp, "����%s������������!", $1);
						yyerror(tmp);
					}
				}
				else
				{
					if ($3 != argnum)
					{
						sprintf(tmp, "����%s������������!", $1);
						yyerror(tmp);
					}
				}
				/*��������ѹջ*/
				fprintf(yyout, "PUSH #%d\n", $3);
				fprintf(yyout, "CALL %d\n", number);
			}
		;
arg_list	:arg_list ',' expr	{$$ = $1 + 1;}
		|expr	{$$ = 1;}
		|	{$$ = 0;}
		;
	/*α����ʽ*/
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
	/*��ȡ����*/
	int opt;
	char input_file[255];
	char output_file[255];
	memset(input_file, 0, sizeof(input_file));
	memset(output_file, 0, sizeof(output_file));
	/*
	*-o ����ļ��� ֻ����һ��������ж���������һ��Ϊ׼
	*-f ������ļ��� ֻ����һ��������ж���������һ��Ϊ׼
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
			fprintf(stderr, "���ļ�%sʧ��!\n", input_file);
			return -1;
		}
	}
	/*��ָ������ļ�*/
	if ( (yyout = fopen(INSTRUCT_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "���ļ�%sʧ��!\n", INSTRUCT_FILE);
		return -1;
	}
	/*�򿪳����ַ�������ļ�*/
	if ( (csout = fopen(CSTRING_FILE, "wb")) == NULL)
	{
		fprintf(stderr, "���ļ�%sʧ��!\n", CSTRING_FILE);
		return -1;
	}
	/*��ȡ����������*/
	if (fcInit(getenv("FC")) < 0)
	{
		fprintf(stderr, "��ȡ����������Ϣʧ��!\n");
		return -1;
	}
	/*����*/
	yyparse();

	fclose(yyin);
	fclose(yyout);
	fclose(csout);
	yyin = NULL;
	yyout = NULL;
	csout = NULL;

	if (hasMain == 0)
	{
		fprintf(stderr, "������mainû�ж���!\n");
		return -1;
	}
	/*�ϲ������ַ����ļ���ָ���ļ�*/
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
			fprintf(stderr, "���ļ�%sʧ��!\n", output_file);
			return -1;
		}
	}
	if ( (in = fopen(CSTRING_FILE, "rb")) == NULL)
	{
		fprintf(stderr, "���ļ�%sʧ��!\n", CSTRING_FILE);
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
		fprintf(stderr, "���ļ�%sʧ��!\n", INSTRUCT_FILE);
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
*�������еı�ʶ���Ƿ��Ѿ�������
*/
void CheckIDValid(const char * id_name, Token &ttt)
{
	/*���ȼ��ñ����Ƿ���ȫ�ֵ�(Ԥ�ȶ���ı���)*/	
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
	sprintf(tmp, "����%sû�ж���!", id_name);
	yyerror(tmp);
}
/*
*�������Ƿ��ظ�����
*/
void CheckIDAlreadyExist(const char* id_name)
{
	/*���ȼ��ñ����Ƿ���ȫ�ֵ�(Ԥ�ȶ���ı���)*/	
	int gIdx;
	char tmp[256];
	if (isGlobalID(id_name, &gIdx) == 1)
	{
			sprintf(tmp, "����%s��Ԥ�ȶ����ȫ�ֱ����������ظ�����!", id_name);
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
			sprintf(tmp, "����%s�ظ�����!", id_name);
			yyerror(tmp);
		}
	}
	return;
}
/*
*���ű���ջ
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
*������ű�
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
*ÿ����һ�Σ�����һ��Ψһ�ı�ǩ��
*/
unsigned int GetNewLabel()
{
	static int label_index = 0;
	return ++label_index;
}
