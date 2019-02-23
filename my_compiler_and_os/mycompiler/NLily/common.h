#ifndef __COMMON_H_INCLUDED__
#define __COMMON_H_INCLUDED__

/*��ʶ����󳤶�*/
#define ID_MAX 20

/*��������*/
#define TYPE_INTEGER 	0
#define TYPE_FLOAT 	1
#define TYPE_STRING	2
#define TYPE_MEMBLOCK	3	/*�������ͣ��ڴ��*/

/*��������*/
#define SYMBOL_TERMINATOR 1		/*�ս��*/
#define SYMBOL_NONTERMINATOR 2	/*���ս��*/
#define SYMBOL_DOT 3			/*��Ŀ��ĳ���ӵĵ�*/

////////////////////////////
#define QSTRING_BUF_MAX 1000

/*���ű����Ԫ��*/
typedef struct
{
	char tk_name[ID_MAX];
	int tk_type;
} Token;
/*α���ս������������*/
typedef struct 
{
	unsigned int label_false;	/*ѭ�����������������ʱ����תĿ��*/
	unsigned int label_true;	/*for ����������÷�*/
	unsigned int label_goto;	/*���һ��ѭ�������תĿ��*/
} Labels;


///////////////////////////

/*���ŵ�����*/
typedef union 
{
	char const_string_val[QSTRING_BUF_MAX];
	int const_int_val;
	float const_float_val;
	char id_val[ID_MAX];
	int expr_type;
	char sign_val;
	Labels fake_val;
	int arg_count;
} YYLVAL;

#endif