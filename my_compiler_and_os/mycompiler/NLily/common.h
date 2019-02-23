#ifndef __COMMON_H_INCLUDED__
#define __COMMON_H_INCLUDED__

/*标识符最大长度*/
#define ID_MAX 20

/*数据类型*/
#define TYPE_INTEGER 	0
#define TYPE_FLOAT 	1
#define TYPE_STRING	2
#define TYPE_MEMBLOCK	3	/*新增类型：内存块*/

/*符号类型*/
#define SYMBOL_TERMINATOR 1		/*终结符*/
#define SYMBOL_NONTERMINATOR 2	/*非终结符*/
#define SYMBOL_DOT 3			/*项目的某处加的点*/

////////////////////////////
#define QSTRING_BUF_MAX 1000

/*符号表里的元素*/
typedef struct
{
	char tk_name[ID_MAX];
	int tk_type;
} Token;
/*伪非终结符的属性类型*/
typedef struct 
{
	unsigned int label_false;	/*循环体进入条件不满足时的跳转目标*/
	unsigned int label_true;	/*for 语句中特殊用法*/
	unsigned int label_goto;	/*完成一轮循环后的跳转目标*/
} Labels;


///////////////////////////

/*符号的属性*/
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