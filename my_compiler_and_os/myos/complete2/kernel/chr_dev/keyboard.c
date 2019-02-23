#include "cycle_buf.h"
#include "global.h"
#include "tty.h"

/**
 * 类似下面这种把变量初始化为0的情况，变量会被放置到bss段
 * 而bss段在编译链接过程中是没有的，只是在load的时候才被清0
 * 但是我们用的是flat binary格式执行码，没有load的过程，所以
 * 下面的定义不能被正确初始化为0，除非初始化为非0的值，gcc才会
 * 把它们放到data段，被放到执行文件里
 * 使用section属性可以避免这个问题
 */
static  TProcess * g_wait  __attribute__ ((section ("data"))) = NULL;
static  CycleBufHead *g_pstKBBufHead  __attribute__ ((section ("data")))  = NULL;
static  CycleBufHead *g_pstCharList  __attribute__ ((section ("data"))) = NULL;

static  unsigned char g_KBBuf[sizeof(CycleBufHead)+100] ;
static  unsigned char g_CharListBuf[sizeof(CycleBufHead)+100] ;



#define	KB_IN_BYTES	32	/* size of keyboard input buffer */
#define MAP_COLS	3	/* Number of columns in keymap */
#define NR_SCAN_CODES	0x80	/* Number of scan codes (rows in keymap) */

#define FLAG_BREAK	0x0080		/* Break Code			*/
#define FLAG_EXT	0x0100		/* Normal function keys		*/
#define FLAG_SHIFT_L	0x0200		/* Shift key			*/
#define FLAG_SHIFT_R	0x0400		/* Shift key			*/
#define FLAG_CTRL_L	0x0800		/* Control key			*/
#define FLAG_CTRL_R	0x1000		/* Control key			*/
#define FLAG_ALT_L	0x2000		/* Alternate key		*/
#define FLAG_ALT_R	0x4000		/* Alternate key		*/
#define FLAG_PAD	0x8000		/* keys in num pad		*/

#define MASK_RAW	0x01FF		/* raw key value = code passed to tty & MASK_RAW
								   the value can be found either in the keymap column 0
								   or in the list below */

/* Special keys */
#define ESC		(0x01 + FLAG_EXT)	/* Esc		*/
#define TAB		(0x02 + FLAG_EXT)	/* Tab		*/
#define ENTER		(0x03 + FLAG_EXT)	/* Enter	*/
#define BACKSPACE	(0x04 + FLAG_EXT)	/* BackSpace	*/

#define GUI_L		(0x05 + FLAG_EXT)	/* L GUI	*/
#define GUI_R		(0x06 + FLAG_EXT)	/* R GUI	*/
#define APPS		(0x07 + FLAG_EXT)	/* APPS	*/

/* Shift, Ctrl, Alt */
#define SHIFT_L		(0x08 + FLAG_EXT)	/* L Shift	*/
#define SHIFT_R		(0x09 + FLAG_EXT)	/* R Shift	*/
#define CTRL_L		(0x0A + FLAG_EXT)	/* L Ctrl	*/
#define CTRL_R		(0x0B + FLAG_EXT)	/* R Ctrl	*/
#define ALT_L		(0x0C + FLAG_EXT)	/* L Alt	*/
#define ALT_R		(0x0D + FLAG_EXT)	/* R Alt	*/

/* Lock keys */
#define CAPS_LOCK	(0x0E + FLAG_EXT)	/* Caps Lock	*/
#define	NUM_LOCK	(0x0F + FLAG_EXT)	/* Number Lock	*/
#define SCROLL_LOCK	(0x10 + FLAG_EXT)	/* Scroll Lock	*/

/* Function keys */
#define F1		(0x11 + FLAG_EXT)	/* F1		*/
#define F2		(0x12 + FLAG_EXT)	/* F2		*/
#define F3		(0x13 + FLAG_EXT)	/* F3		*/
#define F4		(0x14 + FLAG_EXT)	/* F4		*/
#define F5		(0x15 + FLAG_EXT)	/* F5		*/
#define F6		(0x16 + FLAG_EXT)	/* F6		*/
#define F7		(0x17 + FLAG_EXT)	/* F7		*/
#define F8		(0x18 + FLAG_EXT)	/* F8		*/
#define F9		(0x19 + FLAG_EXT)	/* F9		*/
#define F10		(0x1A + FLAG_EXT)	/* F10		*/
#define F11		(0x1B + FLAG_EXT)	/* F11		*/
#define F12		(0x1C + FLAG_EXT)	/* F12		*/

/* Control Pad */
#define PRINTSCREEN	(0x1D + FLAG_EXT)	/* Print Screen	*/
#define PAUSEBREAK	(0x1E + FLAG_EXT)	/* Pause/Break	*/
#define INSERT		(0x1F + FLAG_EXT)	/* Insert	*/
#define DELETE		(0x20 + FLAG_EXT)	/* Delete	*/
#define HOME		(0x21 + FLAG_EXT)	/* Home		*/
#define END		(0x22 + FLAG_EXT)	/* End		*/
#define PAGEUP		(0x23 + FLAG_EXT)	/* Page Up	*/
#define PAGEDOWN	(0x24 + FLAG_EXT)	/* Page Down	*/
#define UP		(0x25 + FLAG_EXT)	/* Up		*/
#define DOWN		(0x26 + FLAG_EXT)	/* Down		*/
#define LEFT		(0x27 + FLAG_EXT)	/* Left		*/
#define RIGHT		(0x28 + FLAG_EXT)	/* Right	*/

/* ACPI keys */
#define POWER		(0x29 + FLAG_EXT)	/* Power	*/
#define SLEEP		(0x2A + FLAG_EXT)	/* Sleep	*/
#define WAKE		(0x2B + FLAG_EXT)	/* Wake Up	*/

/* Num Pad */
#define PAD_SLASH	(0x2C + FLAG_EXT)	/* /		*/
#define PAD_STAR	(0x2D + FLAG_EXT)	/* *		*/
#define PAD_MINUS	(0x2E + FLAG_EXT)	/* -		*/
#define PAD_PLUS	(0x2F + FLAG_EXT)	/* +		*/
#define PAD_ENTER	(0x30 + FLAG_EXT)	/* Enter	*/
#define PAD_DOT		(0x31 + FLAG_EXT)	/* .		*/
#define PAD_0		(0x32 + FLAG_EXT)	/* 0		*/
#define PAD_1		(0x33 + FLAG_EXT)	/* 1		*/
#define PAD_2		(0x34 + FLAG_EXT)	/* 2		*/
#define PAD_3		(0x35 + FLAG_EXT)	/* 3		*/
#define PAD_4		(0x36 + FLAG_EXT)	/* 4		*/
#define PAD_5		(0x37 + FLAG_EXT)	/* 5		*/
#define PAD_6		(0x38 + FLAG_EXT)	/* 6		*/
#define PAD_7		(0x39 + FLAG_EXT)	/* 7		*/
#define PAD_8		(0x3A + FLAG_EXT)	/* 8		*/
#define PAD_9		(0x3B + FLAG_EXT)	/* 9		*/
#define PAD_UP		PAD_8			/* Up		*/
#define PAD_DOWN	PAD_2			/* Down		*/
#define PAD_LEFT	PAD_4			/* Left		*/
#define PAD_RIGHT	PAD_6			/* Right	*/
#define PAD_HOME	PAD_7			/* Home		*/
#define PAD_END		PAD_1			/* End		*/
#define PAD_PAGEUP	PAD_9			/* Page Up	*/
#define PAD_PAGEDOWN	PAD_3			/* Page Down	*/
#define PAD_INS		PAD_0			/* Ins		*/
#define PAD_MID		PAD_5			/* Middle key	*/
#define PAD_DEL		PAD_DOT			/* Del		*/



typedef struct
{
	uint32_t chr; 	/*shift没有按下的时候的字符*/
	uint32_t schr;   /*shift按下的时候的字符*/
	uint32_t	xx; 	/* 什么东西我还不知道*/
} TKey;
//static const TKey keymap[]  __attribute__ ((section ("data"))) = {
static const TKey keymap[]   = {
	/* scan-code			!Shift		Shift		E0 XX		*/
	/* ==================================================================== */
	{/* 0x00 - none		*/	0,		0,		0},
	{/* 0x01 - ESC		*/	ESC,		ESC,		0},
	{/* 0x02 - '1'		*/	'1',		'!',		0},
	{/* 0x03 - '2'		*/	'2',		'@',		0},
	{/* 0x04 - '3'		*/	'3',		'#',		0},
	{/* 0x05 - '4'		*/	'4',		'$',		0},
	{/* 0x06 - '5'		*/	'5',		'%',		0},
	{/* 0x07 - '6'		*/	'6',		'^',		0},
	{/* 0x08 - '7'		*/	'7',		'&',		0},
	{/* 0x09 - '8'		*/	'8',		'*',		0},
	{/* 0x0A - '9'		*/	'9',		'(',		0},
	{/* 0x0B - '0'		*/	'0',		')',		0},
	{/* 0x0C - '-'		*/	'-',		'_',		0},
	{/* 0x0D - '='		*/	'=',		'+',		0},
	{/* 0x0E - BS		*/	BACKSPACE,	BACKSPACE,	0},
	{/* 0x0F - TAB		*/	TAB,		TAB,		0},
	{/* 0x10 - 'q'		*/	'q',		'Q',		0},
	{/* 0x11 - 'w'		*/	'w',		'W',		0},
	{/* 0x12 - 'e'		*/	'e',		'E',		0},
	{/* 0x13 - 'r'		*/	'r',		'R',		0},
	{/* 0x14 - 't'		*/	't',		'T',		0},
	{/* 0x15 - 'y'		*/	'y',		'Y',		0},
	{/* 0x16 - 'u'		*/	'u',		'U',		0},
	{/* 0x17 - 'i'		*/	'i',		'I',		0},
	{/* 0x18 - 'o'		*/	'o',		'O',		0},
	{/* 0x19 - 'p'		*/	'p',		'P',		0},
	{/* 0x1A - '['		*/	'[',		'{',		0},
	{/* 0x1B - ']'		*/	']',		'}',		0},
	{/* 0x1C - CR/LF		*/	ENTER,		ENTER,		PAD_ENTER},
	{/* 0x1D - l. Ctrl	*/	CTRL_L,		CTRL_L,		CTRL_R},
	{/* 0x1E - 'a'		*/	'a',		'A',		0},
	{/* 0x1F - 's'		*/	's',		'S',		0},
	{/* 0x20 - 'd'		*/	'd',		'D',		0},
	{/* 0x21 - 'f'		*/	'f',		'F',		0},
	{/* 0x22 - 'g'		*/	'g',		'G',		0},
	{/* 0x23 - 'h'		*/	'h',		'H',		0},
	{/* 0x24 - 'j'		*/	'j',		'J',		0},
	{/* 0x25 - 'k'		*/	'k',		'K',		0},
	{/* 0x26 - 'l'		*/	'l',		'L',		0},
	{/* 0x27 - ';'		*/	';',		':',		0},
	{/* 0x28 - '\''		*/	'\'',		'"',		0},
	{/* 0x29 - '`'		*/	'`',		'~',		0},
	{/* 0x2A - l. SHIFT	*/	SHIFT_L,	SHIFT_L,	0},
	{/* 0x2B - '\'		*/	'\\',		'|',		0},
	{/* 0x2C - 'z'		*/	'z',		'Z',		0},
	{/* 0x2D - 'x'		*/	'x',		'X',		0},
	{/* 0x2E - 'c'		*/	'c',		'C',		0},
	{/* 0x2F - 'v'		*/	'v',		'V',		0},
	{/* 0x30 - 'b'		*/	'b',		'B',		0},
	{/* 0x31 - 'n'		*/	'n',		'N',		0},
	{/* 0x32 - 'm'		*/	'm',		'M',		0},
	{/* 0x33 - ','		*/	',',		'<',		0},
	{/* 0x34 - '.'		*/	'.',		'>',		0},
	{/* 0x35 - '/'		*/	'/',		'?',		PAD_SLASH},
	{/* 0x36 - r. SHIFT	*/	SHIFT_R,	SHIFT_R,	0},
	{/* 0x37 - '*'		*/	'*',		'*',    	0},
	{/* 0x38 - ALT		*/	ALT_L,		ALT_L,  	ALT_R},
	{/* 0x39 - ' '		*/	' ',		' ',		0},
	{/* 0x3A - CapsLock	*/	CAPS_LOCK,	CAPS_LOCK,	0},
	{/* 0x3B - F1		*/	F1,		F1,		0},
	{/* 0x3C - F2		*/	F2,		F2,		0},
	{/* 0x3D - F3		*/	F3,		F3,		0},
	{/* 0x3E - F4		*/	F4,		F4,		0},
	{/* 0x3F - F5		*/	F5,		F5,		0},
	{/* 0x40 - F6		*/	F6,		F6,		0},
	{/* 0x41 - F7		*/	F7,		F7,		0},
	{/* 0x42 - F8		*/	F8,		F8,		0},
	{/* 0x43 - F9		*/	F9,		F9,		0},
	{/* 0x44 - F10		*/	F10,		F10,		0},
	{/* 0x45 - NumLock	*/	NUM_LOCK,	NUM_LOCK,	0},
	{/* 0x46 - ScrLock	*/	SCROLL_LOCK,	SCROLL_LOCK,	0},
	{/* 0x47 - Home		*/	PAD_HOME,	'7',		HOME},
	{/* 0x48 - CurUp		*/	PAD_UP,		'8',		UP},
	{/* 0x49 - PgUp		*/	PAD_PAGEUP,	'9',		PAGEUP},
	{/* 0x4A - '-'		*/	PAD_MINUS,	'-',		0},
	{/* 0x4B - Left		*/	PAD_LEFT,	'4',		LEFT},
	{/* 0x4C - MID		*/	PAD_MID,	'5',		0},
	{/* 0x4D - Right		*/	PAD_RIGHT,	'6',		RIGHT},
	{/* 0x4E - '+'		*/	PAD_PLUS,	'+',		0},
	{/* 0x4F - End		*/	PAD_END,	'1',		END},
	{/* 0x50 - Down		*/	PAD_DOWN,	'2',		DOWN},
	{/* 0x51 - PgDown	*/	PAD_PAGEDOWN,	'3',		PAGEDOWN},
	{/* 0x52 - Insert	*/	PAD_INS,	'0',		INSERT},
	{/* 0x53 - Delete	*/	PAD_DOT,	'.',		DELETE},
	{/* 0x54 - Enter		*/	0,		0,		0},
	{/* 0x55 - ???		*/	0,		0,		0},
	{/* 0x56 - ???		*/	0,		0,		0},
	{/* 0x57 - F11		*/	F11,		F11,		0},
	{/* 0x58 - F12		*/	F12,		F12,		0},
#if 0
	{/* 0x59 - ???		*/	0,		0,		0},
	{/* 0x5A - ???		*/	0,		0,		0},
	{/* 0x5B - ???		*/	0,		0,		GUI_L},
	{/* 0x5C - ???		*/	0,		0,		GUI_R},
	{/* 0x5D - ???		*/	0,		0,		APPS},
	{/* 0x5E - ???		*/	0,		0,		0},
	{/* 0x5F - ???		*/	0,		0,		0},
	{/* 0x60 - ???		*/	0,		0,		0},
	{/* 0x61 - ???		*/	0,		0,		0},
	{/* 0x62 - ???		*/	0,		0,		0},
	{/* 0x63 - ???		*/	0,		0,		0},
	{/* 0x64 - ???		*/	0,		0,		0},
	{/* 0x65 - ???		*/	0,		0,		0},
	{/* 0x66 - ???		*/	0,		0,		0},
	{/* 0x67 - ???		*/	0,		0,		0},
	{/* 0x68 - ???		*/	0,		0,		0},
	{/* 0x69 - ???		*/	0,		0,		0},
	{/* 0x6A - ???		*/	0,		0,		0},
	{/* 0x6B - ???		*/	0,		0,		0},
	{/* 0x6C - ???		*/	0,		0,		0},
	{/* 0x6D - ???		*/	0,		0,		0},
	{/* 0x6E - ???		*/	0,		0,		0},
	{/* 0x6F - ???		*/	0,		0,		0},
	{/* 0x70 - ???		*/	0,		0,		0},
	{/* 0x71 - ???		*/	0,		0,		0},
	{/* 0x72 - ???		*/	0,		0,		0},
	{/* 0x73 - ???		*/	0,		0,		0},
	{/* 0x74 - ???		*/	0,		0,		0},
	{/* 0x75 - ???		*/	0,		0,		0},
	{/* 0x76 - ???		*/	0,		0,		0},
	{/* 0x77 - ???		*/	0,		0,		0},
	{/* 0x78 - ???		*/	0,		0,		0},
	{/* 0x78 - ???		*/	0,		0,		0},
	{/* 0x7A - ???		*/	0,		0,		0},
	{/* 0x7B - ???		*/	0,		0,		0},
	{/* 0x7C - ???		*/	0,		0,		0},
	{/* 0x7D - ???		*/	0,		0,		0},
	{/* 0x7E - ???		*/	0,		0,		0},
	{/* 0x7F - ???		*/	0,		0,		0}
#endif
};

static uint32_t flags ;

void keyboard_init()
{
	flags = 0;
	if (cycle_buf_MemAttach(g_KBBuf, sizeof(g_KBBuf), 1, &g_pstKBBufHead, 1) != 0)
	{
		panic("cycle_buf_MemAttach() failed!");
	}
	if (cycle_buf_MemAttach(g_CharListBuf, sizeof(g_CharListBuf), 1, &g_pstCharList, 1) != 0)
	{
		panic("cycle_buf_MemAttach() failed!");
	}
}


static int keyboard_put_scan_code_to_buf(uint8_t c)
{
	char cFull = 0;

	if (cycle_buf_IsFull(&cFull, g_pstKBBufHead) != 0)
	{
		panic("cycle_buf_IsFull() failed!");
	}
	if (cFull)
	{
		//简单丢弃
		return 0;
	}
	if (cycle_buf_push(g_pstKBBufHead, &c) != 0)
	{
		panic("cycle_buf_PutScanCode() failed!");
	}
	return 0;
}
static int keyboard_get_scan_code_frm_buf(uint8_t * pc)
{
	char cEmpty = 0;

	if (cycle_buf_IsEmpty(&cEmpty, g_pstKBBufHead) != 0)
	{
		panic("cycle_buf_IsEmpty() failed!");
	}
	if (cEmpty)
	{
		return 0;
	}
	if (cycle_buf_pop(g_pstKBBufHead, pc) != 0)
	{
		panic("cycle_buf_GutScanCode() failed!");
	}
	return 1;
}
void keyboard_intr_handle()
{
	uint8_t scan_code ;

	scan_code = in_byte(KEYBOARD_DATA_PORT);
	keyboard_put_scan_code_to_buf(scan_code);


}
void keyboard_read(char * pc)
{
	char cEmpty = 0;

again:
	if (cycle_buf_IsEmpty(&cEmpty, g_pstCharList) != 0)
	{
		panic("cycle_buf_IsEmpty() failed!");
	}
	if (cEmpty)
	{
		sleep_on(&g_wait);
		goto again;
	}
	if (cycle_buf_pop(g_pstCharList, pc) != 0)
	{
		panic("cycle_buf_GutScanCode() failed!");
	}      
}



int keyboard_do_task()
{
	int i;
	int iret;
	uint8_t scan_code;
	uint8_t is_break_code;
	uint32_t tmp;
	const TKey * pKey = NULL;
	char chr;

	for (i = 0; i < 10; ++i)
	{
		_cli();
		iret = keyboard_get_scan_code_frm_buf(	&scan_code);
		if (iret != 1)
		{
			/* 没有扫描码需要处理了 */
			_sti();
			return 0;
		}
		_sti();
		is_break_code = (scan_code & 0x80);
		scan_code = scan_code & 0x7f;
		if (scan_code == 0 || scan_code >= (sizeof(keymap)/sizeof(TKey)) )
		{
			continue;
		}
		pKey = & (keymap[ scan_code ]);
		if (pKey->chr == SHIFT_L )
		{
			if (is_break_code) /* left shift up */
			{
				tmp = FLAG_SHIFT_L;
				tmp = ~tmp;
				flags = flags & tmp;
			}
			else /* left shift down */
			{
				flags = flags | FLAG_SHIFT_L;
			}
			continue;
		}
		if (pKey->chr == SHIFT_R )
		{
			if (is_break_code) /* right shift up */
			{
				tmp = FLAG_SHIFT_R;
				tmp = ~tmp;
				flags = flags & tmp;
			}
			else /* right shift down */
			{
				flags = flags | FLAG_SHIFT_R;
			}
			continue;
		}
		if (pKey->chr == ENTER && is_break_code == 0)
		{
			chr = '\n';
			print_chr(chr);
			_cli();
			cycle_buf_push(g_pstCharList, &chr);
			if (g_wait)
			{
				wake_up(&g_wait);
			}
			_sti();
			continue;	
		}
		if (pKey->chr == CTRL_L)
		{
			if (is_break_code) /* left ctrl up */
			{
				tmp = FLAG_CTRL_L;
				tmp = ~tmp;
				flags = flags & tmp;
			}
			else /* left ctrl down */
			{
				flags = flags | FLAG_CTRL_L;
			}
			continue;
		}
		if (pKey->chr == CTRL_R)
		{
			if (is_break_code) /* right ctrl up */
			{
				tmp = FLAG_CTRL_R;
				tmp = ~tmp;
				flags = flags & tmp;
			}
			else /* right ctrl down */
			{
				flags = flags | FLAG_CTRL_R;
			}
			continue;
		}

		if (!is_break_code)
		{
			static  int16_t line __attribute__ ((section ("data"))) = 0;
			if (flags & FLAG_CTRL_L || flags & FLAG_CTRL_R)
			{
				if (pKey->chr == 'f')
				{
					line += 5;
					if (line > 100) line = 100;
					//__asm__("cld;cld;cld");
					//printk("VVVVVV%d\n", line);
					tty_scroll_screen(line);
					continue;
				}
				else if (pKey->chr == 'b')
				{
					line -= 5;
					if (line < 0) line = 0;
					//printk("^^^^^^^^^%d\n", line);
					tty_scroll_screen(line);
					continue;
				}
			}
			if (flags & FLAG_SHIFT_L || flags & FLAG_SHIFT_R)
			{
				print_chr(pKey->schr);
				_cli();
				cycle_buf_push(g_pstCharList, &pKey->schr);
				wake_up(&g_wait);
				_sti();
			}
			else
			{
				print_chr(pKey->chr);
				_cli();
				cycle_buf_push(g_pstCharList, &pKey->chr);
				wake_up(&g_wait);
				_sti();
			}
		}
	}
	return 1;
}
