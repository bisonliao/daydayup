#ifndef _TTY_H_INCLUDED_
#define _TTY_H_INCLUDED_

#include "struct.h"
#include "const_def.h"


/*滚动屏幕，使得显存第line行位于屏幕第一行*/
void tty_scroll_screen(int16_t line);


#endif
