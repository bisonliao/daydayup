#include "tty.h"
#include "global.h"
#include "const_def.h"

void tty_scroll_screen(int16_t line)
{
	int total_line;

	if (line < 0)
	{
		line = 0;
	}

	total_line = (GS_END  - GS_START) / (SCR_WIDTH*2);
	if (line + SCR_HEIGHT > total_line)
	{
		line = total_line - SCR_HEIGHT;
	}

	out_byte( CRTC_ADDR_REG,
				CRTC_DATA_IDX_START_ADDR_H);
	out_byte( CRTC_DATA_REG,
			 ((SCR_WIDTH*line*2) >> 8) & 0xff);
	out_byte( CRTC_ADDR_REG,
				CRTC_DATA_IDX_START_ADDR_L);
	out_byte( CRTC_DATA_REG,
			 (SCR_WIDTH*line*2)  & 0xff);
}
