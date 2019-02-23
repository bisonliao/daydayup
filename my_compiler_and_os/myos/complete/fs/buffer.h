#ifndef _BUFFER_H_INCLUDED_
#define _BUFFER_H_INCLUDED_

#include "types.h"

typedef struct
{
	uint32_t    abs_sect;
	uint8_t     cmd;    /* write  / read */
	uint16_t    pid;
	uint8_t     flags;   /* BUFFER_DIRTY, BUFFER_UPDATE etc. */

	uint8_t     data[512];
} TBuffer;

typedef struct
{
	uint32_t 	abs_sect;
	uint8_t 	cmd;
	uint16_t	pid;
} TBufferRequest;

typedef struct
{
	char *		file_name;
	uint8_t		file_name_len;
	uint8_t		cmd;
	uint16_t	pid;
	uint32_t	offset;

	unsigned char * buffer;
	uint16_t	buffer_size;
} TFileRequest;

void buffer_init(uint32_t buffer_start, uint32_t buffer_size);


#endif
