/**
 * This code shows how to use lame library to encode mp3 and decode mp3
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <lame/lame.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#pragma pack(1)
typedef struct 
{
	char   			riffType[4];	
	unsigned int   	riffSize;		
	char   			waveType[4];	
	char   			formatType[4];	
	unsigned int	formatSize;		
	unsigned short	compressionCode;
	unsigned short 	numChannels;	
	unsigned int   	sampleRate;		
	unsigned int   	bytesPerSecond;	
	unsigned short 	blockAlign;		
	unsigned short 	bitsPerSample;	
	char   			dataType[4];	
	unsigned int   	dataSize;		
}WAV_HEADER;

#pragma pack()

int is_little_endian()
{
	short a= 1;

	if ( *((char*)&a) == 1)
	{
		return 1;
	}
	else 
	{
		return 0;
	}
}
// check and print information in wave header
int check_wave(const WAV_HEADER *header)
{
	if (header == NULL) { return -1;}
	if (is_little_endian() == 0)
	{
		fprintf(stderr, "Not little endian\n");
		return -1;
	}
	if (memcmp(header->riffType, "RIFF", 4) != 0)
	{
		fprintf(stderr, "RIFF flag error\n");
		return -1;
	}
	if (memcmp(header->waveType, "WAVE", 4) != 0)
	{
		fprintf(stderr, "wave flag error\n");
		return -1;
	}
	if (memcmp(header->formatType,"fmt ", 4) != 0)
	{
		fprintf(stderr, "FMT flag error\n");
		return -1;
	}
	if (header->compressionCode != 1)
	{
		fprintf(stderr, "Not PCM data\n");
		return -1;
	}
	printf("channel num:%u\n", header->numChannels);
	printf("sample rate:%u\n", header->sampleRate);
	printf("bytes per second:%u\n", header->bytesPerSecond);
	printf("sample depth:%u\n", header->bitsPerSample);

	if (header->bitsPerSample != 16 ||
		header->numChannels != 2 ||
		header->sampleRate != 44100)
	{
		fprintf(stderr, "wav params dismatch!\n");
		return -1;
	}
	return 0;
}
// print information in mp3 header
int print_mp3_header(mp3data_struct * p)
{
	if (p == NULL ) return -1;
	if (!p->header_parsed)
	{
		printf("header has not been parsed!\n");
		return 0;
	}
	printf("channel:%d, sample rate:%d, framesize:%d\n",
		p->stereo,
		p->samplerate,
		p->framesize);
	return 0;

}
// decode a mp3 file
int decode(const char * m)
{
	hip_t hip=NULL ;
	int fd = -1;
	int iRet = -1;

	hip = hip_decode_init();

	fd = open(m, O_RDONLY);
	if (fd < 0)
	{
		perror("open:");
		goto clean;
	}
	unsigned char buffer[1024*20];
	short pcm_l[1024*200];
	short pcm_r[1024*200];
	int len;
	mp3data_struct mp3header;


	while (1)
	{
		len = read(fd, buffer, sizeof(buffer));
		
		if (len < 0)
		{
			perror("read:");
			goto clean;
		}
		if (len == 0)
		{
			break;
		}
		len = hip_decode_headers(hip,
			buffer,
			len, 
			pcm_l,
			pcm_r,
			&mp3header);
		if (len < 0)
		{
			fprintf(stderr, "hip_decode() failed! %d\n",
				len);
			goto clean;

		}
		printf("I got %d pcm data\n", len);
		print_mp3_header(&mp3header);
	}

	iRet = 0;
clean:
	if (hip) { hip_decode_exit(hip);}
	if (fd >=0 ) { close(fd);}
	return iRet;
}
// convert wav file to mp3
int wav2mp3(const char * w, const char * m)
{
	int fdin = -1, fdout = -1;
	lame_global_flags * pLame = NULL;
	int iRet = 0;

	fdin = open(w, O_RDONLY);
	if (fdin < 0)
	{
		perror("open:");
		iRet = -1;
		goto clean;
	}
	off_t length = lseek(fdin, 0, SEEK_END);
	if (length < 0)
	{
		perror("lseek:");
		iRet = -1;
		goto clean;
	}
	unsigned long sample_num  = (length - 44) / sizeof(uint16_t) ;
	printf("sample number:%lu\n", sample_num);


	lseek(fdin, 0, SEEK_SET);
	WAV_HEADER header;
	read(fdin, &header, sizeof(header));
	if (check_wave(&header) != 0)
	{
		iRet = -1;
		goto clean;
	}


	fdout = open(m, O_WRONLY|O_CREAT);
	if (fdout < 0)
	{
		perror("open:");
		iRet = -1;
		goto clean;
	}

	pLame = lame_init();
	if (pLame == NULL)
	{
		fprintf(stderr, "lame_init() failed!\n");
		iRet = -1;
		goto clean;
	}
	lame_set_num_samples(pLame, sample_num);
	lame_set_in_samplerate(pLame, 44100);
	lame_set_num_channels(pLame, 2);
	lame_set_quality(pLame, 5);
	lame_init_params(pLame);

	unsigned char buffer[10240*2];
	unsigned char buffer2[1024*100];
	int len;

	while (1)
	{
		len = read(fdin, buffer, sizeof(buffer));
		if (len < 0)
		{
			perror("read:");
			iRet = -1;
			goto clean;
		}
		if (len == 0)
		{
			break;
		}
		if ( len % 4) 
		{
			fprintf(stderr, "pcm sample bytes error!\n");
			iRet = -1;
			goto clean;
		}

		len = lame_encode_buffer_interleaved(pLame,
			(short*)buffer,
			len / 4, 
			buffer2,
			sizeof(buffer2));
		if (len < 0)
		{
			fprintf(stderr, "lame_encode_buffer_interleaved()  failed! %d\n",
					len);
			iRet = -1;
			goto clean;
		}
		write(fdout, buffer2, len);
	}
	len = lame_encode_flush(pLame, buffer2, sizeof(buffer2));
	if (len < 0)
	{
		fprintf(stderr, "lame_encode_flush() failed! %d\n", len);
		iRet = -1;
		goto clean;
	}
	write(fdout, buffer2, len);
	iRet = 0;

clean:
	if (fdin >= 0) {close(fdin);}
	if (fdout >= 0) {close(fdout);}
	if (pLame != NULL) {lame_close(pLame);}

	return iRet;

}

int main(int argc, char ** argv)
{
	int fdin, fdout;
	const char * wavfile = NULL, *mp3file = NULL;

	if (argc < 3)
	{
		printf("usage:%s wav2mp3 wavfile mp3file\n", argv[0]);
		printf("or   :%s dec  mp3file\n", argv[0]);
		return -1;
	}
	if ( strcmp(argv[1], "wav2mp3") == 0)
	{
		wavfile = argv[2];
		mp3file = argv[3];
		return wav2mp3(wavfile, mp3file);
	}
	else if (strcmp(argv[1], "dec") == 0)
	{
		
		return decode(argv[2]);
	}
	else 
	{
		printf("I do NOT know!\n");
	}
}
