#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <lame/lame.h>


#pragma pack(1)
typedef struct
{
	char magic[5];
	uint8_t channel_num;
	uint16_t sample_rate;
	uint32_t frame_size;
	uint16_t step_size[2];
	int16_t first_sample[2];
} frame_head_t;

#pragma pack()

typedef struct
{
	uint32_t sample_rate;
	uint8_t channel_num;
	uint32_t pcm_len;
	short pcm_l[3*60*44100];
	short pcm_r[3*60*44100];
} pcm_data_t;

static pcm_data_t g_pcm_data;



int DM_get_frame_size(uint32_t sample_rate, uint8_t channel_num);
int DM_encode_frame(uint32_t sample_rate, uint8_t channel_num,
				short pcm_l[], short pcm_r[], unsigned char * frame_buffer);
int comp_float_by_abs(const void * a, const void * b);

int DM_decode(const char * filename, uint32_t *sample_rate, uint8_t *channel_num,
		short pcm_l[], short pcm_r[], uint32_t pcm_len)
{
	if (filename == NULL || sample_rate == NULL || channel_num == NULL
		||pcm_l == NULL || pcm_r == NULL)
	{
		return -1;
	}
	int fd = open(filename, O_RDONLY);
	if (fd < 0)
	{
		return -2;
	}

	int first_flag = 1;



	int frame_nm = 0;
	while(1)
	{
		short pcm1[100*1000];
		short pcm2[100*1000];
		int iRet = DM_decode_frame(fd, sample_rate, channel_num, pcm1, pcm2);
		if (iRet < 0)
		{
			printf("decode frame failed:%d\n", iRet);
			break;
		}
		if (iRet == 0)
		{
			break;
		}
		if ((1+frame_nm)*(*sample_rate) > pcm_len)
		{
			break;
		}
		memcpy(pcm_l+frame_nm*(*sample_rate), pcm1, *sample_rate*sizeof(short));
		memcpy(pcm_r+frame_nm*(*sample_rate), pcm2, *sample_rate*sizeof(short));
		frame_nm++;
	}
	close(fd);
	return frame_nm*(*sample_rate);

}
int DM_decode_frame(int fd, uint32_t *sample_rate, uint8_t *channel_num, short pcm_l[], short pcm_r[])
{
	frame_head_t head;
	static int tt = 0;
	tt++;

		int len = read(fd, &head, sizeof(head)) ;
		if (len == 0)
		{
			return 0;
		}
		if (len != sizeof(head))
		{
			printf("%s:%d\n", __FILE__, __LINE__);
			return -1;
		}
		if (sample_rate)
		{
			*sample_rate = head.sample_rate;
		}
		if (channel_num)
		{
			*channel_num = head.channel_num;
		}
		int data_len = head.frame_size - sizeof(head);
		if (data_len == 0)
		{
			return 0;
		}
		if (data_len < 0 || data_len != (head.sample_rate*head.channel_num) )
		{
			printf("%s:%d\n", __FILE__, __LINE__);
			return -1;
		}
		if (tt == 44) { printf(">>stepsize=%d\n", head.step_size[0]);}

		unsigned char buf[data_len];
		len = read(fd, buf, data_len);
		if (len != data_len)
		{
			printf("%s:%d\n", __FILE__, __LINE__);
			return -1;
		}

		int i, j;
		int offset = 0;	
		
		for (j = 0; j < head.channel_num; ++j)
		{
			short * pcm = pcm_l;
			if (j == 1) { pcm = pcm_r;}

			int prev = head.first_sample[j];

			for (i = 0; i < head.sample_rate; ++i)
			{

				 int val, neg=0;
				 val = *(char*)(buf+offset);
				 offset++;

				 if (val < 0) { val = -val; neg =1 ;}
				 val =  val*256;

				pcm[i] = val;
				if (neg) { pcm[i] = -pcm[i];}
			}
		}

	return 1;

}

int DM_encode(uint32_t sample_rate, uint8_t channel_num, 
			 const short pcm_l[], const short pcm_r[], uint32_t pcm_len,
			 const char * filename)
{
// every 1 second samples as a frame
// frame : header, ch_l, ch_r 
// ch_l/ch_r: 16bit S1, 4bit delta1, 4bit delta2,...
// delta is normal distribute
	
	if ( filename == NULL || (sample_rate % 2) != 0 
		|| channel_num ==0 || channel_num > 2)
	{
		return -1;
	}
	if (pcm_l == NULL || channel_num > 1 && pcm_r == NULL)
	{
		return -1;
	}

	int frame_num = pcm_len / sample_rate;
	int i;
	int frame_size = DM_get_frame_size(sample_rate, channel_num);
	unsigned char buffer[frame_size+1024];

	int fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC);
	if (fd < 0)
	{
		return -2;
	}

	for (i = 0; i < frame_num; ++i)
	{
		int frame_size = DM_encode_frame(sample_rate, channel_num,
			(pcm_l + i*sample_rate),
			(pcm_r + i*sample_rate),
			buffer);
		if (frame_size < 0)
		{
			close(fd);
			return -1;
		}
		write(fd, buffer, frame_size);

	}
	if ( pcm_len > frame_num * sample_rate)
	{
		short left[sample_rate];
		short right[sample_rate];
		memset(left, 0, sizeof(left));
		memset(right, 0, sizeof(right));

		int len = pcm_len - frame_num * sample_rate;
		memcpy(left,  (pcm_l+frame_num * sample_rate), len*sizeof(short) );
		memcpy(right,  (pcm_r+frame_num * sample_rate), len*sizeof(short) );

		int frame_size = DM_encode_frame(sample_rate, channel_num,
			left, right,
			buffer);
		if (frame_size < 0)
		{
			close(fd);
			return -1;
		}
		write(fd, buffer, frame_size);

	}
	close(fd);
	
	return 0;
}

int DM_get_frame_size(uint32_t sample_rate, uint8_t channel_num)
{
	return sizeof(frame_head_t) + (sample_rate) * channel_num;
}


int DM_encode_frame(uint32_t sample_rate, uint8_t channel_num,
				short pcm_l[], short pcm_r[], unsigned char * frame_buffer)
{
	uint16_t step_size;
	frame_head_t *head = (frame_head_t *)frame_buffer;

	memset(head, 0, sizeof(frame_head_t) );

	head->channel_num = channel_num;
	head->sample_rate = sample_rate;
	memcpy(head->magic, "bison", 5);


	if (pcm_l != NULL)
	{
		analyze_distribute(pcm_l, sample_rate, &step_size);	
		head->step_size[0] = step_size;
		head->first_sample[0] = pcm_l[0];
		//printf("step size left:%u\n", head->step_size[0]);
	}
	if (pcm_r != NULL)
	{
		analyze_distribute(pcm_r, sample_rate, &step_size);	
		head->step_size[1] = step_size;
		head->first_sample[1] = pcm_r[0];
		//printf("step size right:%u\n", head->step_size[1]);
	}
	int16_t * pcm[2];
	pcm[0] = pcm_l;
	pcm[1] = pcm_r;
	int8_t * codes = (int8_t*)(frame_buffer+sizeof(frame_head_t));

	int i, j, k;
	int offset = 0;
	for (i = 0; i < 2; ++i) // two channels
	{
		//printf("--------------------\n");
		if (pcm[i] == NULL) { break;}
		int16_t prev = pcm[i][0];

		int overflow = 0;

		for (j = 0; j < sample_rate; j +=1)  // encode one channel samples
		{

			// encode 2 sample once

			int8_t * code = codes+offset;
			offset++;

			int neg = 0;
			int val = pcm[i][j];
			if (val < 0) { val = -val; neg = 1;}

			int delta1 = round( val/256.0);
			if (neg) { delta1 = -delta1;}

			*code = delta1;
		}
		printf("overflow:%d, %f, step_size:%d\n", 
			overflow, overflow/(double)sample_rate,
			head->step_size[i]);
	}
	printf("offset=%d\n", offset);
	head->frame_size = sizeof(frame_head_t) + offset;
	return head->frame_size;
}

int comp_float_by_abs(const void * a, const void * b)
{
	short aabs = abs( *(short*)a );
	short babs = abs( *(short*)b );
	if (aabs < babs ) { return 1;}
	if (aabs > babs ) { return -1;}
	return 0;
}

int alaw(int x)
{
	int s = 0;
	if (x < 0) { x = -x; s = 1;}

	uint32_t mask = 0xFFE0;
	uint32_t result = 0x20;
	uint32_t mask2 = 0x1e;


	int i;
	for (i = 0; i < 8; i++)
	{
		if ( (x & mask) == result || (x & mask) == 0)
		{
			int ret = ((x&mask2) >> (i+1)) |  (i<<4);
			if (s)
			{
				ret = -ret;
			}
			return ret;

		}
		mask = (mask << 1);
		result = (result <<1);
		mask2 = (mask2 << 1);
	}
	printf("%s:x=0x%x, exit!\n", __FUNCTION__, x);
	exit(-1);
}
int ialaw(int x)
{
	int s = 0;
	if (x < 0) { x = -x; s = 1;}

	int i;
	uint32_t tail = 1;
	uint32_t head = 32;

	for (i = 0; i < 8; ++i)
	{
		if ( ((x & 0x70) >> 4) == i)
		{
			int ret = ((x & 0xf) << (i+1)) + tail + head;
			if (s) { ret = -ret;}
			return ret;
		}
		tail *=2;
		head *=2;
	}
	printf("%s:x=0x%x, exit!\n", __FUNCTION__, x);
	exit(-1);
}

int analyze_distribute(const short pcm[], uint32_t pcm_len, uint16_t * step_size)
{
	int i;
	short delta[pcm_len];
	

	delta[0] = 0;
	double sum = abs(pcm[0]);
	int count = 1;
	for (i = 1; i < pcm_len; ++i)
	{
		delta[i] = pcm[i] - pcm[i-1];
		if (pcm[i] != 0)
		{
			sum += abs(pcm[i]);
			count++;
		}
	}
	double avg = sum / count;
	if (avg < 0.0000001) { avg = 1.0;}

	qsort(delta, pcm_len, sizeof(short), comp_float_by_abs);
	
/*
	printf("vvv:");
	for (i = 0; i < 12; ++i)
	{
		printf("%d ", delta[i]);
	}
	for (i = 12; i > 0; --i)
	{
		printf("%d ", delta[pcm_len-1-i]);
	}
	printf("\n");
*/


	int outnum = (pcm_len) * 0.05;
	
	short cut = delta[outnum];

	
	
	*step_size = round((abs(cut)) / 127.0);
	/*
	if (*step_size > avg*0.01)
	{
		*step_size = avg*0.01;
	}
	*/
	if (*step_size == 0)
	{
		*step_size = 1;
	}
	printf("cut:%d, avg_abs:%.2f, step_size:%d, step_size/avg_abs:%.2f\%\n", 
		abs(cut), avg, *step_size, *step_size * 100/ avg);
	return 0;
}

int load_pcm_from_mp3(const char * filename)
{
	hip_t hip=NULL ;
	int fd = -1;
	int iRet = -1;

	hip = hip_decode_init();

	fd = open(filename, O_RDONLY);
	if (fd < 0)
	{
		perror("open:");
		goto clean;
	}
	unsigned char buffer[1024*20];
	short pcm_l[1024*400];
	short pcm_r[1024*400];
	int len;
	mp3data_struct mp3header;

	int samplePrinted = 0;
	int i;


	int flag = 1;

	g_pcm_data.pcm_len = 0;
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
		if ( (g_pcm_data.pcm_len+len) >= sizeof(g_pcm_data.pcm_l)/sizeof(short))
		{
			break;
		}
		memcpy(g_pcm_data.pcm_l+g_pcm_data.pcm_len, pcm_l, len*sizeof(short));
		memcpy(g_pcm_data.pcm_r+g_pcm_data.pcm_len, pcm_r, len*sizeof(short));
		g_pcm_data.pcm_len += len;

		if (flag && mp3header.header_parsed)
		{
			flag = 0;

			g_pcm_data.sample_rate = mp3header.samplerate;
			g_pcm_data.channel_num = mp3header.stereo;
		}
		
	}

	iRet = 0;
clean:
	if (hip) { hip_decode_exit(hip);}
	if (fd >=0 ) { close(fd);}
	return iRet;
}
int save_pcm_to_mp3(const char * m)
{
	int  fdout = -1;
	lame_global_flags * pLame = NULL;
	int iRet = 0;


	fdout = open(m, O_WRONLY|O_CREAT|O_TRUNC);
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
		iRet = -2;
		goto clean;
	}
	lame_set_num_samples(pLame, g_pcm_data.pcm_len);
	lame_set_in_samplerate(pLame, g_pcm_data.sample_rate);
	lame_set_num_channels(pLame, g_pcm_data.channel_num);
	lame_set_quality(pLame, 5);
	lame_init_params(pLame);

	unsigned char buffer2[1024*100];

	int offset = 0;
	const int segsize = 1152;

	while (offset < g_pcm_data.pcm_len)
	{
		int inputsize = segsize;
		if (inputsize+offset > g_pcm_data.pcm_len)
		{
			inputsize = g_pcm_data.pcm_len - offset;	
		}
		int len = lame_encode_buffer(pLame,
			g_pcm_data.pcm_l+offset,
			g_pcm_data.pcm_r+offset,
			inputsize,
			buffer2,
			sizeof(buffer2));
		if (len < 0)
		{
			fprintf(stderr, "lame_encode_buffer_interleaved()  failed! %d\n",
					len);
			iRet = -3;
			goto clean;
		}

		write(fdout, buffer2, len);
		offset += inputsize;
	}
	int len = lame_encode_flush(pLame, buffer2, sizeof(buffer2));
	if (len < 0)
	{
		fprintf(stderr, "lame_encode_flush() failed! %d\n", len);
		iRet = -4;
		goto clean;
	}
	write(fdout, buffer2, len);
	iRet = 0;

clean:
	if (fdout >= 0) {close(fdout);}
	if (pLame != NULL) {lame_close(pLame);}

	return iRet;

}
void debug_info()
{
	return;

	int start= 43*44100;
	int i;
	for (i = 0; i < 30; ++i)
	{
		printf(">>%d\n", g_pcm_data.pcm_l[i+start]);
	}
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("usage:%s sourcefile.mp3  destfile.mp3\n",  argv[0]);
		printf("%s will:\n", argv[0]);
		printf("step1:load pcm from source file\n");
		printf("step2:encode pcm use my dpcm codec to bison.data\n");
		printf("step3:decode pcm use my dpcm codec from bison.data\n");
		printf("step4:save pcm to destfile.mp3\n");
		return 0;
	}

	{
		// load pcm from mp3
		g_pcm_data.pcm_len = 0;
		g_pcm_data.sample_rate = 0;
		g_pcm_data.channel_num = 0;
		int iRet = load_pcm_from_mp3(argv[1]);
		if (iRet  < 0)
		{
			printf("load_pcm_from_mp3 return %d\n", iRet);
			return -1;
		}
		printf("load %d pcm from mp3, sample rate:%d, channel num:%d\n",
				g_pcm_data.pcm_len,
				g_pcm_data.sample_rate,
				g_pcm_data.channel_num);
		
		// encode pcm, save them in bison.data
		int iret = DM_encode(g_pcm_data.sample_rate, g_pcm_data.channel_num,
			g_pcm_data.pcm_l, g_pcm_data.pcm_r, g_pcm_data.pcm_len,
			"./bison.data");
		printf("DM_encode return %d\n", iRet);

		// decode to pcm, save as mp3
		g_pcm_data.pcm_len = 0;
		g_pcm_data.sample_rate = 0;
		g_pcm_data.channel_num = 0;
		iRet = DM_decode("./bison.data", &g_pcm_data.sample_rate, &g_pcm_data.channel_num,
			g_pcm_data.pcm_l, g_pcm_data.pcm_r, sizeof(g_pcm_data.pcm_l)/sizeof(short));
		printf("DM_decode return %d\n", iRet);
		if (iRet > 0)
		{
			g_pcm_data.pcm_len = iRet;
		}

		// save to mp3 file
		iRet = save_pcm_to_mp3(argv[2]);
		printf("save_pcm_to_mp3 return %d\n", iRet);

	}

	return 0;
}
