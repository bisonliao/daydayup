#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) ;
int my_img_convert(AVFrame* pFrameRGB, const AVFrame* pFrame, AVCodecContext *pCodecCtx);

int main(int argc, char **argv)
{
	av_register_all();

	FILE * audioOutFile = fopen("audio.pcm", "wb+");


	AVFormatContext *pFormatCtx = NULL;

	// Open video file
	if(avformat_open_input(&pFormatCtx, argv[1], NULL,  NULL)!=0)
	{
		return -1; // Couldn't open file
	}

	// Retrieve stream information
	if(avformat_find_stream_info(pFormatCtx, NULL)<0)
	{
		return -1; // Couldn't find stream information
	}
	
	int i;
	AVCodecContext *pCodecCtx;
	// Find the first video stream
	int videoStream=-1;
	int audioStream=-1;
	for(i=0; i<pFormatCtx->nb_streams; i++)
	{
		if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) 
		{
			videoStream=i;
		}
		if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_AUDIO) 
		{
			printf("find audio stream\n");
			audioStream=i;
		}

	}
	if(videoStream==-1)
	{
		fprintf(stderr, "can NOT find video codec\n");
		return -1; // Didn't find a video stream
	}
	if(audioStream==-1)
	{
		fprintf(stderr, "can NOT find audio codec\n");
		return -1; // Didn't find a video stream
	}
	// Get a pointer to the codec context for the video stream
	pCodecCtx=pFormatCtx->streams[videoStream]->codec;
	
	AVCodec *pCodec;
	// Find the decoder for the video stream
	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL) 
	{
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Open codec
	if(avcodec_open2(pCodecCtx, pCodec, NULL)<0)
	{
		return -1; // Could not open codec
	}
	//////////////////////////////////////////////////////////////
	// Get a pointer to the codec context for the audio stream
	AVCodecContext* pCodecCtxAudio=pFormatCtx->streams[audioStream]->codec;
	
	AVCodec *pCodecAudio;
	// Find the decoder for the video stream
	pCodecAudio=avcodec_find_decoder(pCodecCtxAudio->codec_id);
	if(pCodecAudio==NULL) 
	{
		fprintf(stderr, "Unsupported audio codec!\n");
		return -1; // Codec not found
	}
	// Open codec
	if(avcodec_open2(pCodecCtxAudio, pCodecAudio, NULL)<0)
	{
		return -1; // Could not open codec
	}
	printf("open audio codec successfully\n");

	AVFrame *pFrame, *pFrameRGB;
	// Allocate video frame
	pFrame=av_frame_alloc();

	// Allocate an AVFrame structure
	pFrameRGB=av_frame_alloc();
	if(pFrameRGB==NULL)
	{
		return -1;
	}

	uint8_t *buffer;
	int numBytes;
	// Determine required buffer size and allocate buffer
	numBytes=avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width,
		pCodecCtx->height);
	buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_RGB24,
		pCodecCtx->width, pCodecCtx->height);

	int frameFinished;
	AVPacket packet;
	i=0;
	while(av_read_frame(pFormatCtx, &packet)>=0) 
	{
			// Is this a packet from the video stream?
			if(packet.stream_index==videoStream) 
			{
					// // Decode video frame
					avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
					// // Did we get a video frame?
					if(frameFinished) 
					{
							// // Convert the image from its native format to RGB
							my_img_convert(pFrameRGB, pFrame, pCodecCtx);
							i++;	
							//if(++i<=5)
							{
									SaveFrame(pFrameRGB, pCodecCtx->width,
													pCodecCtx->height, i);
							}
					}
			}
			if (packet.stream_index==audioStream)
			{
				printf("find audio packet, size=%d\n", packet.size);
				while(packet.size>0)
				{    
					int out_size;
					int len = avcodec_decode_audio4(pCodecCtxAudio, pFrame, &out_size, &packet);
					if (len < 0) 
					{
						printf("Error while decoding\n");
						continue;
					}
					if (out_size ) 
					{
						int data_size1 = av_samples_get_buffer_size(NULL, pCodecCtxAudio->channels, 
									pFrame->nb_samples, pCodecCtxAudio->sample_fmt, 1);    
						fwrite(pFrame->data[0], 1, pFrame->linesize[0], audioOutFile);
						fflush(audioOutFile);
					}
					packet.size -=len;
					packet.data +=len;
				} 
			}
			av_free_packet(&packet);
	}
	fclose(audioOutFile);
	// Free the RGB image
	av_free(buffer);
	av_free(pFrameRGB);
	// // Free the YUV frame
	av_free(pFrame);
	// // Close the codec
	avcodec_close(pCodecCtx);
	// // Close the video file
	avformat_close_input(&pFormatCtx);
	return 0;

}
//int my_img_convert(AVPicture* pFrameRGB, AVPicture* pFrame, AVCodecContext *pCodecCtx)
int my_img_convert(AVFrame* pFrameRGB, const AVFrame* pFrame, AVCodecContext *pCodecCtx)
{
	struct SwsContext *img_convert_ctx;

 	img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,  
				pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
    	if(img_convert_ctx == NULL)
   	{
        	fprintf(stderr, "Cannot initialize the conversion context!\n");
        	return -1;
    	}

	sws_scale(img_convert_ctx,
                 pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
                 pFrameRGB->data, pFrameRGB->linesize);
	return 0;
}

void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) 
{
	return;
		FILE *pFile;
		char szFilename[32];
		int y;
		// Open file
		sprintf(szFilename, "frame%d.ppm", iFrame);
		pFile=fopen(szFilename, "wb");
		if(pFile==NULL)
				return;
		// Write header
		fprintf(pFile, "P6\n%d %d\n255\n", width, height);
		// // Write pixel data
		for(y=0; y<height; y++)
		{
			fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);
		}
		// // Close file
		fclose(pFile);
}
