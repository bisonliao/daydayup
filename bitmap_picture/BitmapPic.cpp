#include "BitmapPic.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

using namespace std;

BitmapPic::BitmapPic(void)
{
}


BitmapPic::~BitmapPic(void)
{
}

static int32_t power(int base, int times)
{
	int res = 1;
	int i;
	for (i = 0; i < times; ++i)
	{
		res = res * base;
	}
	return res;
}
void BitmapPic::modify_headers()
{
	uint32_t width  = (infoHeader.biBitCount * infoHeader.biWidth + 31) / 32 * 4;
	infoHeader.biBitCount = 24;
	infoHeader.biClrUsed = 0;
	fileHeader.bfOffBits = sizeof(fileHeader) + sizeof(infoHeader);
	fileHeader.bfSize = sizeof(fileHeader) + sizeof(infoHeader) + 3 * width * infoHeader.biHeight;
}

int BitmapPic::load(const char * filename)
{
	if (filename == NULL) { return 1;}

	FILE * fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		perror("failed to open file");
		return -1;
	}
	
	if (fread(&fileHeader, 1, sizeof(fileHeader),  fp)!= sizeof(fileHeader))
	{
		perror("fread bitmap file header faile");
		fclose(fp);
		return -1;
	}
	if (fileHeader.bfType != 0x4d42)
	{
		fprintf(stderr, "%s is not a bitmap file\n", filename);
		fclose(fp);
		return -1;
	}
	
	if (fread(&infoHeader, 1, sizeof(infoHeader), fp) != sizeof(infoHeader))
	{
		perror("fread information header faile");
		fclose(fp);
		return -1;
	}
	if (infoHeader.biSize > sizeof(infoHeader))
	{
		printf("more bytes in info header, jump it...\n");
		fseek(fp, (infoHeader.biSize-sizeof(infoHeader)), SEEK_CUR);
	}
	printf("size:%u * %u\n", infoHeader.biWidth, infoHeader.biHeight);
	printf("bits per pixel:%u\n", infoHeader.biBitCount);
	if (infoHeader.biCompression != BI_RGB)
	{
		printf("bitmap is compressed! size=%u, I can NOT analyze it.\n", infoHeader.biSize);
		fclose(fp);
		return -1;
	}
	uint32_t uiColorUsed = infoHeader.biClrUsed;
	uint32_t uiPaletteSize = 0;
	if (uiColorUsed != 0)
	{
		uiPaletteSize = sizeof(RGBQUAD)*uiColorUsed;
		printf("used %u colors, palette size:%u\n", uiColorUsed, uiPaletteSize);
	}
	else
	{
		uiPaletteSize = power(2, infoHeader.biBitCount) * sizeof(RGBQUAD);
	}
	
	
	
	
	long currentPos = ftell(fp);
	if (currentPos < 0)
	{
		perror("ftell() failed!");
		fclose(fp);
		return -1;
	}
	if (fileHeader.bfOffBits < currentPos)
	{
		fprintf(stderr, "offset mismatch current position!\n");
		fclose(fp);
		return -1;
	}
	printf("读完两个header后，文件偏移是0x%x\n", currentPos);
	vector<RGBQUAD> palettes;
	//带有调色板
	if (fileHeader.bfOffBits > currentPos)
	{
		if (fileHeader.bfOffBits - currentPos != uiPaletteSize)
		{
			fprintf(stderr, "palette size mismatch: %u, %u!\n",
				  fileHeader.bfOffBits - currentPos, 
				  uiPaletteSize);
			fclose(fp);
			return -1;
		}
		printf("palette size:%u\n", fileHeader.bfOffBits - currentPos);
		uint32_t quadNum = uiPaletteSize / sizeof(RGBQUAD);
		uint32_t i;
		for (i = 0; i < quadNum; ++i)
		{
			RGBQUAD quad;
			if (fread(&quad, 1, sizeof(quad), fp) != sizeof(quad))
			{
				perror("fread quad failed!");
				fclose(fp);
				return -1;
			}
			//printf("Quad#%u: %02x,%02x,%02x\n", i, quad.rgbRed, quad.rgbGreen, quad.rgbBlue);
			palettes.push_back(quad);
		}
	}
	//继续读取位图数据
	pixColors.clear();
	uint32_t width =  (infoHeader.biBitCount * infoHeader.biWidth + 31) / 32 * 4;
	int i, j;
	if (infoHeader.biBitCount == 16)
	{
		for (i = 0; i < infoHeader.biHeight; ++i)
		{
			vector<PixColor> line;
			for (j = 0; j < width; j+=2)
			{
				uint8_t R,G,B;
				unsigned char v[2];
				if (fread(v, 1, 2, fp) != 2)
				{
					perror("read pixel data failed");
					fclose(fp);
					return -1;
				}
			
				B = (v[0] & 31 );
				G = (v[0] >> 5) | ((v[1] & 3)<<3);
				R = (v[1] & 127) >> 2;

				
 
				PixColor pc;
				pc.B = B << 3;
				pc.G = G  << 3;
				pc.R = R << 3 ;
				if (line.size() < infoHeader.biWidth)
				{
					line.push_back(pc);
				}
		
			}
			//因为位图是从下到上，从左到右的，所以这么恶心
			vector<PixColor>::reverse_iterator it;
			for (it = line.rbegin(); it != line.rend(); ++it)
			{
				pixColors.push_front(*it);
			}
		}
		modify_headers();
	}
	else if (infoHeader.biBitCount == 24)//真彩色
	{
		for (i = 0; i < infoHeader.biHeight; ++i)
		{
			vector<PixColor> line;
			for (j = 0; j < width; j+=3)
			{
				uint8_t R,G,B;
				unsigned char v[3];
				if (fread(v, 1, 3, fp) != 3)
				{
					perror("read pixel data failed");
					fclose(fp);
					return -1;
				}

				PixColor pc;
				pc.B = v[0];
				pc.G = v[1];
				pc.R = v[2] ;
				if (line.size() < infoHeader.biWidth)
				{
					line.push_back(pc);
				}
				
			}
			//因为位图是从下到上，从左到右的，所以这么恶心
			vector<PixColor>::reverse_iterator it;
			for (it = line.rbegin(); it != line.rend(); ++it)
			{
				pixColors.push_front(*it);
			}
		}
	}
	else if (infoHeader.biBitCount == 1)//黑白二色
	{
		if (palettes.size() != 2)
		{
			printf("2色图，调色板的大小应该为2才对!\n");
			fclose(fp);
			return -1;
		}
		
		for (i = 0; i < infoHeader.biHeight; ++i)
		{
			vector<PixColor> line;
			for (j = 0; j < width; j++)
			{
				
				unsigned char v[1];
				if (fread(v, 1, 1, fp) != 1)
				{
					perror("read pixel data failed");
					fclose(fp);
					return -1;
				}
				int  k;
				
				for (k = 7; k >=0 ; --k)
				{
					
					PixColor pc;
					if (v[0] & (1<<k))
					{
						pc.B = palettes[1].rgbBlue;
						pc.G = palettes[1].rgbGreen;
						pc.R = palettes[1].rgbRed;
					}
					else
					{
						pc.B = palettes[0].rgbBlue;
						pc.G = palettes[0].rgbGreen;
						pc.R = palettes[0].rgbRed;
					}
					if (line.size() < infoHeader.biWidth)
					{
						line.push_back(pc);
					}
					
					
				}
				
				
				
			}
			
			//因为位图是从下到上，从左到右的，所以这么恶心
			vector<PixColor>::reverse_iterator it;
			for (it = line.rbegin(); it != line.rend(); ++it)
			{
				pixColors.push_front(*it);
			}
			
			
		}
		modify_headers();
	}
	else if (infoHeader.biBitCount == 4)//16色图，用调色板
	{
		if (palettes.size() != 16)
		{
			printf("16色图，调色板的大小应该为16才对!\n");
			fclose(fp);
			return -1;
		}
		for (i = 0; i < infoHeader.biHeight; ++i)
		{
			vector<PixColor> line;
			for (j = 0; j < width; j++ )
			{
				
				unsigned char v[1];
				if (fread(v, 1, 1, fp) != 1)
				{
					perror("read pixel data failed");
					fclose(fp);
					return -1;
				}
				PixColor pc;
				uint32_t index;
				
				index = v[0]>> 4;

				pc.B = palettes[index].rgbBlue;
				pc.G = palettes[index].rgbGreen;
				pc.R = palettes[index].rgbRed;
				if (line.size() < infoHeader.biWidth)
				{
					line.push_back(pc);
				}
			

				index = v[0] & 0x0f;
				pc.B = palettes[index].rgbBlue;
				pc.G = palettes[index].rgbGreen;
				pc.R = palettes[index].rgbRed;
				if (line.size() < infoHeader.biWidth)
				{
					line.push_back(pc);
				}
			

				
			}
			//因为位图是从下到上，从左到右的，所以这么恶心
			vector<PixColor>::reverse_iterator it;
			for (it = line.rbegin(); it != line.rend(); ++it)
			{
				pixColors.push_front(*it);
			}
			
			
		}
		modify_headers();
	}
	else if (infoHeader.biBitCount == 8)//256色图，用调色板
	{
		if (palettes.size() != 256)
		{
			printf("256色图，调色板的大小应该为256才对!\n");
			fclose(fp);
			return -1;
		}
		for (i = 0; i < infoHeader.biHeight; ++i)
		{
			vector<PixColor> line;
			for (j = 0; j < width;j++ )
			{
				unsigned char v[1];
				if (fread(v, 1, 1, fp) != 1)
				{
					perror("read pixel data failed");
					fclose(fp);
					return -1;
				}
				PixColor pc;

				pc.B = palettes[v[0]].rgbBlue;
				pc.G = palettes[v[0]].rgbGreen;
				pc.R = palettes[v[0]].rgbRed;
			
				if (line.size() < infoHeader.biWidth)
				{
					line.push_back(pc);
				}
				
		
			}
			//因为位图是从下到上，从左到右的，所以这么恶心
			vector<PixColor>::reverse_iterator it;
			for (it = line.rbegin(); it != line.rend(); ++it)
			{
				pixColors.push_front(*it);
			}
			
		}
		modify_headers();
	}
	printf("pix count:%d\n", pixColors.size());
	struct stat stt;
	stat(filename, &stt);
	if (stt.st_size != ftell(fp))
	{
		printf("没有读完，格式可能解释错误了!当前位置：%ld, 文件大小:%ld\n", ftell(fp), stt.st_size );
		fclose(fp);
		return 0;
	}

	fclose(fp);
	return 0;
}
int BitmapPic::get_pix(PixColor&p, uint32_t line, uint32_t col) const
{
	
	if (line >= infoHeader.biHeight || col >= infoHeader.biWidth)
	{
		return -1;
	}
	p= pixColors[ line * infoHeader.biWidth + col];
	return 0;
}
int BitmapPic::dump(const char * filename) const
{
	if (filename == NULL) { return 1;}

	FILE * fp = fopen(filename, "wb+");
	if (fp == NULL)
	{
		perror("failed to open file");
		return -1;
	}
	
	fwrite(&fileHeader, 1, sizeof(fileHeader), fp);
	fwrite(&infoHeader, 1, sizeof(infoHeader), fp);
	uint32_t width  = (infoHeader.biBitCount * infoHeader.biWidth + 31) / 32 * 4;
	

	int i, j;
	for (i = infoHeader.biHeight-1; i >=0 ; --i)
	//for (i = 0; i < infoHeader.biHeight; ++i)
	{

		for (j = 0; j < infoHeader.biWidth; ++j)
		{
			int index = i*infoHeader.biWidth+j;
			fwrite(&(pixColors[index].B), 1, 1, fp);
			fwrite(&(pixColors[index].G), 1, 1, fp);
			fwrite(&(pixColors[index].R), 1, 1, fp);
		}
		int k;
		for (k = infoHeader.biWidth*3; k < width; ++k)
		{
			uint8_t zero = 0;
			fwrite(&zero, 1, 1, fp);
		}
	}
	fclose(fp);
	fp = NULL;
	return 0;
}
