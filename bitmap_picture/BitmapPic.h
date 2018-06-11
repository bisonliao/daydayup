/**
 * bmp图片解析的类
 **/
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <deque>
#include <vector>
using namespace std;

#pragma push(1)

#ifndef _WINDOWS_

typedef uint16_t WORD;
typedef uint32_t DWORD;
typedef long LONG;
typedef uint8_t BYTE;

#define BI_RGB        0L

typedef struct tagBITMAPFILEHEADER {
        WORD    bfType;
        DWORD   bfSize;
        WORD    bfReserved1;
        WORD    bfReserved2;
        DWORD   bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER{
        DWORD      biSize;
        LONG       biWidth;
        LONG       biHeight;
        WORD       biPlanes;
        WORD       biBitCount;
        DWORD      biCompression;
        DWORD      biSizeImage;
        LONG       biXPelsPerMeter;
        LONG       biYPelsPerMeter;
        DWORD      biClrUsed;
        DWORD      biClrImportant;
} BITMAPINFOHEADER;


typedef struct tagRGBQUAD {
        BYTE    rgbBlue;
        BYTE    rgbGreen;
        BYTE    rgbRed;
        BYTE    rgbReserved;
} RGBQUAD;


#endif

typedef struct
{
	uint8_t B;
	uint8_t G;
	uint8_t R;

}PixColor;
#pragma pop()

class BitmapPic
{
private:
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	deque<PixColor> pixColors;
	void modify_headers();
	
public:
	BitmapPic(void);
	virtual ~BitmapPic(void);
	int load(const char * filename);//从文件中读取bmp图片
	int get_pix(PixColor&p, uint32_t line, uint32_t col) const;//获得第line行第col列的像素
	int dump(const char * filename) const ;//以24bit bmp图片写入到文件中

	friend class BlockPic;


};

