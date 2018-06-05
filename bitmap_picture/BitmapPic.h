/**
 * bmpͼƬ��������
 **/
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include<windows.h>
#include <WinGDI.h>
#include <deque>
#include <vector>
using namespace std;

#pragma pack(1)

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
	int load(const char * filename);//���ļ��ж�ȡbmpͼƬ
	int get_pix(PixColor&p, uint32_t line, uint32_t col);//��õ�line�е�col�е�����
	int dump(const char * filename);//��24bit bmpͼƬд�뵽�ļ���
};

