#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <deque>
#include <vector>
#include "BitmapPic.h"
using namespace std;

#pragma push(1)

typedef struct
{
	uint8_t values[8][8];
} block1_t; // 8*8的block

typedef struct
{
	int8_t values[8][8];
} block2_t; // 8*8的block


typedef struct
{
	uint16_t height_blocks;// Y分量的纵轴的block个数，每个block =8*8的像素区域
	uint16_t width_blocks;// Y分量的横轴的block个数
	uint16_t height_pix; //RGB位图的高度
	uint16_t width_pix;//RGB位图的宽度
} BlockPicInfoHeader;

#pragma pop();

const int QMatrix[8][8] = {
							16,11,10,16,24,40,51,61,
							12,12,14,19,26,58,60,55,
							14,13,16,24,40,57,69,56,
							14,17,22,29,51,87,80,62,
							18,22,37,56,68,109,103,77,
							24,35,55,64,81,104,113,92,
							49,64,78,87,103,121,120,101,
							72,92,95,98,112,100,103,99
							};

//BitmapPic经过YUV转换、下采样、DCT变换、量化，就成为了BlockPic
class BlockPic
{
private:

	BlockPicInfoHeader infoHeader;
	deque<block2_t> Y;
	deque<block2_t> U;
	deque<block2_t> V;
	uint8_t QP;


	static double A[8][8];
	static double A_transposed[8][8];

	static void get_coeff(double A[][8]);
	static void transpose(double A[][8]);
	static void matrix_mul(double A[][8], double B[][8], double result[][8]);
//	static void int_array_to_double(const int A[][8], double B[][8]);
//	static void double_array_to_int(const double A[][8], int B[][8]);
	

public:
	BlockPic(void);
	void set_qp(uint8_t qp);
	virtual ~BlockPic(void);
	int from_bitmap(const BitmapPic & bmp);
	int to_bitmap(BitmapPic & bmp);
	void inverse_dct(block1_t &b1, const block2_t & b2);
	//void inverse_dct2(block1_t &b1, const block2_t & b2);
	void dct(const block1_t &b1, block2_t & b2);
	//void dct2(const block1_t &b1, block2_t & b2);

	static const double Pi = 3.1415926535;

	static void test();



	friend class InitCodeHelper;
};




