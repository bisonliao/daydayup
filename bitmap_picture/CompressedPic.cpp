#include "CompressedPic.h"
#include <math.h>
#include <gmp.h>

//void cosin_array(unsigned int x, unsigned int u, mpq_t  result);
static void print_hex(const unsigned char * buf, int buf_sz);
static int double2int(double d);

double BlockPic::A[8][8];
double BlockPic::A_transposed[8][8];


//一个辅助类，用于在main函数进入前做一些初始化
class InitCodeHelper
{
public:
	InitCodeHelper();
};
InitCodeHelper::InitCodeHelper()
{

	BlockPic::get_coeff(BlockPic::A);
	memcpy(BlockPic::A_transposed,  BlockPic::A, sizeof(BlockPic::A) );
	BlockPic::transpose(BlockPic::A_transposed);
	
	//printf("hello world\n");
};
static InitCodeHelper initcode;
//辅助类结束

BlockPic::BlockPic(void)
{
	QP = 8;
}


BlockPic::~BlockPic(void)
{
}

static void RGB2YUV(const PixColor &pc, uint8_t & Y, uint8_t &U, uint8_t &V)
{
	Y =       0.299 * pc.R   + 0.587 *    pc.G    + 0.114* pc.B;
	U = 128 - 0.168736 *pc.R - 0.331264 * pc.G    + 0.5 * pc.B;
	V = 128 + 0.5 * pc.R     - 0.418688 * pc.G    - 0.081312 * pc.B;
}
static void YUV2RGB(uint8_t  Y, uint8_t U, uint8_t V, PixColor &pc)
{
	pc.R = Y                          + 1.402 * ((int)V - 128);
	pc.G = Y - 0.344136 *((int)U-128) - 0.714136*((int)V - 128);
	pc.B = Y + 1.772 * ((int)U - 128);
}

#if 0 
void BlockPic::dct2(const block1_t &b1, block2_t & b2)
{
	int u, v, x, y;
	block2_t g;
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			g.values[u][v] = b1.values[u][v] - 128;
		}
	}
	mpq_t G[8][8];
	mpq_t sum, temp1, temp2, temp3;
	mpq_init(sum);
	mpq_init(temp1);
	mpq_init(temp2);
	mpq_init(temp3);
	


	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			mpq_set_ui(sum, 0, 1);

			for (x = 0; x < 8; ++x)
			{
				for (y = 0; y < 8; ++y)
				{
					//sum += g.values[x][y]* cos( (2*x+1)*u*Pi / 16) * cos((2*y+1)*v*Pi/16);

					mpq_set_si(temp1, g.values[x][y], 1);
					cosin_array(x, u, temp2);
					mpq_mul(temp3, temp1, temp2);

					cosin_array(y, v, temp2);
					mpq_mul(temp1, temp3, temp2);

					mpq_set(temp2, sum);
					mpq_add(sum, temp1, temp2);
				}
			}
			/*
			double alpha_u = (u == 0 ? 0.70710678:1);
			double alpha_v = (v == 0 ? 0.70710678:1);
			G[u][v] = 0.25*alpha_u*alpha_v*sum;
			*/
			mpq_set_ui(temp1, 1, 4);
			mpq_init(G[u][v]);
			mpq_mul(G[u][v], temp1, sum);

			mpq_set_str(temp1, "7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937671638207864/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 10);

			if (u == 0)
			{
				mpq_set(temp2, G[u][v]);
				mpq_mul(G[u][v], temp1, temp2);
			}
			if (v == 0)
			{
				mpq_set(temp2, G[u][v]);
				mpq_mul(G[u][v], temp1, temp2);
			}

		}
	}

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			//b2.values[u][v] = G[u][v] /  QMatrix[u][v];
			//b2.values[u][v] = G[u][v] / QP; 

		#if 1
			mpq_set_ui(temp1, QP, 1);
			mpq_div(temp2, G[u][v], temp1);
			b2.values[u][v] = mpq_get_d(temp2);
		#else
			b2.values[u][v] = mpq_get_d(G[u][v]);
		#endif
			mpq_clear(G[u][v]);

		}
	}
	mpq_clear(sum);
	mpq_clear(temp1);
	mpq_clear(temp2);
	mpq_clear(temp3);


}
#endif


void BlockPic::dct(const block1_t &b1, block2_t & b2)
{
#if 0 
//按公式来计算
	int u, v, x, y;
	block2_t g;
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			g.values[u][v] = b1.values[u][v]-128;
		}
	}
	double G[8][8];
	


	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			double sum = 0;
			for (x = 0; x < 8; ++x)
			{
				for (y = 0; y < 8; ++y)
				{
					sum += g.values[x][y]* cos( (2*x+1)*u*Pi / 16) * cos((2*y+1)*v*Pi/16);
				}
			}
			double alpha_u = (u == 0 ? 0.70710678:1);
			double alpha_v = (v == 0 ? 0.70710678:1);
			G[u][v] = 0.25*alpha_u*alpha_v*sum;
		}
	}
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			//b2.values[u][v] = G[u][v] /  QMatrix[u][v];
			b2.values[u][v] = G[u][v] / QP;  
			//量化和保存为整数都导致损失了精度，使得不能无损的恢复
		}
	}
#else

//用矩阵计算的方法来计算

	int u, v;
	double X[8][8];
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			X[u][v] = b1.values[u][v];
			X[u][v] -= 128;

		}
	}

	double Y[8][8];
	double mid[8][8];

	matrix_mul(A, X, mid);
	matrix_mul(mid, A_transposed, Y);

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			b2.values[u][v] = Y[u][v] / QP;
			//量化和保存为整数都导致损失了精度，使得不能无损的恢复
		}
	}



#endif
}
void BlockPic::set_qp(uint8_t qp)
{
	QP = qp;
}
void BlockPic::inverse_dct(block1_t &b1, const block2_t & b2)
{

#if 0
//用公式的方式来计算
	int u, v, x, y;
	double F[8][8];

	
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			//F[u][v] = b2.values[u][v] * QMatrix[u][v];
			F[u][v] = b2.values[u][v] * QP;
			

		}
	
	}
	
	
	block2_t tmp;
	


	for (x = 0; x < 8; ++x)
	{
		for (y = 0; y < 8; ++y)
		{
			double sum = 0;
			for (u = 0; u < 8; ++u)
			{
				for (v = 0; v < 8; ++v)
				{
					double alpha_u = (u == 0 ? 0.70710678:1);
					double alpha_v = (v == 0 ? 0.70710678:1);
					sum += alpha_u * alpha_v * F[u][v] * cos( (2*x+1)*u*Pi / 16) * cos((2*y+1)*v*Pi/16);
				}
			}
			
			tmp.values[x][y] = 0.25*sum;
			
			
		}
		
	}

	
	

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			b1.values[u][v] = tmp.values[u][v]+128;
		}
	}
#else
//用矩阵计算的方法来计算

	int u, v;
	double X[8][8];
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			X[u][v] = b2.values[u][v] * QP;
		}
	}

	double Y[8][8];
	double mid[8][8];

	matrix_mul(A_transposed, X, mid);
	matrix_mul(mid, A, Y);

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			b1.values[u][v] = double2int(Y[u][v]+128);
			printf("%f\t", Y[u][v]);
		}
		printf("\n");
	}
	printf("\n");
	print_hex((unsigned char*)&Y[7][0], sizeof(double));
	printf("\t%f\n", Y[7][0]);
#endif
	
	
}
#if 0
void BlockPic::inverse_dct2(block1_t &b1, const block2_t & b2)
{

	int u, v, x, y;
	int F[8][8];

	
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			//F[u][v] = b2.values[u][v] * QMatrix[u][v];

			F[u][v] = b2.values[u][v] * QP;
			

		}
	
	}
	
	
	block2_t tmp;
	
	mpq_t sum, temp1, temp2, temp3;
	mpq_init(sum);
	mpq_init(temp1);
	mpq_init(temp2);
	mpq_init(temp3);


	for (x = 0; x < 8; ++x)
	{
		for (y = 0; y < 8; ++y)
		{
			mpq_set_ui(sum, 0, 1);
			for (u = 0; u < 8; ++u)
			{
				for (v = 0; v < 8; ++v)
				{
					/*
					double alpha_u = (u == 0 ? 0.70710678:1);
					double alpha_v = (v == 0 ? 0.70710678:1);
					sum += alpha_u * alpha_v * F[u][v] * cos( (2*x+1)*u*Pi / 16) * cos((2*y+1)*v*Pi/16);
					*/
					mpq_set_si(temp1, F[u][v], 1);
					cosin_array(x, u, temp2);
					mpq_mul(temp3, temp1, temp2);

					cosin_array(y, v, temp2);
					mpq_mul(temp1, temp3, temp2);

					mpq_set_str(temp2, "7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937671638207864/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", 10);

					if (u == 0)
					{
						mpq_set(temp3, temp1);
						mpq_mul(temp1, temp2, temp3);
					}
					if (v == 0)
					{
						mpq_set(temp3, temp1);
						mpq_mul(temp1, temp2, temp3);
					}
					mpq_set(temp2, sum);
					mpq_add(sum, temp1, temp2);

				}
			}
			/*
			tmp.values[x][y] = 0.25*sum;
			*/

			mpq_set_si(temp1, 1, 4);
			mpq_mul(temp2, temp1, sum);
			tmp.values[x][y] = mpq_get_d(temp2);
			
			
			
		}
		
	}

	
	

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			b1.values[u][v] = tmp.values[u][v] + 128;
		}
	}

	mpq_clear(sum);
	mpq_clear(temp1);
	mpq_clear(temp2);
	mpq_clear(temp3);

	
	
}
#endif



int BlockPic::from_bitmap(const BitmapPic & bmp)
{
	infoHeader.height_pix = bmp.infoHeader.biHeight;
	infoHeader.width_pix = bmp.infoHeader.biWidth;


	infoHeader.height_blocks = infoHeader.height_pix / 8;
	if ( infoHeader.height_pix % 8)
	{
		infoHeader.height_blocks++;
	}
	if (infoHeader.height_blocks % 2)//偶数适合uv下采样优雅的操作
	{
		infoHeader.height_blocks++;
	}

	infoHeader.width_blocks = infoHeader.width_pix / 8;
	if ( infoHeader.width_pix % 8)
	{
		infoHeader.width_blocks++;
	}
	if (infoHeader.width_blocks % 2)//偶数适合uv下采样优雅的操作
	{
		infoHeader.width_blocks++;
	}
	///////////// 
	// RGB to YUV
	deque<block1_t> y,u,v;
	int i, j, k, m;
	for (i = 0; i < infoHeader.height_blocks; ++i)//每行block
	{
		for (j = 0; j < infoHeader.width_blocks; ++j)//每列block
		{

			block1_t yblock, ublock, vblock;
			for (k = 0; k < 8; ++k)//同一block里的每行像素
			{
				for (m = 0; m < 8; ++m)//同一block里的每列像素
				{
					//像素坐标
					int p_x, p_y;
					p_y = i * 8 + k;
					p_x = j * 8 + m;
					PixColor pc;

					if (p_y < infoHeader.height_pix && 
						p_x < infoHeader.width_pix)
					{
						bmp.get_pix(pc, p_y, p_x);
					}
					else
					{
						pc.B = 0; 
						pc.G = 0;
						pc.R = 0;
					}
					uint8_t one_y, one_u, one_v;
					RGB2YUV(pc, one_y, one_u, one_v);
					yblock.values[k][m] = one_y;
					ublock.values[k][m] = one_u;
					vblock.values[k][m] = one_v;
				}
			}
			y.push_back(yblock);
			u.push_back(ublock);
			v.push_back(vblock);

			
	
		}
	}

	//u v下采样
	deque<block1_t> ds_u, ds_v;
	int r;
	for (r = 0; r < 2; ++r)
	{
		deque<block1_t> & blocklist_refer = (r == 0? u : v);
		deque<block1_t> & ds_blocklist_refer = (r == 0? ds_u : ds_v);
		for (i = 0; i < infoHeader.height_blocks; i+=2)//每两行block
		{
			for (j = 0; j < infoHeader.width_blocks; j+=2)//每两列block
			{
				block1_t b1, b2, b3, b4, b;
				//一次访问周边4个block，下采样为1个block，即YUV420的方式
				b1 = blocklist_refer[i*infoHeader.width_blocks+j];
				b2 = blocklist_refer[i*infoHeader.width_blocks+j+1];
				b3 = blocklist_refer[(i+1)*infoHeader.width_blocks+j];
				b4 = blocklist_refer[(i+1)*infoHeader.width_blocks+j+1];

				//左上
				for (k = 0; k < 4; ++k)
				{
					for (m = 0; m < 4; ++m)
					{
						//b.values[k][m] = (b1.values[2*k][2*m] + b1.values[2*k][2*m+1] + b1.values[2*k+1][2*m]+b1.values[2*k][2*m+1]) / 4;
						b.values[k][m] = b1.values[2*k][2*m];
					}
				}
				//右上
				for (k = 0; k < 4; ++k)
				{
					for (m = 0; m < 4; ++m)
					{
						//b.values[k][m+4] = (b2.values[2*k][2*m] + b2.values[2*k][2*m+1] + b2.values[2*k+1][2*m]+b2.values[2*k][2*m+1]) / 4;
						b.values[k][m+4] = b2.values[2*k][2*m];
					}
				}
				//左下
				for (k = 0; k < 4; ++k)
				{
					for (m = 0; m < 4; ++m)
					{
						//b.values[k+4][m] = (b3.values[2*k][2*m] + b3.values[2*k][2*m+1] + b3.values[2*k+1][2*m]+b3.values[2*k][2*m+1]) / 4;
						b.values[k+4][m] = b3.values[2*k][2*m];
					}
				}
				//右下
				for (k = 0; k < 4; ++k)
				{
					for (m = 0; m < 4; ++m)
					{
						//b.values[k+4][m+4] = (b4.values[2*k][2*m] + b4.values[2*k][2*m+1] + b4.values[2*k+1][2*m]+b4.values[2*k][2*m+1]) / 4;
						b.values[k+4][m+4] = b4.values[2*k][2*m];
					}
				}
				ds_blocklist_refer.push_back(b);
			}
		}
	
	}

	//至此 y, ds_u, ds_v保留了y/u/v分量的block
	//DCT变换，并量化
	deque<block1_t>::iterator it;
	for (it = y.begin(); it != y.end(); ++it)
	{
		block2_t b;
		dct(*it, b);
		
		Y.push_back(b);
	}
	for (it = ds_u.begin(); it != ds_u.end(); ++it)
	{
		block2_t b;
		dct(*it, b);
		U.push_back(b);
	}
	for (it = ds_v.begin(); it != ds_v.end(); ++it)
	{
		block2_t b;
		dct(*it, b);
		V.push_back(b);
	}


	return 0;
}
int BlockPic::to_bitmap(BitmapPic & bmp)
{
	int r, i, j;

	//反向量化、DCT逆变换
	deque<block1_t> y, ds_u, ds_v;
	deque<block1_t> u, v;

	deque<block2_t>::iterator it;
	for (it = Y.begin(); it != Y.end(); ++it)
	{
		block1_t b;
		inverse_dct(b, *it);
		
		y.push_back(b);
	}
	for (it = U.begin(); it != U.end(); ++it)
	{
		block1_t b;
		inverse_dct(b, *it);
		
		ds_u.push_back(b);
	}
	for (it = V.begin(); it != V.end(); ++it)
	{
		block1_t b;
		inverse_dct(b, *it);
		
		ds_v.push_back(b);
	}

	//上采样，把ds_u ds_v插值, 保存到u,v里

	
	for (r = 0; r < 2; ++r)
	{
		deque<block1_t> & list = (r == 0? u : v);
		deque<block1_t> & ds_list = (r == 0? ds_u : ds_v);
		
		for (i = 0; i < infoHeader.height_blocks; i++)//每行block
		{
			for (j = 0; j < infoHeader.width_blocks; j++)//每列block
			{
				//根据i,j转换出下采样的块的两维下标
				int ds_i = i /2;
				int ds_j = j / 2;
				//根据两维下标，获得一维下标
				int ds_block_nr_per_line = infoHeader.width_blocks/2;
				int ds_index = ds_i * ds_block_nr_per_line + ds_j;

				//找到对应的下采样的块
				block1_t & ds_block = ds_list[ds_index];
				block1_t b;

				bool left = j % 2 ? 0:1;
				bool top  = i % 2 ? 0:1;

				int k, m;
				for (k = 0; k < 8; ++k)
				{
					for (m = 0; m < 8; ++m)
					{
						if (left && top)//左上
						{
							b.values[k][m] = ds_block.values[k/2][m/2];
						}
						if (left && !top)//左下
						{
							b.values[k][m] = ds_block.values[k/2+4][m/2];
						}
						if (!left && top)//右上
						{
							b.values[k][m] = ds_block.values[k/2][m/2+4];
						}
						if (!left && !top)//右下
						{
							b.values[k][m] = ds_block.values[k/2+4][m/2+4];
						}
					}
				}
				list.push_back(b);
			}

		}
	}

	

	//YUV2RGB
	memset(&bmp.infoHeader, 0, sizeof(bmp.infoHeader));
	memset(&bmp.fileHeader, 0, sizeof(bmp.fileHeader));
	bmp.pixColors.clear();

	bmp.infoHeader.biBitCount = 24;
	bmp.infoHeader.biHeight = infoHeader.height_pix;
	bmp.infoHeader.biWidth = infoHeader.width_pix;
	bmp.infoHeader.biCompression = BI_RGB;
	bmp.infoHeader.biPlanes = 1;
	bmp.infoHeader.biSize = sizeof(bmp.infoHeader);

	bmp.fileHeader.bfType = 0x4d42;
	uint32_t width  = (bmp.infoHeader.biBitCount * bmp.infoHeader.biWidth + 31) / 32 * 4;
	bmp.fileHeader.bfOffBits = sizeof(bmp.fileHeader) + sizeof(bmp.infoHeader);
	bmp.fileHeader.bfSize = sizeof(bmp.fileHeader) + sizeof(bmp.infoHeader) + 3 * width * bmp.infoHeader.biHeight;
	
	
	//printf("ysize %d, usize %d, vsize%d\n", y.size(), u.size(), v.size());

	for (i = 0; i < bmp.infoHeader.biHeight; ++i)
	{
		for (j = 0; j < bmp.infoHeader.biWidth; ++j)
		{
			int block_line = i / 8;
			int block_col = j / 8;
			int block_index = block_line * infoHeader.width_blocks + block_col;

			int line_in_block = i % 8;
			int col_in_block = j % 8;

		

			uint8_t yy, uu, vv;
			yy = y[block_index].values[line_in_block][col_in_block];
			uu = u[block_index].values[line_in_block][col_in_block];
			vv = v[block_index].values[line_in_block][col_in_block];

			PixColor pc;
			YUV2RGB(yy,uu,vv, pc);
			bmp.pixColors.push_back(pc);
		}
	}
	//printf("pix count:%d\n", bmp.pixColors.size());
	return 0;
}
static int double2int(double d)
{
	char buf[50];
	snprintf(buf, sizeof(buf), "%.0f", d);
	return atoi(buf);
}

static void print_hex(const unsigned char * buf, int buf_sz)
{
	int i;
	for (i = 0; i < buf_sz; ++i)
	{
		if ( (i % 16) == 0)
		{
			printf("%08X ", i);
		}
		printf("%02X ", buf[i]);

		if ( ((i+1)%16) == 0)
		{
			printf("  ");
			int j;
			for (j = i - 15; j <= i; ++j)
			{
				if ( buf[j] >= 'a' && buf[j] <= 'Z')
				{
					printf("%c", buf[j]);
				}
				else
				{
					printf(".");
				}
			}
			printf("\n");
		}

	}
}

void BlockPic::test()
{
#if 1
	block1_t b;
	int i, j;
	BlockPic bpic;
	uint8_t buffer[][8] = {
							52, 55, 61, 66, 70, 61, 64, 73,
							63, 59, 55, 90, 109, 85,69,72,
							62, 59, 68, 113, 144, 104, 66, 73,
							63, 58, 71, 122, 154, 106, 70, 69,
							67, 61, 68, 104, 126, 88, 68, 70,
							79,65, 60, 70, 77, 68, 58, 75,
							85, 71, 64, 59, 55, 61, 65, 83,
							87, 79, 69, 68, 65, 76, 78, 94,
						};
	memcpy(&b.values[0][0], buffer, sizeof(buffer));

	int r;
	for (r = 0; r < 1; ++r)
	{
		printf("\ndct之前：\n");
		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				printf("%d ", b.values[i][j]);
			}
			printf("\n");
		}

		block2_t b2;
		bpic.dct(b, b2);
		printf("\ndct之后\n");
		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				printf("%d ", b2.values[i][j]);
			}
			printf("\n");
		}

		bpic.inverse_dct(b, b2);
		printf("\nidct后：\n");
		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				printf("%d ", b.values[i][j]);
			}
			printf("\n");
		}

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				b.values[i][j] = b.values[i][j] >> 2;
			}
			
		}

	}



	/*
	printf("\n");
	PixColor pc;
	pc.B = 127;
	pc.G = 27;
	pc.R = 200;
	printf("BGR:%d %d %d\n", pc.B, pc.G, pc.R);
	uint8_t Y, U, V;
	RGB2YUV(pc, Y, U, V);
	printf("YUV:%d, %d, %d\n", Y, U, V);
	YUV2RGB(Y, U, V, pc);
	printf("BGR:%d %d %d\n", pc.B, pc.G, pc.R);
	*/
#else

	double X[][8] = {
							-52.0, -55.0, -61.0, -66.0, -70.0, -61.0, -64.0, -73.0,
							-63.0, -59.0, -55.0, -90.0, -109.0, -85.0,69.0,72.0,
							-62.0, -59.0, -68.0, -113.0, -144.0, -104.0, -66.0, -73.0,
							-63.0, -58.0, -71.0, -122.0, -154.0, -106.0, -70.0, -69.0,
							-67.0, -61.0, -68.0, -104.0, -126.0, -88.0, -68.0, -70.0,
							-79.0,65.0, -60.0, -70.0, -77.0, -68.0, -58.0, -75.0,
							-85.0, -71.0, -64.0, -59.0, -55.0, -61.0, -65.0, -83.0,
							-87.0, -79.0, -69.0, -68.0, -65.0, -76.0, -78.0, -94.0,
						};

	double mid[8][8];
	double Y[8][8];

	matrix_mul(BlockPic::A, X, mid);
	matrix_mul(mid, BlockPic::A_transposed, Y);


	////////////////////////////////
	double XX[8][8];
	matrix_mul(BlockPic::A_transposed, Y, mid);
	matrix_mul(mid, BlockPic::A, XX);

	int i, j;

	printf("\nX:\n");
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			printf("%f\t", X[i][j]);
		}
		printf("\n");
	}
	printf("\nA:\n");
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			printf("%f\t", A[i][j]);
		}
		printf("\n");
	}
	printf("\nY:\n");
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			printf("%f\t", Y[i][j]);
		}
		printf("\n");
	}
	printf("\nXX:\n");
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			printf("%f\t", XX[i][j]);
		}
		printf("\n");
	}
#endif
		
}
void BlockPic::get_coeff(double A[][8])
{

	int i, j;
	for (i = 0; i < 8; ++i)
	{
		double c = sqrt(2.0/8);
		if (i == 0)
		{
			c = sqrt(1.0/8);
		}
		for (j = 0; j < 8; ++j)
		{
			A[i][j] = c * cos(Pi*(j+0.5)*i/8);
		}
	}

}
void BlockPic::transpose(double A[][8])
{
	int i, j;
	for (i = 0; i < 8; ++i)
	{
		for (j = i+1; j < 8; ++j)
		{
			double temp = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = temp;
		}
	}
}
void BlockPic::matrix_mul(double A[][8], double B[][8], double result[][8])
{

	int i, j, k;
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			double sum = 0;
			for (k = 0; k < 8; ++k)
			{
				sum += A[i][k]*B[k][j];
			}
			result[i][j] = sum;
		}
	}

}
/*
void BlockPic::int_array_to_double(const int A[][8], double B[][8])
{
	int i, j;
	for (i = 0; i < 8 ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			B[i][j] = A[i][j];
		}
	}
}
void BlockPic::double_array_to_int(const double A[][8], int B[][8])
{
	int i, j;
	for (i = 0; i < 8 ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			B[i][j] = A[i][j];
		}
	}
}
*/
