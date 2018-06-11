#include "CompressedPic.h"
#include <math.h>



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



void BlockPic::dct(const block1_t &b1, block2_t & b2)
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
	double G[8][8];
	

#define Pi (3.1415926)

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
#undef Pi
	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			//b2.values[u][v] = G[u][v] /  QMatrix[u][v];
			b2.values[u][v] = G[u][v] / QP; 
		}
	}
}
void BlockPic::set_qp(uint8_t qp)
{
	QP = qp;
}
void BlockPic::inverse_dct(block1_t &b1, const block2_t & b2)
{

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
	

#define Pi (3.1415926)

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

#undef Pi
	
	

	for (u = 0; u < 8; ++u)
	{
		for (v = 0; v < 8; ++v)
		{
			b1.values[u][v] = tmp.values[u][v] + 128;
		}
	}

	
	
}



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

void BlockPic::test()
{
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
	for (r = 0; r < 2; ++r)
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
	/*
	mpz_t t;
	char tmp[512]={0};
	mpz_init(t);
	mpz_set_str(t, "0123456789", 10);
	mpz_out_str(stdout, 2, t);
	mpz_tstbit(t, 1);
	gmp_fprintf(stdout, "\n%Zd\n",t);
	mpz_export(tmp, NULL, 1, 1, 1, 0, t);
	mpz_clear(t);
	*/

}
