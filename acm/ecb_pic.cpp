/* 演示用ecb模式加密一张位图，加密后的图片还能够看出原图的大概内容*/

#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
// g++ -o ttx ecb_pic.cpp -lcrypto  -I/usr/include/opencv4/opencv -I/usr/include/opencv4 -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -ldl -lpthread /usr/local/zlib/lib/libz.a /usr/lib/x86_64-linux-gnu/libtbb.so

#include <openssl/evp.h>

#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/objects.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PRIME_LEN (512)

int main()
{
    int i, ret;
    
    int keylen;
    int err, ret1, ret2;
   
    BIGNUM * a = NULL;
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_CIPHER_CTX_init(ctx);
    unsigned char key[16];
    unsigned char iv[16];
    static unsigned char plaintext[1024*1024*10];
    static unsigned char ciphertext[1024*1024*10];
    int offset = 0;
    int len = sizeof(ciphertext);
    int cipherlen;

    Mat src = imread("./duck.bmp", cv::IMREAD_GRAYSCALE);
    cout<<"channels:"<<src.channels()<<" row:"<<src.rows<<" col:"<<src.cols<<" size:"<<src.size()<<endl;

    
    if (RAND_load_file("/dev/urandom", 256) != 256)
    {
        fprintf(stderr, "RAND_load_file failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    RAND_bytes(key, sizeof(key));
    RAND_bytes(iv, sizeof(iv));
    // encrypt
    ret = EVP_EncryptInit(ctx, EVP_aes_128_ecb(), key, iv);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptInit failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    ret = EVP_EncryptUpdate(ctx, ciphertext+offset, &len, (const unsigned char *)src.data, src.size().height * src.size().width);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    len = sizeof(ciphertext) - offset;
    ret = EVP_EncryptFinal(ctx, ciphertext+offset, &len);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_EncryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    cipherlen = offset;
    printf("cipher text len:%d\n", cipherlen);

    memcpy(src.data, ciphertext, src.size().height * src.size().width);

    cv::imwrite("./duck2.bmp", src);


    return 0;

#if 0

    // decrypt
    ret = EVP_DecryptInit(ctx, EVP_aes_128_ecb(), key, iv);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptInit failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset = 0;
    len = sizeof(plaintext);
    ret = EVP_DecryptUpdate(ctx, (unsigned char*)plaintext+offset, &len, ciphertext, cipherlen);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptUpdate failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    printf("decrypt len:%d\n", len);
    offset += len;
    len = sizeof(plaintext) - offset;
    ret = EVP_DecryptFinal(ctx, (unsigned char*)plaintext+offset, &len);
    if (ret != 1)
    {
        fprintf(stderr, "EVP_DecryptFinal failed\n");
        err = ERR_get_error();
        fprintf(stderr, "%s\n", ERR_error_string(err, NULL));
        goto end;
    }
    offset += len;
    printf("plain text len:%d\n", offset);
#endif
    
end:
    return 0;
      
  

}

