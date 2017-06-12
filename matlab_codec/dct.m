
%dct�任��Ҫ��A�����ͼƬ��256*256��С��
A=imread('cameraman.tif');
D=zeros(256);
for (i=[1:32])
      for (j=[1:32])
        MB=A( (i-1)*8+1:i*8, (j-1)*8+1:j*8 );
        C=dct2(MB);
        C(abs(C)<10)=0;
        D( (i-1)*8+1:i*8, (j-1)*8+1:j*8 ) = C;
      end
end

%dct��任
for (i=[1:32])
      for (j=[1:32])
        MB=D( (i-1)*8+1:i*8, (j-1)*8+1:j*8 );
        C=idct2(MB);
        AA( (i-1)*8+1:i*8, (j-1)*8+1:j*8 ) = C;
      end
end