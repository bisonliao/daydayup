% 对图像进行均值滤波

A=imread('cameraman.tif');
D=zeros(256);
D(1,1:256)=A(1,1:256);
D(256, 1:256)=A(256,1:256);
D(1:256, 1)=A(1:256,1);
D(1:256, 256)=A(1:256,256);

for i=[2:255]
  for j = [2:255]
     sum = uint32(0);
     for k1=[i-1:i+1]
	    for k2=[j-1:j+1]
		    sum = sum+uint32(A(k1,k2));
		end
	 end
	 
     D(i,j) =  uint8(sum / 9);
  end
end

D = uint8(D);

subplot(1,2,1);imshow(A);
subplot(1,2,2);imshow(D);