[sampledata, FS]=audioread('d:\short.wav');
left= sampledata(:,1).';
right= sampledata(:,2).';

y=left;
N=2^nextpow2(length(y));
Y=fft(y,N)/N*2;
A=abs(Y);
f = FS/N*(0:1:N-1); %频率
figure;plot(f(1:N), A(1:N));title('左声道频域');
Y(abs(Y)<0.00005)=0; %量化，高频部分抹除
A=abs(Y);
figure;plot(f(1:N), A(1:N));title('左声道高频抹除后');
y2=ifft(Y*N/2, N); %恢复到时域
left2=y2(1:length(left));
figure;plot(left-left2);title('左声道傅立叶变换前后差值');


y=right;
N=2^nextpow2(length(y));
Y=fft(y,N)/N*2;
A=abs(Y);
f = FS/N*(0:1:N-1); %频率
figure;plot(f(1:N), A(1:N));title('右声道频域');
Y(abs(Y)<0.00005)=0; %量化，高频部分抹除
A=abs(Y);
figure;plot(f(1:N), A(1:N));title('右声道高频抹除后');
y2=ifft(Y*N/2, N); %恢复到时域
right2=y2(1:length(right));
figure;plot(right-right2);title('右声道傅立叶变换前后差值');

 sampledata2=sampledata; %拼合为双声道
 sampledata2(:,1)=left2.';
 sampledata2(:,2)=right2.';
 audiowrite('d:\short_fft.wav', sampledata2, FS);

