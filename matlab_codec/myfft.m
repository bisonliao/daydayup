%对一个复合函数y做傅立叶变换和逆变换

Fs = 128; % 采样频率
T = 1/Fs; % 采样间隔
L = 256; % 采样个数
t = (0:L-1)*T; % 采样时间

y=5 + 7*cos(2*pi*15*t - 30*pi/180) + 3*cos(2*pi*40*t - 90*pi/180);

N = 2^nextpow2(L); %采样点数，采样点数越大，分辨的频率越精确，N>=L，超出的部分信号补为0
Y = fft(y,N)/N*2; %除以N乘以2才是真实幅值，N越大，幅值精度越高
%Y是一组复数，复数的角度代表该频率分量的相位偏移、复数的幅值表示该频率分量的幅值
f = Fs/N*(0:1:N-1); %频率
A = abs(Y); %幅值
P = angle(Y); %相值

figure;
subplot(2,1,1);plot(f(1:N/2),A(1:N/2));title('幅值频谱');xlabel('频率(Hz)');ylabel('幅值');
subplot(212);plot(f(1:N/2),P(1:N/2));title('相位谱频');xlabel('频率(Hz)');ylabel('相位');


 y2=ifft(Y*N/2, N); %恢复到时域
 figure;
 subplot(3,1,1);plot(1:N, y); title('原信号');
 subplot(3,1,2);plot(1:N, y2); title('从频域恢复到时域的信号');
 subplot(3,1,3);plot(1:N, y-y2); title('两者差值');
