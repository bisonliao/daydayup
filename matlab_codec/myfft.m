%��һ�����Ϻ���y������Ҷ�任����任

Fs = 128; % ����Ƶ��
T = 1/Fs; % �������
L = 256; % ��������
t = (0:L-1)*T; % ����ʱ��

y=5 + 7*cos(2*pi*15*t - 30*pi/180) + 3*cos(2*pi*40*t - 90*pi/180);

N = 2^nextpow2(L); %������������������Խ�󣬷ֱ��Ƶ��Խ��ȷ��N>=L�������Ĳ����źŲ�Ϊ0
Y = fft(y,N)/N*2; %����N����2������ʵ��ֵ��NԽ�󣬷�ֵ����Խ��
%Y��һ�鸴���������ĽǶȴ�����Ƶ�ʷ�������λƫ�ơ������ķ�ֵ��ʾ��Ƶ�ʷ����ķ�ֵ
f = Fs/N*(0:1:N-1); %Ƶ��
A = abs(Y); %��ֵ
P = angle(Y); %��ֵ

figure;
subplot(2,1,1);plot(f(1:N/2),A(1:N/2));title('��ֵƵ��');xlabel('Ƶ��(Hz)');ylabel('��ֵ');
subplot(212);plot(f(1:N/2),P(1:N/2));title('��λ��Ƶ');xlabel('Ƶ��(Hz)');ylabel('��λ');


 y2=ifft(Y*N/2, N); %�ָ���ʱ��
 figure;
 subplot(3,1,1);plot(1:N, y); title('ԭ�ź�');
 subplot(3,1,2);plot(1:N, y2); title('��Ƶ��ָ���ʱ����ź�');
 subplot(3,1,3);plot(1:N, y-y2); title('���߲�ֵ');