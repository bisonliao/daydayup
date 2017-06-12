[sampledata, FS]=audioread('d:\short.wav');
left= sampledata(:,1).';
right= sampledata(:,2).';

y=left;
Y=dct(y);
figure;plot(Y);title('左声道DCT频域');
Y(150000:length(Y))=0;
figure;plot(Y);title('左声道DCT频域高频分量置0');
left2=idct(Y);
figure;plot(1:length(left), left, 1:length(left),left2); title('左声道前后差异');

y=right;
Y=dct(y);
figure;plot(Y);title('右声道DCT频域');
Y(150000:length(Y))=0;
figure;plot(Y);title('右声道DCT频域高频分量置0');
right2=idct(Y);
figure;plot(1:length(right), right, 1:length(right),right2); title('右声道前后差异');

sampledata2=sampledata; %拼合为双声道
 sampledata2(:,1)=left2.';
 sampledata2(:,2)=right2.';
 audiowrite('d:\short_fft.wav', sampledata2, FS);