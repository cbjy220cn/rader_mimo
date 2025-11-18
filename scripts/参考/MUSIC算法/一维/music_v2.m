clear; close all;
%%%%%%%% MUSIC for Uniform Linear Array%%%%%%%%
derad = pi/180;      %角度->弧度
c=3e8;
N = 8;               % 阵元个数        
M = 6;               % 信源数目
theta = [-60 -30 0 10 40 75];  % 待估计角度[-90`90]
snr = 10;            % 信噪比
K = 512;             % 快拍数
f= 3e6;              %信号频率
lambda=c/f;          %信号波长
dd = lambda/2;            % 阵元间距 
L=5;                 %迭代次数
d=0:dd:(N-1)*dd;
A=exp(-1i*2*pi*d'*sin(theta*derad)/lambda);  %方向矢量

%%%%构建信号模型%%%%%
S=randn(M,K);             %信源信号，入射信号
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算协方差矩阵
Rxx=X1*X1'/K;
Pss=Rxx^-L;     %通过迭代取代En


 
% 遍历每个角度，计算空间谱
for iang = 1:361
    angle(iang)=(iang-181)/2;
    phim=derad*angle(iang);
    a=exp(-1i*2*pi*d*sin(phim)/lambda).'; 
    Pmusic(iang)=1/(a'*Pss*a);
end
Pmusic=abs(Pmusic);
Pmmax=max(Pmusic)
Pmusic=10*log10(Pmusic/Pmmax);            % 归一化处理
h=plot(angle,Pmusic);
set(h,'Linewidth',2);
xlabel('入射角/(degree)');
ylabel('空间谱/(dB)');
set(gca, 'XTick',[-90:30:90]);
grid on;
