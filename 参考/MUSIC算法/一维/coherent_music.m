clear; close all;
%%%%%%%% MUSIC for Uniform Linear Array%%%%%%%%
derad = pi/180;      %角度->弧度
N = 16;               % 阵元个数
L = 8;               % 子阵元长度(L>M)
M = 6;               % 信源数目
theta = [-60 -30 0 10 40 75];  % 待估计角度
snr = 10;            % 信噪比
K = 512;             % 快拍数
 
dd = 0.5;            % 阵元间距 
d=0:dd:(N-1)*dd;
A=exp(-1i*2*pi*d.'*sin(theta*derad));  %方向矢量

%%%%构建信号模型%%%%%
S0=randn(1,K);             %信源信号，入射信号
S=[S0;S0;S0;S0;S0;S0];
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算协方差矩阵
Rxx=X1*X1'/K;
num_L=N-L+1;             %子阵元个数
R=zeros(L,L);
for i=1:num_L
    R=R+Rxx(i:i+L-1,i:i+L-1);
end
R=R/num_L;

% 特征值分解
[EV,D]=eig(R);                   %特征值分解
EVA=diag(D)';                      %将特征值矩阵对角线提取并转为一行
[EVA,I]=sort(EVA);                 %将特征值排序 从小到大
EV=fliplr(EV(:,I));                % 对应特征矢量排序
                 
 
% 遍历每个角度，计算空间谱
for iang = 1:361
    angle(iang)=(iang-181)/2;
    phim=derad*angle(iang);
    a=exp(-1i*2*pi*d(1:L)*sin(phim)).'; 
    En=EV(:,M+1:L);                   % 取矩阵的第M+1到N列组成噪声子空间
    Pmusic(iang)=1/(a'*En*En'*a);
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
