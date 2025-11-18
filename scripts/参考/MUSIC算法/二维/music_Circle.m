% 均匀圆阵Music算法

derad = pi/180;      %角度->弧度
N=10;%阵元个数; 
n=0:1:N-1; %阵元编号
M=1;%信号个数
c=3e8;
f= 3e6;              %信号频率Hz
lambda=c/f;          %信号波长
R=50;                  %圆阵半径
dw=R/lambda;          %半径波长比   
snr=20;              %信噪比
K = 512;             % 快拍数
fangwei=[90];%信号方位角
yangjiao=[70];

%构建阵列流形
fang=repmat(fangwei,N,1);
gamma=repmat(2*pi*n'/N,1,M);
yang=repmat(yangjiao,N,1);
A=exp(-1i*2*pi*dw*sin(yang*derad).*cos(fang*derad-gamma));
%构建信号模型
S=randn(M,K);             %信源信号，入射信号
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算协方差矩阵
Rxx=X1*X1'/K;
[tzxiangliang,tzzhi]=eig(Rxx);
Nspace=tzxiangliang(:,1:N-M);%噪声子空间对应小的特征值（从小到大排列）

% 遍历每个角度，计算空间谱
for azi=1:1:180
     for ele=1:1:90
      
     az=repmat(azi,N,1);
     el=repmat(ele,N,1);
     daoxiang=exp(-1i*2*pi*dw*sin(el*derad).*cos(az*derad-2*pi*n'/N));
     
     Power=daoxiang'*Nspace*Nspace'*daoxiang; %在1-180度范围内进行计算
     P(ele,azi)=-10*log10(abs(Power));
     end
 end
figure;
mesh(P);
xlabel('方位角/(degree)');ylabel('仰角/(degree)');
zlabel('空间谱/db');
