% 均匀圆阵Music算法
%该算法假设已知俯仰角，求方位角
derad = pi/180;      %角度->弧度
N=10;%阵元个数; 阵元数目要大于2M
n=0:1:N-1; %阵元编号
M=3;%信号个数
c=3e8;
f= 3e6;              %信号频率Hz
lambda=c/f;          %信号波长
R=50;                  %圆阵半径
dw=R/lambda;          %半径波长比   
snr=20;              %信噪比
K = 512;             % 快拍数
fangwei=[30,60,100];%信号方位角
yangjiao=[10,60,80];


%虚拟线阵
L=floor(2*pi*dw);    %最高模式阶数
l=[-L:1:L];
C=diag(1i.^l);
V=sqrt(N)/N*exp(-1i*2*pi/N*n'*l);
F=C*V';

%阵列空间的阵列流形
fang=repmat(fangwei,N,1);
gamma=repmat(2*pi*n'/N,1,M);
yang=repmat(yangjiao,N,1);
A=exp(-1i*2*pi*dw*sin(yang*derad).*cos(fang*derad-gamma));
S=randn(M,K);             %信源信号，入射信号
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
Rxx=X1*X1'/K;
%波束空间的阵列流形
B=F*A;
Y=F*X1;
% 计算协方差矩阵
Ry=Y*Y'/K;
[tzxiangliang,tzzhi]=eig(Ry);
Nspace=tzxiangliang(:,1:2*L+1-M);%噪声子空间对应小的特征值（从小到大排列）

%计算贝塞尔函数
J=zeros(1,2*L+1);

for i=0:2*L
    J(i+1)=besselj(abs(i-L),2*pi*sin(30)*dw);%仰角取值会影响测向精度。
    
end

%求方位角
jieguo=zeros(1,M);

    Gn=diag(J)*(Nspace*Nspace')*diag(J);
    a = zeros(1,2*L+1);
    for j=-2*L:2*L
    a(j+2*L+1) = sum(diag(fliplr(Gn),j));
    end
    a1=roots(a);%求多项式的根
    %除掉大于1的增根（方程式在此条件下变形的）
    a2=a1(abs(a1)<1);
    angle1=angle(a2)/derad;
    angle2=angle1(angle1>0)
