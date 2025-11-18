clear; close all;
%%%%%%%% MUSIC for Plane%%%%%%%%
derad = pi/180;      %角度->弧度
c=3e8;

Nx =5;              % x方向阵元个数 
Ny = 5;              % y方向阵元个数
M = 3;               % 信源数目
snr =5;            % 信噪比
K = 1000;             % 快拍数
f= 3e6;              %信号频率Hz
lambda=c/f;          %信号波长
dd = lambda*0.5;            % 阵元间距 
dx=0:dd:(Nx-1)*dd;
dy=0:dd:(Ny-1)*dd;
fangwei=[10,20,30]; %信号方位角[0-360]
yangjiao=[15,30,35]; %信号俯仰角[0-90]

%%%%%%构建信号模型%%%%%%
S=randn(M,K);             %信源信号，入射信号
Ax=exp(-1i*2*pi*dx'*cos(fangwei*derad).*sin(yangjiao*derad)/lambda);  %x方向阵列流形
Ay=exp(-1i*2*pi*dy'*sin(fangwei*derad).*sin(yangjiao*derad)/lambda);  %y方向阵列流形
A=zeros(Nx*Ny,M); %整体阵列流型
for i=1:Ny
    A(Nx*(i-1)+1:Nx*i,:)=Ax*diag(Ay(i,:));
    
end
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算协方差矩阵
Rxx=X1*X1'/K;
[tzxiangliang,tzzhi]=eig(Rxx);
Nspace=tzxiangliang(:,1:Nx*Ny-M);%噪声子空间对应小的特征值（从小到大排列）
Gn=Nspace*Nspace';
%遍历角度，计算空间谱
for azi=1:1:180
     for ele=1:1:90
         dao=exp(-1i*2*pi*dy'*sin(azi*derad)*sin(ele*derad)/lambda);
       
           for m=1:Ny
           
           daoxiang(Nx*(m-1)+1:Nx*m,:)=exp(-1i*2*pi*dx'*cos(azi*derad)*sin(ele*derad)/lambda)*dao(m);
           end

         Power=daoxiang'*Gn*daoxiang; %在1-180度范围内进行计算
         P(ele,azi)=-10*log10(abs(Power));
     end
 end
figure;
 mesh(P);
 
 xlabel('方位角');ylabel('仰角');
 zlabel('空间谱/db');
 grid;





