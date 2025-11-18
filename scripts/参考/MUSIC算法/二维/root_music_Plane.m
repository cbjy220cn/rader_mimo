clear; close all;
%%%%%%%% MUSIC for Plane%%%%%%%%
derad = pi/180;      %角度->弧度
c=3e8;
Nx =8;              % x方向阵元个数 
Ny =8;              % y方向阵元个数
M = 3;               % 信源数目
snr = 20;            % 信噪比
K = 100;             % 快拍数
f= 3e6;              %信号频率Hz
lambda=c/f;          %信号波长
dd = lambda*0.5;            % 阵元间距 
dx=0:dd:(Nx-1)*dd;
dy=0:dd:(Ny-1)*dd;
fangwei=[15,25,35]; %信号方位角[0-360]
yangjiao=[10,20,30]; %信号俯仰角[0-90]
N=800;

xaxaxa=0;
yayaya=0;


%%%%%%构建信号模型%%%%%%
for n=1:N 

S=randn(M,K);             %信源信号，入射信号
Ax=exp(-1i*2*pi*dx'*cos(fangwei*derad).*sin(yangjiao*derad)/lambda);  %x方向阵列流形
Ay=exp(-1i*2*pi*dy'*sin(fangwei*derad).*sin(yangjiao*derad)/lambda);  %y方向阵列流形
%构建X方向信号
Sx=zeros(M,K*Ny);
for i=1:Ny
    Sx(:,K*(i-1)+1:K*i)=diag(Ay(i,:))*S;
end
Xx=Ax*Sx;
X1=awgn(Xx,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算x协方差矩阵
Rxx=X1*X1'/K;
% 特征值分解并取得噪声子空间
[Ux,Dx] = eig(Rxx);                                       % 特征值分解
%[Dx,Ix] = sort(diag(Dx));                                % 将特征值排序从小到大
%Ux = fliplr(Ux(:, Ix));                                  % 对应特征矢量排序，fliplr 之后，较大特征值对应的特征矢量在前面
%Unx = Ux(:, M+1:Nx);                                     % 噪声子空间
Unx=Ux(:,1:Nx-M);
Gnx = Unx*Unx';
% 提取多项式系数并对多项式求根
coex = zeros(1, 2*Nx-1);
for i = -(Nx-1):(Nx-1)
    coex(i+Nx) = sum(diag(Gnx,i));
end
rx1 = roots(coex);                                       % 利用roots函数求多项式的根
rx = rx1(abs(rx1)<1);                                      % 找出在单位圆里的根
[lamda,I] = sort(abs(abs(rx)-1));                      % 挑选出最接近单位圆的K个根
Thetax = rx(I(1:M));
angle1=angle(Thetax)/pi;


%------------------------------------------------------------------------
%构建Y方向信号
Sy=zeros(M,K*Nx);
for i=1:Nx
    Sy(:,K*(i-1)+1:K*i)=diag(Ax(i,:))*S;
end
Xy=Ay*Sy;
Y1=awgn(Xy,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算x协方差矩阵
Ryy=Y1*Y1'/K;
[Uy,Dy] = eig(Ryy);                                       % 特征值分解
%[Dy,Iy] = sort(diag(Dy));                                % 将特征值排序从小到大
%Uy = fliplr(Uy(:, Iy));                                  % 对应特征矢量排序，fliplr 之后，较大特征值对应的特征矢量在前面
%Uny = Uy(:, M+1:Ny); 
Uny=Uy(:,1:Ny-M);% 噪声子空间
Gny = Uny*Uny';
% 提取多项式系数并对多项式求根
coey = zeros(1, 2*Ny-1);
for i = -(Ny-1):(Ny-1)
    coey(i+Ny) = sum(diag(Gny,i));
end
ry = roots(coey);                                       % 利用roots函数求多项式的根
ry = ry(abs(ry)<1);                                      % 找出在单位圆里的根
[lamda,I] = sort(abs(abs(ry)-1));                      % 挑选出最接近单位圆的K个根
Thetay = ry(I(1:M));
angle2=angle(Thetay)/pi;


%L1=[X1;Y1];
A = [Ax;Ay];
L1=awgn(A*Sy,snr,'measured');
RL=L1*L1'/K;
%对协方差矩阵进行特征分解
[V3,D3]=eig(RL);
Un3=V3(:,1:Nx+Ny-M);
Gn3=Un3*Un3';
azi=zeros(1,M);
ele=zeros(1,M);
%生成M！种排列的导向向量
for i=1:3
for j=1:4-i
    
    a1 = exp(-1i*(0:Nx-1).'*dd*2*pi*angle1(1)/lambda);
    a2 = exp(-1i*(0:Ny-1).'*dd*2*pi*angle2(j)/lambda);
    a = [a1;a2];
    t(j)=a'*Gn3*a;
end
t;
[t1,t2]=sort(t);
t=[];
%disp(t);
%disp(t2);
%angle2;
%angle2(t2);
t2=t2(1); %选中a2中第几个
fai=atan(angle2(t2)./angle1(1))*180/pi;
the=asin(sqrt(angle1(1).^2+angle2(t2).^2))*180/pi;
azi(i)=fai;
ele(i)=the;

angle1(1)=[];
angle2(t2)=[];
end




scatter(ele,azi,'k','.');
hold on;

fafafa=sort(azi);
xaxaxa=xaxaxa+(fafafa-fangwei).^2;
yayaya=yayaya+abs(fafafa-fangwei);
end
rrrr=(xaxaxa/N).^0.5;
vvvv=sum(rrrr)/M;

qqqq=(yayaya/N);
wwww=sum(qqqq)/M;
disp('RMSE');
disp(vvvv);

disp('绝对误差');
disp(wwww);
