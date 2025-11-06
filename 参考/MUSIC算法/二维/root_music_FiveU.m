clear; close all;
%%%%%%%% MUSIC for Uniform Linear Array%%%%%%%%
derad = pi/180;      %角度->弧度
c=3e8;
Nx =5;              % x方向阵元个数 
Ny = 5;              % y方向阵元个数
M = 3;               % 信源数目
snr = 20;            % 信噪比
K = 1000;             % 快拍数
f= 3e6;              %信号频率Hz
lambda=c/f;          %信号波长
dd = lambda*0.5;            % 阵元间距 
%dx=-dd:dd:(Nx-2)*dd;
%dy=-dd:dd:(Ny-2)*dd;
dx=[-100,-50,0,50,100];
dy=[-100,-50,0,50,100];
fangwei=[10,20,30]; %信号方位角[0-360]
yangjiao=[15 ,25,35]; %信号俯仰角[0-90]

xaxaxa=0;
yayaya=0;
cishu=200;
%------------------x方向信号处理-----------------
%%%%%%构建信号x模型%%%%%%
for i=1:cishu
    
S=randn(M,K);             %信源信号，入射信号幅度
Ax=exp(-1i*2*pi*dx'*cos(fangwei*derad).*sin(yangjiao*derad)/lambda);  %x方向阵列流形
X=Ax*S;                    %构造接收信号x
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中

% 计算x协方差矩阵
Rxx=X1*X1'/K;
[V,D]=eig(Rxx);
Un=V(:,1:Nx-M);%噪声子空间对应小的特征值（从小到大排列）
Gn=Un*Un';
%找出多项式的系数，并按阶数从高至低排列
a = zeros(2*Nx-1,1)';
for i=-(Nx-1):(Nx-1)
    a(i+Nx) = sum( diag(Gn,i));
end
%使用ROOTS函数求出多项式的根
a1=roots(a);%求多项式的根
%除掉大于1的增根（方程式在此条件下变形的）
a2=a1(abs(a1)<1);
%挑选出最接近单位圆的N个根
[ind,I]=sort(abs(abs(a2)-1));
f1=a2(I(1:M));
angle1=angle(f1);

%----------------y方向信号处理-----------------
Ay=exp(-1i*2*pi*dy'*sin(fangwei*derad).*sin(yangjiao*derad)/lambda);  %y方向阵列流形
Y=Ay*S;                    %构造接收信号y
Y1=awgn(Y,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算y协方差矩阵
Ryy=Y1*Y1'/K;
[V2,D2]=eig(Ryy);
Un2=V2(:,1:Ny-M);
Gn2=Un2*Un2';
%找出多项式的系数，并按阶数从高至低排列
b = zeros(2*Ny-1,1)';
for i=-(Ny-1):(Ny-1)
    b(i+Ny) = sum( diag(Gn2,i) );%Gn的第i个对角元的和，放进行向量a中。
end
%使用ROOTS函数求出多项式的根
b1=roots(b);%求多项式的根
%除掉大于1的增根（方程式在此条件下变形的）
b2=b1(abs(b1)<1);
[ind2,I2]=sort(abs(abs(b2)-1));
f2=b2(I(1:M));
angle2=angle(f2);

%--------对x，y信号进行配对---------------------

A=[Ax;Ay];  %整体导向向量
L=A*S;
L1=awgn(L,snr,'measured'); %将白色高斯噪声添加到信号中
RL=L1*L1'/K;
%对协方差矩阵进行特征分解
[V3,D3]=eig(RL);
Un3=V3(:,1:Nx+Ny-M);
Gn3=Un3*Un3';

%生成M！种排列的导向向量

daoxiang1=exp(-1i*dx'*angle1'/dd);


m1=perms([1:M]);%所有排列可能
for i=1:factorial(M)
    m2=m1(i,:);%索引矩阵
    m3=angle2(m2);
    daoxiang2=exp(-1i*dy'*m3'/dd);
    dao=[daoxiang1;daoxiang2];
    for j=1:M
        t(j)=dao(:,j)'*Gn3*dao(:,j);
    end
    pmusic(i)=sum(t);
    
end
pmusic;%正确的角度对应的导向向量与噪声矩阵正交，得到的t接近于0

[p1,p2]=sort(pmusic);%根据顺序排列，看看那一组对应的值最小，就是那一组
p3=m1(p2(1),:);%从m1中找到对应的一行
angle2(p3);%对应的angle2

azi=atan(angle2(p3)./angle1)*180/pi;
alpha=angle1*lambda/(2*pi*dd);
beta=angle2(p3)*lambda/(2*pi*dd);
ele=asin(sqrt(alpha.^2+beta.^2))*180/pi;

for i=1:length(azi)%调整范围
    if azi(i)<0
        azi(i)=azi(i)+180;
    end
end

scatter(azi, ele,'k','.');
hold on;


fafafa=sort(azi);
xaxaxa=xaxaxa+(fafafa-fangwei').^2;
yayaya=yayaya+abs(fafafa-fangwei');

end

rrrr=(xaxaxa/cishu).^0.5;
vvvv=sum(rrrr)/M;
qqqq=(yayaya/cishu);
wwww=sum(qqqq)/M;
disp('RMSE');
disp(vvvv);

disp('绝对误差');
disp(wwww);

