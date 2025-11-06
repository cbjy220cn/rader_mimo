%ROOT_MUSIC ALOGRITHM
%DOA ESTIMATION BY ROOT_MUSIC
clear all;
close all;
% %% 
clc;
tic
N=100;
source_number=2;%信元数
sensor_number_x=9;%阵元x数
sensor_number_y=9;%阵元y数
N_x=512; %信号长度
snapshot_number=N_x;%快拍数
w=[pi/4 pi/6].';%信号频率
l=((2*pi*3e8)/w(1)+(2*pi*3e8)/w(2))/2;%信号波长  
d=0.5*l;%阵元间距
snr=0;%信噪比


fai1=30;fai2=40;%方位角
theta1=30;theta2=40;%俯仰角角度
fangwei=[30,40];
xaxaxa=0;
yayaya=0;

for n=1:N
Ax=[exp(-1i*(0:sensor_number_x-1)*d*2*pi*cos(fai1*pi/180)*sin(theta1*pi/180)/l);exp(-1i*(0:sensor_number_x-1)*d*2*pi*cos(fai2*pi/180)*sin(theta2*pi/180)/l)].';%阵列流型

s=10.^((snr/2)/10)*exp(1i*w*[0:N_x-1]);%仿真信号
%x=awgn(s,snr);

x=Ax*s+(1/sqrt(2))*(randn(sensor_number_x,N_x)+1i*randn(sensor_number_x,N_x));%加了高斯白噪声后的阵列接收信号

R=(x*x')/N_x;
%对协方差矩阵进行特征分解
[V,D]=eig(R);
Un=V(:,1:sensor_number_x-source_number);
Gn=Un*Un';

%找出多项式的系数，并按阶数从高至低排列
a = zeros(2*sensor_number_x-1,1)';%1*15的0矩阵
for i=-(sensor_number_x-1):(sensor_number_x-1)
    a(i+sensor_number_x) = sum( diag(Gn,i) );%Gn的第i个对角元的和，放进行向量a中。
end
%使用ROOTS函数求出多项式的根
a1=roots(a);%求多项式的根
%除掉大于1的增根（方程式在此条件下变形的）
a2=a1(abs(a1)<1);
%disp('a1');
%disp(a1);
%disp('a2');
%disp(a2);
%挑选出最接近单位圆的N个根
[lamda,I]=sort(abs(abs(a2)-1));
%disp('lamda');
%disp(lamda);
%disp('I');
%disp(I);
f1=a2(I(1:source_number));
angle1=angle(f1)/pi;
%计算信号到达方向角
%source_doa=[asin(angle(f(1))/pi)*180/pi asin(angle(f(2))/pi)*180/pi];
%source_doa=sort(source_doa);

%disp('source_doa');
%disp(source_doa);
Ay=[exp(-1i*(0:sensor_number_y-1)*d*2*pi*sin(fai1*pi/180)*sin(theta1*pi/180)/l);exp(-1i*(0:sensor_number_y-1)*d*2*pi*sin(fai2*pi/180)*sin(theta2*pi/180)/l)].';
s2=10.^((snr/2)/10)*exp(1i*w*[0:N_x-1]);%仿真信号
%x=awgn(s,snr);
x2=Ay*s2+(1/sqrt(2))*(randn(sensor_number_y,N_x)+1i*randn(sensor_number_y,N_x));%加了高斯白噪声后的阵列接收信号

R2=(x2*x2')/N_x;
%对协方差矩阵进行特征分解
[V2,D2]=eig(R2);

Un2=V2(:,1:sensor_number_y-source_number);
Gn2=Un2*Un2';

%找出多项式的系数，并按阶数从高至低排列
b = zeros(2*sensor_number_y-1,1)';%1*15的0矩阵
for i=-(sensor_number_y-1):(sensor_number_y-1)
    b(i+sensor_number_y) = sum( diag(Gn2,i) );%Gn的第i个对角元的和，放进行向量a中。
end
%使用ROOTS函数求出多项式的根
b1=roots(b);%求多项式的根
%除掉大于1的增根（方程式在此条件下变形的）
b2=b1(abs(b1)<1);
%disp('a1');
%disp(a1);
%disp('a2');
%disp(a2);
%挑选出最接近单位圆的N个根
[lamda2,I2]=sort(abs(abs(b2)-1));
%disp('lamda');
%disp(lamda);
%disp('I');
%disp(I);
f2=b2(I2(1:source_number));
angle2=angle(f2)/pi;



A = [Ax;Ay];
x3=A*s2+(1/sqrt(2))*(randn(sensor_number_x+sensor_number_y,N_x)+1i*randn(sensor_number_x+sensor_number_y,N_x));
R3=(x3*x3')/N_x;
%对协方差矩阵进行特征分解
[V3,D3]=eig(R3);
Un3=V3(:,1:sensor_number_y+sensor_number_x-source_number);
Gn3=Un3*Un3';
azi=zeros(1,source_number);
ele=zeros(1,source_number);
for i=1:2
for j=1:3-i
    
    a1 = exp(-1i*(0:sensor_number_x-1).'*d*2*pi*angle1(1)/l);
    a2 = exp(-1i*(0:sensor_number_y-1).'*d*2*pi*angle2(j)/l);
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


scatter(azi,ele,'k','.');
hold on;

fafafa=sort(azi);
xaxaxa=xaxaxa+(fafafa-fangwei).^2;
yayaya=yayaya+abs(fafafa-fangwei);

end
rrrr=(xaxaxa/N).^0.5;
vvvv=sum(rrrr)/source_number;

qqqq=(yayaya/N);
wwww=sum(qqqq)/source_number;
disp('RMSE');
disp(vvvv);

disp('绝对误差');
disp(wwww);

