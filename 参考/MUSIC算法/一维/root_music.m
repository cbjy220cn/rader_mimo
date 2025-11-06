clear; close all;
%%%%%%%% MUSIC for Uniform Linear Array%%%%%%%%
derad = pi/180;      %角度->弧度
c=3e8;
N = 8;               % 阵元个数        
M = 5;               % 信源数目
theta = [-60,-45,10,45,50];  % 待估计角度[-90`90]
snr = 20;            % 信噪比
K = 200;             % 快拍数
f= 3e8;              %信号频率
lambda=c/f;          %信号波长
dd = lambda/2;            % 阵元间距
L=5;                 %迭代次数
d=0:dd:(N-1)*dd;
A=exp(-1i*2*pi*d'*sin(theta*derad)/lambda);  %方向矢量
lalala=0;
bababa=0;
for i=1:1000

%%%%构建信号模型%%%%%
S=randn(M,K);             %信源信号，入射信号
X=A*S;                    %构造接收信号
X1=awgn(X,snr,'measured'); %将白色高斯噪声添加到信号中
% 计算协方差矩阵
Rxx=X1*X1'/K;
Pss=Rxx^-L;     %通过迭代取代En
%%%root求根%%%%
a = zeros(2*N-1,1)';
for i=-(N-1):(N-1)
    a(i+N) = sum( diag(Pss,i) );
end
a1=roots(a);                                                      %使用ROOTS函数求出多项式的根                            
a2=a1(abs(a1)<1);                                           %找出在单位圆里且最接近单位圆的N个根
[www,I]=sort(abs(abs(a2)-1));                      %挑选出最接近单位圆的N个根
fangxiang=a2(I(1:M));                                                    %计算信号到达方向角
source_doa=asin(angle(fangxiang)*lambda/(2*pi*dd))*180/pi;
source_doa=sort(source_doa);

lalala=lalala+(theta-source_doa').^2;
bababa=bababa+abs(theta-source_doa');

%disp('角度');
%disp(theta);
%disp('source_doa');
%disp(source_doa);

end

aa=(lalala/10000).^0.5;
RMSE=sum(aa)/M;

bb=bababa/10000;
judui=sum(bb)/M;
disp('RMSE');
disp(RMSE);
disp('绝对误差');
disp(judui);



