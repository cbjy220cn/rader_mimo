clear; clc; close all
%% Radar parameters
c = physconst('LightSpeed'); %speed of light
BW = 50e6; %bandwidth
f0 = 3000e6; % carrier frequency
numADC = 361; % # of adc samples
numChirps = 256; % # of chirps per frame

numCPI = 10;
T = 10e-3; % PRI 
PRF = 1/T;
F = numADC/T; % sampling frequency
dt = 1/F; % sampling interval
slope = BW/T;
lambda = c/f0;



N = numChirps*numADC*numCPI; % total # of adc samples
t = linspace(0,T*numChirps*numCPI,N); % time axis, one frame

% size(0:dt:dt*numADC-dt)
t_onePulse=0:dt:dt*numADC-dt;
Vmax = lambda/(T*4); % Max Unamb velocity m/s


%%

%% Targets

r1_radial = 660;
tar1_theta = 30;
tar1_phi=60;
r1_x = cosd(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_y = sind(tar1_phi)*sind(tar1_theta)*r1_radial;
r1_z = cosd(tar1_theta)*r1_radial;

v1_radial = 0.001; % velocity 1
v1_x = cosd(tar1_phi)*cosd(tar1_theta)*v1_radial;
v1_y = sind(tar1_phi)*cosd(tar1_theta)*v1_radial;
v1_z = sind(tar1_theta)*v1_radial;
r1 = [r1_x r1_y r1_z];


tar1_loc = zeros(length(t),3);
tar1_loc(:,1) = r1(1) + v1_x*t;
tar1_loc(:,2) = r1(2) + v1_y*t;
tar1_loc(:,3) = r1(3) + v1_z*t;

scatter3(tar1_loc(1,1),tar1_loc(1,2),tar1_loc(1,3),'r','filled') %[output:750f0dcb]
hold on %[output:750f0dcb]

r2_radial = 600;
tar2_theta = 30.5;
tar2_phi = 60.5;
r2_x = cosd(tar2_phi)*sind(tar2_theta)*r2_radial;
r2_y = sind(tar2_phi)*sind(tar2_theta)*r2_radial;
r2_z = cosd(tar2_theta)*r2_radial;

v2_radial = 0.1; % velocity 1
v2_x = cosd(tar2_phi)*sind(tar2_theta)*v2_radial;
v2_y = sind(tar2_phi)*sind(tar2_theta)*v2_radial;
v2_z = cosd(tar2_theta)*v2_radial;
r2 = [r2_x r2_y r2_z];


tar2_loc = zeros(length(t),3);
tar2_loc(:,1) = r2(1) + v2_x*t;
tar2_loc(:,2) = r2(2) + v2_y*t;
tar2_loc(:,3) = r2(3) + v2_z*t;

scatter3(tar2_loc(1,1),tar2_loc(1,2),tar2_loc(1,3),'blue','filled') %[output:750f0dcb]
xlabel('x'); %[output:750f0dcb]
ylabel('y'); %[output:750f0dcb]
zlabel('z'); %[output:750f0dcb]
hold off %[output:750f0dcb]

%%
numTX = 1;
numRX = 8;

% N_L=floor(2*f0/BW);
% N_L=(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L); % 扩展的虚拟阵列
N_L=1;
N_num=2*(N_L-1)+1 %[output:03206bb2]

% 均匀圆阵参数
R_tx = 0; % 发射阵列半径（根据实际调整）
R_rx = 0.05; % 接收阵列半径

tr_vel=2*pi/N_num/numRX/T;
% tr_vel=0;

% 生成发射天线位置（均匀分布在圆上）
theta_tx = linspace(0, 2*pi, numTX+1); theta_tx(end) = [];
tx_loc = cell(numTX);
tx_loc_t= cell(numTX);
for i = 1:numTX
    tx_loc{i} = [R_tx*cos(theta_tx(i)), R_tx*sin(theta_tx(i)), 0];

    tx_loc_t{i}=zeros(length(t),3);
    tx_loc_t{i}(:,1)=R_tx*cos(theta_tx(i)+tr_vel*t);
    tx_loc_t{i}(:,2)=R_tx*sin(theta_tx(i)+tr_vel*t);
    tx_loc_t{i}(:,3)=0;

    scatter3(tx_loc{i}(1),tx_loc{i}(2),tx_loc{i}(3),'b','filled') %[output:3a0b1a00]
    hold on
end

% 生成接收天线位置
theta_rx = linspace(0, 2*pi, numRX+1); theta_rx(end) = [];
rx_loc = cell(numRX,1);
rx_loc_t = cell(numRX,1);
for i = 1:numRX
    rx_loc{i} = [R_rx*cos(theta_rx(i)), R_rx*sin(theta_rx(i)), 0];

    rx_loc_t{i}=zeros(length(t),3);
    rx_loc_t{i}(:,1)=R_rx*cos(theta_rx(i)+tr_vel*t);
    rx_loc_t{i}(:,2)=R_rx*sin(theta_rx(i)+tr_vel*t);
    rx_loc_t{i}(:,3)=0;

    scatter3(rx_loc{i}(1),rx_loc{i}(2),rx_loc{i}(3),'r','filled') %[output:3a0b1a00]
    hold on
end
xlabel('x'); %[output:3a0b1a00]
ylabel('y'); %[output:3a0b1a00]
zlabel('z'); %[output:3a0b1a00]
hold off %[output:3a0b1a00]

theta_circle = linspace(0, 2*pi, numRX*(2*(N_L-1)+1)+1); theta_circle(end) = [];
circle_loc=cell(numRX*(2*(N_L-1)+1),1);
for i = 1:numRX*(2*(N_L-1)+1)
    circle_loc{i} = [R_rx*cos(theta_circle(i)), R_rx*sin(theta_circle(i)), 0];
end

%%
P1 = rx_loc{1,1};
P2 = r1;


% 计算两点之间的距离
d = sqrt((P2(1) - P1(1))^2 + (P2(2) - P1(2))^2 + (P2(3) - P1(3))^2);

% 计算仰角
theta = acos((P2(3) - P1(3)) / d); % 仰角，单位为弧度

% 计算方向角
phi = atan2(P2(2) - P1(2), P2(1) - P1(1)); % 方向角，单位为弧度

% 如果需要，可以将弧度转换为度
theta_deg = rad2deg(theta);
phi_deg = rad2deg(phi);

% 输出结果
fprintf('仰角 (弧度): %.4f\n', theta); %[output:3e12932f]
fprintf('方向角 (弧度): %.4f\n', phi); %[output:7400464a]
fprintf('仰角 (度): %.4f\n', theta_deg); %[output:85c92fce]
fprintf('方向角 (度): %.4f\n', phi_deg); %[output:8a6169ee]

%%
%% TX

delays_tar1 = cell(numTX,numRX);
delays_tar2 = cell(numTX,numRX);

for i = 1:numTX
    for j = 1:numRX
        delays_tar1{i,j} = (vecnorm(tar1_loc-rx_loc_t{j},2,2) ...
                        +vecnorm(tar1_loc-tx_loc_t{i},2,2))/c; 
        delays_tar2{i,j} = (vecnorm(tar2_loc-rx_loc_t{j},2,2) ...
                        +vecnorm(tar2_loc-tx_loc_t{i},2,2))/c;
    end
end

%%

%% Complex signal
phase = @(tx,fx) 2*pi*(fx.*tx+slope/2*tx.^2); % transmitted
snr=50;
mixed = cell(numTX,numRX);

for i = 1:numTX %[output:group:44a28a60]
    for j = 1:numRX
        disp(['Processing Channel: ' num2str(j) '/' num2str(numRX)]); %[output:3fca16f7]
        for k = 1:numChirps*numCPI
            phase_t = phase(t_onePulse,f0);
            phase_1 = phase(t_onePulse-delays_tar1{i,j}((k-1)*numADC+1:k*numADC)',f0); % received
            phase_2 = phase(t_onePulse-delays_tar2{i,j}((k-1)*numADC+1:k*numADC)',f0);
            
            signal_t((k-1)*numADC+1:k*numADC) = exp(1j*phase_t);
            signal_1((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_1));
            signal_2((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_2));
        end
        % mixed{i,j} =awgn(signal_1+signal_2,snr,'measured');
        
        mixed{i,j} = signal_1+signal_2;
    end
end %[output:group:44a28a60]


%%

figure %[output:3a97c91f]
subplot(3,1,1) %[output:3a97c91f]
p1 = plot(t, real(signal_t)); %[output:3a97c91f]
title('TX') %[output:3a97c91f]
xlim([0 0.1e-1]) %[output:3a97c91f]
xlabel('Time (sec)'); %[output:3a97c91f]
ylabel('Amplitude'); %[output:3a97c91f]
subplot(3,1,2) %[output:3a97c91f]
p2 = plot(t, real(signal_1)); %[output:3a97c91f]
title('RX') %[output:3a97c91f]
xlim([0 0.1e-1]) %[output:3a97c91f]
xlabel('Time (sec)'); %[output:3a97c91f]
ylabel('Amplitude'); %[output:3a97c91f]
subplot(3,1,3) %[output:3a97c91f]
p3 = plot(t,real(mixed{1,1,1})); %[output:3a97c91f]
title('Mixed') %[output:3a97c91f]
xlim([0 0.1e-1]) %[output:3a97c91f]
xlabel('Time (sec)'); %[output:3a97c91f]
ylabel('Amplitude'); %[output:3a97c91f]
%%

%% Post processing - 2-D FFT
% size(cat(3,mixed{:}))



RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX); % radar data cube

RDC_plus=zeros(numADC,(numChirps-2*(N_L-1))*numCPI,numRX*numTX*(2*(N_L-1)+1));
    
for i = 1:numCPI
    for j = 1:numChirps-2*(N_L-1)
        for k = 1:numRX*numTX
            RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L+(k-1)*(2*(N_L-1)+1))=RDC(:,j+(N_L-1)+(i-1)*(numChirps-2*(N_L-1)),k);
            for l =1:N_L-1                
                RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L+l+(k-1)*(2*(N_L-1)+1))=RDC(:,j+(N_L-1)+l+(i-1)*(numChirps-2*(N_L-1)),k);
                RDC_plus(:,j+(i-1)*(numChirps-2*(N_L-1)),N_L-l+(k-1)*(2*(N_L-1)+1))=RDC(:,j+(N_L-1)-l+(i-1)*(numChirps-2*(N_L-1)),k);
            end
        end
    end
end
%扩展后仍需要根据新旋转后的位置对通道进行修正
RDC_plus_rot=zeros(numADC,(numChirps-2*(N_L-1))*numCPI,numRX*numTX*(2*(N_L-1)+1));
if tr_vel==0
       RDC_plus_rot=RDC_plus;
       delta_phi = 0; % 计算固定的方位角偏移量
else
    for i = 1:numChirps-2*(N_L-1)
        RDC_plus_rot(:,i,:)=circshift(RDC_plus(:,i,:), (i-1), 3);
    end
    delta_phi = 360/N_num/numRX; % 计算固定的方位角偏移量
end



numChirps_new=numChirps-2*(N_L-1);

DFmax = 1/2*PRF; % = Vmax/(c/f0/2); % Max Unamb Dopp Freq
dR = c/(2*BW); % range resol
Rmax = F*c/(2*slope); % TI's MIMO Radar doc
Rmax2 = c/2/PRF; % lecture 2.3
dV = lambda/(2*numChirps*T); % velocity resol, lambda/(2*framePeriod)

N_Dopp = numChirps_new; % length of doppler FFT
N_range = numADC; % length of range FFT
N_azimuth = numRX*numTX*(2*(N_L-1)+1);
R = 0:dR:Rmax-dR; % range axis
V = linspace(-Vmax, Vmax, numChirps_new); % Velocity axis
% 
ang_phi = -180:0.1:180; % angle axis
ang_theta=0:0.1:90;


% ang_phi = 50:0.01:70++360/N_num/numRX; % angle axis
% ang_theta=40:0.01:60;


RDMs = zeros(numADC,numChirps_new,numRX*numTX*(2*(N_L-1)+1),numCPI);
for i = 1:numCPI
    RD_frame = RDC_plus_rot(:,(i-1)*numChirps_new+1:i*numChirps_new,:);
    RDMs(:,:,:,i) = fftshift(fft2(RD_frame,N_range,N_Dopp),2);
    % RDMs(:,:,:,i) = RD_frame;
end




figure %[output:88f92da2]
imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))))); %[output:88f92da2]
colormap(jet(256)) %[output:88f92da2]
% set(gca,'YDir','normal')
clim = get(gca,'clim');
caxis([clim(1)/2 0]) %[output:88f92da2]
xlabel('Velocity (m/s)'); %[output:88f92da2]
ylabel('Range (m)'); %[output:88f92da2]
%%

%% CA-CFAR

numGuard = 2; % # of guard cells
numTrain = numGuard*2; % # of training cells
P_fa = 1e-5; % desired false alarm rate 
SNR_OFFSET = -1; % dB
RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

[RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET); %[output:893bbd25]
cfar_ranges_real=(cfar_ranges-1)*3;
figure
h=imagesc(V,R,RDM_mask);
xlabel('Velocity (m/s)')
ylabel('Range (m)')
title('CA-CFAR')
%%

%% Angle Estimation - FFT

rangeFFT = fft(RDC_plus_rot(:,1:numChirps_new,:),numADC);

angleFFT = fftshift(fft(rangeFFT,length(ang_phi),3),3);
range_az = squeeze(sum(angleFFT,2)); % range-azimuth map

figure
colormap(jet)
imagesc(ang_phi,R,20*log10(abs(range_az)./max(abs(range_az(:))))); 
xlabel('Azimuth Angle')
ylabel('Range (m)')
title('FFT Range-Angle Map')
set(gca,'clim', [-35, 0])

doas = zeros(K,length(ang_phi)); % direction of arrivals
figure
hold on; grid on;
for i = 1:K
    doas(i,:) = fftshift(fft(rangeFFT(cfar_ranges(i),cfar_dopps(i),:),length(ang_phi)));
    plot(ang_phi,10*log10(abs(doas(i,:))))
end
xlabel('Azimuth Angle')
ylabel('dB')



%%


%%
M = numCPI; % # of snapshots
% 提取所有传感器位置的x和y坐标
x_coords = cellfun(@(c) c(1), circle_loc);
y_coords = cellfun(@(c) c(2), circle_loc);
X = x_coords(:); % 转换为列向量 K×1
Y = y_coords(:); % K×1

% 生成所有角度组合(theta, phi)
num_theta = length(ang_theta);
num_phi = length(ang_phi);
[theta_grid, phi_grid] = ndgrid(ang_theta, ang_phi);
theta_list = theta_grid(:); % 列向量 M×1
phi_list = phi_grid(:);     % 列向量 M×1

% 计算角度相关的三角函数值
A_sind = sind(phi_list) .* sind(theta_list); % M×1
A_cosd = cosd(phi_list) .* sind(theta_list); % M×1

% 构造坐标矩阵和角度矩阵进行矩阵乘法
coords = [Y, X]; % K×2，列分别为Y和X坐标
A_matrix = [A_sind, A_cosd]; % M×2，列分别为sind(phi)sind(theta)和cosd(phi)sind(theta)

% 计算相位项 (矩阵乘法实现向量化)
phase = (2 * pi / lambda) * (coords * A_matrix.'); % K×M

% 计算导向矩阵
a1 = exp(-1i * phase);


music_spectrum = zeros(K, num_theta, num_phi);
[numSensors, num_angles] = size(a1);
block_size = 1000;  % 根据内存调整

for i = 1:K
    % 1. 计算协方差矩阵
    A_all = squeeze(RDMs(cfar_ranges(i), cfar_dopps(i), :, :));
    Rxx = A_all * (A_all' / M);
    
    % 2. 提取噪声子空间 (假设信号维度=1)
    [Qn, ~] = eigs(Rxx, numSensors - 1, 'smallestreal');
    Qn_H = Qn';
    
    % 3. 分块计算MUSIC谱
    spec = zeros(1, num_angles);
    for blk = 1:ceil(num_angles/block_size)
        idx = (blk-1)*block_size+1 : min(blk*block_size, num_angles);
        a1_block = a1(:, idx);
        a_proj = Qn_H * a1_block;
        spec(idx) = 1 ./ sum(abs(a_proj).^2, 1);
    end
    
    % 4. 重组为角度网格
    music_spectrum(i,:,:) = reshape(spec, [num_theta, num_phi]);
end



%%
%% Angle Estimation - MUSIC Pseudo Spectrum
circle_loc;


M = numCPI; % # of snapshots
% a1=zeros(numRX*numTX*(2*(N_L-1)+1),length(ang_theta)*length(ang_phi));
% for k = 1:numRX*numTX*(2*(N_L-1)+1)
% 
%        for i=1:length(ang_theta)
%             for j=1:length(ang_phi)
%                 % a1(k+(l-1)*(numRX*numTX+N_L-1),i,j ...
%                 %     )=exp(-1i*2*pi*(d_y*(l-1)*cos(ang_phi(j).'*pi/180)*sin(ang_theta(i).'*pi/180) ...
%                 %     +d_tx*(k-1)*sin(ang_phi(j).'*pi/180)*sin(ang_theta(i).'*pi/180)));
%                 a1(k,i+(j-1)*length(ang_theta) ...
%                     )=exp(-1i*2*pi/lambda*(circle_loc{k}(2)*sind(ang_phi(j))*sind(ang_theta(i)) ...
%                                    +circle_loc{k}(1)*cosd(ang_phi(j))*sind(ang_theta(i)) ) );
%             end
%        end
% 
% end
% music_spectrum=zeros(K,length(ang_theta),length(ang_phi));
% for i = 1:K
%     Rxx = zeros(numRX*numTX*(2*(N_L-1)+1),numRX*numTX*(2*(N_L-1)+1));
% 
%     % A = reshape(squeeze(RDMs(:,1,:,1)),numADC,(numRX*numTX+N_L-1)*Ny);
%     % 
%     % Rxx = (A'*A);
% 
%     for m = 1:M
%        A = squeeze(RDMs(cfar_ranges(i),cfar_dopps(i),:,m));
%        Rxx = Rxx + 1/M * (A*A');
%     end
% 
%     [Q,D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
%     [D, I] = sort(diag(D),'descend');
%     Q = Q(:,I); % Sort the eigenvectors to put signal eigenvectors first
%     Qs = Q(:,1); % Get the signal eigenvectors
%     Qn = Q(:,2:end); % Get the noise eigenvectors
%     for j = 1:length(ang_theta)
%         for k = 1:length(ang_phi)
%             music_spectrum(i,j,k)=(a1(:,j+(k-1)*length(ang_theta))'*a1(:,j+(k-1)*length(ang_theta)))/(a1(:,j+(k-1)*length(ang_theta))'*(Qn*Qn')*a1(:,j+(k-1)*length(ang_theta)));
%             % music_spectrum(i,j,k)=1/(a1(:,j,k)'*(Qn*Qn')*a1(:,j,k));
%         end    
%     end
% end
% a1(:,j,k)'*a1(:,j,k)
% (a1(:,j,k)'*a1(:,j,k))/(a1(:,j,k)'*(Qn*Qn')*a1(:,j,k))

%%
%% 方位角坐标修正
% 生成修正后的方位角坐标 (自动处理-180~180循环)
ang_phi_shifted = mod(ang_phi + delta_phi +180, 360) - 180; 

%% 数据与坐标对齐
% 获取修正后坐标的排序索引（保证单调性）
[ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);

% 对频谱数据按新方位角排序（沿phi轴重排）
spectrum_shifted = squeeze(music_spectrum(1,:,:)); 
spectrum_shifted = spectrum_shifted(:, sort_idx); 




figure;
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi_sorted,ang_theta,10*log10(abs(squeeze(spectrum_shifted))));
 



spectrum_data = 10 * log10(abs(reshape(spectrum_shifted, length(ang_theta), length(ang_phi_sorted))));

% 找到最大值及其索引
[max_value, max_index] = max(spectrum_data(:));
[row, col] = ind2sub(size(spectrum_data), max_index);

% 获取对应的角度
theta_max = ang_theta(row);
phi_max = ang_phi_sorted(col);

% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);
% fprintf('修正点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max-tar1_theta*0.1*lambda, phi_max+360/N_num/numRX, max_value);
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value);

% xlabel('方位角');ylabel('仰角');
xlabel('Azimuth angle');ylabel('Elevation angle');
zlabel('Spatial Spectrum/db');
grid;


% % 计算3dB带宽
% threshold = max_value - 20; % 3dB对应的值
% 
% % 找到小于阈值的索引
% [x_values,y_values]= find(spectrum_data >= threshold);
% 
% % 计算带宽
% x_min = min(x_values); % 3dB下降位置的最小值
% x_max = max(x_values); % 3dB下降位置的最大值
% x_bandwidth = x_max - x_min; % 计算带宽
% 
% y_min = min(y_values); % 3dB下降位置的最小值
% y_max = max(y_values); % 3dB下降位置的最大值
% y_bandwidth = y_max - y_min; % 计算带宽
% 
% % 输出带宽
% fprintf('方位角10dB带宽为: %.2f\n', x_bandwidth);
% fprintf('仰角10dB带宽为: %.2f\n', y_bandwidth);
%%
%% 方位角坐标修正
% 生成修正后的方位角坐标 (自动处理-180~180循环)
ang_phi_shifted = mod(ang_phi + delta_phi +180, 360) - 180; 

%% 数据与坐标对齐
% 获取修正后坐标的排序索引（保证单调性）
[ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);

% 对频谱数据按新方位角排序（沿phi轴重排）
spectrum_shifted = squeeze(music_spectrum(2,:,:)); 
spectrum_shifted = spectrum_shifted(:, sort_idx); 


figure;
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi_sorted,ang_theta,10*log10(abs(squeeze(spectrum_shifted))));
 

spectrum_data = 10 * log10(abs(reshape(spectrum_shifted, length(ang_theta), length(ang_phi_sorted))));

% 找到最大值及其索引
[max_value, max_index] = max(spectrum_data(:));
[row, col] = ind2sub(size(spectrum_data), max_index);

% 获取对应的角度
theta_max = ang_theta(row);
phi_max = ang_phi_sorted(col);

% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);

% fprintf('修正点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max-tar2_theta*0.1*lambda, phi_max+360/N_num/numRX, max_value);
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value);

xlabel('Azimuth angle');ylabel('Elevation angle');
zlabel('Spatial Spectrum/db');
grid;
%%



RDC_plus_rot_frame=RDC_plus_rot(:,1:numChirps_new,:);

M = numCPI; % # of snapshots

A = reshape(squeeze(RDC_plus_rot_frame(:,1:100,:)),numADC*100,numRX*numTX*(2*(N_L-1)+1));

Rxx = (A'*A);



[Q,D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
[D, I] = sort(diag(D),'descend');
Q = Q(:,I); % Sort the eigenvectors to put signal eigenvectors first
Qs = Q(:,1); % Get the signal eigenvectors
Qn = Q(:,3:end); % Get the noise eigenvectors
% for j = 1:length(ang_theta)
%     for k = 1:length(ang_phi)
%         music_spectrum(2,j,k)=(a1(:,j+(k-1)*length(ang_theta))'*a1(:,j+(k-1)*length(ang_theta)))/(a1(:,j+(k-1)*length(ang_theta))'*(Qn*Qn')*a1(:,j+(k-1)*length(ang_theta)));
%     end
% end

% 预计算噪声空间投影矩阵
QnQn = Qn * Qn';

% 提取所有角度对应的导向矢量矩阵 (numElements x numAngles)
A_all = a1; % 维度 [numRX*numTX*(2*(N_L-1)+1), length(ang_theta)*length(ang_phi)]

% 批量计算分子：每个导向矢量的自相关 (等效||a||^2)
numerator = sum(conj(A_all) .* A_all, 1); % 结果为行向量 [1 x numAngles]

% 批量计算分母：a'*(QnQn)*a
QnQn_A = QnQn * A_all;                     % 矩阵乘法加速核心
denominator = sum(conj(A_all) .* QnQn_A, 1); % 结果为行向量 [1 x numAngles]

% 计算MUSIC谱并重塑为二维矩阵
music_spectrum_2D = reshape(numerator ./ denominator, length(ang_theta), length(ang_phi));







%%


%% 方位角坐标修正
% 生成修正后的方位角坐标 (自动处理-180~180循环)
ang_phi_shifted = mod(ang_phi + delta_phi + 180+180, 360) - 180; 

%% 数据与坐标对齐
% 获取修正后坐标的排序索引（保证单调性）
[ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);

% 对频谱数据按新方位角排序（沿phi轴重排）
spectrum_shifted = squeeze(music_spectrum_2D); 
spectrum_shifted = spectrum_shifted(:, sort_idx); 




figure;
mesh(ang_phi_sorted,ang_theta,10*log10(abs(squeeze(spectrum_shifted))));
 
xlabel('Azimuth angle');ylabel('Elevation angle');
zlabel('Spatial Spectrum/db');
grid;
spectrum_data = 10 * log10(abs(reshape(spectrum_shifted, length(ang_theta), length(ang_phi))));

smoothed_spectrum=spectrum_data;

% 检测区域最大值
regional_max = imregionalmax(smoothed_spectrum);

% 获取峰值位置和数值
[max_rows, max_cols] = find(regional_max);
peak_values = smoothed_spectrum(regional_max);

% 合并信息并按峰值强度降序排序
peaks_info = sortrows([max_rows, max_cols, peak_values], -3);

% 提取前两个峰值
if size(peaks_info, 1) >= 1
    main_peak = peaks_info(1, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
            ang_theta(main_peak(1)), ang_phi_sorted(main_peak(2)));
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(main_peak(1)), mod(ang_phi_sorted(main_peak(2))+360/N_num/numRX+180, 360) - 180);
end

if size(peaks_info, 1) >= 2
    second_peak = peaks_info(2, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
            ang_theta(second_peak(1)), ang_phi_sorted(second_peak(2)));
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(second_peak(1)), mod(ang_phi_sorted(second_peak(2))+360/N_num/numRX+180, 360) - 180);
else
    disp('未找到明显的次峰值。');
end

%%
% DBF



% music_spectrum=zeros(K,length(ang_theta),length(ang_phi));
% 
% 
% A = reshape(squeeze(RDC_plus_rot_frame(:,1,:)),numADC*1,numRX*numTX*(2*(N_L-1)+1));
% Rxx = (A'*A);
% 
% 
% 
% % 提取所有角度对应的导向矢量矩阵 (numElements x numAngles)
% A_all = a1; % 维度 [numRX*numTX*(2*(N_L-1)+1), length(ang_theta)*length(ang_phi)]
% 
% % 批量计算分子：每个导向矢量的自相关 (等效||a||^2)
% numerator = sum(conj(A_all) .* A_all, 1); % 结果为行向量 [1 x numAngles]
% 
% % 批量计算分母：a'*(QnQn)*a 
% QnQn_A = Rxx * A_all;                     % 矩阵乘法加速核心
% denominator = sum(conj(A_all) .* QnQn_A, 1); % 结果为行向量 [1 x numAngles]
% 
% % 计算MUSIC谱并重塑为二维矩阵
% music_spectrum_2D = reshape(numerator ./ denominator, length(ang_theta), length(ang_phi));
% 
% 
% 
% 
% %% 方位角坐标修正
% % 生成修正后的方位角坐标 (自动处理-180~180循环)
% ang_phi_shifted = mod(ang_phi  + 180+180, 360) - 180; 
% 
% %% 数据与坐标对齐
% % 获取修正后坐标的排序索引（保证单调性）
% [ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);
% 
% % 对频谱数据按新方位角排序（沿phi轴重排）
% spectrum_shifted = squeeze(music_spectrum_2D); 
% spectrum_shifted = spectrum_shifted(:, sort_idx); 
% 
% 
% 
% figure;
% mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum_2D))));
% 
% xlabel('方位角');ylabel('仰角');
% zlabel('空间谱/db');
% grid;
% spectrum_data = 10 * log10(abs(reshape(music_spectrum_2D, length(ang_theta), length(ang_phi))));
% 
% 
% smoothed_spectrum=spectrum_data;
% % 检测区域最大值
% regional_max = imregionalmax(smoothed_spectrum);
% 
% % 获取峰值位置和数值
% [max_rows, max_cols] = find(regional_max);
% peak_values = smoothed_spectrum(regional_max);
% 
% % 合并信息并按峰值强度降序排序
% peaks_info = sortrows([max_rows, max_cols, peak_values], -3);
% 
% % 提取前两个峰值
% if size(peaks_info, 1) >= 1
%     main_peak = peaks_info(1, :);
%     fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
%             ang_theta(main_peak(1)), ang_phi(main_peak(2)));
%         fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
%             ang_theta(main_peak(1)), ang_phi(main_peak(2)));
% end
% 
% if size(peaks_info, 1) >= 2
%     second_peak = peaks_info(2, :);
%     fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
%             ang_theta(second_peak(1)), ang_phi(second_peak(2)));
%         fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
%             ang_theta(second_peak(1)), ang_phi(second_peak(2)));
% else
%     disp('未找到明显的次峰值。');
% end
%%


%% Point Cloud


% 
% [~, ang] = max(music_spectrum(2,:,:));
% a_phi_1 = ang_phi(ang(2));
% a_theta_1=ang_theta(ang(1));
% 
% [~, I] = max(music_spectrum(1,:));
% angle2 = ang_phi(I);
% 
% coor1 = [cfar_ranges_real(2)*sind(angle1) cfar_ranges_real(2)*cosd(angle1) 0];
% coor2 = [cfar_ranges_real(1)*sind(angle2) cfar_ranges_real(1)*cosd(angle2) 0];
% figure
% hold on;
% title('3D Coordinates (Point Cloud) of the targets')
% scatter3(coor1(1),coor1(2),coor1(3),100,'m','filled','linewidth',9)
% scatter3(coor2(1),coor2(2),coor2(3),100,'b','filled','linewidth',9)
% xlabel('Range (m) X')
% ylabel('Range (m) Y')
% zlabel('Range (m) Z')
%%

% %% MUSIC Range-AoA map
% rangeFFT = fft(RDC_plus_rot);
% for i = 1:N_range
%     Rxx = zeros(numRX*numTX*(2*(N_L-1)+1),numRX*numTX*(2*(N_L-1)+1));
%     for m = 1:M
%        A = squeeze(sum(rangeFFT(i,(m-1)*numChirps_new+1:m*numChirps_new,:),2));
%        Rxx = Rxx + 1/M * (A*A');
%     end
% %     Rxx = Rxx + sqrt(noise_pow/2)*(randn(size(Rxx))+1j*randn(size(Rxx)));
%     [Q,D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
%     [D, I] = sort(diag(D),'descend');
%     Q = Q(:,I); % Sort the eigenvectors to put signal eigenvectors first
%     Qs = Q(:,1); % Get the signal eigenvectors
%     Qn = Q(:,2:end); % Get the noise eigenvectors
% 
%     for k=1:length(ang_phi)
%         music_spectrum2(k)=(a1(:,k)'*a1(:,k))/(a1(:,k)'*(Qn*Qn')*a1(:,k));
%     end
% 
%     range_az_music(i,:) = music_spectrum2;
% end
% 
% figure
% colormap(jet)
% imagesc(ang_phi,R,20*log10(abs(range_az_music)./max(abs(range_az_music(:))))); 
% xlabel('Azimuth')
% ylabel('Range (m)')
% title('MUSIC Range-Angle Map')
% clim = get(gca,'clim');
%%

%% Angle Estimation - Compressed Sensing

% num_phi = length(ang_phi); % divide FOV into fine grid
% 
% B = a1; % steering vector matrix or dictionary, also called basis matrix
% % P=zeros(length(ang_phi),length(ang_theta))
% 
% for i = 1:K
%     A = squeeze(RDMs(cfar_ranges(i),cfar_dopps(i),:,1));
%     cvx_begin
%         variable s(length(ang_theta)*length(ang_phi)) complex; %alphax(numTheta,1) phix(numTX*numRX,numTheta)...
% 
%         minimize(norm(s,1))
%         norm(A-B*s,2) <= 1;
%     cvx_end
%     cvx_status
%     cvx_optval
% 
%     P=reshape(s,length(ang_theta),length(ang_phi));
% 
%     figure
%     grid on;
%     mesh(ang_phi,ang_theta,10*log10(abs(P)))
%     title('Angle Estimation with Compressed Sensing')
%     xlabel('Azimuth')
%     ylabel('dB')
%     grid;
%     % 找到最大值及其索引
%     [max_value, max_index] = max(P(:));
%     [row, col] = ind2sub(size(spectrum_data), max_index);
% 
%     % 获取对应的角度
%     theta_max = ang_theta(row);
%     phi_max = ang_phi(col);
% 
%     % 输出结果
%     fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max+numRX*numTX*(2*(N_L-1)+1), max_value);
% 
% end



%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":34.4}
%---
%[output:750f0dcb]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdkAAAFjCAYAAABxFnERAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnU+oV8fZx8eAtfelN5abW\/UmvaYBX9OFryGLKndR0y660Wx1lbipIkgkFBQEL7QFBV90Z7hw0WxMV7pwU4UuSv9kIWYTMKVgYhf1UvWCShPhtTeF+vIcMz\/nd+6cc2bO\/H3mfH8g6r1z5szzfZ45n9\/MmWdmzdOnT58KfKAAFIACUAAKQAHvCqwBZL1rigqhABSAAlAAClQKALIIBCgABaAAFIACgRQAZAMJi2qhABSAAlAACgCyiAEoAAWgABSAAoEUAGQDCYtqoQAUgAJQAAoAsogBKAAFoAAUgAKBFABkAwmLaqEAFIACUAAKALKIASgABaAAFIACgRQAZAMJi2qhABSAAlAACgCyiAEoAAWgABSAAoEUAGQDCYtqoQAUgAJQAAoAsogBKAAFoAAUgAKBFABkAwmLaqEAFIACUAAKALKIASgABaAAFIACgRQAZAMJi2qhABSAAlAACgCyiAEoAAWgABSAAoEUAGQDCYtqoQAUgAJQAAoAsogBKAAFoAAUgAKBFABkAwmLaqEAFIACUAAKALKIASgABaAAFIACgRQAZAMJi2qhABSAAlAACgCyiAEoAAWgABSAAoEUAGQDCYtqoQAUgAJQAAoAsogBKAAFoAAUgAKBFABkAwmLaqEAFIACUAAKALKIASgABYwV+Oc\/\/1mVXVpaEk+ePBGbN28W3\/3ud8W3v\/1t4zpQEAoMSQFAdkjehq1QwFIBCdV\/\/etfgv7Qh4C6srIibt26JV577TWxdu3a6meAraW4KD4IBQDZQbgZRkIBMwUkTFWo0pUEUPXvL7\/8UvzlL38R27ZtE+vWrRMEY7qGYCuBa3ZHlIICZSsAyJbtX1gHBToVkKNV+bcOqvVKVMiuX7+++jVBVsJW1iHh3NkIFIAChSoAyBbqWJgFBZoU0I1W5QhU\/t2lng6y8hpZv4Q2ppK71MTvS1YAkC3Zu7ANCnyjQJ\/Rapt4bZBVr6P7ArYIwyErAMgO2fuwvVgFdFC1Ha36gKw6ulXf22KRVLGhB8NqCgCyCAkoUIACbVPAZJ7vd6OmI9m6tOp7W0wjFxB4MKFTAUC2UyIUgAJ5KuB7CtjGyr6QxcjWRmWULUEBQLYEL8KGQSjQlLOqTgPHEsIVsoBtLE\/hPqkVAGRTewD3hwINCpjmrKYQ0BdkdbCV09u+p7hT6IR7QgFAFjEABTJSIOUUsI0MviHbBFu8t7XxCsrmqAAgm6NX0KbBKOAjZzWFWKEgq9qC9J8UnsU9fSsAyPpWFPVBgQ4FuIxW28yIAVnd6BYjW3QvbgoAstw8hvayUyB0zmoKQWJCFrBN4WHc05cCgKwvJVEPFPhGgdg5qymETwFZwDaFp3FPVwUAWVcFcT0UEGK0daDNJvuchUsJWcCWc+QMr+2A7PB8Dos9KJBTzqoHc6yryAGyOtjSzyj1B+k\/1i7FBYEUAGQDCYtqy1Ig55zVFErnBNk22JqeKpRCQ9xzGAoAssPwM6zsoUB9wdLKykp1QDlGSkLkCFnVxUj\/6RHwuCSIAoBsEFlRKUcFunJWyab79++LTZs2CRohDfmTO2R1o1uk\/ww5YtPZDsim0x53zkABm5xVgjAg+8xpXCAL2GbQyQbeBEB24AEwNPNdclYB2efRwg2ygO3Qeno+9gKy+fgCLQmggM+cVUCWP2QB2wCdDFW2KgDIIkCKU8BmCtjGeEC2HMjWYUsj8zVr1ogNGzYg\/cemU6BspwKAbKdEKJC7ArFyVgHZ8iArLaIYun37tpicnMQK8tw7PLP2AbLMHIbmCpEqZxWQLReyqm\/p3\/KLG1Yk44njqgAg66ogro+iQKgpYJvGA7KBIfurXwnx618\/u8lPfvLs71\/+8vm\/bZxlWVbnW4o5+YUOsLUUFMVHCgCyCIYsFejKWU2RpwrIBoLsH\/8oxE9\/2hyHf\/hDcNC2+VaObOlvwDbLx0XWjQJks3bPsBqXw2i1TXFANgBkuwArb\/n0adDOYOJbwDaoC4qtHJAt1rX5G+aSs5rCOpMHcYp2pbintzxZGsESaLs+NG1M08mBPja+BWwDOaHQagHZQh2bo1k+c1ZT2GfzIE7Rvpj3jA5ZMi7gaLaPbwHbmBHH916ALF\/fsWh57lPANiL2eRDb1M+prDfIrlljbnZmkJUNV2FLP8MBEuYuHUJJQHYIXo5oY6yc1YgmjW4FyD5XPTpkabUxLYAK9PHhWzlTg\/SfQE5iWi0gy9RxuTQ7Vc5qCvt9PIhTtDvEPb1BVk3baWsoA8iqzcdReyGijmedgCxPvyVtdUlTwDZCArIBRrJUpcmUceA0nlC+xXtbmx5WZllAtky\/erUqx5xVrwYaVhbqQWx4+6yKeRvJklVdaTyBAUtNCO1bwDar8I3aGEA2qtx8bkaj1ZWVFbG8vCxefPHFquG0oEP9m481floa+kHsp5VxavEKWQlauduTTOkJnLajKhXLt4BtnPjM6S6AbE7eSNgW3RQwnUpCD9PNmzfjZJIIo52E7re+tXfIWrfA7wWxICtbDdj69V\/OtQGyOXsnYNtMclZjP3gCmuulaujxXEZA1ktIVdPUco9kOUskZ4z83AG1pFYAkE3tgYj3t12wBKiMOwd6ALKhuqsOtrRPcoo9ukPZONR6AdmCPe+aswqoALJN3QMj2XAPDqT\/hNM2Rc2AbArVA93Td84qIAvIArKBOqtBtXhvayASgyKALAMntTXRdgrYxlxAFpAFZG16TJiygG0YXWPVCsjGUtrTfWLmrAKygCwg66njeqgGsPUgYoIqANkEotveMuRota0tgCwgC8ja9tbw5QHb8Br7vENSyD569EgcOnRI3Lx5c8ym7du3i8XFRTE1NTX6+eXLl8X8\/Pzo\/ydPnhR79+4du+7GjRti\/\/79Yz+7ePGi2Llzp0\/NgteVyzmrgCwgC8gG7+69b1CHLa1ERvpPbzmDXZgUsrdv3xYHDx4Up0+fbgUhAfbSpUsj8D558kScOHFCzM3NjUAr69q9e7c4duxYJRhdt7CwIM6fPy+2bNkSTETXik1yVl3v0ed6QBaQBWT79Jy41yDXNq7etndLClkaeZ49e3bVqFU1Qo52d+zYMYKnBKgK3jqIqYwOxrYChSqfagrYxh5AFpAFZG16TPqySP9J74N6C5JClsB4\/fp1cerUKTExMaFVxxSyZ86cEffu3VtVV9PPY7vCNWc1dnvpfoAsIAvIpuh57vekPcfpmbNu3bpqQwuaRsbGFu669qkhGWTlKPPOnTvi4cOH4u7du1X79+zZswqUTdPFMzMz1ehW1iX\/rwpBkP3kk09aR8t9hOu6xnfOatf9QvwekAVkAdkQPSt8nbLvElzlswiwDa+77g7JICtHqLOzs2NQ1U37UsPlO1cJY3VBUxtkm+oLITeHKWAbuwFZQLYeLzLGl5aWqj65bds2QX2Y+yiptFiv24MVyTZPPr9lk0G2yQwJ33379lWLmiRAqbw6raxOA9PvaCGUbiQbErIxc1b9ut2sttIePGZWN5caoh5NrznoGMRbt26J1157Taxdu5b9lGRpvm2yB7B1fQrYX58tZOVCJ5mWU0\/FUVcmU8pPLMiWNlptC5nSHjz23WN4I1nT1xzq3sX03k+eJCM3teeWSlJarHfZA9i6Pg3Mr2cB2ePHj69Kw6mPePssfDp37lz1vvbIkSOCoK775JKzau5SfyW7Oqq\/O\/GoqVQ9+nxx1B0QwDmVpDTfmtrD2Wc8nhpCJIOsHIkePnx4bFOJeu4sjWR1kK2Xa0vh0U0jk4MIsBK09H8CLQF369ato8UC9HP1yClu39BdAtG0o7rcg9O1pejh4zVH2yk8sn4Jbw4LbkrxrexPtvZw9BmXZ0cyyJJAutFnPa2naVFT\/VrXzShU4P7ud78b7ZwyJKjWg9a2o3IJ+r7t5KxHn9Fqm06mR91xydvk7Fudn1zs4eKzvv049nVJIUvG1rdLPHDgwNimE1IQguqFCxdG+ujKuW6rSKB99913xUcffdQ4fRzbQSnv59JRU7Y71L056RH6NYcpZNWRlfreNre8TU6+NYlvH\/bgva2J0t1lkkO2u4nxSgCy41r76KjxvBf+Tjnr0TYFTMr4npGxhWzusM3Zt30i26c9gG0fDzy\/BpBV9ANkAdm27uTzweXWbZ9d7XsK2KZNfSGbK2xz862NL3RlQ9gD2PbzCiALyDZGToiO2i9M87gqtR45bc3pCtncYJvat74jPKQ9gK2dtwBZQBaQNewzIR9cTaMRdRpYlpFTv76ngA1lqIr5gqwOtvQzsi2mfbF9a6N1n7Ix7FFhm8JnfXRJcQ0gC8gCsoY9L8aDK+UUsKEMQSDbBNtY6T8xfGujr2vZmPbUYRvLZ64axboekAVkAVnD3hbiwdW2YEnNzzZsYrRivkeyuobHTCUJ4dtoztDcKJU9MX2WUl+bewOygCwga9hjfD24uIxW22SJAVnd6DbUKMmXbw1DKXix1Pbgve1zFwOygCwga\/jI6\/vgCp2zath8r8ViQjYGbPv61quoHivLxR7ANuG2ih7jyVtVSOEZlzKXjurNwY4VmerRNgVMTYi5oMfR5MbLU0A2JGxNfRtKT9\/15mbPkGGLkSxGshjJGj7h2h5cJUwBG8pQFUsJ2RCwzQ1KNr7Qlc3VniHCFpAFZAFZwyea+uCif9NHjlrp33KhUs4LlgxN7SyWA2R1sJUzBbazBblCqdMRDQVyt2dI6T+ALCALyBo8yeihQGBZWloSk5OTgs5QVad+bR\/qBrfMukhOkG2DrekXntyhZBsMXOzRwdbUZ7aapCoPyAKygGyDAvUp4JWVFUF\/Nm7cWP0Z8idHyKr+sE0l4QIl05jjaI+tz0y1SF0OkAVkAdlvFGhbsETfrulz\/\/59sWnTpmpqeMif3CGrG922pf9whFJb\/HG2R57WRDaEStmK2XcBWUB20JC1WbDE+cHl+6HCBbKmsC3NtyXYU8oiKUAWkB0UZF1yVkt4cPmCLTfIdsG2NN+WZA\/Zsry8LJ4+fSp+8IMf+ArhaPUAsoBs0ZD1mbNa0oPL9QnDFbJNsKVpSfoCVsqrgNJilbM9gCwgWxxkbaaAbWDDuaPb2GlSljtk67Alex4\/fixmZ2eLWNRWWqxytgeQBWTZQzbWOaucO7oJOG3KlAJZaTPF0O3bt0fpWbGP2rPR3qRsabHK2R5AFpBlB1ndFDAZEfqcVc4d3eTBbFOmNMiqvpULbkgPrqtbS4tVzvYAsoAsC8iGmgK2AQvnjm5jp0nZkiEr07M4p5KUFquc7QFkAdksIduVs5oiT5VzRzcBp02ZIUC2aZEUzZikiD8b\/5QWq5ztAWQB2Wwgm8Note1Bxrmj2zygTcoOCbIcYVtarHK2B5AFZJNB1iVn1QQEvstw7ui+tRgiZDnBtrRY5WwPIAvIRoOsz5xV39AwqY9zRzexz6bMkCHLAbalxSpnewBZQDYoZHOfArYBC+eObmOnSVlA9rlKcjUy\/U2fHNJ\/SotVzvYAsoCsV8jGylk1AYHvMpw7um8tANnVisqZGtkHUqb\/lBarnO0BZAFZJ8imyln1DQ2T+jh3dBP7bMoAsu1qpT62rbRY5WwPIAvIWkO2pClgG7Bw7ug2dpqUBWRNVBJCnUqOObItLVY52wPIArKdkJU7KclRK11ADwz1j9kjh3cpzh3dt\/KArJ2isWFbWqxytgeQBWS1Twsara6srFT7uU5PT4t169YF37bQ7rEVvzTnju5bLUC2n6KxYFtarHK2B5AFZCsFdFPAa9asEfQw3bx58wiw\/R4tZVzFuaP79gAg66ZoaNiWFquc7QFkBwpZ3YIlOf1LktAUMefAdnsE6q+GHs91AWT9RFio9J\/SYpWzPYDsgCBru2CJc2D7eQSO1wI9ANkQcUV16mCrfum1vW9pscrZHkC2YMi65qxyDmzbh5JJeegByJrEiWsZH+k\/pcUqZ3sA2YIgq5sCllO\/6t+mDwHOgW1qo0056AHI2sSLa1mX97alxSpnewBZ5pC1nQK26ficA9vGTtOy0AOQNY0Vn+X6wLa0WOVsDyDLDLJtC5Zc3uHoHgqcA9vnQ07WBT0A2RBxZVqnDWxLi1XO9gCyDCAbcrTa1sE5B7bpg8umHPQAZG3iJVRZE9iWFquc7UkK2UePHolDhw6JmzdvjsXj9u3bxeLiopiamhr9\/MaNG2L\/\/v2j\/+\/Zs0ecOnVKTExMNJahX1y8eFHs3LnTKN4\/+eQT8e6774qPPvpI7Nixw+iaEIVyOWeVc2CH8Av0AGRDxFXfOuuwlds2Un2lxSpne5JClnYTOnjwoDh9+nQrCC9fviyuX78+gqqE8+zs7Ohnsq7du3eLY8eOVXFL1y0sLIjz58+LLVu2dMZyKsi2TQFTo+W2hp0GeC7AObA9S1FVBz0A2RBx5VqnClv5vCDg3r9\/X2zatKna\/pT7h3PfSwpZGp2ePXt21ahVDQgJ1H379om9e\/eOjVqPHz8+AigB9dKlS2N1PXnyRJw4cULMzc2NXdsUcDEhm2oK2KazcQ5sGztNy0IPQNY0VlKVk+k\/tCXq48ePq8FFqi\/pPjXg3PeSQrY+QtU5xQTEdN2ZM2fEvXv3Vk0hN\/1cd6+QkHXNWfUZsKZ1cQ5sUxttykEPQNYmXlKWXV5eFktLS2JyclKsX7++Ai3nES3nvpcMsnKUeefOHfHw4UNx9+7dKibr71oliI8ePSref\/\/90fvbAwcOjKaFZV0zMzOjn8kAJ8gSPOvveENDVjcFLKdy1L9TdsSue3MO7C7b+vweegCyfeImxTUyVuX2qPT\/mEft+baZc99LBlnde1VyTH3alyB54cIF8fLLL4+9W1VHqHQdTQvrIKubRm4KANeRLIcpYJvg5xzYNnaaloUezw+SoFESrYPYtm2boLURnEdJ5P\/SfFu3x2RFsmk\/SFGOs3+SQbbJUfV3sBKy9VXC6qIpWo2cArJtC5Z856wisFMoMH5Pzh29r3pNrznond+tW7fEa6+9JtauXct6lDQEyEr\/c4Ut576XLWQphYZWCRNkr127tmqFsArjt99+OxpkSxuttj18OQd2X6gMXQ\/T1xzqKTx01jD1CzklqaaShPBDiDpLi\/Uue7jBtsueEDHhq87sIduUhqMb8doufDp37lz1vvbIkSNVXqxuujiXnFVfDreph3Ng29hpWrZUPfp8cdQddac+uElTeh\/IZWVrab41tYeLz0ztMe3LMcslg6yc7j18+PBYek09d7atHKXwUI4tLVNvS+HRvaslkQmqErT0fwla+hlNQZNj6VM\/ZzWmg1Lei3Ngh9CtFD18vOZoO09W1i\/hzWHBTSm+VaeFbfJkc\/cZZ\/8kgywFgy69RpfWo1shTD+jj9x4wnUzChW4VPeuXbuq+rl8EwdUQigwXifnjt5ntNqmqOmh7T6ObQvv2fIXPtlomKPPOPe9pJAlxxNU5+fnRzGgpuaogWFSrr71Il3PcVtFmw4RsiznwA6hCyc9Qr\/mMIWsOrJS39vmlrfJybcmse3Dnpze2\/qwx0S3EGWSQzaEUX3rdE3h6XvfXK\/jHNghNM1Zj7Yp4BAzMraQzR22Ofu2Tyz7tCcH2Pq0p4+eLtcAsop6gGw506MunaLp2tw6uu8pYBvN+kI2V9jm5lsbX+jKhrAnJWxD2OOqsen1gCwg2xgrnAPbtAPYlEutR05bc7pCNjfYpvatTRyalA1pTwrYhrTHRE+XMoAsIAvIGvag2B1dNwWsTv2mXJTnC7I62EobY9oX27eGIde7WAx7VNiG9lkMe3qL3XEhIAvIArKGvStGR085BWwoQ1XMN2SbYBsr\/SeGb230dS0b0546bEP4LKY9rtrXrwdkAVlA1rBXhejobQuWct6aMxRkVVfETCUJ4VvDsApSLJU9oXyWyh4fzgFkAVlA1rAn+eroXEarbbLEgKxudBtilET38eVbw1AKXiy1Pb7f26a2x8VhgCwgC8ga9qC+HT10zqph870WiwnZGLDt61uvonqsLBd7fME2F3v6uAiQBWQBWcOeY9rR26aA6VYxF\/QYmmZdLAVkQ8LW1LfWQiW6IDd7XGGbmz02bgVkAVlA1rDHtHX0EqaADWWoiqWEbAjYcn6I6\/yWqz19YZurPSZ9BpAFZAFZk55Se29HnZ4+ctRK\/5YLlXJesGRoamexHCCrg62cKbCdLeD8EOcE2b4+4+wfQBaQBWQ7kfIMpgSWpaUlMTk5KegMVXXq1\/ahbnDLrIvkBNm2B7fpFx7OD3GOkLX1GWf\/ALKALCDboEB9CnhlZUXQn40bN1Z\/hvzJEbKqP2xTSTg\/xDlD1tRnnP0DyAKygOw3CrQtWKIREX1szugsGcK5Q1Y3UmpL\/+H8EC8FstIOeVoT+UT6jHPfA2QB2UFD1mbBUmkPYpcvAVwgawrb0nxbgj3qIqk1a9ZUr2xeffXVCrycPoAsIDsoyLrkrJbw4PL1cOIG2S7YlubbkuwhW5aXl6s1EVu3bgVkfXXiFPXgqLtx1UvoqD5zVkvQw1e\/4grZJtjS6Ii+gG3atIndQ7y06eLS7MFIFiPZ4kayNlPANtABZJ+rxR2yddiSPY8fPxazs7NFLGorLVY52wPIArLsIRvrnFXOHd3my4RJ2VIgqy62uX379ig9i1KyOKdllRarnO0BZAFZdpDVTQGTEfKhGOrhyLmjm4DTpkxpkFV9KxfckB6hDiSw0bpP2dJilbM9gCwgywKyoaaAbR5gnDu6jZ0mZUuGrFy9qksl4bKytbRY5WwPIAvIZgnZrpzVFA87zh3dBJw2ZYYA2aZFUjRTkiL+bPxTWqxytgeQBWSzgWwOo9W2Bxnnjm7zgDYpOyTIcoRtabHK2R5AFpBNBlmXnFUTEPguw7mj+9ZiiJDlBNvSYpWzPa2QffTokTh06JDYsWOHOHbsmLafnjlzRlB+6eLiopiamvLdl6PWhzzZcbl9B7bPnNWogfHNzXzrkcIGX\/ccMmQ5wLa0WOVsjxFkb968KQ4cOKAFLSDr67GVXz0+Ajv3KWAb1X3oYXO\/nMsCss+9I1cj09\/0ySH9p7RY5WyPEWTXr18vPv74Y7Fnzx5x6tQpMTExMYowQDbnR6Fb2\/oEdqycVTfL+l3dR49+d8r\/KkB2tY\/kTI3sAynTf0qLVc72GEH26NGjVUTt379\/FWgB2fwfiH1baBLYuilg+W1e\/btvG3K6zkSPnNobsi2AbLu6tkft+fZVabHK2R5jyO7cuVPcuHGjAu327dtH72ABWd\/dI5\/6mgK7pClgG7U5d3QbO03KArImKonq5BiZbxtzZFtarHK2xwqyFFZ10H744YdY+GTW39iVkoEtd1CSo1YyhB4Y6h92xvVoMOeO3sPc1ksAWTtFY8O2tFjlbI81ZCm0aI\/PgwcPVlG2a9cu8de\/\/hWri+36XPal6dv3yspK5evp6Wmxbt264NsW5i4K547uW1tAtp+isWBbWqxytqcXZFXQ3r17d2z6uF\/o5XHVkFN4dFPAdFAyPUw3b97MerN0X9HFuaP70kDWA8i6KRoatqXFKmd7nDajkHm0FG7Ik3XrdLGv1i1YktO\/1BaaIuYc2CH0hB7PVQVk\/USYClvZ73wccFFarHK2xwmyfsIsn1pKH8naLljiHNghogp6ALIh4orq1MFW\/dJre9\/SYpWzPYCsEr2lQdY1Z5VzYNs+lEzKQw9A1iROXMv4SP8pLVY52wPIFgRZ3RSwnIJS\/zZ9CHAObFMbbcpBD0DWJl5cy7q8ty0tVjnbA8gyh6ztFLBNx+cc2DZ2mpaFHoCsaaz4LNcHtqXFKmd7AFlmkG1bsOTyDkf3UOAc2D4fcrIu6AHIhogr0zptYFtarHK2B5BlANmQo9W2Ds45sE0fXDbloAcgaxMvocqawLa0WOVsT1LIyhQgOuVH\/ajbNuoC9fLly+LSpUur0obkblTqNRcvXhS0JaTJJ5eFT7mcs8o5sE38bVsGegCytjETsnwdtnLbRrpnabHK2Z6kkJU7R50+fdoYhPIa2oVIzc2VP9+9e\/foSD6C8cLCgjh\/\/rzYsmVLZ7yngmzbFDA12kfeXKfxmgKcA7uPvV3XQA9AtitGUvxeha18XhBw79+\/LzZt2lRtf8r9w7nvJYUsjTzPnj1rvJHFkydPxIkTJ8TVq1dX7TKlG93K8nNzc2Lv3r2dcRYTsqmmgDtFUApwDmwbO03LQg9A1jRWUpWT6T+0Jerjx4+rwUWqL+k+NeDc95JClsB4\/fr1VWfUNjlHln\/jjTfEb3\/72zE402lA9+7d0553q\/u57h4hIeuas+ozYE3r4hzYpjbalIMegKxNvKQsu7y8LJaWlsTk5KSg88AJtJxHtJz7XjLIylHmnTt3xMOHDwXtgUwf3cHw9HMa9R4\/frya+v3000\/H3snKumZmZkZTxTLAbY7i8wlZ3RSwnMpR\/07ZEbvuzTmwu2zr83voAcj2iZsU18hYlduj0v9jHrXn22bOfS8ZZOWip9nZ2bHRp27aV5bdt29fNe1bL9MG2aZFUiFGshymgG2Cn3Ng29hpWhZ6iOpsVPrQKInWQWzbtk1QH+Y8SiJ7SvNt3R6TFcmm\/SBFOc7+SQbZJkfVgUrl6lPBuUC2bcGS75xVBHYKBcbvybmj91Wv6TUHvfO7deuWeO2118TatWtZj5KGAFnpf66w5dz3soXsjh07qqlfdZpYrhBOCdnSRqttD1\/Ogd0XKkPXw\/Q1h3oKD501TP1CTkmqqSQh\/BCiztJivcsebrDtsidETPiqM3vI0ij2woULjfaePHmymkLus\/Dp3Llzgt7DHjlyRBDUde9kc8lZ9eVwm3o4B7aNnaZlS9WjzxdH3VF36oObNKX3gVxWtpbmW1N7uPjM1B7TvhyzXDLIyrzWw4cPj6XXmOTO6t6ztqXw6BZEkcgEVQla+r8ELf2MNsQgx9Knfs5qTAelvBfnwA6hWyl6+HjN0XaerKzRaj4yAAAgAElEQVRfwpvDgptSfKtOC9vkyebuM87+SQZZ3btW+plJWo8OqK6bUajApVHxrl27qnjl8k0cUAmhwHidnDt6n9Fqm6Kmh7b7OLYtvGfLX\/hko2GOPuPc95JCVkJ1fn5+FAMHDhxYlYZTD5DSt1W06RAhy3IO7BC6cNIj9GsOU8iqIyv1vW1ueZucfGsS2z7syem9rQ97THQLUSY5ZEMY1bdOn3myfduQ03WcAzuEjjnr0TYF7HNGRoU35bZ\/73vfs0rhyenBrcZIzr7tE8s+7cnBZz7t6aOnyzWArKIeIFvO9KhLp2i6NreO7nsKWGd3G7zXrFkjKEedyti+d83hwQ3I2vWSlD7Lre\/ZKAfIArKN8cI5sG06gWnZ1HrE2prTFt4uD1+Xa039ZlIutW9N2mhTJqQ9KXwW0h4bXfuUBWQBWUDWsOfE7ui6UaQ69etrUZ6v97fqw1e207SNLtcauq+1WGzf+mhzWx0x7Inpsxj2hPIJIAvIArKGvStGR7cdRRo2fVWxkPdxefjWr7Wdhu6rRwzf9m1bn+ti2hPDZzHt6aN32zWALCALyBr2qhAdve2dp8+tOUNCtU0+l3QQl2sNXToqFsK3tm3wWT6VPaF8lsoeHz4BZAFZQNawJ\/nq6DGAFwvehtJV2y723ZwixjtAX7411SN0udT2+PZZantc\/AXIArKArGEP6tvRfb3z7GpmDHh3taHr9y4PX5drTdpls0NSV32pf983Vn2325fPcrGnjz6ALCALyBr2HNOO3jaKpFuZLgbqalZuo9Wu9qq\/d3n4ulzb1EZT39rYmLJsbva4+iw3e2x8C8gCsoCsYY9p6+ixRpGx7mMoiXMxl4evy7X1hnN+iOuckKs9fX2Wqz0mHQCQBWQBWZOeUjvYmzo9feRokv4tFyr5XLDEebRqKOtIx77bLqoPbjlTYDtbwPkhzgmysq22PuPsH0AWkAVkDWhAnZz2611aWhKTk5OCzlBVp35tH+pttyxttGog76hI35GO\/MIjQS19Y\/qFh\/NDnCNk22Cr8xln\/wCygCwg26BAHXYrKyuC\/mzcuLH64+vTBlVTSPhqSy711GFrexC87Wpmzg9xzpBV297mM87+AWQBWUD2GwW6pmapmI8VqLr7hBoV5wLNvu2wnVas38d0ZMz5IV4KZKUdcjaCfKJ+ufLR9\/rGoct1gCwgO2jI2kzNujyIbe7j0qFLvVZ+MQmVa+vi2xw1L8Ee9QsSHUZB\/3\/11Vcr8HL6ALKA7KAg65KzavPg6hoVc3tQ5PRQs50KVtveNLK18W1OWjS1pSR7yJbl5eVqTcTWrVsBWQ4B2NRGHHU3rkwJHbUNduoUrUncdumB0aqJiv7KmE4FN02nqquZ6UsP\/X\/Tpk3sHuKlTReXZg9GshjJFjeSDQW7OmQxWvUHTJeafMCWRkmPHz+uDqD3uajNxS6Xa7u+ELrUneJazvYAsoAse8hKqErokUGhclY\/\/\/xzsWHDhur9kPzI9B2faTwpHmTc7+kCW4qh27dvj9KzyJec\/ckZShjJcu+JLe3HdDGP6eKYq3PrAP\/HP\/5RAfyVV16pHsJ4t5rfA6EPbFUoyevllzWOfgZk84lLjGQxkmUxkg01Bawzvitvte\/ORPl0+2G0xAa2OijpUkm4fKkCZPOJcUAWkM0SsjHfd\/YBeP0BbrtZQj6PgPJbYgLbNiiZXJ+bioBsPh4BZAHZbCDbB3Z9upLvFcf1rfw4v8vroyeXa1RYUpvV964mUOIEWxN7uPiN2snZHkAWkE0GWZecVdsHRGiAS3D33SzB1h6U76+ADrZUm2kKDwfYcoaSzrOc7QFkAdlokPU5gux6xMacbq63xWWzhC678Hu\/Ckhf0Z7UlMKzZcsW41XFOcOWM5QAWb8xnlVtWF087g4fHTX0CFJtccx7mQRuzg9hk\/YPoYyMmTt37oh\/\/\/vf4jvf+Y5Yv3691crxtmnoVBr66Lup2g7I5qS857YAsu6QjZWzKt\/TqCNW+lmI\/FjXMANsXRX0d31XfMppY\/KZXMxmuqI4p1cGgKy\/mHGtCdPFmC52mi7WTctShaE2aMhttGrTAQFbG7X8lO0bn66+Sv3KAJD1Ez8+agFkAVlryMYEXczFUT46lEkdrg9wk3sMuYzP+HT1lev1ff0IyPZVzv91gCwg2wlZOSqVo4KQ07J9Rx7+u0b4GnN8lxfeav93iLHIzRWWrtfbqgbI2ioWrjwgC8hqo4tGA7TikvZznZ6eFuvWrcMUcKB+WIet7bvAQM3Kulqfo1UbQ11h6Xq9aVsBWVOlwpcDZAHZSgHdQ4sOSqbTSTZv3myc1mASsjFGHibtyLFM6nd5OWrSFJ8pF7m5zkKEhi0gm08kA7IDhWwb6EgSmiL22VFTjTzy6Wp2LQn9ELZrTfzSJvEZv1Wr7+gTtrLf+dgxzGffzUXn+\/fvszzvF5AdEGRtQefSUTFa9fNoGhJsbePTj8L+anGZhdDBWo7U+7TQpe\/2uV\/oazjbA8gWDNmunMCu\/D\/bwOb+kAz9oHCpv0TYusani54hr3WBrZwad92e07bvhtTDR92c7QFkC4Ks75W5XYFd6kPSx0MhVB2cYes7PkNp7KteV1+5XN\/Vd33ZGKsezvYAsswhG3L0qAvskPeL1WFLuI\/ru8BYGiBenp0gQzrQ331Wjve5njOUdLHJ2R5AlhlkY77rpHt9\/vnnYsOGDdUDQn5C7eYU68Ff0n18v8tz1SZmfLq2Nfb1fWCpttHmes5QAmRjR2bE++W6d3HM0UD9IfmPf\/yj+vb9yiuvjL6FR3QJbmWhgOu7QItbjRWNGZ9925jTdTawbAJO18gYkM3H40lHso8ePRKHDh0SN2\/eHFNk+\/btYnFxUUxNTVU\/f\/LkiThx4oS4evXqqNzJkyfF3r17x667ceOG2L9\/\/9jPLl68KHbu3GmkeC6Qjb2VYNtDkgDb1aGNxEWhaAq4PsS7Gho7Prvaw\/X3rn6qXy+nokkPQDafqEgKWdpN6ODBg+L06dONIJQgnp2dFadOnRITExPVLkR03e7du8WxY8cqNXU\/u3z5slhYWBDnz5+vzons+qSCbNsUG7XZR96canufKb22Dt2lK36fRgHXh7hsdez4TKNWuru6vl9vemXANa+0afTO1Z6kkKWR59mzZ8dGrXWBqczx48dXgZIAeunSpdG19f+rI+C5ublVo16dI2NCNvYUm6\/7uT4Q0j3KhnvnPrD1FS\/DVd3ech99S74y6HMIvX2L413BeWSeFLIExuvXr49GqDYuq49Sz5w5I+7du7eqrqafx4Zs7HSXPqNVG\/1l\/a75fDb3RFk3BdpgGzs+3Swp\/2rX9+vLy8tiaWlJTE5OWh9Cn6O6gGwPr8j3rHfu3BEPHz4Ud+\/erWrZs2ePEXQJnjTypHe3NIVM72xnZmZG08eySWo5+Y63qbk+R7I6yKlTv76ngKnuVKMP1wdCj\/DBJQ4KyAfWV199JV544QXxn\/\/8R7z44otVjVg57iBsgEv79i3pY7k9Kv2\/T\/pQAJN6VQnI9pBN966VqtFN+9arl+9fDx8+XE0DS2DrIGtSn6zfFbKxIZfbApQ+05I9QgeX9FSgHi80pUiHQJDfCLL0QA7x5a9nc3GZooBt36pDyfb63MQHZD16RMJ337592veoEqh0S7kQKhVkQ0\/J1mWV91NHrTmOPrh3aI\/hnLQqm\/jsO2JKauAAb27at5qgZHp9btICsh49IiG7Y8eOVVO\/Eqb0rkGX4hNjJJvDaJXLlB7XDu0xnKNX5RqfMl2L+\/RidOEj37Crb3VBqev6yOZ03q7Lns4KEhZIuvBJZ3cTZJsAK+vos\/Dp3Llz1XvdI0eOCIK6bro49pSszegjYdxY3Zpbh7YyLnHhUPEJnyV2rOHtm\/xkCiX1ejkrluMrA1N7DGWLWiwZZOvvVaXVutxZuclE26KothQe3QiX7kdQlaCl\/0vQ0s9oQwxyLH3UI6dCBKDr6CNqxDjcjEuHdjAx+KVtX8LkQ9JnIwBbn2qGq6vuJ7mJzKZNm6rnV9dHxlWu2QKAbJcHG36vG33W03okdKenp1vzaV03o1CBS+3atWtX1eoQUC1xtGoTAnXYcl71aGN337I5fAkDbPt6L+51tGHDl19+KR48eFDB9fvf\/77YuHGjVSNyfD8PyFq5cLwwQXV+fn70wwMHDoy9iyXgXbhwofEO6raJOW+rmMOD0sFNwS7NsUMHM9aw4pxzVgFbQydGKtYUK3LV+NOnT0eDBdsBQ06+BmQjBVTo27im8KjtG\/po1dZXOXVo27a7ltfFijqLYvtwdG2PyfWY+jdRyX8Z21jxMQ2cQ98EZP3HUpIaXSGL0aq723Lo0O5WdNdQSqz4eIh3qzXsEr5ixXXWKGXfBGQL6QO2kPUV\/IXI59WMlB3aqyHfVDaEmQ3Xh3gI3TnWGTpWXPuW6\/V9fALI9lEtw2u6IGs7VZOhieyalKJD+xJpqF\/COPvMl+9t60kRK65+ivnKAJC1jahMy5vmyXLZDCJTmXs1K2aH7tXAhr2jZfqXmgbWt35u17k+xLnZa9PeUPnNNm2QZV39VO+bIbIFANk+ns3wGglZ2q7xRz\/6kZAr84b8oMzNTTrYpgJY27Qe6ZbjgqUU\/nR9iKdos+97cogVH34K9coAkPUdkZHqI6hS2g996N\/0hz6vv\/66oL2TCbSvvvqqUTJ3pCbjNooCoTp0m8gppvVKcbqPhzgnLbjGig8\/+ahD9TUgyynylbbSzk4ffPBBtdMT\/aHPzp07R7tA0a5P7777bvU7GpWY7JzCVArWzfbdoVUxcs5Z5eo0DlP\/fbQtLVZ89CsfdZAvANk+EZnBNTRylXCtN0fdAYpgK4GL48AycFxDE3x0aCxui+ffnKb++1g9lFjx8aXItW8Csn0ilMk19f2N33nnnWp0C9jm60DbDs11Wi9fD9i3LMXUv30rhRhyrPj4UmTbN6WPANk+0crsGvn+lqaX6YOp5Pwd2NSh2xahpFpElb+acVrY9yEcqnWIFb2yrl+KbEfHgGyoCM+0XvkuF7DN1EG1ZlEH\/fvf\/y5WVlbE119\/Lb71rW+JF198cbT6F6uA8\/NjStgOebRqGwmu5w+bjo4BWVvPFFK+\/t4Wi6TycazuQUmbphNoKTVr3bp1mPLPx12NLYkB25xyVhm4RNtEH35qGx0Dsj0jo37CTv0EHllt\/aQe9eQdWcb1BJ6eJlSXAbYu6vm5tm1aj+5QH626Tnf5aTVqMVXAx0Nc3ss2VkzbiHLPVgHL0W3fTSl0o2PSlo7xMz0fNydfJDm0\/cmTJ+LEiRNiaWlpdEbso0ePxKFDh8Ts7KygzSAmJiYqnepnzspr5+bmxN69e6syrmfJ+nIIYOtLSbN6fEzruU53mbUUpXwpUH+Iywd5V\/0+YqXrHvj9cwV8wFatQx7dx3HfgiSQlVA8ffp0lZeqjkaPHz8uzp8\/L7Zs2TKCp67c2bNnR4Cmke6lS5fGDnXXwThWJwBswygdMg\/Rx0MhjNWoVadA18KZkLECj5gr4KNfUR3Ly8vVYfRbt25lt19BEsg2uUhO+crpYPq\/Cl15XR3S9dGuLNf0c\/MQcSupS\/85ePAgNrYwlDVFHqKPh4KheSjmSQECKj2E6X37Cy+8INauXVu9c6cP9hn3JLJjNa79Cu9kHR0gL6cR6cLCwmgk2wXZw4cPi7fffruaep6ZmRHHjh0bawlBlkC3uLgopqamPLXSvpo6bCn9h2Ard5Kyr7HcK3KZ1nN9KJTroXwsq8fKV199VY1yyHcbNmzAl9l8XDVqSdcMRFOTAVkPzpTvZAk8EpZt08r79+8XJ0+ebIWsbhrZQ1OdqtCl\/+zatWuwm8nnnocI2DqFu9eLTWMFPvMqe5DKbGELyHpwQ9Oosz7lK8F79+5dlpCVUg35vW0uo1WbsLV9KNjUjbLNCrjECmDLI7JMVvoDso6+JJBeu3ZtNE1cr05N9dmzZ484evSoeP\/996uTctqmi3McydZtGwJsS8pDlKMpaVPfNAXHLlPs5SFiBbDlES7Ikw3kpy7A6m5Lo1laEEWrjmkVcq4Ln2wkKwm2Q8lDNPkGbhMDQywbM1YAWx4RpvMTtRx5spb+k9O+09PTjQuTmt7J1keobSk8ugVRlk2NWpwrbF2m9aIKHOBmeHjbiZo6VjD1b+evVKWRJ+ugvFzk9ODBg8YpYll9fZQqr6UpY5ljm8tmFA6SrLo09\/Qf5CGu9jZgq+8BucYKpv59PrHC1EULRclPFy5cEK+\/\/rpYv369OHLkSOMRpWFa4VZrkjzZ+jaJdRNo1bDczYl+p76Tffnll7VgTrmtopsL2q\/WwZa+XNCK5JiHyKfIWQ2pa8i6hw5bjrGCqf+QPcK8boIqfei5R3\/oQxkn9Ieee\/R79eccgJsEsuaSo6SqQD39h3JtQ8I29bQed+8PCbalxMqQfJZD\/5JHiKpQpXa99957VfMIorqPHHwQfJvK5GAftQGQzcUTFu2ov7f1BVvTPESLpqKoGN80nQShXYi4H69XeqwAtuG6rhytyrO5TaAarjXhawZkw2sc7A66RVK2G1uUMgIJJrLHinULbjgdEj\/EWAFs3TuAbrSqTgHTv0v+ALIFeLcJtroHeIg8xAIkjG4Ch3eAiJXnYQHY2nWRoY1W29QBZO1iJ+vSOti+8cYb1QIpOqicHhT0UeHLfdoya4cYNC6nh3fMnFUDabIsgvQfvVt0UB3SaBWQzbK7+m+UblqGlr3\/8Ic\/FD\/\/+c+r5e906DE++SmQCrZDnAL24f06bIe281fbFDDpm\/tiJB8xYFoHRrKmSjEoJ1cfy2+Q1GR11d4777xTnf5Do9eY6T8MpMumiaFhm2vOajYO6NEQDlP\/PcxadQmmgPupCMj20y3LqwioTYsIYqf\/ZCkQo0b5gi3HnFVGbhprqi+f5WJ\/V85q7AVLtL\/C9evXxalTp8TExMSYTE+ePKmOPL169ero5\/JccvkD9XAZ9WLaD19Xpy8\/ALK+lGRSj0zmJiDLc21D5toykSXbZtYf3HJasq3BmAJO606usO2bsxpDbbnZkA6Iul0A5YZHKmibzicP3X5ANrTCmdYfKtc2U3PZN6ttwU3pOatcnccBthymgOunsNVHnaZ717eNhEPGGCAbUl0GdQO2DJxUayKNVO\/cuVP99OuvvxaTk5Ni3bp1ow0usGI8L5\/mBFtuOavqKW1XrlwR9+7dG5valdPEc3NzY1vx6iKA6qLPsWPHogYIIBtV7nxv5mNji3yt498y3RQwAZY+lJ5FK8exoC1vP6dK\/+EwWjXxnO5IU3WqmL54zs\/PV1XV97iX5aiffPzxx6PbHThwIDh0AVkT7xqWUac16JImB9YPSKgfiEDXpjrwALA1dHbgYjY5qzmNlALLUkT1oXf+KjVnVQdZdTGT+rytH5OqO6mNgqnpLHKfgQbIelBTTlksLS2NzsaV35xmZ2fHpjfq7w900x05HN2X+1F7HtyWXRWuC5YA2+xc2tkgH+k\/Q8lZbYPsm2++uWqFsAlAm84s73ScRQFA1kKspqJNjqqvZpPgpaXv6nsBm0PoTd49eDBpVIXswHIzb6xI9qduqJxVwNafj2LVRLEgZy9MNrYoZQrYRt82yO7evXvVtK9uQVT9fvLZffjw4c53ujZtVcsCsn2VM7hOTvnKZeSmkG36BmbyzcygWb2LINe2t3TVhbFzVgFbN3+luLrJZ7nlrKbQpu2dbH3gQu0DZFN4KfI9yckLCwtjh8w3TRfPzMxU38Tk9LH8v9pkCjIaWS4uLoqpqanI1jy\/HVYkm0vvOgVsfqfmkqkW3Pho+1Dr+POf\/yz+9Kc\/iZs3b1Z\/5KfrnNWS9WobfOiei+pq4vqAR+oUI3cWI9lAUdk0aqXb1XceUROm2yBr8s0skDnaagHb1bLknLMq2ybBbzItGTOehn6vpilguTWqPKA89k5LufilCbK6KV\/62fHjx8Xp06fFli1bGgcvMdJ6ANlAEaQbdUqA0i3VhGo1eOh3tD2YbiSbG2SldEOHbQ6jVdsw9rHgxvaeKD+ugE3OquxjErRD1LLtdZkc1MhRfz2FR+pVzwDRZXb41haQ9a3oN8vCr127NjZNTLdpmrJQF07RwiJukB0abEs6ZxXvbQM8AFqqHOKCpbgK53c3QNazT9QdSmiaQv00zf\/Lb2H79u2rVrjluvDJVCpd+g8dfcV1JyKbnFVTjXIrB9iG8UipOath1CqzVkDWk1\/laHR6erpxYVITZOspQKZ7cXpqerBqdLB96623BB1IkPuH4xSwD00BWzcVh5Kz6qbSsK4GZD34W45EHzx4sGqKWK2+aVFTfeSaw2YUHmQZqyL39J9QOau+dYxVH2BrrjSmgM21GmJJQNaD1+vbJNarrL9cN9l+MdW2ih7kaK0il0VSsXNWQ+saqn6k\/6xWFjmroaKtzHoB2TL9mr1VKWA71ClgH8FQh+2Q0n9yPmfVh29RR1gFANmw+qL2DgVCwjbnnFXOgTGE9J9cpoDbzkCVr5+uXr06Cic1555+qCujO\/icczzm3nZANncPDaR9vmCL0Wq8gCnpva1NzmosheUrIx0U1SPedu7cWTVJvrZSQVvPdmjbJCeWXUO7DyA7NI9nbm8dtjL5nqYndZ+SclYzd01j87jCNpfRqk5Ydd2GDrImGQj11EB5H4L32bNnk2\/PyjXebdsNyNoqhvJRFKin\/9BZkZRrSx91Gpj+TwCWEOaaixtF1MA3yR22XHJW1dHnlStXxL1798Z2iNMdj6lzbdPpYDGOdwscaqyqB2RZuWuYja2n\/9CmHT\/72c8qMQDV\/GIiF9iWkLPadvLM0aNHxZ07d8T8\/HwVBPWtBLvy8kMe75ZfVKZrESCbTnvcuUOB+shDboxOD0\/afpJGthw2thiqo1Ok\/+Q8BdwnDtrOUL17966gGR55NnV9hGq6w1yfduEacwUAWXOtUDKyAnIES8d7yaliaoKvRVKRzRns7XSwVaf4XYQpPWe1DbJvvvnm2DQy6aiWp83y6SSa8+fPVyfRyE\/Tu1oXP+DaZgUAWUQHWwUAW36uc03\/GVrOahtkd+\/ePRrFykhQF0R98cUXgGwGXQSQzcAJaIKbAoCtm34pribYygVsXRtblDYFbKN32ztZen0ip4p1kKUR68GDB6szVWWaD5XDwicbD7iXBWTdNUQNmSgA2GbiCItm6BZJ0TQnvU+Uh5VTdQQU+kOwGNKh5W0ncpE+i4uLYmpqaqS4egh5WwqPbhrZwm0oaqEAIGshForyUKAp\/acp15aHVWW3kuBAwP3Nb34zMpTexdNHfR9ftgqrrWuCrByNqiuE6WcETxq5ynew2IwifcQAsul9gBYEUkC+v\/vggw+qO8iNLYY0EgokrXO1bTmrcgQLf40vZJqYmBjTXY5UaeRPn3oKD\/0M2yo6h6pzBYCss4TlVtB1WpDupCBVDfX0odSnCqm5tm0P71\/9Sohf\/\/qZFT\/5ybO\/f\/nL5\/8u19thLbPNWZWzEdJXYVuH2qFAOAUA2XDasq1ZfvtdWloavfOR35pnZ2dXpQ2ohuquzel8XHUqWYXtH\/8oxE9\/2uyyP\/wBoLUN6CEvWLLVCuXLVQCQLde3vS1rWn3YlNyu3ohSCBYWFsZy89r2WZ2bmxN79+7t3da+F6qw\/e\/\/PiCuXj3WWdXTp51FBl2g9JzVQTsXxvdWAJDtLd3wLpRTvvXjtKQSTSd8tK2QrO\/LGltVgi1NCz95sqPz1jRtTNPJ+DxTYGg5q\/A7FOijACDbR7WBXqMbpXaNYuX08czMzKqcPoKvLg0htrw0TUzTxSafoY9mMQVsEiUoAwWeKwDIIhqMFOg6h7Lp922Q1U0jGzXGc6E1a1ZX+P3vvyv+678+EQ8fvicePnx2+g99hgbZHM9Z9ex+VAcFgioAyAaVt5zKu0adTVPJXCE7MfGJeOmlcxVo6fN\/\/7dD\/M\/\/HBHXrnVPK3P3Okar3D2I9uekACCbkzcybUs9oV3XzCYIc4Csmrajs41g+9JL5eba5nTOKs1uXL9+XbuCvZ5SJn2lrhGQi\/bohBr1ozv4PNPuhmYVpgAgW5hDfZtjAtiuqeScFz5JvXRTxnUt\/\/d\/PxGffXaueo\/MeaME25xV3zHVVJ+cDdEBse3LmlqfyQr4WPbgPlCAFABkEQdaBeSIYHp6etX+qPULujYcb0vh0S2ISuESmzzZplzbFO02vWfuU8DqKFUHWflFjg4qVze7r9vfNhI21QrloIBPBQBZn2oWUpd8oD148GDVWZQ6E7tGDzltRtHmIgKt3O1JrjZuS9vJGbacclbV2ZIrV64IXVqXbl9enS\/VDfIL6Y4wg7kCgCxzB4ZoPo0G5ufnG6tWt0ukQl2pPVQm9baKIXSSddZP\/6EN7Xft2hXylqvqLiVntenVgoyxl156SXz22WeV\/fW9euWXw\/Xr14uPP\/54pNGBAwdWpY9FdQ5uNmgFANlBux\/G+1SgfvoPnSIT8gSZ3KeA+2jb9v7+2rVrYzMr9dcUuhkTakNTnX3ah2uggK0CgKytYigPBToU0B2199ZbbzmfgzqEnFVbIJqU71ozgICGAiEVAGRDqou6B6+A6ek\/TUKVOFptCwoTaKrXd+VvU1nd2auDD0wIEE0BQDaa1LjRkBUgWJqck5pTzmoKfwGyKVTHPUMqAMiGVBd1Q4GaAroVybQoTAKYilMOrjxYPuQ73Rydo4OszJGl9p46dUrIw8vrubNNu451rX7PUQe0qRwFANlyfAlLMldAvlOlZn7wwbMdpORHgnVoUK27rGkkqwNlPa2nacMKpPVk3jEKbx4gW7iDYV4+Csj3s+pIlTZWkFPJchRLoJUj2XxaH6clbdPF9S0Tt2\/frt0opb79Yj3lLI4luAsUeKYAIItIgAKRFJDbMTbdTk4lyy0bIzULt4ECUCCgAoBsQHFRNRSAAhUR1\/wAAAUlSURBVFAACgxbAUB22P6H9VAACkABKBBQAUA2oLioGgpAASgABYatACA7bP\/DeigABaAAFAioACAbUFxUnZcC9VWnTRvH1w8z0B29VvKBB3l5Da2BArwVAGR5+w+tN1BA5k8uLS2NUj7kiS2zs7NjGxzUzyPVleNydJ+BNCgCBaBAYAUA2cACo\/r0CjRtEF\/f4EACdd++fWLv3r2jhtfLtR1CPzc3N3Ztl\/Vth4zXR96yrosXL44OLpdfIK5evTq6lW7k3dUO\/B4KQIEwCgCyYXRFrQwUqG\/DR\/8\/e\/asdoMD1Zy249h0B443SSHvr4Ni0+5F9brUA8+3bNki5BcFyrU9duwYAy+giVCgbAUA2bL9C+taFKgfNi9HlUePHhXvv\/++uHnzZnW1+u62DX4mJ8LI5qijVB1kJSypLbQrlO7TNvI2+bKA4IACUCC8AoBseI1xhwwV0I34JPhefvnlscPB1ZErmXLixAkxMzOzaqSom0bWma6OPq9cuSJ0o9\/6vry6epqmwXF+aoYBhyYNVgFAdrCuH7bhulGnhKz6zpNUUqFF++W6QtZk6lmOsl966SXx2WefVZfU4d90ugzOTx12bMP6vBQAZPPyB1oTQYH6e0x1CvfatWtjo1j6nTot+\/bbb0eBrK6N9RFqE2SbppEjSItbQAEoUFMAkC04JHSpJtLcoY52mgBLutTf0Uqt6tDytfCJ6u9zSLmcXqZ3xsePH2\/9UqCuki441GEaFMhWAUA2W9f4aVjTYpwmoPi5a361yC8V09PTjauHm7541N+PtqXw6N7VtqnRB7J0Ws\/i4qL44osvANn8Qg0tggJjCgCyhQeETBNRz9SUK2TJ9FOnTomJiYmiVZAj0QcPHqwa9dUNb3pXS+VkSozPzShcIEt2HTx4UJw+fXpsBTIWPhUdzjCOmQKALDOH2TZXB9ShTRXTyHN+fr5Ruvqh3vXyuu0XfW2rqINs05egevqQ6eYZtjGD8lAACvhTAJD1p2W2NenyQRcWFjpHddkaVFDDmkayukVNurQebEZRUDDAlCIVAGSLdOu4UWpO6HvvvVetjh3KVHHu7m2bLpYzDnfv3q3MoPQhehc7NTU1MgvbKsb1cNMuXfLnTYdOxG0l7paTAoBsTt4I2Bb5rpF2EKIVqYcPH7baYzdg01A1FGClQD2fWn6JJSPqX4JYGYbGBlEAkA0ia36Vym\/aP\/7xj8Xf\/vY3TBXn5yK0iIkC9VOdPvzwQ3HhwgVR38SEiTloZmAFANnAAudSvTqtiFNacvEK2sFVgfpUPqaJuXoyfLsB2fAaZ3MHuWoW37izcQkawlgB2Z9078oZm4Wme1YAkPUsaM7V2ZwSk7MdaBsUSK2AOjNU31M6ddtw\/7wUAGTz8kew1rRtsRjspqgYChSqgBzF\/uIXvxC\/\/\/3vxezs7CA2dinUnUHNAmSDypu+cnXTBHzjTu8PtIC\/AvIL65tvvlmBlfaQ3r9\/\/9i5w\/ythAW+FABkfSmZaT0qZPEuNlMnoVlsFJDTxJ9++umqM4exwpiNG6M2FJCNKjduBgWgAGcF5DRxfStO5Mpy9mrYtgOyYfVF7VAACkABKDBgBQDZATsfpkMBKAAFoEBYBQDZsPqidigABaAAFBiwAoDsgJ0P06EAFIACUCCsAoBsWH1ROxSAAlAACgxYAUB2wM6H6VAACkABKBBWAUA2rL6oHQpAASgABQasACA7YOfDdCgABaAAFAirACAbVl\/UDgWgABSAAgNW4P8BYrBHpqaAhH4AAAAASUVORK5CYII=","height":258,"width":344}}
%---
%[output:03206bb2]
%   data: {"dataType":"not_yet_implemented_variable","outputData":{"columns":"1","name":"N_num","rows":"1","value":"1"},"version":0}
%---
%[output:3a0b1a00]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdkAAAFjCAYAAABxFnERAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnU+Il0eax0sYx+lDYlbDjG3GZJwxTg4im4tu2B3ZXONZYQ96WBRB0rABmxVayAYiuOglGGRFswdzWIiXvegyhyHLzGXMZUDDgOAcZjrjPxg3MQwdZw69PK+p9u233z\/1v56n6vuD0Ka7\/n6fp+rzq3rrqXfd8vLyssIHCkABKAAFoAAUCK7AOkA2uKYoEApAASgABaBAowAgC0eAAlAACkABKBBJAUA2krAoFgpAASgABaAAIAsfgAJQAApAASgQSQFANpKwKBYKQAEoAAWgACALH4ACUAAKQAEoEEkBQDaSsCgWCkABKAAFoAAgCx+AAlAACkABKBBJAUA2krAoFgpAASgABaAAIAsfgAJQAApAASgQSQFANpKwKBYKQAEoAAWgACALH4ACUAAKQAEoEEkBQDaSsCgWCkABKAAFoAAgCx+AAlAACkABKBBJAUA2krAoFgpAASgABaAAIAsfgAJQAApAASgQSQFANpKwKBYKQAEoAAWgACALH4ACUAAKQAEoEEkBQDaSsCgWCkABKAAFoAAgCx+AAlAACkABKBBJAUA2krAoFgpAASgABaAAIAsfgAJQAApAASgQSQFANpKwKBYKQAEoAAWgACALH4ACUAAKQAEoEEkBQDaSsCgWCkABKAAFoAAgCx+AAlDAWIEvv\/yySbu4uKiWlpbUyy+\/rF544QX1ve99z7gMJIQCNSkAyNZkbfQVClgqoKH6zTffKPqPPgTUJ0+eqNu3b6vt27er9evXN78DbC3FRfIqFABkqzAzOgkFzBTQMG1DlXISQNs\/v\/rqK\/X555+rXbt2qQ0bNiiCMeUh2GrgmtWIVFCgbAUA2bLti95BgUkF9GpV\/+yDareQNmQ3btzY\/Jkgq2Gry9BwnmwEEkCBQhUAZAs1LLoFBYYU6Fut6hWo\/jmlXh9kdR5dvoY2tpKn1MTfS1YAkC3ZuugbFPhWAZfV6ph4Y5Bt56N6AVu4Yc0KALI1Wx99L1aBPqjarlZDQLa9um0\/t8UhqWJdDx3rKADIwiWgQAEKjG0BU\/dCPxs1Xcl2pW0\/t8U2cgGOhy5MKgDITkqEBFCApwKht4BteukKWaxsbVRG2hIUAGRLsCL6UIUCQzGr7W3gVEL4QhawTWUp1JNbAUA2twVQPxQYUMA0ZjWHgKEg2wdbvb0deos7h06oEwoAsvABKMBIgZxbwDYyhIbsEGzx3NbGKkjLUQFAlqNV0KZqFAgRs5pDrFiQbfcF4T85LIs6QysAyIZWFOVBgQkFpKxWx7qRArJ9q1usbDG8pCkAyEqzGNorToHYMas5BEkJWcA2h4VRZygFANlQSqIcKPCtAqljVnMInwOygG0OS6NOXwUAWV8FkR8KKLVydaDNJfuShcsJWcBWsufU13ZAtj6bo8cBFOAUsxqgO9ZFcIBsH2zpdxT6g\/Afa5MiQyQFANlIwqLYshTgHLOaQ2lOkB2DrelbhXJoiDrrUACQrcPO6KWDAiWcAnbotlEWjpBtNxzhP0ZmRKIECgCyCURGFTIUkBqzmkNd7pDtW90i\/CeHp6BOQBY+ULUCWK26mV8KZAFbN\/siVzgFANlwWqIkAQqUGLOaQ3ZpkAVsc3gJ6iQFAFn4QdEK1BCzmsOAUiEL2ObwlrrrBGTrtn+RvccWcHyzSofsEGz1c9v4CqKGWhQAZGuxdMH9rD1mNYdpS4FsH2zpd4i1zeFVZdYJyJZp16J7hZjV\/OYtDbJtRRH+k9+\/SmoBIFuSNQvuC7aAeRm3ZMhqpcnn9Bc6hP\/w8j9JrQFkJVmrorbqyY0mc\/rv+eefV\/r2Htzik98RaoAsDknl97MSWgDIlmDFQvowFF7z8OFDtXPnzgay+PBQoCbIArY8fE5qKwBZqZYroN0mMau0or1\/\/77asmULIMvI5jVCFrBl5ICCmgLICjKW9Ka6xKwCsjytXjNkAVuePsm1VYAsV8sU0i7fA0uALE9HAGSf2YV8VB+Sot8i\/Ienz+ZqFSCbS\/lC6w0dswrI8nQUQHatXfROjR4DOJHM03dTtwqQTa14YfXFjlkFZHk6DCA7bhfE2vL02xytAmRzqC68Tt8tYJvuA7I2aqVLC8iaad3eSsbK1kyz0lIBsqVZNEJ\/cr5nFZCNYNAARQKydiICtnZ6lZQakC3JmgH7knK1OtZsQDagUQMWBci6iUnj6sGDB2rDhg1NSBodkkL8t5uWUnIBslIsFbmdJjGrkZvQWzwgm0P16ToB2WmN+lJofya44spGNw2l5QJkpVksUHtdYlYDVW1VDCBrJVeyxICsm9Rdf0b4j5uOknIBspKs5dlWLlvANt0AZG3Uip9W+9Di4qK6c+eO2rVrl9q2bRu2PA2lH\/LnPtjijm5DUZknA2SZG8ineaFjVn3a4poXkHVVLky+IR968uSJun37ttq+fbtav349ni8aym3izwj\/MRRTSDJAVoihTJoZO2bVpA2h05hMSqHrrLk8Ux9qbxfTIR5945FefdEzR3zWKmDjzziRXIYHAbLC7ShxC9hGcptJyaZcpH2mgIsP9T2TxfPFaa9y8WfAdlpXzikAWc7W6WlbzpjVHFK5TEo52impzhA+NHbwSZeP6wX9VrLd3ICtpFH2rK2ArAC7uaw0BHTLqImArJFMk4lC+5Dp6WI8X1xtmhD+DNhOujurBMVC9uzZs43Q8\/PzrAQ3aQzXmFWTtodOE2JSCt0mCeXF9iFTyGqtAIanSoT0566m+nILCf5ZUxuLhOzVq1fVqVOn1JEjR0RAVkrMao6BEXJSytH+VHWm9iFbyAK24SHbpyn9Dq\/aSzXqzOopCrJLS0tqYWFBXbt2rek9Z8iG3r4zM7e8VIDssM1y+pArZGuHbWx\/xvY8vzmuGMhqwFKQ\/AcffKDOnTunZmdn2axkS4hZzeG+sSelHH1yrZOTD\/lCtlbYpvJnHVJF9eGOZNcRFyZfMZBty6GBmxOypvGGYcxYbimpJiWOCnL2oVCQrW3LM7U\/41l4\/pENyAa0Qc7tu4DdYFVU6kkpd+el+FBoyA7BtrRVWC5\/BmzzjWxA1kP7EPGGHtVXkTXXpJRKXKk+FAuybd1LfL6Y258B21Qj+1k9gKyl5lJWGpbdYps896QUQ5gSfCgFZEt8bsvFnwHbGCO7v8yqIXv+\/Hn12Wefqbm5ObVnz55ehWLHG6YztcyauExKPuqV6EMpIVsSbLn5cxu2pDPCf3xGOiC7RgECrAYt\/ZFAS8DduXPnyguV6fftV07h4vPwTjhWIrdJyaT3qWNWTdoUOk0OyJYAW67+rH0WV2GGHilKVb2SbcvZBu7Pf\/7z5hud\/mYXXnaUaKoA10mp2\/4StoBNbULpckJWMmwl+HOJz8JtfDt0WkC2pSiB9tChQ+rjjz8e3D4ObQCUN64A10mJU8xqDh\/iANk+2HLf8uTqz30+hOe2YUZWkZB1lQaQdVUuXj4ukxLnmNV46g+XzAmyY7BtP+rJoVO3Ti7+bKMFYGuj1tq0gCxWsn4eFDl3zkmpti1gG1NyhGy7\/Vy3PHP6s419sbL1VetZfkAWkA3nTRFKSjkpSY1ZjSD7ZJHcIdu3uuVwsUVKf540omMCrGzthANkAVk7j0mcOvakhNWqm0GlQJYbbGP7s5s13XIh\/MdMN0AWkDXzlEypQk9KJcas5jCNNMhygW1of85h+26dXdhy2DHgoItuAyALyHLyxzVt8Z2UaohZzWFAqZDNDVtff85ha5s6uT4Lt+lD6LSALCAb2qeClucyKWELOKgJeguTDtkh2OpVWCwFXfw5Vltilovnts\/UBWQB2Zhjzbtsk0mp9phVb5EdCigFsn2wpd\/Ful7QxJ8dzME2C2Bb6I1Prh6HOFlX5eLl65uUELMaT2\/TkkuDbLvfMbc8a4Ns7u15U3+OmQ4rWaxkY\/qXd9l6UtKXCuhVq15ttH96V4YCjBUoGbJaBPI1\/YUu1GGeWiFbM2wBWUDWeGJNmVBPbjSZf\/HFF+rFF19UGzduXHlZA016+ORToAbIxgBD7ZBNvT2fb4Q8qxmQBWQ5+GHThqHwmocPHzZvRgJY2ZiKxQsCUqsR4vkiILvaan2xttyuwvT1M0AWkPX1Ief8JjGrmJSc5Y2asaaVbFdIH9jCn4fdMuaz8KiDYaJwQBaQTeZ\/LjGrmJSSmceqopoh67ONDH+edrMYz8Kna42XApAFZON518AWsM27ejEpRTWPc+GA7DPpbK4XhD+bu5zPjoF5LfFTArKAbFAvCx2zikkpqHmCFQbIrpVS79ToMdB3Ihn+bO+CpNmDBw\/U8vKy+tGPfmRfQOYcgCwg6+WCsWNWMSl5mSdaZkB2XNqh54vwZzeXlKwbIAvIWnt9ymsLJQ8ua2EFZQBkzYzV3fKk1S2Nny1btuC0vJmETSrJ8wAgC8hOuvrYgaXYx+0lD65JYQUnAGTtjKdhS7p9\/fXXaseOHc3VjfiYKSB5HgBkAdleL0+5Wh0bZpIHl9n0ITMVIOtmNxpXd+7cUc8991xzuQqBFvHf01pKngcAWUC2UcAkZnV6KIRPIXlwhVeDT4mArJsttD8TXPUOUagrG91aJCOX5HkAkK0Usi4xqzmGo+TBlUOvVHUCsm5Kd\/3ZJvzHrcYyckmeBwDZiiDLZQvYZthLHlw2\/ZSSVvvQ4uJis+25a9cutW3bNmx5GhpwyJ\/7YBv7vINhk1kkkzwPALIFQzZ0zGqO0SZ5cOXQK3SdQz705MkTdfv2bbV9+3a1fv36BrJ4vjitvok\/l3q94LQ6wylMdPMpP2ZeQLYgyMaOWY3piENlSx5cOfTyrdPUh9rbxRs2bGie6VNevfrCydl+S9j4czf8p+YvMTa6+Y6B0PkBWeGQlbgFbOPEkgeXTT9zpnXxob5nsni+OG1FF38GbBEnO+1ZQlJ89tln6tChQ+rjjz9We\/bsYdnqsQNLJT7DcZmUWBqOUaNC+NDYwSdd\/tj1gozkSNoUH3+uGbY+uiU1cE9lWMkKWMm6rDRyO1ao+iUPrlAahCgntA+Zni7G88XV1gvhzzXCNoRuIcaRSxnsILu0tKQWFhbUtWvXVvqzf\/9+dfr0aTUzMzPax7Nnz6rLly+vSXPlyhW1d+\/eSX24rGS5xqxOChghgeTBFUEO4yJj+5ApZHWDawRDn7FC+nNXU334zNhJBCUMqVvqbrODLIHy+vXr6tKlS83VY48ePVLHjh1rtm\/n5+cH9dFwnp2dHU03JnAuyI5t31F7az5EInlwpRzMqX3IFrKA7VMFYvhzDc\/CY+iWanyygqwG6sGDB9WBAwdWNLhx44Y6d+6cunjxotq0aVOvNjrviRMnjFatfYWkhGzo7btUDpO6HsmDK7ZWOX3IFbK1wza2P5e6PR9bt5hjlRVkKbj96NGj6syZM6tAOfT7tjCU5uTJk01eWgG7fGJCtoSYVRdNffNIHly+fe\/m5+RDvpCtFbap\/FmHVFF9JcQwp9It9Jil8lhBllasBEq9Vaw7rCF7\/PjxVSvctiBXr15VFy5cUJs3b1a3bt1q\/rR169Y1ZaXaLu7bvmtv\/da8BWzjyJIHl00\/+9Jy9qFQkO2DrR4nJY6R1P5cyrPw1Lr5jt12fhGQHdpGbnek+yyX\/mayAm6X4buSzbl9F9IpOJUleXC56CjFh0JDdgi2JazC2n6Qy5+lwzaXbi5juJunGMgOiUHwvXfvntHpZFvIjh02KTFmNYTD2ZYheXCZ9FWqD8WCbFuzEp8v5vZnqbDNrZvJWB5KUwVkCZ5jh6a0OCaQlbLS8HEKTnklD64hHUvwoRSQ7VvdSl\/ZcvFnabDlopvL3MgKsj4Hn8ZWskOQPX\/+vKK\/zc3NNSFCfZCNHW\/oYrSa8kgeXNpOIn3o3\/5Nqffee9qFf\/zHpz\/ffXfl3ykhWxJsufmzlPAfbrrZzMGsIDsWwtN3IEp3VMfI0v+3L62Yip0lqGrQUl4NWvrd7t27m5g2+rS3fks8jGHjMKnTShxcY1vApB9rH\/rf\/1XqzTeHzfzppw1oc0C2BNhy9Wfts1yvwuSqm8l8yAqy1GDXyyj6TibbhPW0gUtt2LdvX6Mf6wnRxMLC00gZXCVsAaspwGpfWl7OClnJsJXgzxyfhUvQbWiqZQdZk2sVh26B0tvNd+\/ebfpLq1GTZ7FaHJNnssKZJa75XAcXp5jVYEalFSyBdurz7rvqq3feUZ9\/\/nnz0vaNGzdO5Yj6dylbniQCV3\/uMxCn57aSdOtqyQ6yUUfjROGAbE71++vmMrj6toDbOx1F7HiYQlYp9dWXX7KBbN\/KVtuG2yl\/Lv5sM9I5wFaiblpjQLblbYCszdBLkzbn4CpiC9jGTOvWGafmCNl24zlueUpbyXadISdsc84DxoNiICEgC8j6+lDU\/CkH19iBJW4roiiim0KWDj7993+zW8ly3\/KUDtm+HYNUIVUp54HQYwuQBWRD+1TQ8mIPrupWq2PWaYftjKUTBNmcYBiC\/v3799WWLVuaqAXJn5TPwmPPAzHtAMgCsjH9y7vs0INLZMyqt4oWBZisZj\/9VH31+usiVrKctjxLWcmOaUp\/i7G6DT0PWIwI76SALCDr7UTRCpi4EMGkXtExqyYdDJ1mKoyHQZxsiC7ner4oGRYmusd6Fi5ZN0AWkDUZO2nTGE70Q43CFrCnuUh\/fduTDumh257oS8+3n5yXUXj2blX2Lmz1KixkHe2yJMPCRpPQX2Ik6wbIArI2Yyd+2inA6hYsL6+0pciY1fhKe9VQCmS1CKmeL0qGhYvDhIKtZN0AWUDWZezEy2MYq\/nNv\/6r+vJf\/mXl6ktqkI5VLSJmNZ7CQUouDbJtUWJteVIdkmHh4zi+sJWsGyALyPqMnfB5DSFLFX\/5f\/\/X1A+ohjfDVIklQ1b3nWCrn+mHOswjGRZTPmHyd1fYStYNkAVkTcZGujQmp1t7tozTNRA1kQI1QLZvK9kXtpJhEdLzbbfnJesGyAKyIceOV1m0cnjhb\/7GrAx6\/RqddMUniwI1QTYkbCXDIoaj9cG27+IXyboBsoBsjLFjVGbfKeAt\/\/Ef6nv\/\/u\/T+QHZaY0ipqgRsiFgKxkWEd2pKXrsWbhk3QBZQDb22Fkp3zhm1WTL+Nt4zWSNR0WrFKgZsj6wlQyLVEOg71k41S31pixAFpCNOnacYlanwngA2Kg2MykckH2mkvHzxQCXq5jYppQ0bV3XrVvXHEJ75ZVXxF1HCcgCskHHZLCYVYMLEYI2HIVZKQDIrpVL79ToMbBySOrXv1aKTs0PffClcdT3SNcHDx40h+127twJyFqNVGaJ8ao7e4P0bQFTKaFiVrG9Zm+TFDkA2XGV9fPF7\/3612rLP\/3TtElal6tMJ64vheR5ACtZrGStR6zTFrB1LU8zSB5cjl0WkQ2QNTSTadx359pKw9KrSSZ5HgBkAdnJgTp2YCn2e1YlD65JYQUnAGQNjWcKWSoOq9lBUSXPA4AsINvr2ClXq2PTleTBZTgNi0wGyBqazeSkvC4KkAVkDd1KbLKan8lyfc8qIMtzOAGyhnYxhSzivkcFlTwPYCVb6UrWOGbVcC6JlUzy4IqlCYdyAVlDK7TDdkayfPN3f6fu\/9d\/NQcGcRf3WqEkzwOAbEWQ5bIFbDg9NckkDy6bfkpJq31ocXFR3blzR+3atUtt27ZNXFhFUr0NVrPf\/M\/\/qC\/\/9m9X3ipFoI193iGpBp6VSZ4HANmCIRssZtVzgPhklzy4fPrNJe+QDz158kTdvn1bbd++Xa1fv74BggYDl7azaYfl5SoxX7XHRhPLhkieBwDZgiAbO2bVclwESS55cAURIHEhpj7U3i7esGFDc+8s5QVsBwzmcLlK+8aj2nWVPA8AssIhK3EL2IYbkgeXTT9zpnXxob5nssbXC+bsbOa6XfwZsJX92AiQFQbZnDGrOeYnl0kpRzsl1RnCh8YOPgG2w97g4881w9ZHt9xjE5AVAFmXlUZuxwpVv+TBFUqDEOWE9iHT08V4vrjaeiH8uUbYhtAtxDhyKaMoyN64cUMdPnx4lQ5XrlxRe\/fuNdKGS5ws15hVIxEDJ5I8uAJLYVVcbB8yhaxuNGD7VImQ\/tyFrX5ua+UoQhKH1C11l4uBLIUTHD16VL311ltqfn6+0fHq1avqwoUL6tKlS2rHjh2T2uaCrJSY1UkBIySQPLgiyDFYZGofsoWsbniNq7C20WL4cw3b8zF0SzU+i4EsAfWTTz5RFy9eVJs2bWr0W1paUgsLC+qNN95QBw4cmNQ0JWRDb99Ndk5oAsmDK7bkOX3IFbK1wza2P5e6YxBbt5hjtRjInj17Vt27d0+dPn1azczMrGg29Ps+UWNCtoSY1ZiOOFS25MEVWi9OPuQL2Vphm8qfdUgV1VdC+E8q3UKPWSqvCMjqFevs7OzKVrEWiyBL8GyvcIeEDAnZvu07qjfUe1ZjOAPHMiUPLl89OftQKMjWBtvU\/lzK9nxq3XzHbjt\/8ZDt20aOBdmc23chnYJTWZIHl4uOUnwoNGRrgW0uf5YO21y6uYzhbh5AtqWI7Up27LAJ7h0N4Z5hT2OGaVHYUqT6UCzIDsG2lJOzuWEhFba5dfMZ9aOQffTokTp27Jjas2fPmm1Yl+1Yn4aO5R3bLg69kpWy0oildepyJQ+uIa1K8KHYkO2DrX7cIvktNVz8WRpsuejmMv8ZQfbmzZvqyJEjvaC1eebp0kDTPC4Hn86fP988r52bm2u+SPStZGPHG5r2zzpd+xVb9K5K+rz7rlL639YF5skgeXBpxcT60IjJU0G2DVsNBvqd1MM83PxZSvgPN91sZkMjyG7cuFH96le\/Uvv37+89vWt6sMimYbZpx0J4+g5EUfnUbg1a+n8NWvrd7t27V1471d76Zf8t2vKNH7Y6p04vcXCNbQHr1VhqHUPXlxqy7fZLDlPh6s\/aZ\/UXQm5fYrjqZjKujCB74sSJpiy6TakLWi4rWd\/LKNrApT7t27ev6TN7qLatPAVYnXZ52cQ3WKSRMrhK2AK2MXhOyLZXt9Le\/iPBnzl+iZGg29D4MYYsXU2ory2kVZ4OieECWepgKdcq2kx2q9K++aZSBNqpD20b03aygA\/XwcUpZjWHGTlAViJsufpznw9xem4rSbeullaQbYNMg\/ajjz4yjkPNMRnY1Gl7utim7CRpTSFLjRGymuUyuPq2gNs7HaJ2PAI4IyfISoItF3+2cQEOsJWom9bYGrKUUW\/N0r9pW\/W3v\/2t0WUPNobNkVY8ZNetM5cNkJ3UqrYt4ElBWgk4QlYCbCXDIidsJevmBNk2aO\/evdscEjK5UclmEOdIWw1k6YTxp5\/mkNi6zpSDa+zAkrS459iHyzlDljNsU\/qz9WAzzJADtpJ187qMQsfRkm0AWUMPjZmsPbOO1QPIrqhT2mp16uwbfbcKEcUlAbIcYSsZFt0pJWX4j2TdvCAbkxc5yha\/kiXRTLaMQ820CYwUenCVGLOqzTAFWJ0uxJMCSZDtgy39jp6jp36WHtqfEwzBySq6sI0R\/iNZN0C25UJFQHZqphUEWDKN7+CqIWZVu7DpubcQh8slQnYMtqkeB\/j68yTxMieIFf4jWTdAtjTIUn8ItO+997RnOqQnxMyaYQC7DK7StoBNZTeFLJXnu5qVDNm2nrGgMGQzF382tT+ndKGf20rWDZAtEbKcRptnW0wGV+0xq1pikycFobaMS4Gs1iPV+1dN\/NlzyLDKHgq2knUDZAFZVoOy25i+wYWY1X6TmUI2xLm30iDbt5Us9tli7KPlDjOGL2wBWQfROWYp4pksR2E92qQHl35mpletVKQ+tJL68IpHd6JmTXm4vFTIxoZtVFgIOI\/hCtuoukUdlUphJYuVbGQXcyter1ZpMv\/iiy\/Uiy++qOhFFRq29BOftQqYrGZDnH0rHbKxYBsNFlOADfWcINCgsw3\/iaZboP6MFQPIArIJ3MysiqHwmocPH6qdO3c2gMVnXIGpuTYEYKkFtUA2NGyjwcL01BuzA5B9sO076R1NtwQTCiALyCZws\/4qTGJWJQ+uXMKmOFxeG2RDwTaaP5tCljrie7Q8kmOPnfSOplukvrSLBWQB2QRu9rQKl5hVyYMrmbAZKqoVsn2wpd+ZXmwRzZ9NnhMw2zIectu+k96U9v79+2rLli3idrQAWUA26hTtG7MabVKK2uvyC68dsq6wjebPppANcbQ8kXu3t5LXrVvXfEl\/5ZVXANlE+kepBqeL\/WUNHbMabVLy72rVJQCya81vcrFFNH9OebQ8seeTZg8ePGjOAUg8m4GVLFayXkMmdsxqtEnJq9fIDMgO+0C2Z4smq9lQJ98SDwHJ8wAgC8haDxffLWCbCiUPLpt+SksLyE5brC8mNOqzxVRHy6e7HjyF5HkAkAVkJwfE2IGl2BerSx5ck8IKTgDImhsv6bPFFEfLzbseLKXkeQCQBWR7B0LK1erYSJQ8uILNMAwLAmTtjaKfLdLlKj\/84Q\/VD37wA3GHeOx7HSaH5HkAkAVkGwVMYlbDDBe7UiQPLrueykoNyLrZi\/z597\/\/fQPX5eXl5ieF\/+CilXE9Jc8DgGylkHWJWXWbVvxySR5cfj3nnRuQdbNP25\/1l1v6HWALyLp5lLBcpYfwcNkCtnELQNZGrfhp\/\/CHPzSVLC4uKrrucuvWreq1115r7pXGZ1qBPn9uP7OlEkwvtpiurZwUkucBrGQLXsmGjlnNMWQlD64ceoWuk1as9N\/jx4+bn\/QhoD7\/\/PPqO9\/5jnr06FHze\/rdtm3bANsJA4z5M2A7LJ7keQCQLQiysWNWQ0\/gJuVJHlwm\/eOYpr1a1e0jgNLn5ZdfXtNkgiytbAHbaWua+rPJxRbTtZWTwlQ3jj0GZIVDVuIWsM1AkDy4bPqZM20fVPVqlX6abgW3YUv9ITD3QTlnX3PXbevPgO1Ti9nqltvO7foBWWGQzRmzmsNxJQ+uHHqZ1mm7WjUtl9IBtuGliTa\/AAAcJElEQVS3PdtbyTUekpI8DwCyAiBb+mp1bAKXPLhswBQ7bUyojrWd6qWtZPrgua3\/iqxW2EqeBwBZhpDlGrMaGwR95UseXDn00nWOHViy2QIO1QfANuy2Z22wlTwPsIPs0tKSWlhYUNeuXVsZ3\/v371enT59WMzMzo2P+7Nmz6vLly2vSXLlyRe3du3dyvsgVwiMlZnVSwAgJJA+uCHKMFplrtWrTz9oPSYX251pgG1o3G5\/1TcsOsgTK69evq0uXLqkdO3Y0IQLHjh1Te\/bsUfPz84P91XCenZ0dTTcmWErIFrMF3H7FFr2rkj7vvquU\/renh0oeXJ5dn8zObbU62eBWglphG8ufS4dtLN1sfNY1LSvIaqAePHhQHThwYKVPN27cUOfOnVMXL15UmzZt6u2rznvixAmjVWtfITEhW0LM6irNEr3xQ\/Lgch2UY\/kkrFZt+l0bbGP7cxe2+pCUjU04po2tW8w+s4LsnTt31NGjR9WZM2dWgXLo921hKM3JkyebvLQCdvmEhGyJMasrmk4BVidcXnYxw6o8kgeXd+e\/Pak7dBlEjmerIfrUV0YtsE3lz23Ykt7Sb5FKpVsM\/2YFWVqxEij1VrHusIbs8ePHV61w24JcvXpVXbhwQW3evFndunWr+RNd+dYta0xEX8gWswU85WlvvqkUgXbqQ9vGtJ3s8ZE8uFy7Xdpq1UaH0mGb2p\/1l309N0kN\/0mtm43PTqUVAdmhbeR257rPculvJivgdhm2kK0tZnVFK1PIUgbP1azkwTU1+PTfx6Ba0mrVVA9K14UtXeNYwsUWOf1Z8sUWOXWz8du+tMVAdkgIgu+9e\/eMTiebQLaa1eqYZ61bZ+53gOwarfoOLFGisasLzQUvK2VpF1twgIXEQ1IcdHMdWVkgS1u7p06dWmmz3tb905\/+1LtdbLKSHYMswbPv0NT58+cV\/W1ubq45vdwHWcSs9ihrClk6Yfzpp66+2eSTPLjaHa95C9jLAb7NrL+YSL\/YgpM\/S4ItJ91s\/TkLZIca6XPwyQWyBFUNWsqvQUu\/2717dzPB04eeY+iXKtMBguo\/7bCdMTEqhqzk8Bru\/i35YguOsJAAW466mY4TVpAdC+HpOxClO6ljZOn\/25dW2MTOtoFLW8z79u1rigdUB1zJZDVLq1jPeFlJgwurVdNpJ0w6iYekOPszZ9hy1m3Km1lBlhrrehlF38lk27Aek2eyU4JW8\/epMJ4AgOW+XYzVKg9vDwpbXK7S7ODRYzL6yeU0MiAbcKyZXKs4dAuU3m6+e\/du0yLa8h27wKLbbEDW0pAE2vfee5pJh\/QECNtpt4Lb4MJq1dJHEib3gi2+NK6xFCfYcpsHbNya3UrWpvGh0wKyoRX1Ly\/34NJQffz4cRNWQh+Xd636K4ESTBWwhu0UYHXFniflue\/MDOnL4WKL3POAqe\/1pQNkW6oAsj6uFCdvjsGF1WocW6Yu1Ri2pnHfAXZpcvhzKN37YNs+FBqqnr5yJOsGyAKyMceGd9kpBheg6m0m1gVMwtYUstRLz9VsCn9OYYzUF1tI1g2QBWRTjEnnOmIMrrEDS9TQEm4Wcha84IyDF1uYnJQPtGUcw59zmkwfkIp9SEqyboAsIJtzjE7WHWpwYbU6KXU1Cbqw\/ft\/+Aezvlcc9z0lUOxDUqHmgal+xPg7IAvIxvCrYGW6Di6E1wQzQdEFNV++3ntPvfyf\/zndT0B2UqNYsHWdByYbnCABIAvIJnAz9ypsBhdWq+46V5\/TZMs4QOy3jT9Ltklo2ErWDZAFZFmP5bHBVexqNfKFCKwNnqtxU2E8AQBLXZMMCxfThIKtZN0AWUDWZewky9MdXEWvVhNN9MmMJ62instV\/vDP\/6wev\/NO84Ykio\/2\/UiGhU\/ffWErWTdAFpD1GTvR8xJU\/\/jHP6r79+8390h\/97vfLfMyiCnAaqU9Q0iiG6ywCibDfyz7KxkWll3tTe56sYVk3QBZQDbE2AlWRt8W8F\/+8pcGsDTQtm\/f3vxbvxUpWMW5CzKN1QxwIULurkqsPxRsJcMipN1sYStZN0AWkA05dpzKMt0C9t1ycmpcqkymkKX2YDWbyipr6vGFrWRYxBLd5GILyboBsoBsrLEzWK7vgSXbb8HJO+hSocnpVmwZuygbJc\/gxRYTtUmGRRQhW4WOwVayboAsIBt77DTlm65WbRrTB9tUd6natNMorSlkA8RqGrUHiYwUsIWtZFgYCRIgUd+OFRVL5zK2bNki7lERIAvIBhgWa4vwXa3aNspky8m2zKTp22E7YxUDsknNYlMZfZFcXFxsstBJ5L4TyYCsuaJt2K5bt645k\/HKK68AsuYS8kuJt\/D42STGatW2RaKf25qsZgPFa9rqivTmCozBFpA111GnJM0ePHjQvGpy586dgKy9hHxyALJ2tuD8rlWRsJ0K4wFg7Rw0c+q+Q1IbNmwQu+2ZU07JX06wXYztYquxw2G1atNgcbDtuRBBIWzHxuTs0rZhOzMzo9avX69effVVcSuynMICsjnVD1g3VrJrxZQG1SF3EAfbgH6NovIrQGcGCLa3b99Wf\/7zn5tniz\/5yU+C3CKVv3fxWwDIxtc4SQ2ArGomAvrv8ePHzU\/60CGO559\/vvm39HetArZJhlL1lRBU6UP+Rv\/RR598\/+tf\/7ryjHHogFT1AnYEAGQL8YhaIVvKatXGDYuMtbURAGmDKqBh2oYqVUC3k7V\/tiv1vdgiaAeYFwbIMjeQafNqgWzq8BpT\/XOkA2xzqF5GnXq1qn9OQXWo17axtmWoZ9cLQNZOL7apS4ZsjatVW0cTH2tr22Gkt1Kgb7Wqt4BDXIIC2A6bA5C1clW+iUuCLFar7n5GsNUTKk2eRb6QwF2eqnKGWq3aimZysYVtmZLTA7KSrddqu3TIYrUa1hFxSCqsnhJK64NqyNWqrQaA7VPFAFlbz2GaXhpk+6CqTwLTzxAvmWZqqqTNAmyTyp20srEtYGqIPriUtFE9ldV+SAqQze2BgernDlm9BUzd1Xek0r\/pjlT6SA+vCWTGaMUAttGkTVpwri3gEJ2sFbaAbAjvYVAGR8hiC5iBY3SaANjys8lYi8ZiVkMcWMqhRm2wBWRzeNlEnWfPnm1SzM\/PG7eOA2RxYMnYXNkTIvwnuwl6G+ASs8qzJ9OtqgW2gOy0LyRNcfXqVXXq1Cl15MgREZDFajWpewSvDLANLql1gZK3gK0725OhdNgCsiG8JEAZS0tLamFhQV27dq0pjStksVoNYGymRSDWNo1hYsespulF+Fq6sKXrUEs4qwHIhvcV6xI1YOlA0AcffKDOnTunZmdn2axksVq1NqnoDIBtePPVvlq1UbS0iy0AWRvrJ0irgZsTslitJjC0gCpwSMrdSNxiVt17ki+nnod0NILUFxIAsvl8qLfmXJDFapWZIzBqDmA7bQwpMavTPeGZQvLFFoAsM59KBVlAlZnhBTQHsF1tJGwBp3daiYekANlEfvLo0SN17NgxdfPmzZUa+w43mUL2\/PnzisJ25ubm1J49e5p\/Hzp0SH388cfN\/3c\/fVvAlAaXQSRygIKqqRW2JcasSnVLSbAFZJl5mSlkCaoatG+\/\/bbau3fvGshitcrMuIU1pwtb\/UKCUrpZU8yqVJtJgC0gy8y7TCGrm02g\/fDDD1dWs6dPn1avvfaaIuejD+4DZmbgAptTUqwttoBlOihn2AKyzHzKFLK0kr1x40bTeoKs\/vz0pz9Vhw8fbqBbQowZM\/OgOSMK6JWfBpWEV+0hZrUsl+YIW0CWmY+ZQra9gtXPYGnLWG8h0+\/081pmXURzKlCAc6wtVqvlOyAn2AKyQv2NVrJ9B5yoO+3ntZRGA1doV9FswQpwOCSFmFXBDuTZdA4XWwCynkbknL0NW2onHZCi1S0+UCC1Ailhi5jV1NblX18fbFO9txqQ5e8f3i3Uz2\/1s1tsJXtLigIcFYgFW2wBOxqkwmypL7YAZCtzMv0sl7oN2FZmfEbd9YUtYlYZGVNoUwi2jx8\/biIxYl7ZCMgKdRDfZnef2+KQlK+iyO+igClsEbPqoi7ymCgQ+5AUIGtgBXqJ+uXLl1dS7t69W128eFFt2rRpNHf39XWUeP\/+\/YpiWWdmZlbydsvXf7hy5UpzyUTMD2AbU12UbapAX6ytzqtXrfT\/L7zwQvNr\/dO0fKSDAlMKxIItIDuhvH6JugaeBidl68KyWxTB8\/r16+rSpUtqx44dSl+tSNu08\/PzTXLTkJ0pB\/H9O2DrqyDy+yqgQfvw4UP19ddfN8U999xz6vvf\/34DVYq7xQcKxFYgNGwB2RGLDQHwzp076uTJk+rMmTMNPPs+GqgHDx5UBw4cWElCF0jQ+2L1SlinO3HiRPRVq4lzArYmKiFNKAXGDixRHZIutgilCcrhoUAo2AKyI\/YcAuXQ79tFEYiPHj3agLi95dv9vQmwc7gcwn9yqF5+nS4xq6bPbctXDz3MoYAvbAHZEasNgbJv27dbDK1YabWrt4r133WZx48fb1a4tB194cIFtXnzZnXr1q0m2datW9fky+FcVGcXtjiRnMsSMusNGbMK2Mr0gVJa7XqxBSDrAFmT56hDkO2ugrvPbak5Q3DP7awI\/8ltARn1x45ZBWxl+EGprbSFLSCbGbJD1RN87927N3m4Kocj47ltDtX51pkrZhWw5esTtbTM5GILQJYxZAlmJqFCuRwasM2lfN56ucWsArZ5\/QG1KzUGW0B2xENSHHwaW8lyh6xuO2Bb\/jQTews4hIJ9sbaIpw2hLMowVaDvkNSGDRvU\/fv31ZYtW8SFoa1bXl5eNu28S7qxEJ6+k8PtOsZCePSBqJdeekktLCw02doxtybPfF36EzsPYBtb4XTljx1YonhVzjGrfbDl3uZ0lkVNKRRow5YuHlq\/fr169dVXWY+bPl2iQ5YqjX0ZRd8BKa5hPabOifAfU6V4pZOwWrVVjPN7bW37gvSyFKCDonQ3Ml1k9OMf\/1i9+OKL4t7xnQSyZFaTaxUpTXd71\/RaRX2a+O7du40XmV7byN3l+mBLMcND78Hl3p\/S2ucSsypVA+qrXp3TqhY3SEm1JN92E1TpQ\/Me\/Ucf\/T5vmvfo7+3fS7gvPhlk+ZpVTssQ\/pPfViFjVvP3xq0FOCTlphtyrVVAv0K0DVVKRe\/tps\/Qu7v14kPfOcBZW0CWs3UG2obntmmNVuIWcAgFAdsQKtZXhl6t6ndzm0BVskqArGDrAbZxjJcrZjVOb+KXCtjG11hyDX2r1fYWcOmPvgBZyd77bdsBWz8jcotZ9etNvtyAbT7tudVc22p1TH9Alpt3erQHsDUXD1vA5lrZpkSsra1i8tP3QbWm1SogK9+HrXrQha2EwwFWHXRILDlm1aG7LLIAtizMEKURY1vAVOHQgaUojWFeKFayzA3k07zaY22xWvXxnrB5EWsbVs8cpWEL2E11QNZNN3G5agj\/qSlmVZwDfttgwFaO5aZiVmMeWDK5V6FPye59CZTm\/fffb16J2v50y9d\/o0sv2u8uD2EtQDaEioLK0MHctMqV\/l5bxKwKcrxOU3FIip\/tXGNWQ\/fE9YZAfQ3vtm3bVq7YpdsADx8+3NwYpeGZ+spdQDa0hwgpT+ohKWwBC3Eww2YCtoZCRUrGbQt47K57uq\/+zJkzaseOHb1qDL1\/nFat9Jmfn29+ahifOHEi+Kq1r2GAbCTnlVIsd9giZlWKJ\/m1E7D10880N\/eYVZ+3ttEK+JNPPlnzatPu71Pfaw\/Imnpn4em4wBYxq4U72kT3ANvw9ue2Wh3roX6mSivW9rNRDV96xKVXpN1yaMV67969VW9jozQE2QsXLqhLly41q2D9\/5s3b1a3bt1qitm6devK30NbAJANrajw8nLAFlvAwp0mQvO7sNUvJIhQVXFFSo5ZHYKsyXPUIch2t5Ep3fXr11dBdajeEM4ByIZQscAyYob\/IGa1QIeJ1CXE2k4LW1LMagrIDik6BOlpC4ynAGR9FSw8vx7A+jJv1xPJWK0W7igJuofwn2ciS9oCtnGN3JDtvmrVpu1DaQHZECpWUoZNrC1iVitxigzdrBG2OWNWU5o4xcGnsZUsIJvS2qhrUIG+57a7d+9eeaE3bfHRh56j0X\/0oRd84wMFQipQ8iEpLjGrIe1lUtZYCM\/Ro0ebEJ6hyyLGQnj0gShqw8LCQtOU06dPq5mZmebfJs98TdrflwYrWVflKs7XNwEQZA8dOqT27dsHqFbsGzm6XgpsS90CtvWJ2JdR9ME4ZlgPIGvrAUiv9LaxfssGSULgLeEWKZhXrgLSYMs9ZjWnJ5hcq0hputu7ptcqdtPRIuHixYtq06ZNwbsNyAaXtPwCNUy7Pc0R\/lO+2uihrQKcYYvVqq015acHZOXbkF0PYob\/sOssGsRWAQ6wlRyzytawwhoGyAozmKTm9sGWDi3EfHuHJH3Q1jQKpIy1LSlmNY11yq8FkC3fxix6aBP+w6LBaERxCuhLUHR4mb5FSp+Ad+0wtoBdlasjHyBbh53Z9BLPbdmYouqG+MTa1hKzWrWDBOw8IBtQTBRlrgBga64VUsZTwOS5ba0xq\/FUr6tkQLYue6\/prX6pcfsP7Rccx5YHsI2tMMo3UaALW4rVpI++TpT+\/fbbbze\/m5ubMykSaaBAowAgW7Ej6Fixt956a+X1Ud3XQqWSB7BNpTTq6VNAr1Z\/+ctfqps3b64kcb2rGypDAa0AIFuxL\/S95FhfL\/bGG2+oAwcOJFenC1s9ySVvCCosXoGxA0t0Cp7+jgtWineD6B0EZKNLzLeCoVc7xXrlk40SiLW1UQtpTRRwiVnVfogveyYKI02fAoBspX4xdiF233VlOWVC+E9O9eXWjZhVubYrqeWAbEnWtOjLGGT7tpEtio6WVG\/fYQsvmsTiC0bMqngTFtcBQLY4k5p1SCJkdc9wSMrMxjWkQsxqDVaW3UdAVrb9nFsvGbKArbPZxWdEzKp4E1bXAUC2OpM\/6zDng082ZsHK1kYteWmxBSzPZmjxMwUA2Yq9YSyEZ3Z2diV2VopEgK0US423E+9ZLcOO6MVTBQDZij2B02UUIc2A8J+QaqYpS9JqNfctaWksglpCKQDIhlJSaDklTxh6RaSvxsPtPXyc1CVmlUPrS\/1iykHbUtsAyJZqWfRrlQKItc3rEKXErHK8JS2vZVH7lAKA7JRC+HtRCuC5bTpzStoCNlWllMOCpv1FOn8FAFl\/DVGCQAUA2\/BGKz1mVdItaeGtixJdFQBkXZVDviIUAGzdzVhbzGoJseXu1kZOVwUAWVflkK8oBQBbM3OWuAVs1nOlAFlTpZCurQAgC3+AAi0FEP6z2h0Qs\/pMD0AWU4WLAoCsi2rIU7wCXdjWFP5T82p1yrFx8GlKIfy9qwAgC5+AAhMKlB7+IzVmNYfjlnZLWg4Na6sTkK3N4uivswKlPLctJWbV2ZAeGXEZhYd4lWYFZCs1PLrtroBE2GIL2N3e3Zwl35IWTiWUpBUAZOELUMBRAc6wLT1m1dFkyAYFkisAyCaXHBWWpgAH2NYWs1qaD6E\/5SoAyJZrW\/QssQKpw3+wBZzYwKgOCjgoAMg6iIYsUGBMgT7Y7t27V1EYkM8HMas+6iEvFMijACCbR3fUWokCvuE\/WK1W4ijoZrEKALLFmhYd46QAwZJWovTf2MUWiFnlZDW0BQr4KwDI+muIEqCAsQJ9h6QoJEQDmAoiCOut5bm5OeOykRAKQAF+CgCy\/GyCFhWqgH6mSt378MMPV\/VSgxVQLdT46Fa1CgCy1ZoeHU+tgH4+216p0oEovZVc0\/3IqbVHfVAglwKAbC7lUW91CujnsX0d19vIGrTViYMOQ4FCFQBkCzUsugUFoAAUgAL5FQBk89sALYACUAAKQIFCFQBkCzUsugUFoAAUgAL5FQBk89sALYACUAAKQIFCFQBkCzVs7d06e\/asunz58ooMu3fvVhcvXlSbNm0alWZpaUktLCyoa9euraTbv3+\/On36tJqZmVn5Xbd8\/YcrV64oOjGMDxSAAlCAFABk4QfFKXD16lV16tQppYGnwUkd7cKy23mC5\/Xr19WlS5fUjh071KNHj9SxY8eayyHm5+eb5Lq82dnZld8VJyI6BAWgQBAFANkgMqIQLgoMAfDOnTvq5MmT6syZMw08+z4aqAcPHlQHDhxYSUI3Mp07d25lJazTnThxwnnV6rrSpn4cPXpU3b17d6V977\/\/\/qr20h+w0ubikWhH7QoAsrV7QGH9HwLl0O\/b3dcAIxC3t3y7vzcB9pisritt3Ydt27atrMjpC8Dhw4dXVu1YaRfm0OiOeAUAWfEmRAdMQNm37dtVjoBFq129Vaz\/riF7\/PjxZsVIkLxw4YLavHmzunXrVpNs69ata\/L1WcZnpT3UPlq10kdvZ4dYacOroAAUCKMAIBtGR5TCRIGh1ajJc9QhiHVXwd3nttT1oXq7svistAnun3zyyZoDXN3f+660mZgSzYACRSgAyBZhRnSiu+rsbvmGhOyQ2gTfe\/fujR6uGoKxyUp7qHy9stYrcJ+VNjwJCkCBsAoAsmH1RGmJFBg6APT66683B4NyQZbuIB4LFfJZaQ9BtrsC91lpJzIfqoEC1SgAyAo0tZ6o33rrrTUhJN3nhwK759Vkn+1Y04NPYytZDpD1WWl7iY\/MUAAKrFEAkBXqFLRa6ZvQu1uHQrvn3Oyxg0V9K9x2RWMhPPpA1EsvvdRcVkGfdsxtt94YK23TlazPlwBn4ZERCkCBXgUAWaGOoUM32jGSNpcuCO22UbNdQ2SocJPLKPoOSJkeNvJZaZsefAJkjdwEiaBAEgUA2SQyh6+kD6i1bxW3VTa57KFvN8D0WsXuStX22sbubVEmp5PHQnj0gSvSwGSlHd4jUSIUgAJ9CgCygv1i6FRpN85TcBeLbLrrStv0MgqflXaRgqNTUCCjAoBsRvF9q26Hfbz99tu9KxjfOpA\/jgKuK23TaxVdV9pxeltXqfpRTvfFEvr3R44cwZ3XFbkEICvc2HrLk+7RpcM5+lYi4d1C86GAaAX0lyj9kgr9hZg6ZfI2KNGdR+NXKQDICncI\/e34Zz\/7mfrd735ndLWf8C6j+VCAvQL62f7i4mID1Y8++qh59SJehcjedMEbCMgGlzRtge2DOn3vPU3bGtQGBaCAVqC7ZY9t4jp9A5AtwO7dgzQFdAldgAJFKKDHpunp8yI6jU5gu7g0Hxi6mKK0fqI\/UECSAu1dJtO3NEnqH9pqpgBWsmY6sU01dsUi20ajYVCgAgX0Kvadd95Rv\/jFL1T7PcAVdB9d\/FYBQFaoK+gDT9R8fEsWakQ0u1gF9JdfemEFXb958+ZNdfjwYYXnssWafLBjgKxQm7chixOLQo2IZhepgN4m\/s1vfrPqtH83rKfIzqNTaxQAZOEUUAAKQIGACuht4va94lQ8YmUDiiyoKEBWkLHQVCgABaAAFJClACAry15oLRSAAlAACghSAJAVZCw0FQpAASgABWQpAMjKshdaCwWgABSAAoIUAGQFGQtNhQJQAApAAVkKALKy7IXWQgEoAAWggCAFAFlBxkJToQAUgAJQQJYCgKwse6G1UAAKQAEoIEgBQFaQsdBUKAAFoAAUkKXA\/wOMOeunlIb9fAAAAABJRU5ErkJggg==","height":258,"width":344}}
%---
%[output:3e12932f]
%   data: {"dataType":"text","outputData":{"text":"仰角 (弧度): 0.5236\n","truncated":false}}
%---
%[output:7400464a]
%   data: {"dataType":"text","outputData":{"text":"方向角 (弧度): 1.0473\n","truncated":false}}
%---
%[output:85c92fce]
%   data: {"dataType":"text","outputData":{"text":"仰角 (度): 29.9981\n","truncated":false}}
%---
%[output:8a6169ee]
%   data: {"dataType":"text","outputData":{"text":"方向角 (度): 60.0075\n","truncated":false}}
%---
%[output:3fca16f7]
%   data: {"dataType":"text","outputData":{"text":"Processing Channel: 1\/8\nProcessing Channel: 2\/8\nProcessing Channel: 3\/8\nProcessing Channel: 4\/8\nProcessing Channel: 5\/8\nProcessing Channel: 6\/8\nProcessing Channel: 7\/8\nProcessing Channel: 8\/8\n","truncated":false}}
%---
%[output:3a97c91f]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdkAAAFjCAYAAABxFnERAAAAAXNSR0IArs4c6QAAIABJREFUeF7tvV+sXVd1N7rQhQcjHhKfoE82X6ukOKFS9YECyMaC0kAk0iZ5Qch2WxI\/4ETWDQ5qWgdHsgOB2FLcuqQKrmkUF1VO+L46vblPSbjhAVwVZBIFrKZPmESxaJs8ENdRhcgDSFyNdfzb\/p1xxvy71tpr7eOxpcgne68115y\/Ocb4jTHmWHO+7Te\/+c1vGv84Ao6AI+AIOAKOQO8IvM1JtndMvUFHwBFwBBwBR6BFwEnWBcERcAQcAUfAERgIASfZgYD1Zh0BR8ARcAQcASdZlwFHwBFwBBwBR2AgBJxkBwLWm3UEHAFHwBFwBJxkXQYcAUfAEXAEHIGBEHCSHQhYb9YRcAQcAUfAEXCSdRlwBBwBR8ARcAQGQsBJdiBgvVlHwBFwBBwBR8BJ1mXAEXAEHAFHwBEYCAEn2YGA9WYdAUfAEXAEHAEnWZcBR8ARcAQcAUdgIAScZAcC1pt1BBwBR8ARcAScZF0GHAFHwBFwBByBgRBwkh0IWG\/WEXAEHAFHwBFwknUZcAQcAUfAEXAEBkLASXYgYL1ZR8ARcAQcAUfASdZlwBFwBBwBR8ARGAgBJ9mBgPVmHYE+EXjrrbeae+65p3n00UejzX73u99tTp482fzwhz9s\/33f+9634vof\/OAHzcc+9rHmiSeeaD772c\/22UVvyxFwBAwEnGRdLByBBUTg\/PnzLUnefvvtq8gSv1199dXNww8\/3Kxbt64dIb7\/6Ec\/2tx\/\/\/0LOGrvsiOweAg4yS7enHmPHYEZYVokK\/BYEeuDDz7Yfv+tb32rWVpachQdAUdgDgg4yc4BZH+EI9A3ArFIFs8SUn3qqafatPGLL77Y\/NVf\/ZWZQu67b96eI+AIXELASdalwRFYQARySBbXPPfcc+0IfR12ASfau7zwCDjJLvwU+gAuRwRySJbTxrt3716xPns5YuZjdgTGQMBJdgzU\/ZmOQEcEcklWUsZf+tKXmg984AOeKu6Iud\/uCNQgMCrJ\/td\/\/VcjHvZLL73U3HLLLc2hQ4eao0ePNlIVuW3btprx+D2OwGWBQA7Jovjpa1\/7WiMpY11tfFkA5YN0BEZGYDSSff7555v77ruveeyxx5ozZ840p0+fbklW3gcU4t2+fbsT7cjC4Y+fLgIpktWv8fzsZz9rduzY0dx7773+fux0p9V7tgYRGIVkhUj379\/fbN26tSXSf\/qnf5qRrLzTJ\/\/\/5JNPti\/er1+\/fg3C7kNyBLohkCJZ63UdeXXHK4y74e53OwKlCIxCskgT7927t9myZcsqkpUo98iRI06ypbPp1182CMRIVsj0tttua77\/\/e83svEEPtg16ty5c\/6u7GUjKT7QsREYhWRTkax426+\/\/nqbPsZuNWMD5c93BKaEQIhksQ771a9+1dzV6Sc\/+UmbNv7IRz7i1cZTmlDvy5pFYBSSFTRDa7JPP\/10c+DAgebEiRNtlNvlI2TtRVRdEPR7HQFHwBFwBLogMBrJSqe5uhiD2LhxY1sMtWnTpi7jateejh8\/3hw8eNALqDoh6Tc7Ao6AI+AI1CIwKsnWdjp2H4j7jTfeaPdnldSYvw40BNLepiPgCDgCjkAKgTVHslKZLIUde\/bsWVHBnALCf3cEHAFHwBFwBPpGYG4ka6WGY4N5\/\/vf36m6WBdX9Q2ct+cIOAKOgCPgCKQQmBvJ6o5I4dPOnTtXrZlyQVSXddkUyb7wwgspbPx3R8ARcAQcgQkhsHnz5gn1Jq8ro5AsCHDDhg3tDjT6ozenyBvKyqtCJCvk+sBzrzavPPGlVc3+cum69rt3nj8bfOT5625pzl93a\/M\/Tz8cva6mz7hH+hHrQ5e253Wv4LTu\/E8HHYfgJM8RrJbOPjOvoY32nBz5PHvrN5p15882v3X64ap+yjPeWrrWxFPL5TzlNCVPqd9zwcD4Q7Ibwyf3GSXXlYwL8pGyYfz82jkEDtLWvHTvvbd9tfnin36queG9V5ZAOPq1o5Cs3oxCo9DHZhQxkt1y8hfN4Q\/9qrlh08rJku\/\/6IqfNw\/cdE1wYuQa+aSuq53ZZy+8u\/nKd15tDeUDn7pmVR9r28V9p16+0Jx65c3m1CsXmreWrjNxEPxlD2lZ1659jWrfi29vn1OLk\/RTPnqOePx4hmB16q4PVkEDPGJzzg3fcOzHs3npgpM4eje894rk\/MoYxeBL\/0Q2\/vmVC83hD\/961iXuP2RHfnx+x7uakrHJtft+9I52vr795rtncoHvRV\/kd9abkB4xXl0w0nMt\/89jZ5mWvmHcVYJw8SbGwZIJxiMmmyV9sDCC\/Mtc8LghN2gffRA5kc8fvPfKVTIS6gvkBbKi51ffx\/IEHIa0hVqOpH9f\/JObova5BPd5XTsKyaZSuX1sRpEi2e\/ddf0qj+htf\/Hd5sufuiY4iUJMnzh2pp2b2HVdJu8Tx37ckpMYYPn3N3\/9yS7NrbqXx8DjEOWFUZFo\/\/bbb28ef\/zxpjY9I1jiY2GN32S88vmeIknpjxCK\/p4HBKzku1qcgEesj\/xMlpEuOIXGPSOOi7LGciCYiAPGY+Xv8DfwsK4PCRNwELmWZwCPnO+lzZCTEsJI+ha7T\/czhhePO3ceQziw\/n35pmtW2YiUvLAeQYatdvj5FkbSDxAm6wdwuKRby85l6PuY8YjJjnUfj335mcPaQo2ROLhOsgV0gDVZvemEpIqPHTvW+V3ZIUgWSqMFvGDY5qUivPKRNAiURRRMG9Suz5H7tdGEsyDkwQbq61\/\/ehvFlpBs62VvuqIdB9r7ynOvtsYiZIS7kKw8I+aMyFjl+TGiThlNYIYUlXbEanBioxjqmyUHIIAQybLTIXN56uU3s2Uoh0xFHiEvITmyZNTCiMkMJAJisWTFGjueBb0Up7Sr86sx1KnJlLxA7pmE5O9UvzRGpWTK1+c65zGStXSHxw7ZknGlnGF2PGptmDgiTrIV6FkVxzjyrut2iimStYQ+Fcmy4UsJVgkcTDQwNHJ\/jGRF4EXQzZTWKxeC6xZQFDHUeJYQo3ilKUOQGhPjxyTLnnhuhBIzqmgDzwvhxGMN9V0bTfl\/NqxMJoJ1SkZSGOH3VCSL3yUKkrkR0hSHQRtQNpS4R67pSrIpMi0hWQsTOEhwJmGsQ7ISkwcedx8yDMeNnU7IBfAORcyQD7EPGEuNzOSQLMtCV5LVzgWcBHboNMnKGHOCAe3A5+oIXyckK8sTXee35tld7xklXdy1013uj01WShnwuzx\/bJINpQKZWCwingfJgrRBDEOQbCjyYtlIGUREqXAwuN8gWhgfjvgtRecoPkc+c0lWIl12WDTJMvloskk5atzPEGmWfp8zdrmGdS2UGuW2eJw61cwEE8uaxBxTdtxEbuHYiBxYEZxFsiGZTNmVmBOCpSNkPOCcLMvupeUkzgzURLIhkuVxsj6Jgy82kJ3AUEFSzfg1Jk6yuZo1getCk6UjlpDgi4Flko2luHKHywYEAgmDn1p\/02uR1rqJ9sjFgMh9vGbUZyQ7D5KFodYGsQ+SZRJlko1F\/KWGBMYylS4GySLSS5EsR4YlJAsDijVZYBD6HnKGqK80wtAky\/NmYaKdCSuKwxpmCNPUGjXrTopkrfEORbKCDfQ8RLKogbCWT3R2BljnZEFCJCtZFfmkSDbHrubYSSfZHJTompyNKbpuRhHqUl8kixRlKiLJgUaTrAi2fECGaEN781aa1CJZNggcyQ5NskzkOWQSilwsDNmrZoPYN8nCeAmG8yZZRGWcgswhWaTxQLI5xUApktVkOhbJWmnz1BKLTvWCsEJLA\/I7pzh1BCd6J3iAZBDBhUiW+4eahZRd4HqDLiQbcyw0ycJB46WGFMlypsWKZJ1km2Zy6eI+Xt+JCfBaINnYepuV2oqRLCpJu0aybHTh3YZIlo1byElJrcmygQit+eSki5lcLBLtSrIyVvloA5QTyYJkOZ0aI1m9HshGM\/VuYS7JCrmIYa0lWeDB8sZpbrTPjiX+xlqzXpuOkaxV3AO51E4qO6AcaTM24sBw9b+VJdIV2ugfF47VkCxjLvcDDxQaLsvZyrcSckmW6xuQreMqc2mb9YkLGlnudCW1k+wESVYmU17hkY+1UUVKOFO\/dyVZTXAwwvCO9XpRqj\/yO9rg1KcVyfJ1ocrRkNeNSNIyOnhuacqPx8YGgIshEC1zJMvEGiLTHJLFujiTLBN4V5LFmGC84DyUFM2F0sglJAsMYVh5icBaapDruCgltH7IldfASkes+nuQYOj6lLwz4aBfOSQrhCIfjmT5fpE5RO4hfEKyryNRRLKYZ14yyCFZXTg1JsnG9Ai\/sf7DEbCyICmS1aS8LK\/Lrz12sS3SDux2TlYmJYPz\/n1ykawAMGQ0GyoFzxEGGHMmQE2yNenjGMmyUFkkq4VOKwKMEzsB8OD1GpSOIkqEUZMsCNAiWSaYGpKVZ6HwgtcsJRLFWpF8z8Yx9AoRkwWcAytaEyxSJGvhNwTJ8pxbSw3AJhbJ8twLyQAHzDnGogtqupIsR0wYB2REF\/ks69nye9RMsng\/E\/ejzRjJaufYcsBCkezUSBZOHyJZ2CP9PV4JDBVCaZJlsgWWTI56iUYXA\/ZFspbOOMmWWOOMa+Vd2SeffLLTAQGhx9SQrPa+UySbqu7T0a4mWSFEbQQ5qgqlc6RfNSSL9NY8SVb6iteILLxC5KtTd0yyMAy67ZgX3QfJsoOh155DEWuXSDaHZBGRWIZPy8kikSxktXV6Lm4cU0OylgOmlyCmFMlynQbbNq4V6INkkQa2SNYiZaSqrdeeaiNZJ9kMkuxyCYqiZBOEodLF1kvNKUPJ6ywgWXh+8v+cukqRrDawvO5nRZn8GgGMC5QAyod3ZjXJIgrgqIHXojiy0CQbqkqE8ohCYg1GrxctY\/LBGelzsQkiEaTlQIosN6Eir1yS5bRiV5LlubdSXynZCUW41rjhgPH6nU4XW9kN7iOWEkKGbwiSzXXQukayMZLlqnboQyj7g3VdkBTIhQsaeU2c12GZzKx6BytdDD3MTZvCRnAEzxErdCWXZK00KzsacMZySFavOXNf9XNyMoSaL5xkuzDoxXtT1cV9bUhhdTUVyXKqcFmZlncvyiVZVupYtZ1OSYI8UyQr9\/H6E3u4XNABB0ArvE7fhkg2lWrVv3NECNz7IlmLcFHdibQwYwKnhw1pTroYhhDkZ1WLCslazoj1vbQTi2RDzoU2yHr9sw+S5aiE04qYO50uZl1iOdPX4\/9DDpplkDldrDFBP5lUREegL9gghPUAuspOnM7+aJKF82SRLHQaujQ0yXLW6pIuXdrBi+diDJLVNoXnVDsRMQcUGUKtm5bOeLq4B+KdVxO5JMvelGUYsD6r9+\/MJVltyGE0eJMLPIPTeRbJInLhdacakmUDx9G1RVD691yS5bW\/3Eh2SJJlI85GVKfs8ZpQCcmmItwhSJb7HYtku5KsJj92PGLRC+QGjhBkGwVFKZLVRMevkMi9TJacaRLjj2fw99Ie1tq5MpcJAPrPTpguEFv+bbnIx4pkEZFakazcx9W6XUiWSZmdp5xIFss30ga\/bw391+vzvCZu4aExsZZScklZL5\/Niy\/6eM4kC5\/6GFioDZDsls2bV+xpqw1iV5INpYXYOLFBzCFZVIxiyza9fZ5FsjAoushE7yCjC6Q0iaLYCAqXIlmk2nidS8bIJMvGTm+qwcTKhMCp0GVDvVzghEiWDWlOJDsGybIRtcYNYw4Z0kVJsUi2L5IF2TEpQqd4LR3fMcnGsiB9kSxnMuCMhkgWjqkmWSYGfveVZQopaDwPesKRNXRCkyzmlmXMsgu8BCJtlZKs9AXR\/dRIlp1vJtmQIxZyTJ1kC1lR0sX79u1r\/7MOZp9HdXGMZNmzhadspUhC60MgzNhG5zWRbIhkmUygcNxf3qzBShfz9FnEj5Qcj0uTLHu58ITlPivCtSpFNdlwRoDHx69QMMnimVa0EooeUIWL\/rAhtVLvoQKxEGmWfr9MEpeqafU6ujag8v86lbmoJMskwbJgkTw7pLzjEEiWCZDJE86LzvLwvGqZkvZRsa4JVuu5doZ0Rb9cH1ufh5xaJMvpbtZXOBCaZHXRFjtI+FtHpvxeO7CynCeO+LXMal1LkWzu0ouTbM8kO4\/qYk2yLAxQXMub168NWOtDJSQbenUCBlRHoPyaiX45nxUxRLIw4hzJdiFZvXan00ZWQZQ2dsCZ+8Ekyyn5XJJlB8gybJYhY5K1sgJDkazIAHYC4qinL5K1nIyQk6RloSaSZeK3UoTaWQiREzsRIChOu+q9c0Mki1Qx5A7zqP\/l6Bh6wq9tQdc5Zc3YdiVZyGmIZDmlDjzgXIZIVs+f9Q4xk6Yusssl2VA6PEWyWvdTjqmV9i6kn7lfPtd0sWwycfz48axBHjx4sNm2bVvWtXzRyy+\/3Nx5553Na6+91n69cePGFcfmhdLFNSQbWh9iZZU+SNu8rgFlTxWQIGUtBli\/+8gkGotkse7J1c\/y\/C4kixQaFJ3fScVcaK8cmLBRYCIEFnxU3nI\/lzdrh+PChoa9dZYBUfgSktXzqNPwwBDXhbzvUoOB69nxsUhWG1027LFIVs+F5cSEIh70zSJZrtxGm4wJr2EOSbK8WxkXPvG4rexPDsmCbDTJ6rTxFEgW9oTnF\/2qIVlE\/\/q9ZUvWlh2b5TOwoaOcwQuRrK7NQN9D33skW0iFqXRxYXOzy3G83YYNG2av\/+gD4EtJlo08G1FO64FsOM0MItNCxkVUOSTLRowJSkeq6AMUjr1TXl9ORbJsrGBQOLIX5dXrVDkky\/3jogp44Fx0gRS9JtmQgWPvG+PnqFOvNy9fs1ykgn6xkYDx1htehEi21mAwyeK1rxKS1WOAzMVIGWvrOgOjC5k09qyTtSSr+2WlU1nH5G8mCb08U0KyTBxoB7ptOWSaZDnFzKTWlWT1HPKWlZo0uRaD5ZadRHYweJkH31upeA4KdCW7dp74uTlrzqW64SRby35zuk+i2MOHD7f\/rV+\/vn2qJnQh2Zsf\/Mfmf\/3RbSsKn1i4QAJyPytgiGQ1GTHB8bqHXt+tIVldCanXk0Ika0Xd2ojBsMq\/qJJsMbh4pinw4D1+UQ1tvWqhiziAE05LYZzZcHPVsTZuloHTJCvX6LQ6xqa30NMky4ZXfuOtGyHGOpItNSQ6JQYi4XFzf0OkWUuycC4wnzJPFslq2bccO1ZtGF3tSOKariTLssC7WeEgDQsn3l6SnSRkK7gaGLKAV8\/01pSobUDxHeMnf9eki7kACLIWSxez\/OvxaFLOJVk97lQkqwk7J5Jl27lsXy5Fv5wCTpHs8zve1cgeCov0mVu6GO\/Gbt++vbnxxhub3bt3Ny+99FIQq5pTeKRg6uTJk82hQ4caHPqO6HbHjh3Nli1b2j0wS0hWe4nsvTGhMRmlSJY9aaR6LQPOXnyIyLuSrFWWD5LFWhaTrPSDU2gWCYUIArig3RyS1REpzweiiK4ky2QyFMmyIQmRrM6a4J55kKyVFg5hr\/vJRjdUYbusK5eyB2x0Q9\/LNboa2TpijU+r0uMQedEOl3YWLAeUK3753XW9O1ZfkSx0wopkrTGA4LmwiwuzctLF2r5YBz\/wPIVSz7UkC\/vJuqHfvtAOmpPsyO6FFEydPn3aJNmtW7e2a7y5JKs3Ao95b1ppYYT4VQdEP0jd8g4r7J1a623wNC0i196s5RRwZKG97xKS1UQPI6wjsFyS5XHr13m4MEd71nr9J1bBzHiEPGYZh3z0rjdIz3H6mr1+yEROJBsiWb3mzAUsJSSr+6XTrFgnY\/JApKYjHnaGLOx1hDtPkl0e58qD1FMkq7M\/6K\/lLEBPNU68tszzzVmNmkjWeo5uh5dJrJQ59M3KUMTWZIckWdShhHQDr93xzmbLzr0d4cIRc5JdMJI9\/OFftz2W9INWHBBALJK11vF40wIrMtEvy+vXA5hk9Tuc4pFz0REMKxf6WP3V68fWM9hRQDrMWrviFJomWYvQcislGSuLeKw1Vo5kQQZIRfNrOfzKENLFPN\/cDlKnaGfZgH1wRUQVSxdrr9zy1hmTsUgWDiCfWQv11TKkHTlrLhDxcPqTt+9cxqFbJMvriaGI2YpkLZJlp5fN1hgky8VZvEsa+oUoVxf0hXSGHWntYIdej8K4Na6YV\/nXImVNjNJXvb0l111gTBxsaHukdUYCo1MvX2j2\/egdjZPsApHsL5eua37r9MPNnj17mrvvvtskWQyH0zBWsYYmIwh\/iGT5d94zlZVDCy9+C5Es1jQhoBxdhNZL+Rkcvcj38DC5uESneTXJ6rSr3nOZIwhe18LYoIxdSRb9ZydkmSyvn53rapEs+gen5FJ\/L53ow7jib24rh2T1s61XQ7i\/OelijNWqGrdSmjGS5XnUqUp5jrXerUmWZdxaB2fjrcfHGLJhnwfJaseN07jWfIci2ZzvoX+MqVWnkRO9alvDyzqhbE4skmXd0e\/Oa1nTwYBeTrIKD+XZsFlw9IAHr7f\/27NPNEePHm3EXv\/H1nucZGMcm9qvWN9buyZ75MiRFaf3xNZkJZLdcvIXbXGTNszcnxTJ8joevHcIEadwOI2shUxXe+aSLBNXqIijhmStiJVJln9HutgiWemfji4Eh1yS1al4nhcYRPa+9R7GoT6FiJENXy7JhlJc1sYS0ubYJMvFastkvjJS58jJIlm5R2+YPwTJxtZqcyNZjqxYdkKRrCVTwChFspps4PCE0shrhWR1FpDfHghtlxkiWb2c9s7zP21hf\/bCu1vH3yPZGMvO4beS6uLz1906ewfTIllOF9aSLAy8jnT53VVOHwtEoXQx\/2YVqcTWl3JJFh61\/BsiWUS4eiMAPY5QNLJssOIkqyOpduxU4cyOReyEIjxL7y+tiY77it+wHqtPYsGz9Uv9IJ5QJGtFk5jTUPEKnqHJRqc00Sd+RmxNNkSynM3gmgSOXDFO+ddamkBf9fvZy\/flpYtDa9GMubSHVCiToNaNRSBZfeC8JuVQWpi\/13URHMnKdcuEfmZFNic3kuX7IWuxSB0yws4anDDcn0Oyls44yc6BSGOPyH1PVqqLc0lWb6VmpYtDkawm2VB6VaeD9Ibc2ojOi2T1i\/jopy6A0uMHuSFNWBrJWmuCNSSr15lCRDckyVobrVuRrDZIllHSBk4b4z5JVp7F69z6b3YQ2PFwkl3elIFJKFU9zU6+lS4OkSnSrXjHGnpm6SdOEws5v5bscBYJJM3r81akrou0OJLVzpOVLuYMoJNsD4RqpZC7HnOXs+OTRbJ6yzLL6IW8NygBBIw3GOe1TQioTq\/qNchSkk0VcZRGsoji9NqKjmh431gu+KohWVZejpZgsLqQLKKeEpJlUrHeIw1FsjoTESJZyJt2xFitUkUqNSTLm5Hg1Sf9\/jYbff6bU8eYY14OYYJgGdcb6KdIqCSS1W1Zkaz1vmhuuhjOhLWLkX52KF2cQ7KotQCGOs2KcXHEinV1RPIhkuX3l0tIljNwJSRr7QXONgF4gGTRf+gpO+26Kt4j2QLilXda77vvvhVbHsrt8hrOsWPHVn1f0HT0UrzCI5Gsjsw4arVSmqUkC7LQz4FR02lkTU4hcrQqKPW6GhsA3Q4rmvX6CwDUO1iFSNYaR4ggMEa9JgsHRRNsDslCOXXb2qCEUvFWJJvaESmHZLG+xK\/KSB9BbryuydkM4G9hyI4cvx+MylN9SpJFBDBquN\/a4ESugYOIv\/X6rHZ8WGeYZNs+XNwalIu8EAmFCp9i5MRzznrZF8ly+yw3mD9rHLUky+lYJk4uGEJqnPUchNz26aZrVhwW0ToHF5dYLJINFdPpcaPwESTLtiwUyWqShU6HIlmLZPnoP2D+l\/\/nuebUXR\/0zShyyFAXI+l79FaIOW3mXjMEycKg6EhWv+ITinT1mgvSrCFyLCVZa41Hv8rCJKdJVhMdjC3v\/MRVyHJ9KcniGbxBBfrB60s8z5ZXzoqMMS4biOXj8KwsgSZZNkAwsHouNMlaEetaIFkmTh3VanLWJMvvQuooOJZOLY1kQ85TSzSfumYFAbFMWY6plTbVJMvyweMYg2RZH2AXtNPbB8nq1\/pCAQfji7+5Wli\/48xbl+oIl\/VVbJiTbC7LGVsd6luHPuoO6WJe+9PrgKlINrRrD4RPV6\/yPrlIJ7OHh9+tKBP44LehSJaLveBdcgqKjSq\/s8trNxhHimThGfPca0M8I\/uAoYyRrJYpGHx8zxtN6GstkrUqRDWJ6hfrubiDvXUdycpvlnORimT5VSMdyYYMvpUutiJZ611HMd68b3WIZLmwKrRFKRtpGT\/L9DxINuQ8lZIsO5QlJKsrz6EPnELlDfp53jh1XEqyGF9OJMtpW3mOXvoKVU\/DYU6RLKfGebcpa63WSbaAYOXSVCRr7dxU+Ijg5RLJfubPHmzfueqDZPXajkWyvI6C37WSjU2yiFJ0Va1FsjpCKCVZwcPapGJokoVTw+u\/+pSY5WsuVcHC4IcqPrF2z\/sxc8rWWnfSm4OEIvUQyfJhAhD0EpJFhIAxsTGFIRUylZSdEClHMZy9YMLlk4t09bK1MUgJyXIGiBUbpMbOVm72J0SyVluxTIcmWZaxWDqcX7fjdHGIZEUml8nryplDYlVbLztzyydXIaLlSLYLyeo3DvR6Kc8Np5VDkWyKZLVj6pFsIQtKtLpz587mxIkT7Z7C+AjBHjhwYNX3hc1PlmRZuaGQUCwYKiuVy7\/lRrJ4Fq+7hp5hFbjodR5eg+OoGkZQn8OLScB6Eq8xhXaCgscMb5jJw6pS1ql1bRDRB01s7ChYwoJn5ZAsv9YEAuPUF6fE5FldSJYNp45kLYLQBMVp8xjJWoU+HMnqv7FcwGPVjgnPaYxkNRGMRbKMDy89hNaKeV5T6fBSkmUZDUWy7HRw+6Ukq9thBzxVLY9+co0B\/tbzGiJZ\/b206ZFsIQPOY2OKUJd0JAuiYwMA71QbCUtxdCQRWYaBAAAgAElEQVRreYocycIQsseNYodckg0VRIW+74Nk2ajCW4ZXzUbQMgCaVEKRLBP9ECSrX8fSa20sM9rAWpW8vK8vIlp4311IFpG2FclqkuVtJGtJlo0gyJJJBWNnB1EcqmWH4VLEpKP2UpLFuBeFZPU85ZKsXprSa6a8C5x2RgTzPklWYw4nELYIfeEMl14WseY5RrK4HmSKDXz49S+ufEefnpftFb3wqZBtR7icSRbRjLWtHQuAlV7kHWf4PFVrPWneJKsVh0kWa6k6WtYEp6Ng\/L9WcE4n68rHUBQZI1l9IHyfkazepShGslo0taOC3zlVHiriQLoVxlQ7HdqBsSI3jqyxIT6iUrzukUOyWEeWSJXJUx\/rFiNZjB06AIMaI1nWIV5rtLIdVgYGzgvPC5xKXYTG82w5tZzFQBs8j1o3QuniGMmybFlZGx5H3yRrFWKCqHSmxRo3R7K1JIuUNdtYPa8pkuW9r6VPTrIjEGbNIzXJsjCwh8lEUEKyUJhQYZSVfoxFsvAgOcrVhoEjDP1aiDamnAZEGobX4fTuSDqKySFZ7X1bkSx75JhHXpviSDZU8GUZ11i6eAiS5b5bJKsNCYyd\/MvV59q50KSiSZbX3bqSLEhdr51ZZMryc7mTLGcVIM9WZKh\/Y9mWvzXJcnagJpK1akC6kCzbWStdbO1Ips+g5tfYOEoNRbWcSZLnC9k7yRYyXiplXLN3cU4XupCsjnz43TUuTNDeOUe6UChOsTDJ6mIGKMdYJMuGFGu0rPhWJFtKsuxNL4\/3TAuh9rJ1OpyxCq2fYS6sSLSPSLYPktVOhBXJ6nUykGxoPZ8dMSZsjmQ5G9NGHBffqwxFbhZeaI\/XpfE8lnEQkuWwhgiK50w7HXAQUWSFeQhFpqHvY0speIY4QNaa7JAki\/kdmmQ5C2HVN+CUMMGCt3+MRerSZ12\/gXZwyhDmTaeIEb1q2ZHrnWRzGO7iNagulv\/lA9YLmqi+1CJZHJyuDQAbIV29yYbBqlDVv+vUVijdbJEse3tMutprDkW4vD7IxkLaYhLlaIoNLRtmdix0X3jnIH3+ZuuNfufVdt7Yc7fIlIk75lxYJGvhB2Op9wguIVktC9xvjCm11oRInVNwVkFbjGQRpXQhWb2kgef1RbJWSpI3MonpEv+m5d5SepkXK7UdctD09ymSDT2T101ZnrtGsqHUO\/cjtSbLTjvLOBe9WVkIK02O7EwOyWr7ydkXXRCXIln9LrI830m2gPYQxe7du3dFZXFBE9WXapKNrb32TbLaG0ca1Yp0eYApktXrTtqQdCVZ7kvIqPGareV9W05JiExTJKsjCCbReZIsF3fotDDPmTYYXFSC13GAD8jbkhVci\/dvsZUlnER2jCzy0Nfp94H53dhQ5KaXG\/hdYP0aE+tPLsnCmGOZQ0eqKcUPkWZtJBt7HkdhLN8pJ0Ku5ciejydk3eXoj\/vBdkN\/z6936ULKLiSLMcFZtiJZPSbOLjHJcv9DxU5WCtpJNiX99Dsi2a1btzbbtm0ruLP7pbq6mA9J1kLSlWQ1iWrDyYRlFXFgtJpkQ4Yk9H0uyVrp4BzE5blIqelIV3vgOtWMKCoU0VmRbBeSlXutNczUOLXnvzyXy+8jYky5BgP3gvR0ZC\/tilFiQmP54FeicK2uCtaywPUGOksBo9+VZIFhl0i2b5LlYjOrIKomksU4S0mWnTIrHczzimhU2wz5\/y4ki\/a41iIkO5At1hcU8bEMad2wnGy2YRbJ8rN4YwrWy6WzTzfP3v\/Hvq1iyljhd9nIX\/Yufuihh5pNmzbl3tb5uhqShXLE1mStjolgyUeMpbVmG\/JCdVvzJFlRAEkb6rRyLvAxktXGAddK26GIjr8H0VgkK\/hqYuI+s1GBNx4qkrLGapEsV3GyYUBUy1GYfsfXMqIp7DiS5YhHVy+3Mvfcqyt26OExWVXIXHQWWpON4YViFo7i2Em1dgvT215qkuYCmVz50+PWJKvXIEPX5zxPO696fZqdJP7NctDkuz5IVpOoLg6sIVks84RkROuGtQyTQ7IcuetIVvr9P08\/3Dz1N\/c7yeYIp1yjT8vR9\/VR+CR7IF999dUromUh2dtvv705e+s3ZvuaxjwxKIf0TxdA8Fpl7rhD19VEsiGDYZXlc7TJfYhFuTVjShEFlJyNiiZZndoLGcpQajjk8OgN77uQLLx1q5AIz7cKxPBbF5Llam2rermEZDmlJ\/fpM0eX52Z5B6wUyepDFXiOmWStGgaMSe7hHalq0sX8OpIuNByaZLnqW2ciNAZaV7jYR\/odcsxjkSz0y3I6uXAwVlFu1RbAMbBkpIRkUajH\/eRlD20fWJedZAsssnXua8HtWZcKwR4\/frw5ePBgkGR1AYgVscaItE+SjRV5aGULpXpiKaAQaDkGNAvwixeVkCwbVitihbOgDaWVTk31Ua8PWZsuxNoIOWIc+VvrSLzepslXz2vIeC4bth\/Pov0akg2lyTXJYitFa022lmTl2bwFY4pk4XRZBWCpecbvoexPiGRLHC48g3WHyQev4Oj1a3Y6rCJBTl2HiikhC0xG6E9MftjxwvW81BAqImTij0WyXAhmRbJaN\/gaXtfXJAtH9S\/\/93PN33\/lC83jjz\/ukWyOEgxZ+IS233jjjWZpaanZsWOHSbK7vvxIs3nL5llalItJtACEqlBD0WEOBvqaGMmGrk155TmGwyJZifZl28u77767eCh9kaw8GMUymmRLIlhtdENrmKmBWt76zVf+fBVOnBa21p+tjRhyjKRVJCf3pSJZRERCmiBqLRcpOc5xxNiI6jXfP7ri582z9+9o06EhXcL49PF5oSK61HylSLYPmdI1G5zdCOke6wfwgBzJ8ZuogK8h2VR6nQlzmcyW1\/550wcmY8gO76vOzkTsTQXtYHQlWWQgnWRTkn\/x9yELn2Tv43PnzjV79uxp9u\/f3+jiqtBkoQhJG4I+o9UYPCkF4XuHjmS7CDRXMKbEgSNZ9mDxfZ8ky0RhFW+k+oooREhCPhKZvfP8T9ulB1b8FMnG1rtTkSww0mlFy0GLfZfjfDEepSTLThDL0paTvwiSrJUajTlsqfnKJVldvJRqV\/+OPmKbSWw1WkqykCPBKOVYhOQkVffBdkNv7q8dV7mWj4WE7MVkIec34KcDGevVJ57\/LjapdE77vv5tv\/nNb37Td6M57Q1d+BQickyWkDAfTIA+33Dsx80DbXrryvYrGAbxNof8iFC3XuTFPWFThPztN9\/dSIQg\/z6\/413t5adevtDs+9E7msMf+tXsX4wj1B7fg2slij169GjrqFgYpfomCpN6rrSx78W3r2jq8Id\/vYzBc682by1d26w7\/9P2X\/6ex1syHzzO9tkXccrpJ54jsiDYxnCCkUQBEOQGY8JYrL4DD+savl\/+FhwEZ7kW\/w85AIYaK+u6XAy1Xlj3Yezcf5alL5z7vZZkc3VJ5kxwzNGJ1Dgwdq0zqftSv0sf5ZMrR5Y9YYzeWrou2V5MTlK6KTLBMlwqE5a9wDNjv8GWMplCDgQTfFg+GFtg5JFsSiIv\/p7a7Uku61r4FIuWxWsUsl3Uz\/nrbml+uXRds3T2mfbIvuue\/r\/boch3+H+5Rn7P+ch97zx\/NufSXq+RPspHyFQ+ug\/SLyHZ3HGkOsfjlGfLc0vGLcVyUnwRuwfPqGlf7rVwkO8w5791+uF2nuGEhPrC1wMX67sUZn3+LvjJaxh9zWdJ3xgzwUFwHOPz71vvacdfIne6n7V6EZIJSVPDhuRgEpJt2J+Ujmg9EkzkIw5GTD42b97cLmHJv4v0GS2SHRqkGMkuMsEOjZu37wg4Ao7AFBFYNHIFhpMkWSFISVfu2rWrWb9+vTnfOHcWP+oq4iHXfacogN4nR8ARcAQcgekhMCmSZeIcMl08vWnwHjkCjoAj4AisRQRGJ1lZ0N65c+cM240bNzaPPfZY512gPJJdi+LqY3IEHAFHYLEQGIVkQ4VPOuXbBUon2S7o+b2OgCPgCDgCfSAwV5LFLkzo+B133NHce++9DUh3+\/btgx4YYJH7iRMnil9T6QP4odvQ21amMgQpbFK\/y3j0\/E4d2zEw4nkXvOQjOjDVzxgY6Wf26XwPhfMYOOks4NT1TWPfN2bcPvZLmIJuzY1kY7s8zYtkxai9\/vrrszNsZSKefPLJ5tFHHw0WWA2llEO2a21bqceun5\/CpvR3KJAcAFH6ru2Q2KDtMTDiccFAwtGcx5hLnzEGRoLLkSNHZjo5dTkSTMfASXA5fPhw+58Uh4oN3bdvX\/vfPA9cKZWpIfUPbaO2Zyq6NTeSFQC05zXPSNYSQiiHbL04RSKoFWCtgNJOTAlT2Fx77bWrFJixk993797d6POBpxypzRsjli\/gfeHChVbupuBtW7I2b4yk2FF2aduwYcMKTKYsR4LbvHESmbEitanjxDLWN2aCCWzSmTNn2i11p6JbcyVZnSqTDfz5M2RayJpUpDj1ST215DaV+8SZOXny5CxiZ2\/bcihS2Fx\/\/fUrvGaM0zrlSM+x\/P8USWQsjLhWQLb\/nCo+cIrnKUc33nij6axNRa9C\/RhDlhadZPvGTM4l5zblFdCp6NZoJAuB1Wt9qbXDWoXTaSgmiqlMRu3Y9H2igKdPnzZJVu\/lDGPKKTqNzcc\/\/vEVKbwc7Oa1BFCL2VgY8XOnZAgsHOeN0ac\/\/en2jGlxykQeX3rppbZbQzrftfLD980bJ8HHShdb2aQ+xjdEG31jph35KUX1o5MsTyCnk7u+J6sFw0n2LfPAhKFI1lKiIZS1ts2+lTzHEdH7dU\/JEEyFZO+8887mqquumq3JLsJbAmPIkswXFw4NFZzU6lfqvr4xc5JNIW78LpVyt956a28FSU6y8yPZENYVYjDYLX0reYpkrVOhnGSXpxc4IJKVYjku3pm6PM1bloRQ9DPncUZ3n8rYN2ZOsn3OTmVboVN\/UuuKlY8b9TbLKMWKvFLYyJqspPG08bOwk2fLtX1sKDIkiPPGSDCUKO21115bNay+szZ94TYGRpacTZ1k541TaO166jjprKVeoupio2RNlj9TcmAnlS7uyzhY7aQqaL26eGX5v64e1q8HWAohgv3ss89OnmCRauNXIOS7ISuwLfmakiGwdKbvCtBUlTqqi3VxnlUkM6StKG173jiFqvkXiWT7xkzr15R067IhWaSl\/D3ZQ826detW2ZHS92D1O8aLRLAy+DHebdSgT8kQWMQyBkaaUBdhTXYMnBa9ungIzDySLXUPB7g+Z9eiAR47SpOx3VQsw5XCJva7fhYP+JZbbllR5TwKGIGHzhMjqwtTJ1lE\/Jzm5gKbvuUIGOn36adeXTwWTqmTyKaka6FMSUi2EBTJv1hvTdkoJ9mpz7j3zxFwBBwBR8AR6BmByypd3DN23pwj4Ag4Ao6AIxBFwEnWBcQRcAQcAUfAERgIASfZgYD1Zh0BR8ARcAQcASdZlwFHwBFwBBwBR2AgBJxkBwLWm137COgqWGvEUl0tOxk98sgjox6pOO\/zNaf+buval04f4VQQcJKdykx4PxYegameTxw6ZWlowNfibmpDY+btrz0EnGTX3pz6iEZCYIokO+ZmDmOR+0jT7491BEwEnGRdMByBnhAIkSxvdyePkiPJPve5zzXf\/OY3Z8e53XHHHc2uXbva33DEmxySoQ97599zNvqw9qUG8T7zzDOzkdc8izdDsPZfHpPge5pSb8YR6IyAk2xnCL0BR2AZgRKSfeONN2Z7PIOseDcl+e7YsWOza7DbzebNm2c74Eg69oUXXoiu9eac1qIPdch5lr5Hb8sJmZj6kYcuu47A0Ag4yQ6NsLd\/2SBQQrJMltje8a677mpwmgi+k5OPJJq12gYZbt++fXYfg23tD5u6J+Qs8H1yBOX+\/fubDRs2zAg\/NMmLtGn9ZSOoPtC5IuAkO1e4\/WFrGYESkmViBIHt3bt3lh5mksXpNBapxfY\/DhGq3HP8+PHGSjfHziXFs5DWDpE7z3HoGMW1LAc+NkeAEXCSdXlwBHpCYGiS5TVU7rKs5+pDq+X3WNSqN5dHG9Z6rX4WSJadghCEOiLvCWpvxhFYGAScZBdmqryjU0dgaJLNSc8yRjmpYaSHDxw40MhpNzmp4Nx2pW2PZKcutd6\/oRFwkh0aYW\/\/skFgKJKVNVmrsChVvRtL\/epJ4bRz6lk5RIz2fU32shF\/H2gAASdZFw1HoCcEhiRZq+I3571cXd1rRaE6pZvzLF1dHOqLVxf3JFzezMIi4CS7sFPnHZ8aAkOSrIxVH1ptvZuqMbHStdbh1\/pg9Jxn8bouv36EPqQi7anNn\/fHERgCASfZIVD1Nh2BiSAwJtH5jk8TEQLvxqgIOMmOCr8\/3BEYHoGxyM73Lh5+bv0J00fASXb6c+Q9dAQ6I+Cn8HSG0BtwBKoQcJKtgs1vcgQcAUfAEXAE0gg4yaYx8iscAUfAEXAEHIEqBJxkq2DzmxwBR8ARcAQcgTQCTrJpjPwKR8ARcAQcAUegCgEn2SrY\/CZHwBFwBBwBRyCNgJNsGiO\/whEYHQF53\/Wee+5pz461Pl\/96leb+++\/v\/0J1\/7whz9sTp482bzvfe9bccsPfvCD5mMf+1jzxBNPNJ\/97GdHH5t3wBFYywg4ya7l2fWxrRkEQJzvec97ZmSKwf3kJz9pduzY0Z7EA9I8f\/58+\/fVV1\/dPPzww826devay\/H9Rz\/60VXtrBmwfCCOwIQQcJKd0GR4VxyBEAIxkpV7vvWtbzWPP\/54++\/S0lLbjBWxPvjgg+33fJ2j7gg4AsMh4CQ7HLbesiPQGwIpkhXi\/PznP78qPSyk+tRTT7Xfv\/jii+1pPlYKubeOekOOgCOwAgEnWRcIR2ABEEiRrBXJcnr4ueeea0fp67ALMNnexTWFgJPsmppOH8xaRaB0TZZxQNp49+7dK9Zn1ypWPi5HYEoIOMlOaTa8L45AAIFUdXEsQpWU8Ze+9KXmAx\/4gKeKXcIcgTkjMCrJ8pmVt9xyS3Po0KHm6NGjbUXktm3b5gyFP84RmC4CViQbqiC2otivfe1rjaSMdbXxdEfsPXME1gYCo5Hs888\/39x3333NY4891pw5c6Y5ffp0S7JiTCSttX37difatSFjPooeEAili\/H6zmc+85lVr+RoEv7Zz3626lWfHrrmTTgCjkAEgVFIVh8kLcdwgWTlfT75\/yeffLJ98X79+vU+gY7AZY9AbE1Wip5uu+225vvf\/34j77\/iY72uI9d6hfFlL04OwBwRGIVkkSbeu3dvs2XLlpZUmWQlyj1y5IiT7BwFwR81bQRiJIvfzp07N3v\/NUS81rXTHrn3zhFYbARGIdlUJCue9uuvv96mj7FTzWLD7L13BLohkHqFB2njj3zkI21K+JOf\/GTDWy3y0\/la3g2qWw\/9bkfAEbAQGIVkpSOhNdmnn366OXDgQHPixIk2yu3yEbL2IqouCPq9joAj4Ag4Al0QGI1kpdNcXYxBbNy4sS2G2rRpU5dxtetOx48fbw4ePOgFVJ2Q9JsdAUfAEXAEahEYlWRrOx27D8T9xhtvtHu4SurMXwcaAmlv0xFwBBwBRyCFwJojWSmikgKQPXv2NPv372+2bt3qJJuSAv\/dEXAEHAFHYBAE5kayVmo4NqL3v\/\/9naqLdXHVIOh5o46AI+AIOAKOQASBuZGs7oMUPu3cuXPVmikXRHVZl02R7AsvvOCC4Qg4Ao6AI7BACGzevHmBervc1VFIFgS4YcOG9qBp\/dHvzdagGiJZIdevf\/3rDUj2\/HW3NEtnn1nxiF8uXdf+\/zvPn6159OweaXvd+Z92bsfqhPTxraVrV\/W9psPSlh6r9V1N24uMgSUbaxkDPbY+ZcySg770rK925qFn2q701fe+2pkHBpatybG3QrByZvKifUYhWb0ZhRXldt2MIkayt99+e7tmK68IbTn5i+bwh37V3LDpylk35Ls\/uuLnzQM3XZM9nw8892pzw3uvWNHOvhff3t5\/+MO\/nrUj18mnpO1nL7y7+cp3Xm2e3\/GuWTvWd9mdpQtPvXyh2fejd6zAQLIJXzj3e6twSbUvbcmHsbQwsL7LafuB77zanLrrg0kMpB\/ch5y2NQZyjyUb3JbgJHttQ5ZCz5HxirPFc16LgcbXkgMZ\/6lX3iySMUsOrO9SWFq6LBidvfUbq3TKwiXVvqVngsE\/v3JhhZ5J37W8pNrGeFnP+sAAz7Xsiny3dPbp1llOyRHaydUzC5ccDDRuFi7STqmeQae0bc2xt6JroqNf\/JObiuQ6Nd55\/D4KyaZSuX1sRpEiWfGIxDN62198t\/nyp65ZMXHWd6nJ+MSxHzd\/8N4rV7Qj38nne0QM1neptsWwCMn+5q8\/uYKs9XenXrnQfOLYmRXXcdvy+6mXVxpf3PO9u65vbnjvsqMhUb4IvsZFrm2N\/MXrdL8XAQMLawsD6zu5V+YCZCk4icMGWZLfLdnJxSVHDoRMWJ5yZSPVdgkGMp4v33RNlhwAIyFZLU8WLql+WvrTNwasZyW4cN9ZTvB9yNYI6bzyxJdmcpSjZzl2xcIlha9lQ2oxkPu0rci1t60ztemKFTbphmM\/dpJNTSD\/jjVZvemEpIqPHTvW+V3Zvkk2Rby5hJoiWUvBco1IiBhmHvBFEmZCDd2Tqww8pyGCyTEIrWdsKCXaz8UgZVhyDUYuLrL0IBkRrBXl4hYii5bIAxmUvghGxvaV515dQda5uMCRYBnSeq8xEIwkM5JDsin9COmZRO4pJ7TG4SzBJaULMdmQaBZylHI+cm1NSBdYz7TOWfd0wUDLSa5+6OvEWXOSLWHYi9daFcc48q7rdoo5JPvLpWvbyC8nkuVJt4gwV\/BDhhIRShcyWWSSZVwEg9qILWVYcg0GrsuRjRzjKksJqYyGxkATrvyeQyYhQ4ksRpdoJYRLDgYWljFcSvSsFJcuZJLSs1BGI5dghiRZHneurcnVGe3EiG11kh2p8KmCj3u7hVN8tSQbUpZSQ8pRWi3JpqJSRC2S3pOPFnzLaOYQTK7BkOtKcekSrcBwWLhIpJNrMHIw0CmtmHEdAwMr7VmDgYWlJkzGoi\/ZKHFc+yDZmOyk9GxZty6l0UswsCJ8ZH9ylxosByzllMVIlsdbg0uurclxSOT5Yre\/\/ea7VwVEvRHDgA2NsiY74HiSTU+BZGMGIaUYOn0aUga5TlKCsk4sa7dyXa7g5xAMjIi0ibUTi1DHJllEQ3AuYhjkkAkbT\/wt+MJQ5jpgKVxyHY1SeRGSLTWaTAJaNmQcghtnhEoIJuZ8TJ1kgQvWXzFu0YeSDJnGgGUDGIgeQ8\/6ko0+STakZzHZgS1L6YyMVwoHnWST9HbpgpyNKbpuRhHqTopkSwhGok8UO5UQDJMsIk18V+qRxkgWij4kyWqCiRkMzEnKSJRiYK3FaacCJCspU+CBoozcaN4iGDEiQ5IsMhDS11JcgIF2tiwMYsSbItmhMYDcSLo9F4OU85FLMLHMh4wbhMqOxhAkqzFApMs6pZ33vjBIOWVwXCEHrHt9kKzYVikOc5ItINnYpUOfJduFZEEg8Ly6kizIRvDQJCuCijW0mLIsKslivKLA8hEsNQaioEijx4xrCAMmkymTLIpPeIwsG5w6LDGkkCE4AXAu+iJZJhM8I0YwpQ5YFwwYS5AEkwAwkO\/kI8VmFpmknI9cDGLOewwXOO9MsqEsSEw24Mwjs8UZDeASynJMAYN15882by1d5+ninni2PUFHPtZGFV2fUUqyWBMRI8IkK3+LUOvvuLhFlEE+HGmFDKlFMHhFB\/ek1klYGdiIxIxrThQHEshJleYYUsYl5miw8nchWcyVJh1tXFNeN2MVM665GOC6EAaQ9S5ZjpgcWKSTSpnnYpBT5MRkgTVdcYa0TnWNZFMkCz2LkWxINkpJtlQ2+iJZjYHYJOjU0CSLNHfM1qQcDSfZrsyn7h8ymhWS\/cyfPdg89Tf3N1bhEwRBR62LTrLw2DEOgZyLoVIpQXYq5D5ORaccDUuB4HyMSbLa+YhhwCTAaUJOlcLg5hrSHAyAOxtFy2ljcmSnrJRkGQM2zHitqIRkBY9L6cxL5Inv9NojoizgwgRT6mh0wcByZlMki3nSsoFsV45sxBx6HcnGnHcrhQw82OGUZQTBtYZkWTZSjkYsy6HtbQgDJ9meSVbelX3yySc7HRAQ6hJvtGCltuZNsoiGSyNZNiJtuvnlN9shQ6D5b\/0dK5pcp4s0rLXHFMmCeMcmGCuaD0WyuSSrjYi0ZzkajIEQBiLl0Hp9jGQR7U+FZAUDXttlMgkRjDauqWgF7cRINpUd0gQjhKnJhL+zZCNGslzg0xWDElvT6vjFd4EtDFK4xEjWIt6UozEUyXIxITvvINnY+9k901BvzU2uuhhFUfJy\/1DpYuxmlCLZlCG10sVcBZhSBi34VgqHDUJutKKjVnirIF42ikzGOlphMrFINvSdtMkFYdrrZlysaCVGMCEM2nGoampEcaHxWg5JLGIrcTSAeSyC0WQi9\/B38ndXkgXB1DgauYY0JhuaiPRyis6CaAzkd4tgUFyTKgjrk2RTSwR6zlNZjhjJ5jgawEYwjZFsyKmwZAPfhUiW5VnbFR6vZWssXch1NJxkCzk\/VV3c14YUVrd455DUpHNkAkLV31nKIN9pwed0Y8iQpgzCIpGshQHmo5RktaOhC8J4rdXyyi2STTkaFsGUOBqWvITW6+FopEgWuFlr1SUYoMisBgM8u9YB6wsDy\/kA8cLhEjxzCQZFdiAOOFsxObCcWSvLEXI0LIIJRfO5shFaSogRaiiSXXbyrpwVhIUi95jzbmU5OKPBGMSCGifZQpId83ImWRYOKL816THjukwmy+8J6sjDKvCJKYvVjiZeeYYYAE4XawVig1FDMF0wsIwEnAqsv3bBIGRcczEoJZhaMtEkCww43VgSybJshEg2x5CmZCO0JqsNqeCCNdQS5+GN0RAAACAASURBVEPrGetPSHZ4zq0sB77LJVTrOo2L9EuMvq5CRk1DKDvEuIQckhxbU4sLsmGwsV0cDWs5JZdQLXmxlpRitoYxcJIdkzULnw2S3bJ58wojIQIgRko+sagVxo4NQg7J6mglRKggawwrZRBCRjNGJmwUYSjhpeK5qPLMMaQhg8BjFOWPRWw8Xu2wpJyPFEbsaKRwwVoqGwkLAy7SgXNlZTm47\/gbxrfEkNZgUIJLzHiGIrYa2YCeMX4pncqVjRTJop0anbHuCTkaKdlAwaH0B7IewyDlfIT0DLYsJTuWfnDqFzqTctAsuxL6DnrG2cSY\/jjJFhKdpIv37dvX\/mcdzD50dfHND\/5j87\/+6LYVJJtjSHPIhFMvOQrE0S8rQ65BYMG37om1YxkJ7k8qWuFpTxnK1O+5htQyCJYRSRlSPC9lKFO\/94VBymlLyUYKg1LZkOstQ1uCGzIWPPfsxFq\/6+8sfFlGLYelq06FnFDtoNXKRhcM4KyGxsh4xHQqVz90Gt1yUvFdjWxY0b41506yPZPs0NXFQrLnr7t1ZkTYwFkKVmJILcG2CIaVwSJmNoqStmGlChmZmCGVZ2jlDpFsTFnY+IYMYKzvjHWJQdAYsPKnCMgi5tQYWQ5iuJXIBo\/dkpMYbqExAhdLnix8LafMkg2eZ\/49hVtqTvn+vkm2RKdiZGHJaC5urFMp2SjRhZhs8LhT+LMuQHZidkPbGsvRgH7kyoZlb3PtiuhB6KjNQhqa2+VzrS6WTSaOHz+eNbiDBw8227Zty7qWL3r55ZebO++8s3nttdfarzdu3Lji2DxJF3chWfa8YCQswQ4JkqUspQa3L5INCTYriyaYXGXgPvLfXQ0LzzUTIeYiZYwsr9saYw3JpmSjhmRj8hLCstSQhkjWwsAypDwnIWLXBt3qe1fZsHQu1Z\/YGFPjSjmu1v1dCEbrnjXPXWVM60fI1nSRjS4YOMlmUmIqXZzZzKrLcLzdhg0bZq\/\/6APgsRnFf2y9x4xkLSOeUra+SDbXIIRIiw1yjExS47FINnWPFWlZypQypCkMLNmwDEtoTkCoqXlOGd9YP1Pkl8IyF4MaRy43IkvJmBXdhhwwazxjY2CRviUbrAvakSvBwHIIU7qQyl7EnKkafEPjiTmz1j0pXEJym6PHTrK17NjTfRLFHj58uP1v\/fr1baua0JlkrUlNGd9cA5gaUm47IQMVE\/wSgokZwJSypLCqJQErjaW\/q8E3ZURyCZwJKGXMpiZjof6kDGkfBBOas1xdsPqekrEcw435jJFsSDase1JOt0Xcuf3MbTulmyU2osbRSM1Vrq7pfjrJRiwf3o3dvn17c+ONNza7d+9uXnrppeAdNafwSMHUyZMnm0OHDjU49B3R7Y4dO5otW7a05xLefvvtzdlbvzF70T\/UiS6CX0sCljHT3+VEGblGUfdTjzkXg5RSpX6PKXIqErKwLjEiMSJMkVLKmPVFsql2co1WyXhic1LiaMxLxnIxKJGNmB6XOG2p+euiHylb00WHc+TbcjRyx5Nq30k2Nbtz\/l0Kpk6fPm2S7NatW9s13nmQbEqhQgaqRjBzDUvomSkDmNt+zphznlWLgb6vxOOfJ8n26cSk1KsEy77mL8chynlWSd9TRFjquMZ0RUfOuQTT11zlYJejZyX6UYpviX0rxcUj2RRiA\/9eSrI5wtjXNTnGdUjDogW\/JKKp7XtqunPH29fzcw1UznW5fR9Tfkr6mDPmocYyBUcud2w5ONVckztX89SFmnEAx5x7U\/bBIuvnd7yrkS13F+kz1+rioYEZgmRrhCVXYXIVO4VbjuINTbI5Y8nBsvaanPty+hiLYvj+ec9xTt9z5CAHp1wHrLatIe+rwSmX5LvIRk6\/cq7pa45TNiU2Vh3Nx5azYs+p0SEn2Qiiqf2K9a21a7JHjhxZcXpPbE22RqhrhbPWsOjnLWqaR8ZRk0rMVcQcfHPmO2RcaoxbX33vq50SkpBrc9OsfRjSvuZv6HZy56LGIRtTxmodjVyHzGq\/xpY5yeYw0IDX5FYXS+HTv2+9p3lr6boVvamZ9Fylq2m7L8GsVSDc15fhyiHZHCIswTLW99DcWQSjn5mDSW4\/U211kbGUupVgUIplrsOSS\/45Dmcu5jUkWDKeVPs1\/czV41x5SfWxRDZieivt5OhUjhw4yaY0euDfc9+TXRSStYR8jMrIGnLs8lpGSvlLDFQpMZQYhJQ4d8EglZLLlY2U4SoxpDHjHcI5d65SMpY73tzn8dzl6lRINmqcpFzZqJGxFMlaspW6hx1uizBj\/SzRqRSWTrIpiVC\/Wynkrsfc5ez41JVkWRBylSX3upTy5xoENq4hpdLKItfpzc3lGjwzZPi7GMCUcc3FLdQO+tbViKT62RcGVn+ttlMbFeQaxRJ5ihFYSDa6zh+PI7arF64bk2RLHI1cXGrsQe7GMGwjcvtTQpgpGcydK8bVSbaAZOWd1vvuu2\/FlodyuxQvHTt2bNX3BU1HL8UrPLnp4hTBWIKSMoAQGr53CEMae46lLLUkm6uguViFnBhrPGxcu2yZaBmElDPFhJibDsuVjdSzS9qJRR65JBuSjZRTkDvnli6EiNsi1JhspJzDEgyApTU\/IUcu5rimSNT6vS\/ZyCHZqe0c5ySbyYS6GEnfprdCzGw26zKQ7Htv+2rz7TffPduQImRErM39UwpmGcDQcVP6WDceRK4hFSMRU4aQAesSyXKbMUNaYxD4nhRuFsla5MdOBRvk0hNEeH74ObkZgNw5rcEtNc8WCbDsxJyGHAcsRkCWXKecqRIMcG3uYREhgimVjVCmh522LrKRsjVdnJiuJGvJf8rRKJlTy5lyks2iudVbHerbhj7qTtLFXUjWIpgQMWBsKbKwDGTq+LJcgxAypNa5l7HvQgqSS7K5GAxBsha+Ieckduyh1TcmoNDvpUe4lchTTMasudfGNXaEWwlZzOO4yJAT2pVk+zrCrS8MLIJJHRcZw8DSvRKSDTmuOHubx62dLUs\/SoIa1l0n2UySTUWy1vuumU0nLyuJZEUQ9EHC\/J08LCX4uSQr7cih8SKsUPgQKcl1EG751yIL+U4+sucv\/83EjN9jHmlIQRgX\/J2KVlJnYWK8PO4UMcciWcZlaAy0cS2RjVxc2CjWyIbcc+rlN2cyAdlIHWRvGVLru9gh7inZsMbDuODvEgeMSQc6Y+mZ5YjUOmA5DpplV1heeJ4ZA8vWpGSH7Q8wgC7Igekal5BdybUblp6lSLbEMXWSTVLcpQskWt25c2dz4sSJdk9hfIRgDxw4sOr7gqajl+aQLJMOk4koUKsAm65oYgeyWwYhpQy8Xdgnjv24JVEYBBZ86ztNJmI8tUHVhFqiDDFHgw0Ge5y5BkEbUkze2\/7iu+2fuWSSwgrjFUKBp12CQYxAUs5YyohoDE69cqE9MxMYpGTHIpMQwQBfJttS2cA8a+OZkpPY+bE8RszLAzddswqDPkgW55FCz0Ikm4OLliFgwBjV4HKJHD8YxaBGNnJtTQ0ucg\/sY8p5z8FFBzVOsplMOI+NKUJdiZFsatK\/d9cHLxroCy3JigDId2IMWflzSTZEMKUkywQD5dYkK4IvhhVRSwnB6KhVjJQQgcaASXZMDJhgmEzEaMvngede7RThSxtCumxMtByEohXGBWSfcjTkd8hELqGGSBYEU4qBzowgCi7FIIWLYCLj1c5WrjylnK0QwUBOYhFbKhsiv4uMCbZM0Cm7Aida69lyRLva2bJszfK1V8yccwQBbA84atWORijLUepoAINl2b5kI2oxkHmBs7Xu\/Nnm1F0f9G0VM3l2tMtAsru+\/Eiz70fvWFH4lPJCQbLSeRBrimQtQwpl6JNkdRpQK02MYKCIIeJlBYk5Guy9x3Bhg9AVA8uIWCTLxtUimBgG7J2HjIiMV4iQI15tWEAgkJ1QtGIRjLQ9i0ILshxMOikMcgmG043SJ3a2QsY1FdlBXmIkqzFIOR8pgoEcsK5YGMRkI6RnXQlG2xrgAgyQFbAiWZ0VEIeIsUo5GpajHpONeWHgJDsabZY9eAiSZeNqGVJRCK0Mch0MM37HSDiStchTr8mykIcMqUWyiELYiMS+Q+SuDamMg1OcORF+imCAgeAGY2hFZ5YhRZpL+sWGNBbF9UGyTI4WBvq7FAacLo4RTMzRCBnXLgQTczQs2ZBxxBwwjUuMYLhmwSJZ\/g7PnBcGqSgu19FgPYNDD7siegbHJmZrOBXeJ8lay1DIkiG7AVsDRyOUQi4Napxky7iuPUw9dqZszd7FOV0IkayOVkLpMMuQxoyrFckiEhBhZDLJJVmd2poCyfZNMJpkeS01lR63HI2Q85FyKlBYxtdpI2IZxT5IljEQg8XpfotMNC7SZ3EsYo6G3MPRKMYr30EHtC5YZBIj3hjJakcjRjBaP4YgWSuKC31nkUlINjQGVvqUsz8hRyOEAb6Hw943yVoYpBx6J9nlWRnlFB5UF0sH+ID1HJLseo0m2ZooDt5ljnFNRSsxkkUUJwSDNKS1dpIiWVYGS\/BzMWBD2gUDrMtJG0ICKQwsw8IpX13o1RfJhnApJdlQFNdVNmKGlNc1c0k2RTAg40UhWcaAs0M6owF54bXUlJ7lYoDrukbz2tawzsSyHHCiQnqmZQPOnFU8maNnMVy4diUU1GD91bKtHskWsB+i2L17966oLC5oovpSkOzjjz\/ebDn5i1kaS096SwAXIwFef82NZFnw5R6khvG3RRyx76z0Xq532YVkOdLpk2TZU88lWWDAKWROCabIxDKubcR28dUpHbHJ2GW9MYSBLv4KRfMhkg2t8efKhnY0GJcSkgUuGI8eL4xrLslqByxGMBLRwdkK6Zn8bhEmIln5HU4oR7c1GKT0DBiEdCoU3YYwYEcuRjC1JBvCDTKW0plYrUcpBqLnnCLGeHMxcJItoD1Eslu3bm22bdtWcGf3S5lk97349hWTbqUEcyI2y7gORbJdCYYjWRFyXuPRjkYsHZaDy1QxsAyLRTCIZkQuhiCYtU6ycMoE71AVcg4GMcdU2uZMT58ky3OuHY1SguGaBR3dQs9ydMpywGKR7JRIlu2kxgABTQwDJ9lC\/pON\/GXv4oceeqjZtGlT4d31lzPJbt68uW0IXiZ7VlrwS6O4eRAMCBevDcAQhNJhMlYmEx3BhEhWY2Q5FdZ3MK4SKcJQApchI9lUBBPDgKO4lCHNwQXGlcdtRWw1uMwjkk1hoJ02y5CGjGtsuYXJpJRkOeOhI7YS2QDJWs5WH7iw3cnVKQsXpLlRtxAqqIzhElpiYUdD2kfmARkeKwti2ZWQPS3BwEm2kPf0aTn69j4Kn2QP5KuvvnpFtDwkybKhRIQo6Rb5oMgpZjByDEJJSjBFJiEFEcFnpYuRSY7zwbiEyKQUl74MaSxVmjKkJbigAj0mB6UYYP0M1Z3sbKXIpCSa72pIQyRbKjsx4gXRpDBI4RLKaCwKBkKENSQbwgUky7+X2hVrnlPRvHWPk2wByVrnvhbcnnWpEOzx48ebgwcPZpEsDGrIs6o1CNzZ3GglRSAwruwdpoxmzEiALBiDPkkWDgeKLywyKSUYSwhyPfWUkeAIBriVEKolQ4jqSzHQuFhjZCxSsqMdOcuRQDQvGQiJXHJkJ2ZIc0m2xAHTuAyBATZdYRlIZYJyMmApgrHWX3N0hvUMaXTOGJXKBhMr45vSnz4iWSfZLJoLXzRk4RPafuONN5qlpaVmx44dSZLlno5BsjmGtMa45iqDPB9pTfn75gdPNu88f7b5f\/7m\/hk0ubhwxGZJQK4hzTUIucY1RCZsEJhUGA+LJOQ7yYqgeC5lXHOcrRxDahnPXAxqopGUcU05Hzdf+fNGtlC9++67V4gDOx34IVc2rOtqMLBkzNKZFAahe1LOFjB45\/mfzjAqWUpIORqIarVjZUWtlmz0RbLaYee5z9EzyJhHsgXEO2Thk+x9fO7cuWbPnj3N\/v37G11cZaWLU10vIZiU4MvvOgLNUQb2Uq3+QtG7GlKQh5xUJBXY1rp1iHTk+xTJWlG4ZTRDnrjGKmVcSxyNHDlgg1FLspZhySWYIUk2Nf4SMoHOCMlqWQo9p08McnQq5LjK96yjfZKsNXZdjCmV0vz8XFx0P7ElYQ3JisPJUXAXDDTJpuTMsreCwfMvvODbKqbA49+HLnwKETkEWkiYDyaI9R3RihgMfKQyWZSBvzv18oXmhk3LhwjkfkSh31q6tjn84V\/PbpHv5Kxb2Qy75KONg\/RHto48\/KFfzfplfaefIZHH0aNHW0cFGFkYWN9J+\/IpwUGwlI\/GQOOSwsJ6NsbLWOZgYD1LxstYCk5fOPd77WtgLAc3HPtx80C7UXq+LFgYWN+lMLB+t+TJwiWnbWtsMdlYOvv0KlkKPScXgxoZszDoomeS0uX51bIhY7RwscbO+vb\/\/l8fXWVXLFwEA9l4IuQM5MpBXxiEdMrCJSVnIXv7b99+onn2\/j\/2vYtTAMrvqd2e5JquhU+xaFk8ayHb3M8vl65r06f8OX\/dLc268z9d9X1um7hO2pHP0tlnZrfK84Rg+LvSduV6aUfal3Z0\/0vbszD496339Na2Hm\/fGPzW6YdXDNkaTykmcv2QGFiyUdNHyIHGALJR0ybf0ycGlp5p\/ajpr4xVcLicMbB0qm8907amLz2T+ZOlnS\/+6U1OsjUKMMQ9MZItIdgh+uZtOgKOgCPgCJQhgKWrsrvGv3qUbRVTwxaClHTlrl27mvXr15uX49xZ\/KiriIdc90313393BBwBR8ARcAQEgUmRLBPnkOlin3pHwBFwBBwBR2AeCIxOsrLov3PnztlYN27c2Dz22GOdd4HySHYe4uPPcAQcAUfAEYghMArJhgqfdMq3y9RZJGs998SJE9lVxl36M+979Y5aKeclhU3qdxkfNgDBWKeO7RgYsRwIXvK599575y0e2c8bAyP9zD7tQvbACy8cAycdoExd3zSkfWPG7eNVzino1lxJVhvhO+64ozUwMODbt28f9MAAef7rr78+O15PJuLJJ59sHn300eDab6GuTeJya0ctPXbd0RQ2pb9DgWRv6txXpeYJ3hgY8fhgIKED8xx77rPGwEhwOXLkyEwnpy5HguUYOAkuhw8fbv+TuhWxofv27Wv\/m+de8LmypK8bAjM8A8uOU9GtuZFsbJeneZCsJYSYaNkVaopEUCvAWgGlnZgSprC59tprVykwYye\/7969u9FHF045Ups3RixfwPvChQut3E3B27Zkbd4YSR2GbCCzYcOGFZhMWY4Et3njJDJjRWpTx4llrG\/MBBPYpDNnzrS7\/U1Ft+ZGsgKwTm\/MM5K1JhUpTn2IQC25TeU+wfnkyZOziJ29bcuhSGFz\/fXXr\/CaMU7rAIZFSYeOhREvY8jOZFNOF88boxtvvNF01qaiV6F+zBsnOR500Um2b8wEE25T3k6Zim7NlWS1AZYN\/Pkz5NqLTkMxUUxlMvoyJqKAp0+fNknWOsM3hc3HP\/7xFSm8HOzmkZ3ogtdYGPFzp2QILCznjdGnP\/3p9vhLiewlZfzSSy+13RrSLnSRIU5PzlvfrHSxlU3qY3xDtNG3bOls0JSi+tFIFhOnC2pSBTq1E54ikqmm7GrG27cA15Cs1YeasQx1zxgY6a1Ep2QIpkKyd955Z3PVVVfN1mQX4S2BMWRJ5osLh4aym4uif06ymTPF6eSu78nqRzrJvmUemCA4pbApJdlQe5liMJfL5m0YrQMrnGSXpxo4IJKVYjku3pm6PM1bloRQ9DPncXxon4rZN2ZOshWzI+Xot956a29Vv6EDCVLrihVdH\/0WyyjFirxS2MiarKTxtPGzsJNny7V9vOs8JJDzxkgwlCjttddeWzWsvh3KvnAbAyNLzqZOsvPGKbR2PXWcWC77xkzWZPkzJQd29HRxXwYh1U6qgtari1eW\/+vqYf16gEXaItjPPvvs5AkWqTZ+BUK+G7IC25KvKRkCS3\/6rgBNVamjulgX51lFMil9n+fv88YpVM2\/SCTbN2Zav6akW5cNySIt5e\/JHmrWrVu3ygaVvger3zFeJIKVwQ\/xnl4KQw36lAyBRUpjYKQJdRHWZMfAadGri4fAzCPZebqWgWfl7Fo0gW720oXYbio1u2HFsNPP4gHccsstK6qcexlcT43MEyOry1MnWUT8nObmApu+5QgY6Vf9pl5dPBZOqUNSelKTwZpJ7fik9aPEfk9Jty6rSHYwafGGHQFHwBFwBBwBAwEnWRcLR8ARcAQcAUdgIAScZAcC1pt1BBwBR8ARcAScZF0GHAFHwBFwBByBgRBwkh0IWG\/WEXAEHAFHwBFwknUZcAQcAUfAEXAEBkLASXYgYL3ZtY+AftXEGrG8wiTbBT7yyCOjnls870Osp76BxNqXTh\/hVBBwkp3KTHg\/Fh4BvUHHVAYUOspw6P6txS1Lh8bM2197CDjJrr059RGNhMAUSXbMHZPGIveRpt8f6wiYCDjJumA4Aj0hECJZ3lNWHiXnfn7uc59rvvnNb87OTL3jjjuaXbt2tb\/hHFU5JIP3ZNU73uTspmUd\/gDifeaZZ2Yjr3kW7zhkHXIwJsH3NKXejCPQGQEn2c4QegOOwDICJST7xhtvzA5SAFnxloXy3bFjx2bXgGA3b97cHmouH0nHvvDCC9G13pwj0fTJSTnP0vfofZshE1M\/V9hl1xEYGgEn2aER9vYvGwRKSJbJEnu43nXXXQ2O7MJ3crygRLNW2yDD7du3z+5jsK1N2FP3hJwFvk+OoNy\/f3+zYcOGGeGHJnmRToa5bATVBzpXBJxk5wq3P2wtI1BCskyMILC9e\/fO0sNMsjgCziK12EboIUKVe44fP95Y6ebY4d94FtLaIXLnOQ6dVbyW5cDH5ggwAk6yLg+OQE8IDE2yvIbKXZb1XKSQ+ftY1KpPcEEb1nqtfhZIlp2CEIQ6Iu8Jam\/GEVgYBJxkF2aqvKNTR2Boks1Jz+aSLF8HwpUj5XJSwTkpZ7TvkezUpdb7NzQCTrJDI+ztXzYIDEWysiZrFRalqndjqV89KZx2Tj0rh4jRvq\/JXjbi7wMNIOAk66LhCPSEwJAka1X85ryXq6t7rShUp3RznqWri0N98erinoTLm1lYBJxkF3bqvONTQ2BIkpWx6vdkrXdTNSZWula3I\/dIqhiVzbnP4nVdfv0IfUhF2lObP++PIzAEAk6yQ6DqbToCE0FgTKLzHZ8mIgTejVERcJIdFX5\/uCMwPAJjkZ3vXTz83PoTpo+Ak+z058h76Ah0RsBP4ekMoTfgCFQh4CRbBZvf5Ag4Ao6AI+AIpBFwkk1j5Fc4Ao6AI+AIOAJVCDjJVsHmNzkCjoAj4Ag4AmkEnGTTGPkVjoAj4Ag4Ao5AFQJOslWw+U2OQD8I\/OAHP2g+9rGPNV\/96leb+++\/32z0Jz\/5SbNjx452f+LPfvazjbyWc8899zS\/\/\/u\/3\/7\/UB8892\/\/9m+bj370o0M9xtt1BNY0Ak6ya3p6fXBTR0BI9vOf\/3zzP\/7H\/2h27ty5ijRBqI8++mjzxBNPDEqqGisn2alLj\/dvERBwkl2EWfI+rlkEQLKf+tSnmv\/+7\/9uHn744WbdunWz8crvjz\/+eHPu3Lnm9ttvd5Jds5LgA1urCDjJrtWZ9XEtBAIg2a985SvNsWPHmkceeaR53\/veN+v7gw8+2Mj2id\/4xjdmJMvp4j\/8wz9siffqq6+eETR+F2L+1re+1SwtLTXnz59vr3vuuefatnfv3r2K0BG5\/uu\/\/mt7jUTP0idPFy+EKHknJ4qAk+xEJ8a7dXkgAJL9h3\/4h+bv\/u7vVqyzCjHu37+\/+fM\/\/\/PmC1\/4gkmyQpw6rSvEKrstnTx5siVsEKysq2LdV8hbng0S1uu+TMrf\/\/73fU328hBHH+UACDjJDgCqN+kI5CIAkhVCfPHFF5t\/+Zd\/mUWY8ptEo4hWkS62Cp+ELCWtLCQqa7wokpJ+4DcQqnwHEkWbQrr\/+Z\/\/uSK6RVGWk2zubPp1jsBqBJxkXSocgRERYJKVbkjEKinj3\/7t326+\/OUvN7t27WquuuqqNtUbI1mOPDkVDEJ+z3ves6p6WYhVPnv37jWrlb3waUTB8EevGQRGJVk+cuuWW25pDh061Bw9erRdX+Jjt9YM2j4QR0AhwCQrxCqv5giZCrH+\/d\/\/fSNrtb\/85S+TJIuI9bbbbms48uTqZAt8eXXorrvuWtE+rgNxS3Tsr\/C46DoCdQiMRrJ86POZM2ea06dPtyQrRkE88e3btzvR1s2p37VACDDJyvoposvf+Z3faUchEaxO7VrpYi5auummm2ZrrbFIFjCF3rv1SHaBBMm7OlkERiFZfcalnBACkpXXF0KHX08WRe+YI1CJgCZZ\/P\/v\/u7vtlEsFy6F0sUgSenCgQMHmjvuuKONPLnISa+3amL1NdnKCfTbHIEEAqOQLNLEsha0ZcuWllSZZCXKPXLkSPsKwfr1630SHYE1i4AmWUSt\/EpOKpK1iFp2kULa2Kou1sVQXl28ZkXMBzYyAqOQbCqSldcPXn\/99TZ9zC\/mj4yVP94R6B0BTZDyAIkqJV2MLRNjJPvhD3+43XLxM5\/5zIrCJmnjqaeeWvUaD96T5ZQyBuXvyfY+vd6gI9CMQrKCe2hN9umnn25TXidOnGij3NLPyy+\/3Nx5553Na6+91t6Kgion61Ik\/XpHwBFwBByBrgiMRrLSca4uxkA2btzYPPbYY82mTZuKxwaCfeihh2YE7VFxMYx+gyPgCDgCjkBPCIxKsj2NYdaMXtuVH4R477vvvkaIt4a4++6jt+cIOAKOgCNw+SCwpkjWmjYn2ctHmH2kjoAj4AhMDYG5kayVGo6BIZui91Fd7K8DTU3kvD+OgCPgCFw+CMyNZDWkUvgk52cePHhwxaYTXBDVNb2rXxXiPrzwwguXzyz7SB0BR8ARWAMIbN68eeFGMQrJ4hWeAiB7bwAAIABJREFUDRs2tBuZ64+1tlqKrH5NCPcLuX79619vnGRLEfXrHYHpIPDLpeuat5aubZbOPtOpU32106kTfnMWAkKwcgjGon1GIdlYhCkAdt2MIkbiQq6yc86ePXuqXhFatAmu6a\/gL3tITxWjUy9faG7YdGXN0Gb3SBsPfOfV5oFPXVPdFnD69633NDe898rmgZuu6dSntXizlqV9L769WXf+p52xknbkc\/jDv+4Em8jBvh+9o3l+x7s6tSM3bzn5i+bwh35VLE9T17fOwBgNPPDcq80N770iGytgJCS7aNHsKCQbijIxF11eu8FrPDfffLMZJYNk9WS97S++23z5U9d0Vv4hBHLebYYwmnc\/rOeJcn7lO682v\/nrT3bqzqlXLjSfOHam+d5d17cEWfMBTmdv\/UYvsiNjk8\/UyFqw+spzrzZfvumaYqy0LPWlZ5849uMWq+\/d9cGaqbvkbF2Ug67yJA3K2GrkSWM0ZTk49fKbvchnqRxM2SalBHAUkkW0KmuyetMJSRUfO3as6l3ZFMHKc4cm2akqSEoQ9O+SUpfNQPryGvvCZUokK5gJTl8493u9kGyfxNGXMZQxdnVIWJZKjWtIbvvCairyxBjJ2P6g58yIzGGtM4k56CoHPJelcuAkW2rBL15vVRx32aFJIuDjx4+bvQGZ90myFnHUKL+088+vXOjslcMgdlWm1HSKssmn5DkWLjXtTMUodjEY8yIOycxMJerHmEuNawyrU6+82TmjAXmqiUC5byCgPrJhfZGs9EmcLfksuhw4yaas8oR+l8mStROtVDXKbylDXyQLBSlJHULRrdRXbcrPGmMtVjq9V4uVNhhdsOpqXJEm7Mu49pECBXH0QbJwhLqm1plkZS2ua5pXZGetkqzol2Ak0ewNm64ocmbZ1MJ5l3bWAsl+5s8ebO7es6f54p\/eNCFGSXdltHRxumvDXGGRbK0XapFNDXFYBoM97NzUX4pkSwwl1pfkHk0gUyPZmug2lPqqSavV4GFJN2RH1j5LMwXauIpR7UqyrBdor8Tps8YIAnGSDds3YIQrarHqy9mCEyuEX2JDYha8VGfEbt9w7MfNF\/\/kpl7WhIdhF7vVUUg2Z2OKvjaj0MMGyTJxdCFZ7ZX3TbIlhjK2ZlK6ngIlsIxriYIgEpLCmaEi2b5IthSjIVKgLK9TMK5wspxk42a51obEHJGuJAvnvcSGWP1hvXCSLafnUUg21s2ur++kIBiaZGtSPbFItkRBtOcqqVD5sKHMSY+mIpgSkkW6Wdac+yDZGFYlFaIWodaQbMi4tq8oFKb64KB1Na6WHJSsn+P5KTlI6Rr\/DjkQsl7r6eKuDklfkSx0RfDuklrXcpBjQ3RmhTMgNQ6JR7Il2pZxrRQwycfaqCLj9ugl8yLZEkNpEUeNFzpVkhUsRMmHjvpLSJaxggHoSrJCqvIRQitxRDgiFoxKZMcS9hjJlqxfp0hWnpObPubMyKKTrFXfkMIqZJQsOakhWSuDVkOylnxAnkDWpSTLY0Q\/xR6U1DFYdrsrF8zr\/slFsjLwIaPZFMmyoUxNgrW+VKsg2tOcAslCqbqsyQKPEMnmetha0ZlQa9LFfZOsxgqGpVSelkl6ZeQBw5dbga5lh41ibN1ey3uKOEociRjJ1lTThgqfYmvqIEdOw3PtAxyklN7HsiDIPOUSSIxkLZ0J9a0vkmVdApa1JCt9kvoCrutge5CLkYzZSTYllYW\/D7mpf4pkS1I9fZOsGEMUOQ1FsjmCDSOiiaMm5cdKJWLA5FhSIRpT9JrXMGIkm4ORTqmGSFaIMfedR8FKk2xNCllHMBbJ5kQj8yRZGXdoDZpT77z8oB20lLNl\/V5THBQj2ZBjWhPJjkmyIh9CjqKvWvdy9MOSHRROAaPcjIY8X+z2t998d1H0W0g7g10+uUgWRVGyCcJQ6WJdpZYyJjEF0YLSJZLl9ddYNBLqj0VEQtpcaVqiIBZxlCpIjGRBKjlp3hhx4Lcc4lhOXS+\/P6gdKu1c5LQXckhihWMxeYqRbG7UH8OqxCGJGdfSdbVYJJuqOOYojZ2OPkjW0jOZg1iBz1RJNpQNK0nzWksN0JUSB8LSi1qSFfn4oyt+7iRb4gakqou7bEiR6oe1gB4ylKm2QpGslfKLtWVFrTFDmUuyQqjyQbFJ7jqIxgOORCz1m3JE5Nk6ku2bZDHe2DqhNTbpF16bEePaxZh0xWookm2dC9qYIMeB6EqynDJnXEOOaSiStUiW5QlpzVRmxIpku5Cs9YaClh3p2\/Kc2lt3xrJhls6E9IwdTZ0NqyFZvmdKJJsjtym7Pe\/fJxfJDg1AHyQLxYHhYOMA4sA4aqI0uWdqJIu1lT4iWRjFvkk2p4rSIlkxZkjV9+GQdMHKIlmkDUORrF6HhOxABnlsyJbkGCuQEtopfe1Nkyza6ZNkrexPSOf0+quQh6TzGV+uyA9hZEXx2iFhZyuWPdIkq+VTO6YpkmUd0MsPJXM+NZJdd\/5s89bSdVV7Qw\/NKan2nWRpb9aQEdAg6ohOfse6Ww3JWsog75XCk5X2cxREE3OfkWwX4oDCwmBwhWEbRbYn4cR3tslJF+d47JYxRISfE\/VzkQ6nndn5yMHKKlQJrcnGSDYWnTE5YmwhktVLJkzMpSTLa\/eYW3FI+cNEmJMuhoOh9Yud2VQkq6M9dqgsrLqSrC760XaECRWOeohksS4tjoGlK1o\/BN8YyUJ2ddbH0jPYIivDE2qnJl0civplHE6yKSpXv0u6eN++fe1\/1sHsQ1cXy5rsls2b20ILNpQ1JIvIoyWLi5V0IWMSgkkbDlHuLiTLBgOeumUoQ\/2xiCiHOGJGhFNfep1WlHfZkQifqNKFZGWOBU9pvyvJcjWodrZAhjlYxUgWOKL4BDJmRbIWsehIVojOitg4hZlDsojOYPCt3cCkr3r9NaUXIZK11l9rSRYygKyFJo6QQyLfayKyiuZ01C9YMclaBZVMRHDUsW6pbQhjYemK5aBZdgVzHioSsxx1LTtWRkM7JFbGCNs7skPKW0eybvH4nWQLCVYuT5Hs0NXFIFlL+WHQQMDL\/79yPUWThGUU2cMOQaQjulg7JZFsXyTLTodFHKm9VaFo2mBYJGsRCHvJ2ojwqxK5BMLOCyt6biTLRKSLOHiMIEcdwWvDYaVNrXaYZLUccHQGRyKHZHUKUxe8cBbEkgPgr8dgGVdLz3j9PLRsoPWDozOtUzy3+BtOm1UQmEuyQo66OlsXzYUc9RKS1Q5aSGcu2YiVDqkVtersAVLhsUMDYiTLdkW\/Wx6SJ02onD2wCit1O5Anj2QziTZ2So5u4uDBg822bdsyW750GY67e+2119ovN27cuOLYPKzJSiQrH07JsjERBbU8K4s4WPhEiLSClJIsDBjfl1MVHDOuloKE+mV55fBCGaNQIRUiHbnWUnQugmEC0WtpHKVpI8IKGvO0pX1ei9MZgpQR4Oh6KJLlNf5akuU0sMacI9mQHFgkq7MgyDhw+rqWZNEPJk70E5hrZ4wj+xjJ6rX5WApUtxNbYrFSoFr2Yg6JjgChH+yMWWMERoy\/1pVckoX8h5YNxiBZXSxo6dwl56L+\/OdiMunphlHWZFORbO3YcBj8hg0bZq\/\/6APgmWQtguXJtNJhMZJlZeF2rH175fdQ6quUZOFJx4yiVv7YNnuaZOVeq08gSJ3mhXG0IhgrygspkEWymuSZpHMIRGPE7VnprBDJyn3aodKyw3IcMoop4rCcFJ47XR0bkp3UsoFV5JQrTzJO1CRYa4oh2dFzp5cNSkjWegYiUIs4QnNnpdZBjqGiOUsGLZ1hh8RaagjpWW5mhPuRkoNQ\/UKOo84OLvQgFMla+mXpMI\/RSbaWAedwn0Sxhw8fbv9bv359+0RN6EKyNz\/4j835626N9ih30rmREMlaVcilETE8bHleqliBhVj+tpwJKy0jzoAYS4tArLHhOTUEYoEfS4VqsonhrsfGRBQiWctQxqK0EK7yfSgaQZ957mO4xogDbWlHLWRcNamA0MQREtLWxpWjJsso6u80EfH85JIsy5OlH7F2ckg2No6YzmBslvMZMyK6T5AnyxGJ9c0aG3QutORk3cOy1oVkLd3LIdkcnQnpHJ6Zs2w2B6opesTcIlm8G7t9+\/bmxhtvbHbv3t289NJLwc7WnMIjBVMnT55sDh061Kxbt65tG9Htjh07mi1btrQ7h9SQrAizGCSsx1gdtwwmCztezeF3Mq12UgqCXViEMELryiCNmGCH2kk9X\/dZ2kE0zYUbsSgtNG5xIMSYyb8gkJhxjbUjfWICSbWj508rPK\/txXC15IAdkS4kC2cLjpYm2ZQMhsgxh2RjlgVYlWRBQu1Bnqy0dywC1b\/B8OdglEOyGqOUpQ2RY4xkc50kkE2oPiRXh4GRLjLLcUhgY2JbplqEnLKdIV2R+5xkU1I38O9SMHX69GmTZLdu3dqu8eaSLAsp\/oaBCyl6ijC5zRgBpRSEf49FQrHfWEH03ykl0L\/z6x6Wt5w77SF8U+PQ7cv14nwgTQ9CTJGs1U\/OaKCdVH9iywahTEEp5ryOlcI3FS0LYVsEkhqn5WyVkGyofRToaJItlfW+SBZYl5JsyNmKRcTWPSF5EvxDGZ7cdkIY5ZIsXwfZji3JhJbouL8oXLN05fkd72pkN8BF+swtkp0HKDUki6o1y1CzQEAIQore1\/epaAF9yiGzlJFMjanUC009L6W4IRLMbTcWjcQciZTxBrnmynBfspAad+r3FN61eIVwkPmzDGwpHjXOlvUMacd6JSY1j7qtlENTMj7r9bzS\/sj1NQ6jpQPS95h858oYxhDrV0lbIdvkJJuSloF\/ryHZ3ImH8JQolGXkSlOoIaLLIdkcwmYlSylcKNqCknaNsDEmrai5c6T7l4t1qv2QYSyVBY74cxyYUPsh8kiNw3IkU8sNoep7\/X3ps2NOz8Bmom2+pr+hfpXKQej5NX1KZb2sObfmDm9T5Fwfc95ir1mV4BTSOSfZiHak9ivWt9auyR45cqR59NFHZ4VPqTXZXMGOCU9KaUNp4lJvlEmQo4aadriSlv+OebXW9LLzwa\/5lHq0PA8prHO9ZjYcqWg1VmnOJ5EwBqW4l5JsSJ1q0qAhB0kwmkcaNPR8jXvNkkzN3Obqfc58l7bVpzxJ\/2rT6hwthkg25lhoW4GIv1SeLD0K6YqT7Dxc0MgzaqqLc8gAwiNrKTi6zKrWDBlqjqh4X2Ir0oopLBsgFsLciI09UBAitxlK9+V4rtJv3rBCv1ebMlYYD7aMyyl6is0diEi\/vmE5RCkjbe0UJe2UkiwX9PAzc2SQ8eOiF\/19bI0uZBSt13didQdWpoExiskL99fCPVXQE3P0LIxQ\/JZT\/GRlYuDYllRhxxwKtiVWoVYIdytzFSoQS9kQPAPXheQpZs7Z4ddkXVuFrR1tqx0n2ZFJNvc9WVQXy6TGyKDkNZ6YoeTUB1ficgVtTnQWI1m9yUJKQVAYxJXOqSjCUl7r9Z0UriHPld\/\/hFGsKRCTsYd2+2HjwGQQiuC1UWSDYslOzMABq5L3P2NEZG2yUEKycESs03ZqSFYwhCPKG1aEZDH1qpPcF9qEJORcaOLAdaWvOqEdYIR2rI1aLOdF107ACdXbGqacT\/zOWEFPQxXmKecztEOaVWEeK1TS75VreYo5G9a4NVlbr3E5yRaSrJVC7nrMXc6OTzGShTLwcVH60G1r8mNRKdJDrRG6afn4uRLlZ8NxiRiunL02Y1WH5kRnaAvjCaVoYg5AiGRja305JMsbUYhhk0+O5x8yinimtdYTIl7scyuGMbYXa46hBFmDOHh+Yo5EjGRBaDDyqXY0GTBWvDUgZzT08y0Hise2\/IwLbQrTinYsctQRjKUfrAOhXcTkvhDJ5u5klLt72CU9XO2o83h4OSW2yYK0l3LUMe7Qe+kxPeXfMEZ5Hh820IVk2\/5ffOe6tHqagxn9OpB2jpxkC0hW3mm97777Vmx5KLdL8dKxY8dWfV\/QdPRSeYXnM3\/2YPMfW+9pFdLaLjC2UT0bkZTBgLKFjt6CwbXaYYMRMhx8X0xBLEMZGmNMQSwjoMeGLQKFmEI7WllkgJQVxgRiZackNxrhe0BE2Is6tPGCJQcwGugT3pUGgViRLBtVXZ3OJ60wPjFyzIn2eP\/iUCRrEaeFOTayh1MTI1kdeWinDQ4rNjnRTpI2rnJ\/artAa72eMcolWcvZ4kxOSSRryc6M1C4eGhJyREK6r40YjztFsil5QtYrNP85ZM22U7cT2hnLsreY89jhACyDTrKZTKiLkfRteivEzGazLuuTZK1ohL+DJx8jbeukDBhKNmIcVVkDtbZDa4l60xWzDTRChoPbs9Jhoc3LteHQ\/QqRrGUoY44I2rWw6uO0olJnCwYBRIT+gVT0AeV438\/KYgDDHKxCThuTrN74nuc8dZAC71\/cF8mmjnDTqfNaktXj1oY\/Z\/9ikKzgtOxMv9lGl9pp02cOW3JgjduKZFnHrewPz3noIIWYDdE6qZeuLN1nmQnJgV66sk500vYrpmchkl2eh+XMiHycZLNoLn0Kz9BH3VmRbMgrt4Zk7dgS8spReBGCxtrYO2QoteFIKQiMg7Xmpg2+jkKYOEqOcOM+hTa+H5tkrUIfbUy0cdXj4hQyY4UqzRx5wi47VmUn1u454kU0qp02PrZMkwlHdtitLBTt8UEK+loeT2gtTssQO2GWzkAOYljx2jyWcHJ0RpN1jGQt55OxYAKxDlLIIVlLnvT+xSGSlfZZrvT8692aLP2y5EBjpMcGeyNzp51ZpJqF\/Cy7xGvzkFtrxzzICHZ5i9lb2dPg1F0f9M0ocng2Fcla77vmtJtzTSiSzTGKlhGJnZOaE53lKH8oghmSZC0vNOecVGsOtKfNRiBnbGhTR+vslYeIw+pPV5KNOSRsAGFEQ5kMTbKcBWHZ4UhVnq0PdyglWV6ftiIQwbUljotr4VzExMYV7YQIRK\/56tOXckiWMQLuFskKAXPlfw6BxA4jD5Esn8QUSq3rDE8sFcrLC1q\/luf60lnLFhZyjUWyOc6WhRFjyORp6YxkyQRDvayi9cOSA+2Ypmy3YOgkm0JJ\/S7R6s6dO5sTJ060ewrjIwR74MCBVd8XNh+8nEmWDUYJyUrj2gvVEYxcU0uyVhQBbzY0sFi0qn+LRcQYm15b7IoVDAbOjdWGsgQrYMAkGyKOXJLFzkAx4tBt6QI4y7imUupMHCmSDWHEJKvn2opgUOCmsxlcAKerYDnay8FKE4u1Tsc6c2lOV56TmkuyM8P+8pttgVyKZOV3qxhKn5MaWm9kOWkdku+8Oiu6qiXZlA0pIVldFMfOLNcpxPQjRbKhbJiT7EpUR9lWcR4bU4TIaOoky2lCjmBiR9PJWGNEWkqynOaNpfxKCER75XBS5HvrwHZr\/qyonw\/Q1gdslzgkOcQRItmQg5aKZLm92NokV1pbY+JiM4tk4aCFojNtFC3jGiLZdl43XbEqup4nyTKhssPBWPHYl2Xx0slDVhbEcjiWZXV5fZCdOvler8PrjEYs2gdBp0jWKgbkMbKcoFpcr8PnOLOWnJTaEI2Vdl5qghqPZEMWbWLfLwLJArIakrUiuqEUJLZuaREIRytdSJbH2BfJ4vQfRCU5YwtFZzo6TVWrLxvpH7e38fKDloMSA2kRR4pkYRitNcgQyYaimVySlT4JSYP4tLnIiWQZF5CL7pdFvim9QN90Sp2rpZFaj5FsTl0GHFYrQ5ZrQi1nTDumJTJkOVslGSN2SJxkc2dxDVyXItkc4yowWMrAhjI3yrPWU2pIFsUJKEqxFCSVJubpjSlISUqVsWKS5XEj0kqJFwwGni8GI1SsEWvLKl7RRiBHDqZIsjHZySFZjVuIiET+a+SpNILhKF2PrWQ935KHFMmm5JEdBCbZHNnRTijugXPC0W2qHzFnLJUFieEyJMnW2BCPZHMl4eJ1qZRxzd7FOV0Qkr399tubs7d+ozUSy1HE8ovzVsl+qE2LZJHOKVEQvXZiFbzkeJ\/oZyzVU2MUtcFgI5BrTIAVv8oyFMnmYBUj5lCaMCYHvH5dUvjEbfYVyebIQQ5Guh0dwUh\/QwUvJTqTo7P6mprobAiSRZtWMVRqvTJEsiBu2KScLAiTrHZYnWRrJKzfe0ZZk0V1sQyFD1jvd2h2ayDZP3nk\/5ulqWpJtg\/jCmLG2smikGypEdGRfV8ki\/SgzLZV8BIzrlbqq2+SzXVEBA9xQkJyoI1nSlcsZwtjKyFZK6OReraT7Mp3a1N4WenkkvV8dohEfpiYu5CstdaNojlrHV6Ps+9smEeyKUmi3xHF7t27d0VlcUET1ZeCZB9\/\/PHZ+1Y10RkbRct7LDWKljKEijhig48Z1z4j2b5IFu\/O5bRnpYt1xJVDIFbxCtbdxiJZjkZ0IViqGjTmSFjVsakiOm5vKJLNdT6ssfUdyZasM8YciNIUaEyPa0lW2mRdGoJkS+RHLyvURv0yjudfeMHfk81lPkSyW7dubbZt25Z7Wy\/XhUgWm0nkKj+8T5CtFuxSko0VZuQQhyYbNq6lxCFthYxrbGeWVASjPexajHhNFs8MFbxYfRqKZLXsWLKREmKr0Mdal0y1U7P+msIqxxHKIQ65Ru8HnhoP\/x6rYyhpp2Y9P9Q+LxXl2pAcrErSxTGHpMSGoL6Dn10rTzGSlag7d8lB5vzfvv1E8+z9f+ybUeQKuWzkL3sXP\/TQQ82mTZtyb+t8nUWyaLSGQKxUj2UoUx3vi2QtQq0hWemvhUcNRjV4WHjVRPYp4sh97Sc0f5j\/GlxCRrHU+ZgHViVZkJiz1QcBQZ6QXq8loj5JNuaYpnTf+r0mku2LZGPyVCoHoVR4qRz85f9+rvn60aPNU39zv5NsrkDp03L0fbWFT7pdfapP3yTbl6G0UoI1pOIkm5bAWCSbvtu+wkk2jVzfGDnJpjGvsSF9kmzIgSgl2ZjdTqMw7hWjFj5t2LChuffee3tDAAQr0TF2kdKHDUyVZK2UYI2CDE2yNZM1tUgWUYesLVnvQtaMsW8CqY3O0Pca2YlF\/aURTF\/GNebElqznx+a0piBsXtG6y8Ey0k6yhVZpqMIna89jnZaOTZaV2igcWnv5mKRiraeAVOTfnKIFwUi2vfzCud8rehcyhFWfeOjqyZr56ese4HT33XebqfWa5\/SJVW61dQ4B1ZLsEBjVFILljLFk3TJGsqVYMUZod63JAeNV45A6yRZak3kWPpWQbOEw1uzl+jWnHGJOGTGcoLKWQGPFf\/bCu83tBUvHu9aMK2O05eQvVh0dV4rPUNf35WCXbtIRitLWmhzwvNW8Y+0kWyH58yp8kuj2ySefbB599NFm\/fr1s7TDnj175v76UAVMo9wiUezRo0cbxygO\/xA4oUq1ayWvtPPtN9\/dnr\/Z5XPq5QvNvh+9ozn8oV81N2y6srgpxuitpeva+2vaKX7wSDe0yw\/vvaJojJYc7Xvx7W0F9s1X\/rzTSKQ\/by1d2xz+8K87tSNy8MB3Xm0e+NQ1RWPr9FC6GRjxq5d9tT10O6OsyaZ2e5JB1xY+MWChtLTs+CSekX8cgbWKwC+XrmuN69LZZzoPUdp65\/mzndvxBvIR6AtzaUc+a2H+Nm\/e3MjSjPy7SJ9RSHYeAMVS0k6w85gBf4Yj4Ag4Av0hsGjkipFPkmSFICVduWvXrjbFa31w7ix+O3jw4GxjCxBs39XL\/YmLt+QIOAKOgCNwOSAwKZJl4qxNF+M1nptvvrnX14MuB2HwMToCjoAj4Aj0i8DoJCsL2jt37pyNauPGjc1jjz1WtQuUE2y\/wuGtOQKOgCPgCHRDYBSSDRU+ccq3Zliy8cTx48fNW0+cONFce+21ze7du5uXXnppdo18j40rap451Xv0zlcp58WaE8Ym9bvgoPGfOrZjYMTyInjJp88NWfqWxzEw0s\/sahf6xsRqbwycdIAydX3TuPWNGbcvWdFz585NQrfmSrLaCN9xxx0tCDDg27dvH\/TAAL37k369Zx7KOI9nWGvSeuy6HylsSn+3dt+ax9hznzEGRtw3GEjoQG6\/53ndGBgJLkeOHJm9cjd1OZL5GAMnweXw4cPtf1K3IjZ037597X\/z3Au+Vh6HwAx9wbLjVHRrbiQb2+VpHiRrCSEmeseOHWsqmtUKKMIXU8IUNpIB0ArM2CFDoI8unHKkNm+MOFsCvC9cuNDK3VQj2XljJHUY+\/fvb3TB4pTlSHRr3jiJzFiR2tRxYkLuGzPBBDbpzJkzzdLS0mR0a24kKwDr9MY8I1lrUpHivPrqqweNoGu9vdr7BOeTJ082hw4datatW9c2E3MoUthcf\/31K7xm9EuUOobdlJV+LIz41TJJZ005XTxvjG688cZ2OWeMc6ZrdQ12bd76tugk27dsyZGp3Ka8nTIV3ZorybIgW+unQ6696DQUE8VUJqOLouv1iNOnT5ska53hm8Lm4x\/\/+IoUXg5288hOdMHL2uc69m51Xxjxc6dkCCws543Rpz\/96fb4S4nsJWWM2okh7UIXGeL05Lz1zUoXL5KD0rds6WzQlBz80UgWAqoLalIFOrVKkTKSU03Z1Yy3bwGuIVmrDzVjGeqeMTDSW4lOyRBMhWTvvPPO5qqrrpqtyc5zn\/NaWRtDlpCmFrxee+21Zii7WYtJ6r6+MXOSTSF+8XdOJ9e+Jxt6lJPsW+161zwi2RDWmWIwl8v6VvKUIyL7QGv8nWSXpxo4IJKVoyq5eGfq8jRvWRJC0c9ctA14+sbMSbbCbEo5+q233hrc8am0ydCBBKl1xdLnTOF6yyil1mQlTaeNG7CRNdnY77Iego88W66tfdd5XvjNGyPBEFGHHmPfDmVfGI6BkSVnUyfZeeMUWrueOk4sl31jxjaIHbcpZChHTxf3ZRBS7aQqaNfSu7J9V+6lqouBnZDys88+O3mCRaqNX4GQ74aswLbka+qR7Lx9eYCXAAAFRElEQVTlCNXFutrfKpJJ6fs8f583TqFq\/kUi2b4x0\/o1Jd26bEgW3s3rr78+Kwjy92QvmaLS92A1dotEsDLqId7TS2GoDf+UDIFFSmNgpAl1EdZkx8Bp0auLh8CMZXhKunVZkWzOrkXz9ICHfFZsNxXLcKWwif2un8XjuuWWW1ZUOQ855tK254mR1bcpGYIQdmNgpF\/1m3p1MTIjvBzAhUh96xvmKnZISqkujHF9ascnrR8pG+UkO8Ys+jMdAUfAEXAEHIEREbisItkRcfZHOwKOgCPgCFyGCDjJXoaT7kN2BBwBR8ARmA8CTrLzwdmf4gg4Ao6AI3AZIuAkexlOug\/ZEXAEHAFHYD4IOMnOB2d\/iiPgCDgCjsBliICT7GU46T7kfhDQr5pYrcorTLJd4COPPDLbj7efp5e1Mu9DrKe+gUQZen61I1CPgJNsPXZ+pyOwAoGpbm4SOspw6Olbi1uWDo2Zt7\/2EHCSXXtz6iMaCYEpkuyYOyaNRe4jTb8\/1hEwEXCSdcFwBHpCIESyvKesPErO\/fzc5z7XfPOb35ydmXrHHXc0u3btan\/DOapySAbvyap3vMnZTcs6GAPE+8wzz8xGXvMs3nHIOuRgTILvaUq9GUegMwJOsp0h9AYcgWUESkj2jTfemB2kALLirfjku2PHjs2uAcFu3ry5PdRcPpKOfeGFF6JrvTlHoumTk3Kepe\/R+zZDJqZ+rrDLriMwNAJOskMj7O1fNgiUkCyTJfZwveuuuxoc2YXv5PhBiWattkGG27dvn93HYFubsKfuCTkLfJ8cQSln427YsGFG+KFJXqSTYS4bQfWBzhUBJ9m5wu0PW8sIlJAsEyMIbO\/evbP0MJMsjoCzSC12yECIUOWe48ePN1a6OXb4N56FtHaI3HmOQ+c4r2U58LE5AoyAk6zLgyPQEwJDkyyvoXKXZT3XOpw6FrXqE1zQhrVeq58FkmWnIAShjsh7gtqbcQQWBgEn2YWZKu\/o1BEYmmRz0rOMUU5qGOnhAwcONHKkXE4qOLddadsj2alLrfdvaAScZIdG2Nu\/bBAYimRlTdYqLEpV78ZSv3pSOO2celYOEaN9X5O9bMTfBxpAwEnWRcMR6AmBIUnWqvjNeS9XV\/daUahO6eY8S1cXh\/ri1cU9CZc3s7AIOMku7NR5x6eGwJAkK2PV78la76ZqTKx0rW5H7pFUMSqbc5\/F67r8+hH6kIq0pzZ\/3h9HYAgEnGSHQNXbdAQmgsCYROc7Pk1ECLwboyLgJDsq\/P5wR2B4BMYiO9+7ePi59SdMHwEn2enPkffQEeiMgJ\/C0xlCb8ARqELASbYKNr\/JEXAEHAFHwBFII+Akm8bIr3AEHAFHwBFwBKoQcJKtgs1vcgQcAUfAEXAE0gg4yaYx8iscAUfAEXAEHIEqBJxkq2DzmxwBR8ARcAQcgTQCTrJpjPwKR8ARcAQcAUegCgEn2SrY\/CZHwBFwBBwBRyCNgJNsGiO\/whFwBBwBR8ARqELASbYKNr\/JEXAEHAFHwBFII+Akm8bIr3AEHAFHwBFwBKoQcJKtgs1vcgQcAUfAEXAE0gg4yaYx8iscAUfAEXAEHIEqBJxkq2DzmxwBR8ARcAQcgTQCTrJpjPwKR8ARcAQcAUegCgEn2SrY\/CZHwBFwBBwBRyCNgJNsGiO\/whFwBBwBR8ARqELASbYKNr\/JEXAEHAFHwBFII+Akm8bIr3AEHAFHwBFwBKoQcJKtgs1vcgQcAUfAEXAE0gg4yaYx8iscAUfAEXAEHIEqBJxkq2DzmxwBR8ARcAQcgTQC\/z965kGeeis5jAAAAABJRU5ErkJggg==","height":258,"width":344}}
%---
%[output:88f92da2]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdkAAAFjCAYAAABxFnERAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnX9oJcl94L9DFnY1XrhFGtjRnIdbiP4wZNF6OSNFYBwfZxyfpONuyUrZg13FGAklyhp8vhGnQ4JwMOK0ls7nwXNKZGnBkQ7OkWyIwdIlwSYOBDZaHDYzt4GAFdizsqMxjOZszjtaQ4yOeup626\/V3eru1\/Wqq\/rzwF7pveqqb32+9eajqv5Rl05PT0+FFwQgAAEIQAACpRO4hGRLZ0qFEIAABCAAgQYBJMtAgAAEIAABCBgigGQNgaVaCEAAAhCAAJJlDEAAAhCAAAQMEUCyhsBSLQQgAAEIQADJMgYgAAEIQAAChgggWUNgqRYCEIAABCCAZBkDEIAABCAAAUMEkKwhsFQLAQhAAAIQQLKMAQhAAAIQgIAhAkjWEFiqhQAEIAABCCBZxgAEIAABCEDAEAEkawgs1UIAAhCAAASQLGMAAhCAAAQgYIgAkjUElmohAAEIQAACSJYxAAEIQAACEDBEAMkaAku1EIAABCAAASTLGIAABCAAAQgYIoBkDYGlWghAAAIQgACSZQxAAAIQgAAEDBGojWT39\/dlYmKiBePm5qYMDg4aQku1EIAABCBQdwK1kOzBwYFMTU3J8PCwzM7ONnK+s7Mjq6ursr6+Ln19fXUfB\/QfAhCAAAQMEKiFZJVQt7e3ZW1tTbq7uxsYT05OZH5+XoaGhmRsbMwAWqqEAAQgAIG6E6iFZJeXl+Xo6EgWFxelq6urmfOk9+s+KOg\/BCAAAQiUQ8B7yeoZa29vb3OpWKNTkn3zzTdbZrhxWFUZXhCAAAQgYJfAwMCA3QAKtF5rycYtI4cZKrl+9atfbYiYFwQgAAEI2COgBLu1tWUvgIItI9nIudqoZF955RU5Pu6XR4+uFkRc3mGXL9+Xnp67lYmnvJ61XxNskhnCBjZ5v2FVGzM6HiVZ12azSDaDZA8PPy0nJ0\/nHaell+\/q+rFcv\/5nUpV4Su9gGxXCJhkebGCT96tVtTGj40GyeTPZofJFL3xSy8RqJlslqfX03GnMqqsg\/Q6lL3MzsElGBRvYZP4iBQWrNGaQbN7sdbh82i08cRdE6fCqKNkOo6M5CEAAAtYJIFnrKUgPoOjDKJBsxRNLeBCAQC0IIFkH0lzksYpI1oHEEiIEIOA9ASTraYqRrKeJpVsQgIBTBJCsU+nKHiySzc6KkhCAAARMEUCypsharhfJWk4AzUMAAhAQESTr6TBAsp4mlm5BAAJOEUCyTqUre7BINjsrSkIAAhAwRQDJmiJruV4kazkBNA8BCECA5WJ\/xwCS9Te39AwCEHCHADNZd3KVK1IkmwsXhSEAAQgYIYBkjWC1XymStZ8DIoAABCCAZD0dA0jW08TSLQhAwCkCSNapdGUPFslmZ0VJCEAAAqYIIFlTZC3Xi2QtJ4DmIQABCHB1sb9jAMn6m1t6BgEIuEOAmaw7ucoVKZLNhYvCEIAABIwQQLJGsNqvFMnazwERQAACEECyno4BJOtpYukWBCDgFAEk61S6sgeLZLOzoiQEIAABUwSQrCmylutFspYTQPMQgAAEuLrY3zGAZP3NLT2DAATcIcBM1p1c5YoUyebCRWEIQAACRgggWSNY7VeKZO3ngAggAAEIIFlPxwCS9TSxdAsCEHCKAJJ1Kl3Zg0Wy2VlREgIQgIApAkjWFFnL9SJZywmgeQhAAAJcXezvGECy\/uaWnkEAAu4QYCbrTq5yRYpkc+GiMAQgAAEjBJCsEaz2K0Wy9nNABBCAAASQrKdjAMl6mli6BQEIOEUAyTqVruzBItnsrCgJAQhAwBQBJGuKrOV6kazlBNA8BCAAAa4u9ncMIFl\/c0vPIAABdwgwk3UnV7kiRbK5cFEYAhCAgBECSNYIVvuVIln7OSACCEAAAkjW0zGAZD1NLN2CAAScIoBknUpX9mCRbHZWlIQABCBgigCSNUXWcr1I1nICaB4CEIAAVxf7OwaQrL+5pWcQgIA7BJjJupOrXJEi2Vy4KAwBCEDACAEkawSr\/UqRrP0cEAEEIAABJGtoDCwvL8vGxkaz9snJSZmdnW1p7eTkRObn52V3d7f5\/sjIiCwuLkpXV1fzvf39fZmYmGg5dnNzUwYHBxOjR7KGEku1EIAABHIQQLI5YGUpqsV5eHgoa2tr0t3dLQ8fPpTp6Wm5fv16i0CViPf29mR9fV36+vqa5QYGBppCPjg4kKmpKRkeHm6+t7OzI6urq83j4uJCslmyRRkIQAACZgkg2ZL5aikuLS21zDTVbHRubu6cUMfHx2VsbKxl1rqystIUtBLq9vZ283dVUIt8aGio5dhwV5BsyYmlOghAAAIFCCDZAtCKHKKXfPUyb5KMo++r2e7R0dG5JeSk93VsSLZIljgGAhCAQLkEkGy5PBNriy7xRme2+kAt2ZmZGRkdHW2cs+3t7T13PldJVolUL0lHG0ayHUoszUAAAhBIIYBkOzA89DnZ8LnWJMnqsmoZOU2yccvIccvFx8f98ujR1eZHJydPd6DHNAEBCEAAAkqwly\/fl56eu7K1tSXKAS69Lp2enp66EHDcrLNTkg3zUcI9Pn7OBWTECAEIQMBpAj09dxpy1S8kayid0SuIdTOdkmx4Jsss1lCSqRYCEIBAhICaxaoXM1mDQyNJsKpJLnwyCJ6qIQABCFSEAOdkDSRCC\/TKlSuJFyaFz71Gb+EJ3+qTdgtP3AVRujtc+GQgsVQJAQhAICcBJJsT2EXFtTwfPHiQ+rAIVQ8Po7iIJp9DAAIQcJsAki05f2rmubCwkFjrzZs3mw+Q4LGKJcOnOghAAAIVI4BkK5aQssJhubgsktQDAQhAoDgBJFucXaWPRLKVTg\/BQQACNSGAZD1NNJL1NLF0CwIQcIoAknUqXdmDRbLZWVESAhCAgCkCSNYUWcv1IlnLCaB5CEAAAiKCZD0dBkjW08TSLQhAwCkCSNapdGUPFslmZ0VJCEAAAqYIIFlTZC3Xi2QtJ4DmIQABCLBc7O8YQLL+5paeQQAC7hBgJutOrnJFimRz4aIwBCAAASMEkKwRrPYrRbL2c0AEEIAABJCsp2MAyXqaWLoFAQg4RQDJOpWu7MEi2eysKAkBCEDAFAEka4qs5XqRrOUE0DwEIAABri72dwwgWX9zS88gAAF3CDCTdSdXuSJFsrlwURgCEICAEQJI1ghW+5UiWfs5IAIIQAACSNbTMYBkPU0s3YIABJwigGSdSlf2YJFsdlaUhAAEIGCKAJI1RdZyvUjWcgJoHgIQgABXF\/s7BpCsv7mlZxCAgDsEmMm6k6tckSLZXLgoDAEIQMAIASRrBKv9SpGs\/RwQAQQgAAEk6+kYQLKeJpZuQQACThFAsk6lK3uwSDY7K0pCAAIQMEUAyZoia7leJGs5ATQPAQhAgKuL\/R0DSNbf3NIzCEDAHQLMZN3JVa5IkWwuXBSGAAQgYIQAkjWC1X6lSNZ+DogAAhCAAJL1dAwgWU8TS7cgAAGnCCBZp9KVPVgkm50VJSEAAQiYIoBkTZG1XC+StZwAmocABCDA1cX+jgEk629u6RkEIOAOAWay7uQqV6RINhcuCkMAAhAwQgDJGsFqv1Ikaz8HRAABCEAAyXo6BpCsp4mlWxCAgFMEkKxT6coeLJLNzoqSEIAABEwRQLKmyFquF8laTgDNQwACEKjD1cX7+\/syMTFxLtnXrl2T9fV16evr83IgIFkv00qnIAABxwh4O5Pd2dmRhYWFTOm4efOmjI2NZSrrSiEk60qmiBMCEPCZgHeSPTg4kKmpKbl3755cNFsNz3IvKtvOIFDtzM3NnZs5n5ycyPz8vOzu7jarHxkZkcXFRenq6mq+Fzcb39zclMHBwcSwkGw7GeNYCEAAAuUQ8EqyevY6OTkps7OzuQgtLy\/LxsaGlD2rffjwoUxPT8uDBw\/OSVa1ube313xflx0YGGjGr\/9oGB4ebr6n+rm6upq63I1kc6WfwhCAAASMEPBGskpQr7\/+urz66qsts8A81NTM8vbt27kFndaGlnd0pqyFOj4+3rJUrWatKysrsra2Jt3d3aKEur293fxdtaVnwENDQ4nL3Eg2T+YpCwEIQMAMAW8kawZPe7VqYX7uc5+TL33pSy0zTz1DXVpaaln2jb6vJH10dHRuCTnpfR0xkm0vdxwNAQhAoAwCXks27pxnFFp\/f3\/LLLEMqKqOsCzV79FzsknnafVxMzMzMjo62jhn29vbe252rSSrRKpnvNG4tWSPj\/vl0aOrzY9PTp4uq4vUAwEIQAACKQSUYC9fvi89PXdla2tL1KlAl16XTk9PT7Ms1aaVMSFZLXctxzihJkk2vIycJtm4ZeRwP7Vkw+8p4R4fP+dSjokVAhCAgJMEenruNOSqX95JVstKdTBptmcqc1EB2pRseCbLLNZUxqkXAhCAQCsBNYtVL29nsnFX6nZiEMSda7Up2cPDTwty7UTmaQMCEIDAeQJen5O96OIgEwPioodg6NuLuPDJBH3qhAAEIFAtAl5LNnwRkc0nOsXNZNNu4QlfJJV2C0\/cBVF6eHF1cbW+aEQDAQjUk4DXklUpvWhmaeLCp+hQSrrIiYdR1PNLR68hAIH6EPBasvpBEGnptClZHqtYny8aPYUABOpJwFvJ2ry6uApDieXiKmSBGCAAgboT8F6y4ecA1ynZSLZO2aavEIBAVQl4K1kF3MbVxVVJNJKtSiaIAwIQqDMBryWb5UH6viYfyfqaWfrVSuCx4Nd\/BAwEKknAW8nqc7J3737wWKu4DHTiwicbmUeyNqjTZucJINnOM6fFPASQrKENAvIkwURZJGuCKnVWj8ATQUjvVy80IoKAiHgr2bpnF8nWfQTUpf9PBR39SV06TD8dI4BkHUtY1nCRbFZSlHObAJJ1O3\/+R++NZNU52Ndff11effVV6erqKpQ5daHU7du3z+3dWqgyywchWcsJoPkOEbgStPOgQ+3RDATyEfBGsqrb+hGKN2\/elLzPKtZPhypybD7knSmNZDvDmVZsE\/hwEMA\/2A6E9iEQS8Aryaoe6k0B7t27J9euXZP19XXp6+uL7bx6pvDExETjs4vKujZ+kKxrGSPeYgSQbDFuHNUpAt5JVoO7aGOAMGBfZq\/hPiHZTn2FaMcugWeC5t+xGwatQyCBgLeS1f0Nz1bDDHybuUbzi2T5zteDgF6lOqhHd+mlcwS8l6xzGSkpYCRbEkiqqTiBZ4P43q54nIRXVwJI1tPMI1lPE0u3IgQ+Gvz+N5CBQCUJINlKpqX9oJBs+wypwQUCHwuC\/IELwRJjDQkgWU+TjmQ9TSzdihD4ePD7X0IGApUkgGQrmZb2g0Ky7TOkBhcIfDII8vsuBEuMNSSAZD1NOpL1NLF0K0LgU8Hv34UMBCpJAMlWMi3tB4Vk22dIDS4QGA2C\/I4LwRJjDQnUQrL6kYkqv+rBE88\/\/7y89tprjf91d3d7mXYk62Va6dQ5Av82eOePYQOBShLwWrJxG7dryU5NTTUSkvbYxUpmLGNQSDYjKIo5TuDFIP5vOt4PwveVgNeSDT\/0X81elVhnZmYamwfoxy5OTk56setOdIAiWV+\/svSrlcDLwa\/\/AzAQqCQBbyWrZ7HXr1+XxcVFeffdd1skq7a1m5+fl8PDQ1lbW\/Nu2RjJVvL7RlClE5gMatwovWYqhEAZBLyX7MDAQGOmqnfn0TNZBU\/NdJWMkGwZQ4k6IGCDwKtBo7dtNE6bELiQgLeS1TNVRSBtJqs\/L7rR+4WELRVgJmsJPM12mMAXgva+0uF2aQ4C2Qh4K1nV\/fAm7uFzsqOjo42l4t3dXeGcbLaBQikIVJPAQhDWzWqGR1S1J+C1ZPVsVsk07tXf3+\/lUrHqKzPZ2n+36wHg8UCyP0ey9Ui4e730WrI6HXEbuPs6g9V9RrLufRmJuACBDwWSfQ\/JFqDHIR0gUAvJdoBj5ZpAspVLCQGZINATSPYYyZrAS53tE\/BWsnEPokjDNTIy0rhAypcLoJBs+18OanCAwC8Hkv17JOtAtmoZIpINpd2nc7RItpbf5\/p1+lcCyf4tkq1f8t3osbeSVfj1vbHqyuLoLFXdI7u3t9d8rKJvT4BCsm58AYmyTQL\/PJDsXyPZNklyuCEC3ko2ep9sdBk4+rniq27rUa8vf\/nLhnB3rlok2znWtGSRwK8Hkv3TFRF532IgNA2BeALeSlafk9VPfIrrfvSJT\/p3Nat1\/YVkXc8g8Wci8BuBZL\/1ByLyINMhFIJAJwl4L1n97OKkmWz42cVItpNDj7YgUAKB3wok+4dqg4B3SqiQKiBQLgFvJasw6V144q4c1p\/p+2XDGwqwXFzuIKM2CBgj8B8Dyb72XRH5K2PNUDEEihLwWrIX3cajryZW8Kanp+Xu3buNTd3VVniuv1gudj2DxJ+JwH8JJPuffiAif5LpEApBoJMEvJasBhn3xKewTON26OlkEky0hWRNUKXOahF4TOT3585C+p0DEflGtcIjGgiISC0kayPTWtz37t1rNB93D27cs5Xjlrb39\/dlYmKipRubm5syODiY2DUkayPrtGmGwGNBtf8Yqf6KyP\/87bP3\/p266Eld\/MQLAtUigGQN5ENJcWVlpbn5QNIG8dF7deOuiNayHh4ebuyLq15qZr66utq8xzeuC0jWQGKp0hKBq0G79yPtf1Lkf3\/87L1PiciP1Z6yP7EUI81CIJ6A95LVFzglDYCyn\/Kkhdrb29uUompby3JpaakxA9VCHR8fbzkHHBW0Eur29nbLbkG6jaGhocTzx0iWr7w\/BPqCrqgl4fDrZZH\/+8wHkv3rPxaRt\/3pNj3xgoDXko07FxvNWtmSVTKdm5sTJdO+Pv2Pw\/mxEpWuLhF9X\/2RcHR0FPvEqrj3dT1I1ovvJ51oEPhowOFvgv8GYpWXpef07N7Y4xeviHxLfa4vfoouLYMSAnYIeCvZ8PnOi85flolez0Rv3brVWDLWe9lGz7WqckrG6+vrLTIOX4SlN5ePzopVvNEHaUT7gGTLzCp12SXw2aB5dZvOsyKPf+zs95dE5r9+9jjFxe8siKgLje\/oJWUlXD3zZQnZbv7q3bq3ks3yxCcTqQ\/PnsNyjy77Jkk2vIycJtm4ZeRwf7Rkj4\/75dEjfU5L5OTkaRPdpk4IlEjgKRG5EtR3VaTnk2c\/H78v8stPiPxq8NHLIoef+XDjl4\/I38l7S09+MJFVjv2pfgKUekiFkq\/+Xf2XRzCWmDCqSiCgBHv58n3p6bkrW1tbop5A6NLr0unp6WlSwLYlG73fNnoOtlOSDfM5Pv6MHB\/\/qw7kWF8NmqWprGXTysV9llQ++n47v6cdG\/4s7ueUz38pxC1aTP8e99+0suqz8DHRn5M+eyJyXPj36M9xv6uuPCki6jP1P\/27ei\/8s\/KqiHzosZ\/JU\/ITuRII8arclw\/LPzQ+e0bekWflbfm4\/GXj9yvfOZab\/\/rsuIXfU\/8n8v3Hfq3x+9vyrBzI2emad+QZuS9X5UEgbvXfn8pTH1wj9TMRUf9TL\/2zdrD6Xf+s\/qv\/p8qqn9WqtP487me9ah39TP0e\/iz6c9Lvqt2sn0XL6mH1C\/1DuKJw4aw\/p5XL81m0rI4v7v200wBFP0trPxxLWlzny\/X0\/C\/p6fng3m3vJKu6nHQ+MwlbGe\/rmWx0iTp6QVSnJBueyTKLLSPD1NE+AW1dVZP6WRlWW1fPYrV1nxR5LzDgyJNqytqcyT7+4vvyk8axIp+Vr8sfvf1S60xWrxYrR6uJ7C\/0TFYtH6uftVnV71HLck63\/TzXuwY1i1Uvb2eyqnNxt7+YTru+p\/UiyXLhk+lMUL9\/BJSI1ew0WC\/+l1fk1797NlP409ufOTsn+1P15Cf1UuvF0Vt+\/CNCj6pPwPtzsupRiWmvsq8uTlqm1u\/fuHHjwlt4whdEpd3CE3dBlO4rFz5V\/8tHhFkJBLPa5j2wgWR7PiVPPzgT6Y8nr4q8rs69qouj1AvBZqVLObMEkGx\/f8s9qGXgjntYhHrvjTfeaLkVh4dRlEGbOvwnEL1PVl8U9ZLI\/wsEPCoif8EmAf6PBfd66K1kbaci+ijEuMcl8lhF21mifTcIqMc5qZeepeqof1Xkz4PPlGTfWwpdFeRGz4jSfwK1l6w6d6pulenu7vYq2ywXe5XOmnfmxaD\/34xw6BPZeensvTF14dJXas6J7leRgPeS7fRjFauSZCRblUwQR\/sEgk0Azm0A8JTI7796Vv3vqPOxauN2XhCoFgGvJZvlsYq+7B8bHVZItlpfNKJpg8CHgj1j3zt7ulPLq7lp+\/dFgvtn22iJQyFQOgFvJavPd7711luNRxeqV\/iZwll2simddgcrRLIdhE1TZgn8s0Cy\/ydGsr8bfPbfvxPctmM2FGqHQF4C3ko2eitN9BYaBUotJauX3kIuL7wql0eyVc4OseUiMBiIdD9Gsr8VfPaHasP26C49uVqhMASMEKiNZOO2h4tuK2eEsKVKkawl8DRbPoF\/E4j02zGS\/Y3gs29tcG9s+eSpsQQC3ko2bl\/X6MwVyZYwgqgCAqYJ\/LdApP8+RrK\/Fnz2FzGfmY6L+iGQgYC3ktXLwRsbG6IfcRg9D3vRdnEZ+FW2CDPZyqaGwPIS+PNApP8iRqRpS8l526E8BAwQ8Fqy+vnA9+7da4i2p6dHpqamRP2uX5OTk5yTNTCwqBICpRE4DSR7KUayvxJ89rfMZEvjTUWlEvBasoqUEm34quLwk5jinsJUKl2LlTGTtQi\/8k3H7Xun96FTn0V\/1vvg6f3qor+rDkf2s\/ulJz7YWEd9pDfbUUXD28WqJySq7Y7PtoVV+9k19mV\/7in1gH+R35Y\/kPuX1ho\/f+T0N+Ur8gXZfz94dvHfiYj6n3q9LWdb14U32tF7tetNdsIb7fwiumed\/lC\/r3fhuWhvu7g97FRA4b3sKj8gCNAgAe8la5BdpatGspVOD8FlJnBVfu\/0bD\/Z\/3zp2ZBVgwrSbu\/J3AYFIWCOQK0lq2a5r732WuN\/PFbR3CCjZgi0Q+Dx0xuNw39+aeV8Nf8kWC7+KcvF7TDmWHMEvJNslgfz63tm1TZ4ZW91Zy5V+WpmJpuPF6UrTOCNQKRDMSJ9PPjs50i2whmsdWheSTbpMYrhc6\/hZxlfu3at8TSovj69lZY\/YwHJ+pPL2vfkPwQi\/a9xIp0L8KgdeHhBoHoEvJFseNs4fcuOwq3F+7WvfU2+\/e1vy+7ubiML4TLVS0v7ESHZ9hlSQ0UI\/GYg2T+Kk+wXgiDZgaci2SKMCAFvJBt9jKLuZ3hpWL3n64YA0ZGNZPmue0Mg9YETwS48ctub7tIRvwh4L9noRgE+Lg3HDUkk69cXtda9+afBTPbduJnsZwM0X681IjpfXQK1kaxKweLionR1dVU3GyVGhmRLhElVdgmkbXUnLwexsZes3STRehIBJOvp2ECynia2lt1Ku7jppYCI2oWHFwSqRwDJVi8npUSEZEvBSCWVIJC2JDwaRKj2k+UFgeoR8E6y6t7XPC\/uk81Di7IQsEHgxaDRb8Y0\/pngvT+xERhtQuBCAki2v1\/W1tZ44tOFQ4UCELBFIE2knwqC+q6t4GgXAqkEvJEseW4lwHIxI8IfAp8MuvL9mC6lfeYPAXriLgEk627uUiNHsp4mtpbdUhsDqJfaaif6Cnbkkb+qJRk6XX0CSLb6OSoUIZIthI2DKklAP\/b0ICa6jwXv\/aCSkRMUBJCsp2MAyXqa2Fp2S282e7blXevro8GvZ\/vP8oJA1Qgg2aplpKR4kGxJIKmmAgTULu\/qpXdhD4f0keAXvXt7BcIlBAiECCBZT4cDkvU0sbXs1hNBr9+P6X3aUnItYdHpihFAshVLSFnhINmySFJPtQk8E4T3TrXDJLraEkCynqYeyXqaWLoVIYBkGRLVJoBkq52fwtEh2cLoONApAleDaO87FTXB1ocAkvU010jW08TSrQiBK8HvDyADgUoSQLKVTEv7QSHZ9hlSgwsE0q48diF+YvSdAJL1NMNI1tPE0q0IgSeD338GGQhUkgCSrWRa2g8KybbPkBpcIJB2e48L8ROj7wSQrKcZRrKeJpZuRQggWYZEtQkg2Wrnp3B0SLYwOg6EAAQgUBoBJFsaympVhGSrlQ+igQAE6kkAyXqadyTraWLpFgQg4BQBJOtUurIHi2Szs6IkBCAAAVMEkKwhsvv7+zIxMdGsfXJyUmZnZ1taOzk5kfn5ednd3W2+PzIyIouLi9LV1dV8L1qX+mBzc1MGBwcTo0eyhhJLtRCAAARyEECyOWBlLaqkODc3J+vr69LXd7ZLyPLycuO\/YdGq9\/b29prlHj58KNPT0zIwMNAsd3BwIFNTUzI8PNx8b2dnR1ZXV1vqj8aGZLNmi3IQgAAEzBFAsiWz1bNTVW14RhoVrxbq+Pi4jI2NtcxaV1ZWZG1tTbq7u0UJdXt7u\/m7KqjbGBoaajk23BUkW3JiqQ4CEIBAAQJItgC0tEOySlbPUJeWllqWfaPvq9nu0dHRuSXkpPd1bEi25MRSHQQgAIECBJBsAWgXHZK0XByWZVwZVa+W7MzMjIyOjjbO2fb29p47n6skq0SqZ7wsF1+UFT6HAAQg0HkCSNYQc70cfPfu3UYLN2\/ePLcsHD1vq8qFl5HTJBu3jBy3XHx83C+PHuntwNRS89OGeky1EIAABCAQJqAEe\/nyfenpuStbW1uN621cel06PT09rWLAcbPMqBSTZrJlSzbMRwn3+Pi5KiIjJghAAAJeEejpudOQq34h2ZLSG17uDV\/QFL3QqVOSDc9kmcWWlGSqgQAEIHABATWLVS9msiUPlaQLmvQFUfr8Khc+lQye6iAAAQhUkADnZEtOSpI8ozPZtFt4wudq027hibsgSneHq4tLTizVQQACEChAAMkWgHbRIXG318TJkodRXESSzyEAAQi4TQDJGsqfkurCwkKz9rjHJfJYRUPwqRYCEIBARQgg2YokouwwWC4umyj1QQDsyGBDAAAOxUlEQVQCEMhPAMnmZ+bEEUjWiTQRJAQg4DkBJOtpgpGsp4mlWxCAgFMEkKxT6coeLJLNzoqSEIAABEwRQLKmyFquF8laTgDNQwACEBARJOvpMECyniaWbkEAAk4RQLJOpSt7sEg2OytKQgACEDBFAMmaImu5XiRrOQE0DwEIQIDlYn\/HAJL1N7f0DAIQcIcAM1l3cpUrUiSbCxeFIQABCBghgGSNYLVfKZK1nwMigAAEIIBkPR0DSNbTxNItCEDAKQJI1ql0ZQ8WyWZnRUkIQAACpgggWVNkLdeLZC0ngOYhAAEIcHWxv2MAyfqbW3oGAQi4Q4CZrDu5yhUpks2Fi8IQgAAEjBBAskaw2q8UydrPARFAAAIQQLKejgEk62li6RYEIOAUASTrVLqyB4tks7OiJAQgAAFTBJCsKbKW60WylhNA8xCAAAS4utjfMYBk\/c0tPYMABNwhwEzWnVzlihTJ5sJFYQhAAAJGCCBZI1jtV4pk7eeACCAAAQggWU\/HAJL1NLF0CwIQcIoAknUqXdmDRbLZWVESAhCAgCkCSNYUWcv1IlnLCaB5CEAAAlxd7O8YQLL+5paeQQAC7hBgJutOrnJFimRz4aIwBCAAASMEkKwRrPYrRbL2c0AEEIAABJCsp2MAyXqaWLoFAQg4RQDJOpWu7MEi2eysKAkBCEDAFAEka4qs5XqRrOUE0DwEIAABri72dwwgWX9zS88gAAF3CDCTdSdXuSJFsrlwURgCEICAEQJI1ghW+5UiWfs5IAIIQAACSNbTMYBkPU0s3YIABJwigGSdSlf2YJFsdlaUhAAEIGCKAJI1RdZyvUjWcgJoHgIQgABXF\/s7BpCsv7mlZxCAgDsEmMm6k6tckSLZXLgoDAEIQMAIASTbBtbl5eXG0bOzs+dqUZ9tbGw03+\/v75e1tTXp7u5uvndyciLz8\/Oyu7vbfG9kZEQWFxelq6ur+d7+\/r5MTEy0tLG5uSmDg4OJ0SPZNhLLoRCAAARKIoBkC4Lc2dmRhYUFmZycPCdZ\/ZkWoZapaiosUCXivb09WV9fl76+Pnn48KFMT0\/LwMBAs86DgwOZmpqS4eHh5nuq\/tXV1eZxcV1AsgUTy2EQgAAESiSAZHPCjM4+o5LVn\/f29rbIV8lybm5OlpaWWoQ6Pj4uY2NjLbPWlZWV5qxXCXV7e7tlFqzbGBoaajk23BUkmzOxFIcABCBggACSzQFVy+3w8FBu3bolSoZRmerZaFSe0ff1DFVJN7zsG31fzXaPjo7OLSEnva+7g2RzJJaiEIAABAwRQLIFwabNWNXyblSe0aVgdZ5VzWz1UrEOQ0t2ZmZGRkdHG+dsoyJXZZVklUij53mjkj0+7pdHj642e3ly8nTBHnMYBCAAAQjkIaAEe\/nyfenpuStbW1uNU4EuvS6dnp6e2go4r2Sj5ZMkG57xpkk2bhk5brk4\/J4S7vHxc7aQ0S4EIACB2hDo6bnTkKt+IdmcqXdFsuGZLLPYnEmmOAQgAIGCBNQsVr2YyRYE6IpkDw8\/Lci1YJI5DAIQgECbBDgnWxBgkmS58KkgUA6DAAQg4CEBJFswqUmSzTrDTZJx9Fxt2i08cRdE6e5wdXHBxHIYBCAAgRIJINmCMJNkqqrjYRQFoXIYBCAAAc8IINmCCU2TrKqSxyoWBMthEIAABDwigGQ9Sma4KywXe5pYugUBCDhFAMk6la7swSLZ7KwoCQEIQMAUASRriqzlepGs5QTQPAQgAAE2bfd3DCBZf3NLzyAAAXcIMJN1J1e5IkWyuXBRGAIQgIARAkjWCFb7lSJZ+zkgAghAAAJI1tMxgGQ9TSzdggAEnCKAZJ1KV\/ZgkWx2VpSEAAQgYIoAkjVF1nK9SNZyAmgeAhCAAFcX+zsGkKy\/uaVnEICAOwSYybqTq1yRItlcuCgMAQhAwAgBJGsEq\/1KqyZZNdDU5sXHx8\/Zh1OxCGCTnBDYwCbv17VqYwbJ5s2gI+WrKNnr1\/9M2ET+\/ADSX0LYwCbPPy+Mm3haVeOCZPOMaofKaskeH\/fLo0dXrUeuZrE9PXelKvFYBxIKADbJ2YANbPJ+V6s2ZnQ8W1tbMjAwkLc7VstfOj09PbUaQcUbf+WVV0TJlhcEIAABCNgjoOT6+c9\/HsnaS4GZlhGsGa7UCgEIQCAPAddmsLpvzGTzZJmyEIAABCAAgRwEkGwOWBSFAAQgAAEI5CGAZPPQoiwEIAABCEAgBwEkmwMWRSEAAQhAAAJ5CCDZPLQoCwEIQAACEMhBAMnmgEVRCEAAAhCAQB4CSDYPLcpCAAIQgAAEchBAsjlgURQCEIAABCCQhwCSzUOrImWXl5dlY2OjGc3k5KTMzs5WJDr7YTx8+FCmp6flxo0bMjg4aD8gSxHs7+\/LxMRES+ubm5u1ZhKXCvV9Ui++Q2d0ouOmv79f1tbWpLu729JIdrtZJOtQ\/k5OTmR+fl4ODw+bg14L5fr167K4uChdXV0O9aj8UDWj3d1dqbNQDg4OZGpqSoaHh5vy2NnZkdXVVVlfX5e+vr7y4TtYo2KysLAg\/KF6ljzNI\/zdUX+E7O3tMW4Kjm8kWxCcjcP0P5xLS0stsxH1l+fc3FztvwSaz7179xrpqbNk1T+W29vbLTMQ\/QfI0NCQjI2N2RjClWkz\/MeYCgrJimgmvb29LbN6\/Yf8+Ph47cdNkQGMZItQq9gxenmnzlIJz9xeeOGFxiwu+sdIxdJmNBw1+zg6Ojq3upH0vtFgKlZ5eEXo1q1bsrKyIlGxVCxkq+FoyapnB7Oknj8VSDY\/s8odwTJga0qSZvyVS5yhgJJmJKo5JVm16QXn2M7gp7EylB7nqtXfp5mZGWayBbKHZAtAq9Ih\/JV5PhtI9uzcfdzsLG4ZuUrjudOxINmLifOH2cWM0kog2fb4WT+aLwCSjRJIEweSbaWFZNP\/CeNUVPv\/xCPZ9hmWXkP0Ah7VwM2bN88t1dTtqr+sXJjJMpPN+qVEssmktGDj\/u3JypdyIkjW0VFQN8HmSVPdJavPvXLh08WjBsnGM0KwF4+drCWQbFZSFSmnBXLlyhUuXknICZI9u98x6RYerqT9YOAg2fjl87rfZ17mP\/dItkyahuvSFzk9ePCg9vfEpqFGsiI8jCLblxHJtnLST5NjiTjb+MlSCslmoVSRMvppLEnh8MU4I4NkzzjwWMWLv7hI9gNGcdc8hAmOjIzwVLmLh9S5Eki2ADQOgQAEIAABCGQhgGSzUKIMBCAAAQhAoAABJFsAGodAAAIQgAAEshBAslkoUQYCEIAABCBQgACSLQCNQyAAAQhAAAJZCCDZLJQoAwEIQAACEChAAMkWgMYhEIAABCAAgSwEkGwWSpSBAAQgAAEIFCCAZAtA4xAIQAACEIBAFgJINgslynhL4Itf\/KJkfU6rfuLW5uamDA4OZmJieivCTm6orR\/reePGjcz9zwJJP3VpaGgo86bgiqt6zc7OZmmCMhCwRgDJWkNPw1UgoB89ODk5mfoPthaMinltbU26u7szhW9DsuqPgdXV1dKfb21KbOoPhbm5OVlaWpK+vr5MXE0JP1PjFIJADgJINgcsivpHQM+iDg8PU+VZdOsv05KNy4iJbRCLiDDraFF\/FLzxxhu5n4tb9LiscVEOAmUQQLJlUKQOpwnoZeC0DRaKissXyZqaxaqBU7RuNoJw+mtXm+CRbG1STUeTCOilx+vXr8fOpuK2jdNy2NjYaFYbJ+kkyeotxfTBSTucRHfSibYRPic7Ojoq8\/PzjXPM+qWWwdVrb28vdvk4yx8BSTILL0u\/9dZbsrCw0Ggr3JdwP+P6GLfsG+3ztWvXEpe+Vf1xm9Mz2iFQFQJItiqZIA6rBNJmqtFznHqJOSyzsNTCF+NEJZZ2bH9\/f8uSddLWhmHRZpHsCy+8IFNTUzIzM9NyYZEW3MDAQOr56KRzvPr9T3ziE\/KNb3yjJX9a7uE\/QlSB6LlvJdSVlZVmv+O259MVx11wZur8s9XBSONeEUCyXqWTzhQlkDRb1VJU9S4uLkpXV5do+UWFoWdtYRlEJauPjc7q9LG6Ti3ABw8eNGdxcTPuuKuLo38wJJ131kK76GrppNmi7kt4pqljvHv3roT\/aNBxXrly5dwfEuHzsXF\/7CTlRuUkax+KjguOg0C7BJBsuwQ53hsCcUun0X\/E46SrAcRtAB6uUwlaLefGXWQVFeEPf\/hDmZiYkLTzxKrdLJJNklGWpeK0\/iady0661SlpVh++dUf\/sZG2RBwecJ28hcmbgU5HOkoAyXYUN41VmUDcrCgqhvBMLakv0XOSb775ZmP2piUbnhWH6wgvfR4fHzcke9EsM6tko0vDWZeK08olyTRpCTfKMu6KZd2fe\/funVt+jrsnNms\/qjzuiM1vAkjW7\/zSuxwEorO2d99999y5TFclqzAUkbhJyUbPx0b\/4NAXUun30y6cuui8co5hQFEIlEoAyZaKk8pcJ6D+4VcPRlhfXxd1xez29nbLOcS05dO4vldluTi6tPzOO++InmGnPVgjy3JxdLaddSab9dYdHYPKh8pL+IEVLBe7\/o3zP34k63+O6WEOAnrmpm6HuXPnjvT29p678jbpwqe4p0eVeeFTnGyyLhcrBPr4H\/3oR6KWo4eHhzM9ljDp3G07y8UqnunpaQk\/ojFNpklXf3PhU47BTVErBJCsFew0WmUCcVfNhuNNWzKOXrCT5xae6LFJt\/CEr2qOk2z4uLhbZtS5XvW66Hyv7nPaLTxqSbfITFYxjHuUYtotPHGPvuQWnip\/k4hNEUCyjAMIRAhocT3\/\/POpj\/rL8kCJTj6MYmxsrNGT8B8B0fOYFz14I24wJC3JtjOT\/d73vpf4KMW4i5\/irrLOu3TPQIeADQJI1gZ12oSAJQJFz2FW8clKPFbR0iCi2VwEkGwuXBSGgNsE0p5sldYzkxsEFCXKBgFFyXFcJwkg2U7Spi0IWCAQfZTjRdv6JYWY9WrgTnSRre46QZk2yiCAZMugSB0QqDCBsGSTNiLIEn6VxFYl4WdhR5n6EkCy9c09PYcABCAAAcME\/j91TSUhMMvzqgAAAABJRU5ErkJggg==","height":258,"width":344}}
%---
%[output:893bbd25]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"在当前文件夹或 MATLAB 路径中未找到 ca_cfar，但它存在于:\n E:\\code\\matlab\\空中雷达\\MIMO_FMCW\\FMCW-MIMO-Radar-Simulation\n\n<a href = \"matlab:internal.matlab.desktop.commandwindow.executeCommandForUser('cd ''E:\\code\\matlab\\空中雷达\\MIMO_FMCW\\FMCW-MIMO-Radar-Simulation''')\">更改 MATLAB 当前文件夹<\/a> 或 <a href = \"matlab:internal.matlab.desktop.commandwindow.executeCommandForUser('addpath ''E:\\code\\matlab\\空中雷达\\MIMO_FMCW\\FMCW-MIMO-Radar-Simulation''')\">将其文件夹添加到 MATLAB 路径<\/a>"}}
%---
