clear; clc; close all
%% Radar parameters
c = physconst('LightSpeed'); %speed of light
BW = 50e6; %bandwidth
fc = 3000e6; % carrier frequency
numADC = 1000; % # of adc samples
numChirps = 256; % # of chirps per frame

numCPI = 10;
T = 10e-3; % PRI 
PRF = 1/T;
F = numADC/T; % sampling frequency
dt = 1/F; % sampling interval
slope = BW/T;
lambda = c/fc;



N = numChirps*numADC*numCPI; % total # of adc samples
t = linspace(0,T*numChirps*numCPI,N); % time axis, one frame
t_onePulse = 0:dt:dt*numADC-dt;

% Ny=4;
Ny=8;
numTX = 1;
numRX = 8;

N_L=floor(2*fc/BW);
(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L) % 扩展的虚拟阵列 %[output:76d6e724]
N_L=1;

d_y=lambda/2/2;
d_tx= lambda/2;
d_rx = numRX*d_tx; % dist. between rxs
% d_tx = 4*d_rx; % dist. between txs

% tr_vel=-d_tx/2/T;
tr_vel=0;
%%

%% Targets

r1_radial = 2900;
tar1_theta = 40;
tar1_phi= 70;
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

scatter3(tar1_loc(1,1),tar1_loc(1,2),tar1_loc(1,3),'r','filled') %[output:7c807e23]
hold on %[output:7c807e23]

r2_radial = 600;
tar2_theta = 30;
tar2_phi = 60;
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

scatter3(tar2_loc(1,1),tar2_loc(1,2),tar2_loc(1,3),'blue','filled') %[output:7c807e23]
xlabel('x'); %[output:7c807e23]
ylabel('y'); %[output:7c807e23]
zlabel('z'); %[output:7c807e23]
hold off %[output:7c807e23]

%%
tx_loc = cell(numTX,Ny);
tx_loc_t=cell(numTX,Ny);
for j=1:1
    for i = 1:numTX
        % tx_loc{i,j} = [(i-1)*d_tx (j-1)*d_y 0];
        tx_loc{i,j} = [0 0 0];

        tx_loc_t{i,j}=zeros(length(t),3);
        tx_loc_t{i,j}(:,1)=tr_vel*t+tx_loc{i,j}(1);
        tx_loc_t{i,j}(:,2)=tx_loc{i,j}(2);
        tx_loc_t{i,j}(:,3)=tx_loc{i,j}(3);
    
        scatter3(tx_loc{i,j}(1),tx_loc{i,j}(2),tx_loc{i,j}(3),'b','filled') %[output:28c0a39d]
       hold on
    end
end


rx_loc = cell(numRX,Ny);
rx_loc_t=cell(numRX,Ny);
for j = 1 : 2:Ny %[output:group:7f3f1511]
    for i = 1 :2: numRX
        % rx_loc{i,j} = [tx_loc{numTX,1}(1)+d_tx+(i-1)*d_rx (j-1)*d_y 0];
        rx_loc{i,j} = [(i-1)*d_rx-((numRX-1)*d_rx)/2 (j-1)*d_y-((Ny-1)*d_y)/2 -5];
        rx_loc_t{i,j}=zeros(length(t),3);
        rx_loc_t{i,j}(:,1)=tr_vel*t+rx_loc{i,j}(1);
        rx_loc_t{i,j}(:,2)=rx_loc{i,j}(2);
        rx_loc_t{i,j}(:,3)=rx_loc{i,j}(3);
        scatter3(rx_loc{i,j}(1),rx_loc{i,j}(2),rx_loc{i,j}(3),'r','filled') %[output:28c0a39d]
        hold on
    end
end %[output:group:7f3f1511]






xlabel('x');
ylabel('y');
zlabel('z');



plane_loc=cell(numTX*numRX,Ny);
for i =1:numTX
    for k = 1:numRX
        for j = 1:Ny
            plane_loc{i+(k-1)*numTX,j}=rx_loc{k,j}+tx_loc{i,j};
            scatter3(plane_loc{i+(k-1)*numTX,j}(1),plane_loc{i+(k-1)*numTX,j}(2),plane_loc{i+(k-1)*numTX,j}(3),'b','filled')
            hold on
        end
    end
end
hold off
% plane_loc_plus=cell(numTX*numRX+N_L-1,Ny);
% for i =1:numTX*numRX+N_L-1
%     for j=1:Ny
%         if i<=(N_L-1)/2           
%              plane_loc_plus{i,j}(1)=plane_loc{i+(N_L-1)/2,j}(1)-d_tx*((N_L-1)/2+1-i);    
%         end
%     end
% end
%%
P1 = rx_loc{1,1};
P2 = r1;


% 计算两点之间的距离
d = sqrt((P2(1) - P1(1))^2 + (P2(2) - P1(2))^2 + (P2(3) - P1(3))^2);

% 计算仰角
theta = asin((P2(3) - P1(3)) / d); % 仰角，单位为弧度

% 计算方向角
phi = atan2(P2(2) - P1(2), P2(1) - P1(1)); % 方向角，单位为弧度

% 如果需要，可以将弧度转换为度
theta_deg = rad2deg(theta);
phi_deg = rad2deg(phi);

% 输出结果
fprintf('仰角 (弧度): %.4f\n', theta);
fprintf('方向角 (弧度): %.4f\n', phi);
fprintf('仰角 (度): %.4f\n', theta_deg);
fprintf('方向角 (度): %.4f\n', phi_deg);

%%
%% TX

delays_tar1 = cell(numTX,numRX,Ny);
delays_tar2 = cell(numTX,numRX,Ny);
for k = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            delays_tar1{i,j,k} = (vecnorm(tar1_loc-rx_loc_t{j,k},2,2) ...
                            +vecnorm(tar1_loc-tx_loc_t{i,k},2,2))/c; 
            delays_tar2{i,j,k} = (vecnorm(tar2_loc-rx_loc_t{j,k},2,2) ...
                            +vecnorm(tar2_loc-tx_loc_t{i,k},2,2))/c;
        end
    end
end


% r1_at_t = cell(numTX,numRX);
% r2_at_t = cell(numTX,numRX);
% tar1_angles = cell(numTX,numRX);
% tar2_angles = cell(numTX,numRX);
% tar1_velocities = cell(numTX,numRX);
% tar2_velocities = cell(numTX,numRX);
%%

%% Complex signal
phase = @(tx,fx) 2*pi*(fx.*tx+slope/2*tx.^2); % transmitted
snr=20;

mixed = cell(numTX,numRX,Ny);
% for l = 1:Ny
%     for i = 1:numTX
%         for j = 1:numRX
%             % disp(['Processing Channel: ' num2str(j) '/' num2str(numRX)]);
%             for k = 1:numChirps*numCPI
%                 phase_t = phase(t_onePulse,fc);
%                 phase_1 = phase(t_onePulse-delays_tar1{i,j,l}(k*numADC),fc); % received
%                 phase_2 = phase(t_onePulse-delays_tar2{i,j,l}(k*numADC),fc);
% 
%                 signal_t((k-1)*numADC+1:k*numADC) = exp(1j*phase_t);
%                 signal_1((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_1));
%                 signal_2((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_2));
%             end
%             % mixed{i,j,l} = signal_1 + signal_2;
%             % mixed{i,j,l} = signal_1 ;
%             mixed{i,j,l} =awgn(signal_1+ signal_2,snr,'measured');
%         end
%     end
% end

% 假设变量已定义: numTX, numRX, Ny, numChirps, numCPI, numADC, fc, t_onePulse, delays_tar1, delays_tar2, snr

numSamplesPerChirp = numADC;
numChirpsTotal = numChirps * numCPI; % 总的chirp/CPI段数
numSamplesTotal = numChirpsTotal * numSamplesPerChirp; % 每个通道的总样本数

mixed = cell(numTX, numRX, Ny); % 预分配结果单元数组

% 确保 t_onePulse 是一个行向量，以便进行广播/隐式扩展
t_onePulse_row = reshape(t_onePulse, 1, numSamplesPerChirp);

% --- 外层循环仍然保留 ---
for l = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            % --- 向量化内部计算 ---

            % 获取当前通道对应的延迟向量 (假设长度为 numChirpsTotal)
            % 注意：如果原始索引 k*numADC 有特殊含义，需要相应调整这里的索引
            current_delays1 = delays_tar1{i, j, l}(1:numChirpsTotal); % 取前 numChirpsTotal 个延迟
            current_delays2 = delays_tar2{i, j, l}(1:numChirpsTotal); % 取前 numChirpsTotal 个延迟

            % 将延迟向量变为列向量，以便与行向量 t_onePulse_row 进行隐式扩展
            current_delays1_col = reshape(current_delays1, numChirpsTotal, 1);
            current_delays2_col = reshape(current_delays2, numChirpsTotal, 1);

            % --- 计算信号 ---
            % 利用隐式扩展 (implicit expansion, R2016b+) 或 bsxfun (旧版本)
            % 计算每个 chirp 段的基准相位 (对所有 chirp 都相同)
            % phase_t_matrix 的维度将是 [numChirpsTotal, numSamplesPerChirp]
            phase_t_matrix = phase(t_onePulse_row, fc); % 实际上 phase_t 对每个chirp段内部点是一样的

            % 计算每个 chirp 段接收信号的相位 (考虑延迟)
            % t_onePulse_row (1 x N_adc) - current_delays1_col (N_chirp x 1)
            % 结果维度是 [numChirpsTotal, numSamplesPerChirp]
            phase_1_matrix = phase(t_onePulse_row - current_delays1_col, fc);
            phase_2_matrix = phase(t_onePulse_row - current_delays2_col, fc);

            % 计算相位差并得到复信号矩阵
            % phase_t_matrix 会自动扩展以匹配 phase_1_matrix 和 phase_2_matrix 的维度
            signal_1_matrix = exp(1j * (phase_t_matrix - phase_1_matrix));
            signal_2_matrix = exp(1j * (phase_t_matrix - phase_2_matrix));

            % 将信号矩阵按列优先顺序（MATLAB默认）展开成长向量
            % 需要转置使其按行填充，再reshape
            signal_1 = reshape(signal_1_matrix.', numSamplesTotal, 1);
            signal_2 = reshape(signal_2_matrix.', numSamplesTotal, 1);

            % --- 组合信号并添加噪声 ---
            % 根据原始代码的注释，选择是加和还是只用 signal_1
            % mixed_signal = signal_1 + signal_2;
            mixed_signal = signal_1+signal_2;

            % 添加 AWGN 噪声
            mixed{i, j, l} = awgn(mixed_signal, snr, 'measured');

        end % 结束 j 循环
    end % 结束 i 循环
end % 结束 l 循环

% --- 清理临时变量 (可选) ---
clear t_onePulse_row phase_t_matrix phase_1_matrix phase_2_matrix signal_1_matrix signal_2_matrix signal_1 signal_2 mixed_signal current_delays1 current_delays2 current_delays1_col current_delays2_col numSamplesPerChirp numChirpsTotal numSamplesTotal;


%%
% 
% figure
% subplot(3,1,1)
% p1 = plot(t, real(signal_t));
% title('TX')
% xlim([0 0.1e-1])
% xlabel('Time (sec)');
% ylabel('Amplitude');
% subplot(3,1,2)
% p2 = plot(t, real(signal_1));
% title('RX')
% xlim([0 0.1e-1])
% xlabel('Time (sec)');
% ylabel('Amplitude');
% subplot(3,1,3)
% p3 = plot(t,real(mixed{1,1,1}));
% title('Mixed')
% xlim([0 0.1e-1])
% xlabel('Time (sec)');
% ylabel('Amplitude');
%%

%% Post processing - 2-D FFT



RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX*Ny); % radar data cube


RDC_plus=RDC;

% numChirps_new=(numChirps-(N_L-1));
numChirps_new=numChirps;
DFmax = 1/2*PRF; % = Vmax/(c/fc/2); % Max Unamb Dopp Freq
dR = c/(2*BW); % range resol
Rmax = F*c/(2*slope); % TI's MIMO Radar doc
Rmax2 = c/2/PRF; % lecture 2.3

dV = lambda/(2*numChirps*T); % velocity resol, lambda/(2*framePeriod)
Vmax = lambda/(T*4); % Max Unamb velocity m/s

N_Dopp = numChirps_new; % length of doppler FFT
N_range = numADC; % length of range FFT
N_azimuth = (numRX*numTX+N_L-1)*Ny;
R = 0:dR:Rmax-dR; % range axis
V = linspace(-Vmax, Vmax, numChirps_new); % Velocity axis


ang_phi = -180:0.1:180; % angle axis
ang_theta=0:0.1:90;


RDMs = zeros(numADC,numChirps_new,(numRX*numTX+N_L-1)*Ny,numCPI);
for i = 1:numCPI
    RD_frame = RDC_plus(:,(i-1)*numChirps_new+1:i*numChirps_new,:);
    RDMs(:,:,:,i) = fftshift(fft2(RD_frame,N_range,N_Dopp),2);
    % RDMs(:,:,:,i) = RD_frame;
end




figure
imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1))))));
colormap(jet(256))
% set(gca,'YDir','normal')
clim = get(gca,'clim');
caxis([clim(1)/2 0])
xlabel('Velocity (m/s)');
ylabel('Range (m)');
%%

%% CA-CFAR

numGuard = 2; % # of guard cells
numTrain = numGuard*2; % # of training cells
P_fa = 1e-5; % desired false alarm rate 
SNR_OFFSET = -5; % dB
RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

[RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
cfar_ranges_real=(cfar_ranges-1)*3;
figure
h=imagesc(V,R,RDM_mask);
xlabel('Velocity (m/s)')
ylabel('Range (m)')
title('CA-CFAR')
%%

%% Angle Estimation - FFT

rangeFFT = fft(RDC_plus(:,1:numChirps_new,:),numADC);

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

    % 
    % doas(1,:) = fftshift(fft(rangeFFT(cfar_ranges(1),cfar_dopps(1),:),length(ang_phi)));
    % plot(ang_phi,10*log10(abs(doas(1,:))))
% figure
% hold on; grid on;
% for i = 1:K
%     plot(ang_ax,abs(doas(i,:)))
% end
% xlabel('Azimuth Angle')

%%

%% Angle Estimation - MUSIC Pseudo Spectrum


M = numCPI; % # of snapshots
% a1=zeros((numRX*numTX+N_L-1)*Ny,length(ang_theta)*length(ang_phi));
% for k = 1:numRX*numTX+N_L-1
%     for l = 1:Ny
%        for i=1:length(ang_theta)
%             for j=1:length(ang_phi)
%                 % a1(k+(l-1)*(numRX*numTX+N_L-1),i,j ...
%                 %     )=exp(-1i*2*pi*(d_y*(l-1)*cos(ang_phi(j).'*pi/180)*sin(ang_theta(i).'*pi/180) ...
%                 %     +d_tx*(k-1)*sin(ang_phi(j).'*pi/180)*sin(ang_theta(i).'*pi/180)));
%                 a1(k+(l-1)*(numRX*numTX+N_L-1),i+(j-1)*length(ang_theta) ...
%                     )=exp(-1i*2*pi/lambda*(d_y*2*(l-1)*sind(ang_phi(j))*sind(ang_theta(i)) ...
%                                    +d_tx*(k-1)*cosd(ang_phi(j))*sind(ang_theta(i)) ) );
%             end
%        end
%     end
% end
%%
% 参数计算
N_theta = length(ang_theta);
N_phi = length(ang_phi);

% 提取所有天线位置坐标（从plane_loc中提取）
numAntennas = numTX * numRX; % 总天线数
M_num = numAntennas * Ny; % 总行数：天线数×快拍数
all_coords = zeros(M_num, 3); % 存储所有坐标点[x,y,z]

row = 1;
for j = 1:Ny
    for i_antenna = 1:numAntennas
        % 获取当前天线在j快拍的位置
        current_coord = plane_loc{i_antenna, j};
        all_coords(row, :) = current_coord;
        row = row + 1;
    end
end

% 生成所有角度组合 (theta, phi)
[theta_grid, phi_grid] = ndgrid(ang_theta, ang_phi);
theta_all = theta_grid(:);
phi_all = phi_grid(:);

% 计算波数向量（假设theta为俯仰角，phi为方位角）
kx = sind(theta_all) .* cosd(phi_all); % x方向分量
ky = sind(theta_all) .* sind(phi_all); % y方向分量
kz = cosd(theta_all);                 % z方向分量
k_matrix = [kx, ky, kz]; % 组合成N×3矩阵（N=N_theta*N_phi）

% 计算相位贡献：坐标与波数向量的点积
phase_contribution = all_coords * k_matrix.'; % M×N矩阵

% 计算复数相位并生成导向矩阵
% lambda = ...; % 确保此处已定义波长lambda
ph = (-1i * 2 * pi / lambda) .* phase_contribution;
a1 = exp(ph);

%%
% % 参数计算
% nn = numRX * numTX + N_L - 1;
% Ny_val = Ny;
% N_theta = length(ang_theta);
% N_phi = length(ang_phi);
% 
% % 生成所有角度组合 (i,j)
% [theta_grid, phi_grid] = ndgrid(ang_theta, ang_phi);
% theta_all = theta_grid(:);
% phi_all = phi_grid(:);
% 
% % 计算列相关的项 (M = Nθ*Nφ)
% col_term_A = sind(phi_all) .* sind(theta_all);
% col_term_B = cosd(phi_all) .* sind(theta_all);
% 
% % 生成所有k和l的组合 (N = nn*Ny_val)
% [k_mesh, l_mesh] = ndgrid(0:(nn-1), 0:(Ny_val-1));
% k_vals = k_mesh(:);
% l_vals = l_mesh(:);
% 
% % 计算行相关的项
% row_term_A = d_y * 2 .* l_vals;
% row_term_B = d_tx .* k_vals;
% 
% % 矩阵相乘计算相位
% A_matrix = row_term_A * col_term_A.';
% 
% B_matrix = row_term_B * col_term_B.';
% phase = (-1i * 2 * pi / lambda) .* (A_matrix + B_matrix);
% 
% % 指数运算得到最终的a1矩阵
% a1 = exp(phase);

%%
music_spectrum=zeros(K,length(ang_theta),length(ang_phi));
% 预计算参数
N_theta = length(ang_theta);
N_phi = length(ang_phi);
total_angles = N_theta * N_phi;
D = (numRX*numTX+N_L-1)*Ny;  % 协方差矩阵维度
noise_sub_dim = D - 1;       % 噪声子空间维度

% 预生成角度映射表
[theta_map, phi_map] = ndgrid(1:N_theta, 1:N_phi);
angle_indices = theta_map + (phi_map-1)*N_theta;

% 优化后的主循环
for i = 1:K
    % 矩阵化协方差计算
    B = reshape(RDMs(cfar_ranges(i), cfar_dopps(i), :, :), [], M);
    Rxx = (B * B') / M;
    
    % 特征分解优化
    [Q, D] = eig(Rxx, 'vector');
    [~, order] = sort(D, 'descend');
    Qn = Q(:, order(2:end));  % 直接获取噪声子空间
    
    % 批量计算所有角度组合
    QnH = Qn' * a1;  % 矩阵预乘
    denominator = sum(abs(QnH).^2, 1);
    numerator = sum(abs(a1).^2, 1);
    
    % 结果重塑
    music_spectrum(i,:,:) = reshape(numerator ./ denominator, N_theta,N_phi);
end


%%

spectrum_data = 10 * log10(abs(reshape(music_spectrum(1,:,:), length(ang_theta), length(ang_phi))));

% 找到最大值及其索引
[max_value, max_index] = max(spectrum_data(:));
[row, col] = ind2sub(size(spectrum_data), max_index);

% 获取对应的角度
theta_max = ang_theta(row);
phi_max = ang_phi(col);

% reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi));
figure;
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(1,:,:)))));
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value);
xlabel('方位角');ylabel('仰角');
zlabel('空间谱/db');
grid;
% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);
% fprintf('修正点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max-2, phi_max-2, max_value);
%%

spectrum_data = 10 * log10(abs(reshape(music_spectrum(2,:,:), length(ang_theta), length(ang_phi))));

% 找到最大值及其索引
[max_value, max_index] = max(spectrum_data(:));
[row, col] = ind2sub(size(spectrum_data), max_index);

% 获取对应的角度
theta_max = ang_theta(row);
phi_max = ang_phi(col);
% reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi));

figure;
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(2,:,:)))));
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value);
xlabel('方位角');ylabel('仰角');
zlabel('空间谱/db');
grid;


% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);
% fprintf('修正点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max-2, phi_max-2, max_value);

%%
RDC_plus_rot_frame=RDC_plus(:,1:numChirps_new,:);

M = numCPI; % # of snapshots

A = reshape(squeeze(RDC_plus_rot_frame(:,1:100,:)),numADC*100,(numRX*numTX+N_L-1)*Ny);

Rxx = (A'*A);



[Q,D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
[D, I] = sort(diag(D),'descend');
Q = Q(:,I); % Sort the eigenvectors to put signal eigenvectors first
Qs = Q(:,1); % Get the signal eigenvectors
Qn = Q(:,6:end); % Get the noise eigenvectors
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
ang_phi_shifted = mod(ang_phi  +180+180, 360) - 180; 

%% 数据与坐标对齐
% 获取修正后坐标的排序索引（保证单调性）
[ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);

% 对频谱数据按新方位角排序（沿phi轴重排）
spectrum_shifted = squeeze(music_spectrum_2D); 
spectrum_shifted = spectrum_shifted(:, sort_idx); 

% reshape(music_spectr2um(1,:,:),length(ang_theta),length(ang_phi));
figure;
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi_sorted,ang_theta,10*log10(abs(spectrum_shifted)));
 
xlabel('方位角');ylabel('仰角');
zlabel('空间谱/db');
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
%% 改进版抛物线插值函数（带边界检查）
function [refined_angle, is_valid] = safe_parabolic_interpolation(angles, spectrum_values)
    % 输入参数检查
    if numel(spectrum_values) < 3
        refined_angle = angles(round(numel(angles)/2));
        is_valid = false;
        return;
    end
    
    % 三点抛物线插值核心算法
    y1 = spectrum_values(1);
    y2 = spectrum_values(2);
    y3 = spectrum_values(3);
    denominator = y1 - 2*y2 + y3;
    
    % 防止分母过小导致数值不稳定
    if abs(denominator) < 1e-6
        refined_angle = angles(2);
        is_valid = false;
    else
        delta = 0.5 * (y1 - y3) / denominator;
        refined_angle = angles(2) + delta * (angles(2)-angles(1));
        is_valid = true;
    end
end

%% 改进后的峰值处理流程
% 初始化存储精确角度的矩阵
peaks_info = [peaks_info, zeros(size(peaks_info,1), 2)];

% 对每个检测到的峰值进行亚网格插值
for i = 1:size(peaks_info,1)
    row = peaks_info(i,1);
    col = peaks_info(i,2);
    
    %% 仰角(theta)方向插值 --------------------------------
    % 动态调整索引范围确保3个点
    theta_start = max(1, row-1);
    theta_end = min(length(ang_theta), row+1);
    if (theta_end - theta_start) < 2 % 边界补偿
        if theta_start == 1
            theta_end = min(theta_start+2, length(ang_theta));
        else
            theta_start = max(1, theta_end-2);
        end
    end
    
    theta_indices = theta_start:theta_end;
    theta_vals = ang_theta(theta_indices);
    spectrum_theta = smoothed_spectrum(theta_indices, col);
    
    % 执行安全插值
    [refined_theta, theta_valid] = safe_parabolic_interpolation(theta_vals, spectrum_theta);
    
    %% 方位角(phi)方向插值 --------------------------------
    % 动态调整索引范围确保3个点
    phi_start = max(1, col-1);
    phi_end = min(length(ang_phi_sorted), col+1);
    if (phi_end - phi_start) < 2 % 边界补偿
        if phi_start == 1
            phi_end = min(phi_start+2, length(ang_phi_sorted));
        else
            phi_start = max(1, phi_end-2);
        end
    end
    
    phi_indices = phi_start:phi_end;
    phi_vals = ang_phi_sorted(phi_indices);
    spectrum_phi = smoothed_spectrum(row, phi_indices);
    
    % 执行安全插值
    [refined_phi, phi_valid] = safe_parabolic_interpolation(phi_vals, spectrum_phi);
    
    %% 结果整合 ------------------------------------------
    if theta_valid && phi_valid
        peaks_info(i,4:5) = [refined_theta, refined_phi];
    else % 插值失败时使用原始网格值
        peaks_info(i,4:5) = [ang_theta(row), ang_phi_sorted(col)];
    end
end

%% 结果输出（示例）
if size(peaks_info, 1) >= 1
    fprintf('精确仰角: %.3f°, 精确方位角: %.3f°\n', peaks_info(1,4), peaks_info(1,5));
end

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
% rangeFFT = fft(RDC_plus);
% for i = 1:N_range
%     Rxx = zeros((numRX*numTX+N_L-1)*Ny,(numRX*numTX+N_L-1)*Ny);
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
% 
% %% Angle Estimation - Compressed Sensing
% 
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
% end
% 
%%
% figure
%  grid on;
% mesh(ang_phi,ang_theta,10*log10(abs(P)))
% title('Angle Estimation with Compressed Sensing')
% xlabel('Azimuth')
% ylabel('dB')
% grid;
% % 找到最大值及其索引
% [max_value, max_index] = max(P(:));
% [row, col] = ind2sub(size(spectrum_data), max_index);
% 
% % 获取对应的角度
% theta_max = ang_theta(row);
% phi_max = ang_phi(col);
% 
% % 输出结果
% fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);

%%


%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":22.5}
%---
%[output:76d6e724]
%   data: {"dataType":"not_yet_implemented_variable","outputData":{"columns":"1","name":"ans","rows":"1","value":"119"},"version":0}
%---
%[output:7c807e23]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOsAAACwCAYAAADuQ5nhAAAAAXNSR0IArs4c6QAAGZhJREFUeF7tnU+MF8USx3tPyktYDSi66AIxyA3zvODjQvTqHjwJJzgYCBfIu0AwgURN2AQiN4nJBuIBvOyadwPeUT0RuZBwc9fD8t8\/YFCSB554+Q7Uz975zUz\/maqent+vJtksy\/TMdFf1Z6q7urpm4smTJ0+MHioBlUD2EphQWLPXkVZQJVBIQGHVjqAS6IkEFNaeKEqrqRJQWLUPqAR6IgGFtSeK0mqqBBRW7QMqgZ5IQGHtiaK0mioBhVX7gEqgJxJQWHuiKK2mSkBh1T6gEuiJBBTWnihKq6kSUFi1D8hK4NNPjfnss6fPePfdp78\/+eTvf8s+faTurrCOlDrTN+bx48cGPy+++OLKh3\/3nTHvvVdfoW+\/VWAD1aWwBgpMi6+UAED9+eefzauvvmqef\/75pyddoNItdMNXUHdSWIPENb6F6yxoJaywqADWdWA4jGGyHl4SUFi9xKSFKqE0phgCD1lWX1ghVrWu3p1LYfUW1XgXDIJ1YsJfWAqrt6wUVm9RjXdBEVjhHYajSQ8vCSisXmLSQkGw2ss1TaJTWIM6lsIaJK7xLRwEK8TkMxTW5ZugDqWwBolrfAsHw+pavlFQgzuTwhossvG8oAzrgwcPCkH88ssv5v79++a1114zr7zyyt9rrTgJYCl6iZZydLkmugMprNGiG68LAevi4qJZt25dsVyDg4IgfvvtNzM5OWmQ1Rb\/h2imQYDEeIlJtLUKq6h4+31zCoTA7z\/++MPcu3fPbN682Tz33HOD8ELb4qK1sLj4P4WWX\/cKK79MWe5YG3PLcvf6m9Dwln6jJCwl1WdFWGFNUATKKrT8ilJY+WXKcsc6hw7LzUs3KQMKq2j\/oHiwg+nZNQotn8YUVj5Zst5JElaykmXrSVa0qiExsNJ91NLydA2FlUeO7HfhhrVueNsEqN2oNrAqtDzdQ2HlkSP7XThgpSGo7b0tD299K84BqyS0Xc3xfeXHUU5h5ZCiwD1iYI0Z3vpWnRNWCWhj5OXb9lzKKay5aKJUD9\/O13Z469t8CVg5ofWVl297cyynsOaolQbvK6rLObz1bb4krBzQKqy+mtRy7BKwOx\/NOUO8t9wVSgFrG2gVVm6N6\/28JYCYW\/wgWgg\/OCgp2VByMu+7xhdMCWsMtAprvG71ykAJkHOIfv\/111\/m4cOHZnp62rzwwgudx9p2AWsItAprYIfrc\/EuXP9NziEssQzlNupQwF3C6gOtwtph50j96FTK9vXepqqPr5xzgLUJWpzL6eXmK9eQcmPhDb5y5Yr54YcfzMGDB2tlIwVHeXiLClBgQtPcU6o+IZ3DLpsTrFXQTkxMFPHLGzdu7HzKECtj13VjA+vu3bvN+fPnzbZt2yplwgmHr\/VsUg5nfVydwOd8jrDa0MIZd+vWLfP6668Pb4L3aWAPyojC+s0335hjx44NxHD8+HHz4YcfDv4un5+ZmTGzs7Nm1apVRZmffvrJ7Nu3z9y5c6f4e\/369ebMmTPFnkqf8\/QgWFZpWOsAJSsa2hcU1jCJQV7Xr18vrOqoboIXgxUgLiwsmLm5ObNmzRrz+++\/m\/3795udO3cWwBKIJ06cMO+8887Q+UePHpmjR4+aqakpc\/jw4UJzn3\/+ubl7924BNI6m8wQ8yknAGju89e2CCquvpJ6Ws+WFv0dxa54IrATa9u3bhyzp5cuXC9guXLhglpeXByBCwJhXzs\/PF+dv375tTp48WfwAdhwA\/siRI8UPjqbzZH05YeUY3vp2QYXVV1LDsFJKGchwlKAVgbVOzLC2BOvp06fNpk2bVsAMa0sALi0tDcAlK0kvgV27dhWPILCrzsNacwyDuYe3vl1QYfWVVD2sdIdRgTYZrPaw9sCBA8UQtmx5AevHH39sMDS+evXqAOwyjLgOB4Ffdd6eG4cMg8lDS8NcPKeLyKHcYfXKbhjGW6vSPvLqO7TJYMUQ99SpU8UcFnDlBCs6HiKG8LJ46aWXViQE6yK0rzwHyyFTIGR048aNIpoKDhwcOWU39IG175Y2Cay2xcRcsm5Om9Kyloe3WKdDBr8NGzYMfxi41Ts\/7uKQzhf3BPdVtozwMss5u2GMvCQsrWQknDisZa8vdRF4dnGQpxf\/Ls9ZyRKTg6k8Z206T3PWL774ovAG4wf\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\/c8\/\/yzC+9588032ULGWTWC5vAtoJZewqoTSNbQKa8uuSrAiHxM+BF3n8u+iM7dsWtTl0h2aY3gb1TDrIuk21tVPYW2hOQCK4\/Tp0wWo+\/btM9u2bWsMildowwQe6hwKu3u70ql1qbC20BfBilvAwuIHsB48eLD43XSkVnSLZra6NMazmoP1DGl0Kl0qrCFacZS1sx0qtCuF5YK2DtDYL+UxqtX7VtLQKqzeqvAvqNDWy4qgxRY9gAjveHme3\/e9tVLQ9hbWcvbC48ePr\/gQVarshk0IK7TD1hX\/A2DhHX\/48GGxnIVv3GzcuNH\/bdiTktzQ9hJWgLiwsDDIqF\/+\/mrKvME+\/WacoXUFxruGxz7yzb0MF7S9g7XuK3EA+PLly8VnHC9cuGCWl5fN4cOHB3qU+j5rSEcZB2jJexsaoaXQuntS72Cta5INK5ZSUn2f1S3i4RKjBi2n91ahre9RIwErWdupqSlz4MABc\/ToUbN9+\/ahjynTd1yvXr06sMLljynjOhxkpavO2x9pjoGVrukrtNJrn\/awEbLKcQdMG73Ttb7DY3s9f+\/evcXSYMh3g3zqmmw\/K4a49NU3wNUXWLmgTdGhOa2nT+dBmXGG9syZM4O1e8gCS4H4wRcMXWv4vvK1yyWB1f6i+ebNm03qL5\/HCKbumraWlhtaqcD4UJn5WqDQ++ZUHrr\/\/vvvzdmzZwfVIkBhSaUPcVjLXl9qUKrvs0oJsCtoUwfGh8pv1KC1h7ckC0zjcNjRcKFyiikvCiuto547d64YGthHqu+zxggl5BooE84yGgaFhjH6WNouhrchMqgq21doASCmbBSammJ46ytrMVibQEXlcltn9RVYXbkYaHGvKs+qtHOobVtDru8DtE3WM8Xw1leeIrBSAMS1a9eG6oGdL3Nzc2bNmjXJvs\/qKwyOcja0GC75Kvv69evFBniKGJqcnBzsDOp7aB\/kmhu0ZUClnUMcfUsEVo6K9f0ePtBWDW\/pw0aIxx3F5ZCuoK0a3tLc0\/VChX+FnEq2sUEfrTJM9rTPdT6knyusIdKKKFuGdvfu3YWVcQXGj3rgQQpo2w5vq1YtAO7du3eLKDwsQZb\/LofZus6HdCmFNURakWXRaWyHBRbNsQneZ3jbVcaDyKYGX8YJLbdzCI4mBOlgPRVLjuRrocAdTOWOHDlS\/NB5AnzXrl1FeqCm82Wnq0t4CqtLQgzn7Q3wuB15j0PmtApttSLaWs8m9drhsRQlZ5eHk\/TkyZPFD8ClA9YUobRvv\/124\/nQKDuFlQHGmFv4zGmr7qvQGlMHKHfkEMUCALxjx44V6rDnrHZUXhlWlN2xY8cgaq\/qvL2JxacPKaw+UhIso9BWC9ceHi8uLpoff\/yxcu3T5RxqozpyLNn7sG1Al5aWGmFUWCOlD69cef6AW+W2AR51Chkec875IkUrclmV9cSDsMRy\/vx5kWeWb1oVZWc7nfC9HYp3V8vKpBJyn9+7d2\/IWQBHz4kTJ4oIq\/IGeXunEA1ZbO8eqocNCdhJVHW+ap7T1CQ7hHEcoS074uy1T5KbRIB8nU5csGJOSs4mcjDhXvactem8zllLksewZc+ePWbr1q3FGYBJgs015DEm7pia3SdLS95bcrjRCwq\/JYe3vjbAToZQ3obp4+1Vb7CvpI0Z7O6BYNeuXTv0FqQ3oP2Gsz18mJPMz88P1tTwaNs1j7+bzoe65stNC4H200+N+eyzp3d4992nEUP\/\/vcD869\/PS72VWKZiHt\/ZYAqBkUlvbcx9Wm6xtY16bLsVHKto7rOh9R5bBxMfd6m1wTtd98Z89579Sr\/738fm3\/+80EBbxfQtln7LOsMrXRFBLnOh8Bhv5wvXrxYXKoRTKESjCjfZ1ipuWVot249aI4caU5UjmufPEkbm8thPcmqIWOIHZTgslSu8xFdJ5tL1LL2ILVM3fD4P\/85aB49csP6ySfGYJiMQ2pOyx0YD38Cphg4yM9Q5dGXjBjKhtJnFRlbWNH+vm+Ax\/AXw2CfA9bVPtpC2yYw3lVfmhceOnSo0BHBmjpiyFXP1OfHGtZcvcG+nWBiorrkqlVXhixuGVa6MgRajuGtq2005wSoZadg6oghV11Tnx9rWPu+Ab4KVoA6Pb3b\/O9\/28z9+0+HyfAOf\/ttc9eqghb7kVNnTbBHO2U\/g8L6pO6dm\/q9Ifu8Ks8inphDBFNsy+3lGvseAHbt2i\/MP\/5xpYAWjqhLl9xzW5oaANyvv\/56cEvffZ+x7aDryjAqrCslOjaWtW1HyvX6uqEw6mtD2\/TFvLrhbeqEYPYm77K8EZ+bOmIoN50rrLlpJLA+rnVWDH9hYSmUj6CtG97mEDlEIihbVvUGj8kwOJCBXhUHsBS9RN5he7mGGmPv8MH\/pRrexgqzauriWkd1nY+tSw7XqWXNQQuJ6oBhLY6UwfBtmpZDBFOb+nNfq7ByS7Rn9yOP+J07d4qaz8zMrIiFdoXvuc73TBxZV1dhzVo9spWr+lpClwnBZFvb\/7srrP3XYdGCsoVcv379ipjaqmZW5Riyh56pE4KNiCrEmqGwiok23Y1dm+RDNsHbsKIFKROCpZNYP5+ksPZTbytqXRUzW5fGxtVcO+9t6hxDrrqN+3mFdQR6gCujge8meDsuF9eMe3hfbl1DYc1NIxH1qZp71n0Dt+72VeUV1ghlCF6isAoKN9Wt28JaNeclp1XKhGCp5NXX5yisfdWcVe82sJIX+f333x9kaKRbj3t4X25dQ2HNTSMR9akarlYl+yrfuglUKusK33Odj2iOXlIjAYV1BLpGrDe4aZcLfbbQFaHkOj8C4s2mCQprNqqIrwjnOmt8LfRKaQkorNISTnT\/mAimRFXTxzBJQGFlEqTeRiUgLQGFVVrCen+VAJMEeg1r3fpg3f8zyUxvoxLoRAK9hhUSs2NZ6bN7VVu\/OpGuPlQlwCiB3sNa\/kxjHcCMMtNbjZAEKLslLVWhafTlQfsjyjk0ufewQoiub6bmIGitQ74SsPsPplD79+8vUt\/QN3dzqflIwIo3IWJY8QEjHPYHknMRtNYjXwnYo7Pl5WWDXFVzc3PG\/pp5DrUfCVhth9KmTZvMwsJClsLOQeFah2oJ2Mne7SFxTvIaCVjteer09LSZmprKbgiTk9K1LsMSIOuK\/jM7O2tCsmukkufIwGpH8OT6ZkylVH1OuATsOOncHEvUmpGBlRxNuc43wruPXpFKAuT9xUv+xo0b5ssvv3Qmm0tVN\/s5IwOrBkJ00X36\/8xyv6G\/0bLchsMjA2vdV+L63520BZISwPD30qVLKyypzz5fyTrV3bv3sNr7KXWu2kUX0memkkDvYU0lKH2OSqBrCSisXWtAn68S8JTA\/wFa0Sw6ZWbeyQAAAABJRU5ErkJggg==","height":128,"width":171}}
%---
%[output:28c0a39d]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOsAAACwCAYAAADuQ5nhAAAAAXNSR0IArs4c6QAAHktJREFUeF7tXU+oV8f1n7eJWtCIilGJiVB5v24izUbrIiEhm1Y3LdRIIWYVsYu40yooaEAhJi5aEmhF04UGiobQRQlp6EKTjT43QroN1H9oAhpMhKhZxB+fed\/zffOdN\/fOmblz7vd7v99z4fFM3twzM2fOZ86fOffM1OPHjx8bfZQDyoGR58CUgnXk10gHqBywHFCwqiAoBzrCAQVrRxZKh6kcULCqDCgHOsIBBWtHFkqHqRxQsKoMKAc6wgEFa0cWSoepHFCwqgwoBzrCAQVrRxZKh6kcULCqDCgHOsIBBWtHFkqHqRxQsKoMKAc6wgEFa0cWSnKYDx8+NPjBc+\/ePTM1NWUWLFhgli5dahYuXCjZtdJO4ICCNYFZ49QUoCRw0rwATPwArA8ePLAAxn8raEdj5RWso7EO4qOoAyeB1B8EwIr38PvRo0cWxM8884xqW\/HVCnegYB0S46W7JXC6Jq4LSmjLlOebb74xX331lVmxYoV58sknVdumMK9QWwVrIUYOmwyB0gUnxkSgTAVnSMt+\/fXXlh71oSZyu6uuYG2X38V6A2C+++47a57+9NNPfbqlwFkF1lWrVlkz2DWRFbTFlrWWkIK1HT4X6cX3OwFU\/Dz11FPWNJWM3AKc0KwEVpqQC1rS5E21eBFmjSERBesIL2osKIShhwAkMaUqsFaBVrVt+VVQsJbnaTbF1KBQDEDZAwm8mNKXmsglOT9HS8Eqw1cW1aZBoRQAsQZU0yinLwVtU64Pvq9gLcvPWmpNwRkL+khOJQesIRNZzeP8VVKw5vOO9SZMWwSBcE65ZMkS+w4FYKqSEViEjbER2VHxWTlj9oNRClwO19QMTuNSQutQUAiZPzhmQfZPyUhp18DqshF8Il4paHkCppqVx6fKVnVBIdKcUqCSohuarFRf6tfyBVDByueVbRnyO2NpfJKC3iUzuI7V4BFcBVggK1eu1HTGALMUrBGwlggKKVh5OyL4dO3aNZvcQVcwwW0o6TrwRjKarRSs3rq4ubVk4rpBoRzBUbDyhN\/lE96gL37wb\/Vr9a4bK0VuUIhS+GCKkXnbNI1PwZoOVpfn6tfO8m8iNWtdGh+B18+B5YlbuJWClce9GJ8mHbQTAdaUNL6YwPDEbrCVBE0Kdo1LgCllPpMK2qJgxcfJO3fuNLdu3bLSumbNGnPy5Emzfv36HBnPfqdJUEgCWBI0OcJ9+LAxb701y8aXXpr9fejQ3L9TGCw1B3cMqX1MWpJFMbCiZs+BAwfM6tWrzd69e+0avPvuu+b27dvm6NGjZtGiRSmykdS2ZFAoVWA4A5WgWQfWCxeMefnl6pGdP58OWKk5NAGr++4kJFkUAyu06rFjx+zPsmXLLB+\/\/fZbs2\/fPvtTWruG\/M4SaXwSQilBswqsMaCSgD9+zNlm5tpIzaEUWInOOJvIxcA6MzNjzp49O6BFSdtu377dbNq0KU06vNahD6\/RBB9eN82xLS0w\/kSlBD1EFxoVgI09MIdhJnMfqTlI8X4cQVsMrB999JG5ePFiEKybN28227Zt48qFbRcLCgGgEsEVCaGUoFmlWblgxfsp2lVqDlJgDWla\/L8uJ1mMDFhTg0JSwiNBV4JmFVinpvh74iSA1QUtpTPi66cuJlm0Dtb33nvPXL582ezevdtMT0\/3c22JqdyCX20CgC\/+4ZZtjpULVkSHEWjiPlJzkNasIfpdrdBYDKzwWY8fP25OnDjRDzCFfFYAlQD72WefZX\/bKSU8EnQlaFZpVve4pg6IkwzWrlZoLAbWlGgwALtjxw5z5swZs3HjRu7mPtCuTQBkDdB5qe2xcrRr6vGN1ByGoVnrKjSOsnlcDKwp56wK1qbwn32\/CkCx45tUoNb1VWYm9fMp1Udswxn1JItiYAVDuRlMCtYy4lcnfAAsZS\/RUU7qcU2bWq+NDSEGVne+o5hkURSsXBFUsHI5Vd8uRfhiPbrn2KErH0v2VTUW6T5y6I\/Sea2C1ZOcnAWNAUGCZlNNlHrlI\/qTONduU3s3WYdRAK2CdULAmnqOTZuBe+UjvvXFcVvT73u7pFn9sfp+bZtJFgrWMQZrXf50SsULCCgSCm7evCl65WMTzRezbppaIjHQthFFVrCOEVhjd+M00YgEJLpBDmwrLaBdAqtvvpMFUponbj8K1g6DFQJy\/fp1e4McFRiLVVrkaKBQGx9IEj5cV8FK\/CILBGuxbt26XFZXvqdg7RhY\/XpRd+7csZ8fLliwQLQKYBWQSoK262AtbWr7qFWwjjhY264XlRv8KQFaBWu9MlawjhhYQ1HbKtNWWrhzjlV80JIPx7EJpecjTV81a8UqSzFegm6MZm7UNkaXAxBum9S+QkccsSIBqX1wx+76lF0+K1bNOgTNWipqKy3cOZo1BCBu6p70fKTpq2YdA81aF7WNaZs67dGG8JXUSjG\/Vno+0vQVrB0EK\/mdGDrdzyoRtW1D+EqC1aUVOpeUno80fQVrR8Ba53cSaEtW+ZcWDJ\/tEoIeCkaBj6X5JLHh5EbNU\/1st736rJk+a13U1jdtJQR9HMDqa1pc93j\/\/n2zdu1aW7Wy9CO1DqV8+9h8FaxMsPqmLb3GqRklJSRSdENC00Zf0Kr4Jnrx4sX9JI8mPn0b1kGbfShYa8AKASUT1gdnihBJCboU3WGB1Z0PmcgYS6l82zb4JdmHgtWRTDAaptiNGzf6u7t77WNuIrzUAkrRHQWwEq9jEeSY6diWidqGXzzRYA35nXQ\/K3ymUn6TFKik6I4SWH2\/FnPO1bRt8Euyj4kDayxbSILZEjTHKcCUopWaaFqpdWhLexcFq18wbevWrcEb5NqswVR3DUfI75RYUAmakwrWJppWah06B1YC6ttvv92\/hKrqykdJsIZMWzCTE7WVAoCUkEjRHUUzuMo3dTUtrXNVFYw2+CXZRzHNGrqYCgDev3+\/AYDdKx9Lg7UuaptSvkTBWh2ukRTCFDO4DrSxCPKozyEWLCsG1lBHUmClFL7SUVsFa3fB6o6cUhn9YJSCtWY7gLY9d+7cwP03aJ6qWduK2ipYxwOsVX4tYhSS6YxS8kPzEdOsuPV8165dZs+ePfMuUuaAdRhRWylmS+3oUnS75LPGTEdaU8gTpTPCJUt1jzj9SMmPKFjp3puqS5RDYE39xlNKUCXoStCUFgxfOKXm0FYkFf246YwoMgfA5ia61PnOUh+4Z2lWmLcHDx7sj\/fIkSP9m81DF1S5E6PrHunqxw0bNtgLlugZZtRWCgBSgi5Fd9w0qx\/Acu9nxd9KFuqWXJMssFbtKnR8s2XLFrN3795gM\/d+VhztvPjii7ZdqlkixRQJuhI0pTaWYWgMH0xtfSJHsRCy6nIzo9qyDoqBlQNUmhTHZ435CF0CQJfGOklgdedaFUGOyWGb7kIxsEJLnjp1Kji306dPDwSZFKypIhBuL7UJjLsZXKe53SSLHE0ruSbFwJoifgrWFG5Vt5UUjDY1xrDM4LpVyAWt5JooWL0Vk2C2BM3SPqvez1ptvaTcYyO11hidgnVCwar3sy5MMm9cTVsXQVawBtgqxRQJuhI0UzVrKAvMjcKHovGugOI7X9RH6mpCQSq\/6gJtpGnRxvdrpdZaNWtLm4DUAsboxrLAUlQL7mdFxH\/FihX21rquJRSUAqsfQfaPffD3kUqKSFnkUFsNMDXl4Oz7PlhTs8BSRkF9uQkFOdHSWFBHStAlwOoGxkjbTk1N2XV59tlni2dHqc\/aYZ9V72dN2W7mb25pb8dbA6SwQJCDPD09rWB1dzOJXThmWsaXbH6LkjRH+X7WusALh28l+RTqT5q+pPZWn7UDPmsX72cNBV4UrBwO1LdRM3jEzOBQ1LZL97OSdkk5m5S2ltqir5q1YrORMmkk6MZo5kZtY3Sb7+VzFHL6Ss0CyukjZY7S9BWsYwjWUlHbNoSvhFbiglZ6PtL0FaxjANbO3M96+LAxb701y\/GXXpr9feiQ\/XcJQfezgNpMKJAGUolNLWYlqM8q4LOS3wnSnbif9cIFY15+uVpWzp83D3\/1q6KH\/eBLmwkFCtbYVlDx93FMiujs\/awxoPbW8OGDB0XB6mqiNhIKFKwTDNa6qG2n7meFRgVgI8\/DffvM13\/8o+hFx0gouHnzpnn66aftPUNdqo+kZnCNAJXwoULkq+j6pi29y6kZ1fZYY8Ab+DsTrHjn6v\/+JwZW0nzXrl2zIH38+LEdZlfqIylYhwxWgIz8Th+cY3M\/69QUG9ttgJWy0ojvtAYl8pClNk2XgZJ9aIDJ4TQYPXH3szLBagNM\/\/iHuGYNpZByj35iu44kkFSzCmvWkN85cfezusc1dfweIlhDwagcTatgjW1ngb8PMxocyxaSWFAJmkWjmwztCq36aPNmkcBPqlbK1bRS69B5M7jqUipMrE2wduJ+1ppkhLq9sJjwRY5v7v3znzYxIsVPz9jDkxMvYkkW\/hiK8UvY4qsiL+KzUlX+K1eumJMnTw5c9ygN1pBpS1FF93cVQyQWtJJm7Izz\/Pm5TKLAgEuMtb+Z\/fvfZulf\/mJ7WXjp0mxvyF7CRtLS02Q+oSQL\/+inCX0uCyT7EAErrtc4e\/asnZ9\/N6sEWOuitqNQ6T+4gDGgknT0jjCqhOXq1asG1Qm455J1lgZnM+MKbWW7FlMa205nLOqaBBhYHKwzMzPm+PHj9vY4FP6WACul8HX6flbu+WZEu8X8t6aWRmNwEoHY5lQ4pTHEFwxFomCBy6POaFb3msfly5cHbz3P0axtRW2ldsbgAnLBikFFtCuNm641REQbRc0osSDFDSgGTpdQDKi9thIpjS5oKdIvUXKFptsZsEKT4sGlVE0DTMOI2rYKVkYEti\/vEbD6pV4glPQsWbJE9GyUBW7mxiSZ0ggQUTpjVys0FjODyfw9ceKEWbZsWTJYU7\/xlNrBJOgGaXLBik\/VEGhyHi6vYiYyC2glGjHBiq4ks6RoHbpaoTELrKH7WRHoqLqYyr2\/dRLvZ7VC6AeCmMkIODKxRye90qMUTKsq9RLC1tBBy92YWgIrXUwlwReJzZ7WNAusnM22ygyexPtZXZ8Si9mPUi5aFGUlkhGQ6lfC75QQzugE0IAJVumUxiogleTLWIEVa9dmUkStMAWOEazf9ItfiPl5A4Jx6ZJZ9Yc\/VA7x4aefWqCmHj\/FAOSOgTaB0n0MjIFpRQwLrG5wqO5qjBhfpWIeQ9OsIwHWSHQS2mzpb39b\/HtK3+\/ERwP3\/\/Uvs\/bvf7d\/evLKldkmLSUj+KDNybflCLBtw9Cu0imNKVqPk2RR5XJIHQ+JmcF1izhUzZpwjFDy4+fUW9tK9m2zkCpqK9E6kUYZMNMXpt20VgvcEUhpTAFrSNtyNrOcPrgb3uSBlRuZbKjdUnOSK\/3aJoCJbUyBdMaS\/pu7Edj5DTmlsQmQuHxp0kcMtArWGg7hkJ6r4UKJGykRW3cYRXzKGFCpw4ozXK5whtg31JTGFtMZQ\/6+gjUgEdlMYfhO1B3O\/KpMnxA4afHc37Hdsu7vjXzKQhYEZ+NogxdRPsY2J8F0RoyN5AT\/Vp\/VWy1xsKJW7qef2nKZ6AtZQVgQpPGR5igNTi5wOb6TLS3KKIRm+0xIZyRe4OMB8AJfWNHDqUcVBV1OgxhQezQl0hl990WvfCypWZnHCG4yAtLU6ObvBQsWmMWLF9uvXESPOyqElm2eJlgQHLC6AbLvv\/\/e\/Pjjj+aJJ54w4MeweNFnEXNjkkxnJNDqlY8lwQpaDEGmZATyO+k3Gyw5GiLhneg4GHO03QXSGfG\/OSmN0TEkzKdRUyZY0YdkOiMBVs3gUmYw6ETMJk4ywqgIaqVf+\/bbc8c1dUjogbVJUGjovOBuTArWxD3x8GFz+Z13zI61a82Zn\/\/cbITP07tPJZHS\/HzbCAE3EALAlqiMMHRBdebsH+SvWr06ylLkHd\/75S+L+J2cYFR0QDkNmGCVzpAaH83qaLPLixbNgvXGjVmw4omUMAmtYQwoBE7XrMO\/JQIhQUH985+jyQg5shl7h8aCDakunRGm\/sJf\/9qSK+l\/N4pixyYX+jszDqFg5TDXMzuDYAUdRlQyBloEP6iiO4IfriBKF\/yindUe\/v\/ud9WcydiYOGz228B3evTZZ30Lwk1nfLh\/P\/sMOadveic3bS+5T4Z2lU5nHA\/N6gUAKsGakTXkB0LcSOVQPrxmHiPkbkx1QlwVFKLjBIkrKfrjiaQ0xqygZHD6L4xAOuNkgZWhXbmBEHHhqJIubmQyY2Pyu0xNaRQxT2Obk2dFSKxLPxYx5HTG8QBrwET5v+lps\/GHH8zuu3fn\/NYAWJtmx0gIR60W4IIVubIJ6Yx9E\/vhQ5ukgR88JVIaWUkWoUnHgErvBNwbf13cDCCOlo2V\/eHQyGrDsCC6fXQTACtM4feWLzeXf\/azOdBu2mSzhgaitj2ONg0KtRapZPhOJCQ48yOfOhTgabpRcYSx0WbG3ZhqrAiutuec+3Lmm90mtjH1LAjMp9tgrYnWuaAFI1977TWzY8eOfnSydFCIKxzZi8oFq5fOiP7gW2K+CIy1ndKYBVouWBnuDZq4wSjiBfzsplZE9lrSizGgOhZE98E6K4m1PLNa9je\/sVUk8GzcuNHs3r3b\/pZ6sgQ0NhjmMQJlDhEor1+\/blP4kNaIdMaVK1faDYv71U9sWNy\/J21m3I0pAaxoipS9+\/fvW15QJULUTRraw92UDh0yiLJ3W7OCy7HdyQlEuHWaOglahhC7tZXclEbSMABNti9ZSKqjxy6MedqhZKY0jgwvuGDtxSG6D1YCLFUsoC9CavyZzoKWeYxA\/moIWyJaPxPElWNJtCJCPjgnQNY0GJU57bnXuJvSMMDqlxrdunWrOXr0qFlUUY0PlQx37txpbt26ZSe4Zs2agQupUFP49ddfH+DZhg0bDNUYjjEzGbSMMiZ1fSaZgh6h0scIrQXGYovQK4VKnwymVGgsldLYZF0Y06tuwgVrLw7RmmYl4OGOmk2bNhm6EuPVV18127ZtmzchujFu9erVthI\/HlTmv337dh\/gAP\/FixdrAc9hpgtatH\/zzTetX9t\/EkxtTn9khpFfGTJL3WMUyaDQ0AQ1wKgBTRep0Ahz370uspQPHjXRuQvMaZdgQeA0ozWwAlgoSE3Aw1ygGXErXEi7AtzHjh2zP6jEjwcA37dvn\/1Zv369BS8elyaHR1VtfNBav\/a558zGffviZAukNCIQVPXhdenodWhCJKj4EGEp8o\/xwC\/Ek\/lRRJxx81tQ5PPH\/\/zHPPXXv9oGbac0tuYuMAKkWIP3\/\/tf88Ybb1glUmpjIs7Pq8EEYK1bt25Ai4YASQRCQCZtu337dgNz98CBA2bz5s1BzZwjJO47VOF\/98cfDyZXVBHOyBzy75JBpBLP0D5CF7AguOtQZUnARUJpVXyQLhIYYyQjzDPRmxSb8xni8RynF3je7\/3Gv6E08AOLVOIUYwCsBDIfWHWXTIVMXJfOK6+8Ynbt2mW+\/PLL\/vRT\/FWuEJUsY8JJaRyaLxkDKjEs04Ko0uT4\/yEz37ckimu62HxbSGfE3GHNzbzzjrk8M2MTeSw4f\/jBbPzTn+y\/B9wxttCmNRQH6\/PPP2+DT+QDY3h1mjpt+E5rz0yhZIt56Yx4xRPkJplCrfuSzGMElDDBmV\/Op291UVuumV8EtDGg1mxMoc2UO\/Y+OGdmZrXn++\/3BQ1xkrbA6WNBHKx1QamipnEFWGkXfPPuXZuHDL+ijZRGEVMQq8cEK5pSOmNsLJJBskabGXeuEdeGG4yCS1UFTinTNkU5BX1WEHCDQTGfFTedu8cwrs+KSfpPlbmdMvB5bSsidn46I95rM6UxBpTkOXOPEXoWRJWg1iXCp2iglPFzQdOnyQVrwFoKjcvX9nDhSIu6mXOSfmcKv2o1K\/5YOhp89+5dkwrm7AnVCDICAggGYCGwMJ3MjAJjuGD1soYAFErjAxkExxDR9rOnsnmf8CLbRObOlQlW63fOzJgvvvhiIIZCgaE2\/M4ENs1rOk+zlj5nhRZFgGnPnj02SoYHfZw5c8bs37+\/MtEia1IxH6cXjEhOssgazNxLjUxBv++EMz8kI7gmLkiJR20TeBUFLResFemMGErMtMXfaeNOGPpQmgavz6jLYApFhmMZTJRYQRHhWEZUI04AsMyUxrZBi3klm4IhZjCEmHKPqz4tHHoKnzOvysh6wsZEt8OHwCl9pNJIXhNeHspdNwnjYzWNbRYxItHMqDrtlpmMENUqdYNmlFKlrKHY3Itq\/Vhnkb+HxhKr0IiYxMyWLQa\/fb8T3Y26aZvCss6DlZPuyGVIMDPK\/UyPaWZz+0O7FNC6RyqlSqm6Yy2i9VMmX9OW+OJXaLTg7J1zugkJnCOV1E3db4\/h+nnvhabLItN5sHLSHVmc8BrBnKLztbbTGTEUmK\/4GUb5EncDWfW3v5mFx47NcifTisjhP72Dsbz3+9+bhZcuDWQLETi5Ryo5mzqCUX5wtMlcmr7bebDG0h1DR0cpTCNtK5nO6I4HSeD2RvTex9eoGoEPsOlD9NL5ppW8ELAiuHwnvxO8d01biuIjOJn65GzqpT5ASR1rVfvOgzWW7hhKyshiXuEzP1dzDJi3vT8AnKiUQGVNip\/X1jEhBlR6t1BKIx2puOBEF75pSyDO8UNzNvXSH6BkyZ3zkoKVy8FABJaSuW1mlPtEhDjVtE3xa7nTqW3H3Zh6ZUxytH3dkUoOGGPzTt3UyWz+5JNP+qSH6a9iEArW2CrT3wNg3fH00wNJ3TYPGWfJOM91nlKV+VoDLResvZRGjtYf9pFKKlhD33H7n35yRadUu86DNRQEiKU7ZjGv5syPsqNA19ZC\/vhjMz09PS8hoWk5Vdd0ps\/B8P84YEmaM+Mctz+WBw\/6F0674wiZtnTeiXdztad\/Zg9ap0+f7ifcVM0zFaxVdIZpGncerDmBgyTBdRszPkCm4wR8BoivjSCgOV++cMcoctzCBauTOYQUvs8\/\/9wO+8MPP+wPn3Okwp0r2oWqkJw7dy5aIqjUpq5gTVktr21OSD67u0jg5TKOOHrHG\/RRfBs5yJhPUROZmTlkffZDh+Z9QubmXmfzOvBiyAzlWlGpm3rVN9yh4gwl51hHq\/OaFZNLPexuxNwG6Yxt5KAWA21NQM1mC9EH2MLVEdy1qvr6iwOg1E099GUYNosjR46YgwcP9ksYNZKlxJfHAqyhOY9ShcZoZpQ\/gYbVGX1Ni\/+mJAu2fFy4YC5v2WKzhVxw4n18GwyNmut3ssfgNaxKUuCaprFN3afjR4RFKpwkMGMswVr6yyF8qVLqgHxeZlSL6YyQi1gwKnik8txztki7PaLKqGGVII+1TZuCtdQ4hkVnLMFa+ptc6QqNbaQzVvm1J0+etLLnli7J\/UoFmunUqVN9WUaVv7qKln57vFj3joKVbtgd1nYh0G+XKjS2nc6I\/hC1xW+3iF3TqC2AhO+TAX5sbr514y9zTrWQUQz6CIhvJcmx06ydrdCYkIiQcms6nXf62pPAWSpyG9ogQxYOSSKdl7pFCWKC3yQaHKPdhb8rWHulbPwbA1zQt1KhsSb6ugklLx88mJOnSDpjrDpC6Zq2VccnseLw0MSoeglNzH1yz1m59Ee5nYKVAdZWKjRWgJWSLPq3xAfSGau+Ummr8FeVlqz7xCx0\/1HMxwWQcjOYRhmE3LF1Gqz+wlFo\/YMPPrDz71SFRuaF05TOCGGPfaXCFYKm7XLAChPZzzyqM5ubjnEc3u80WKsWoHQ0uLUKjQnpjJh706BQnQCnFGLPAWuo77qbH8YBbE3nMJZgLX3O2lqFRmY6Y2mf0xci4h8+evev5fQTC1D8DncZ4dIyv2h7nc+qYE2H7liCFWzobIXGhHTG9OWOv0F8e+GFF2zFCu4duqnR4FDWUYo2j89k\/FqMLVjHb6nkZ+QejYRM\/7oRpJ6zhoJPVCG\/WHUPeZa12oOCVYjdqbnJ7jA4ielCw+6TzSkWVpfBFPJH\/YgwkuQVqNUrq2AVkPpUn9kHKlL2hi24OWAVYKWSdDigYBUQh9RoNIZAEdU7d+6Y5cuXG1xEPUwto2AVEIyGJBWsDRkYej01N5kCYlevXrXHMZI3xbsbA+UGhz79UrAKCEZDkgrWhgz0X8\/JTXZp5CS4F56CJadgleBqM5oK1mb8m\/e2grUwQ5VcnwMK1sLCoGAtzFAlp2AtJQP+cQXKYqLSHx5ubvIomsGl+KN0ynFANWs5XvYp5USD6eVR8VkF2KIkG3JAwdqQgaHXm5yzKlgFFmRMSCpYhRayLoOpDpAKVqEFGQOyCtYxWESdwmRwQME6GeussxwDDihYx2ARdQqTwQEF62Sss85yDDigYB2DRdQpTAYH\/h+\/KT9ntAKQ1QAAAABJRU5ErkJggg==","height":128,"width":171}}
%---
