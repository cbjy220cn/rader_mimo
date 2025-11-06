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
Ny=4;
numTX = 4;
numRX = 4;

N_L=floor(2*fc/BW);
(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L) % 扩展的虚拟阵列 %[output:8be1bd6b]
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

scatter3(tar1_loc(1,1),tar1_loc(1,2),tar1_loc(1,3),'r','filled') %[output:7d33c344]
hold on %[output:7d33c344]

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

scatter3(tar2_loc(1,1),tar2_loc(1,2),tar2_loc(1,3),'blue','filled') %[output:7d33c344]
xlabel('x'); %[output:7d33c344]
ylabel('y'); %[output:7d33c344]
zlabel('z'); %[output:7d33c344]
hold off %[output:7d33c344]

%%
tx_loc = cell(numTX,Ny);
tx_loc_t=cell(numTX,Ny);
for j=1:Ny
    for i = 1:numTX
        % tx_loc{i,j} = [(i-1)*d_tx (j-1)*d_y 0];
        tx_loc{i,j} = [(i-1)*d_tx-((numTX-1)*d_tx)/2 (j-1)*d_y-((Ny-1)*d_y)/2 5];

        tx_loc_t{i,j}=zeros(length(t),3);
        tx_loc_t{i,j}(:,1)=tr_vel*t+tx_loc{i,j}(1);
        tx_loc_t{i,j}(:,2)=tx_loc{i,j}(2);
        tx_loc_t{i,j}(:,3)=tx_loc{i,j}(3);
    
        scatter3(tx_loc{i,j}(1),tx_loc{i,j}(2),tx_loc{i,j}(3),'b','filled') %[output:32c807f5]
       hold on
    end
end

rx_loc = cell(numRX,Ny);
rx_loc_t=cell(numRX,Ny);
for j = 1 : Ny
    for i = 1 : numRX
        % rx_loc{i,j} = [tx_loc{numTX,1}(1)+d_tx+(i-1)*d_rx (j-1)*d_y 0];
        rx_loc{i,j} = [(i-1)*d_rx-((numRX-1)*d_rx)/2 (j-1)*d_y-((Ny-1)*d_y)/2 -5];
        rx_loc_t{i,j}=zeros(length(t),3);
        rx_loc_t{i,j}(:,1)=tr_vel*t+rx_loc{i,j}(1);
        rx_loc_t{i,j}(:,2)=rx_loc{i,j}(2);
        rx_loc_t{i,j}(:,3)=rx_loc{i,j}(3);
        scatter3(rx_loc{i,j}(1),rx_loc{i,j}(2),rx_loc{i,j}(3),'r','filled') %[output:32c807f5]
    end
end
xlabel('x'); %[output:32c807f5]
ylabel('y'); %[output:32c807f5]
zlabel('z'); %[output:32c807f5]



plane_loc=cell(numTX*numRX,Ny);
for i =1:numTX
    for k = 1:numRX
        for j = 1:Ny
            plane_loc{i+(k-1)*numTX,j}=rx_loc{k,j}+tx_loc{i,j};
            scatter3(plane_loc{i+(k-1)*numTX,j}(1),plane_loc{i+(k-1)*numTX,j}(2),plane_loc{i+(k-1)*numTX,j}(3),'b','filled') %[output:32c807f5]
            hold on
        end
    end
end
hold off %[output:32c807f5]
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
fprintf('仰角 (弧度): %.4f\n', theta); %[output:945b79d4]
fprintf('方向角 (弧度): %.4f\n', phi); %[output:3f910af1]
fprintf('仰角 (度): %.4f\n', theta_deg); %[output:6e18f158]
fprintf('方向角 (度): %.4f\n', phi_deg); %[output:99974e03]

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




figure %[output:1df90618]
imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))))); %[output:1df90618]
colormap(jet(256)) %[output:1df90618]
% set(gca,'YDir','normal')
clim = get(gca,'clim');
caxis([clim(1)/2 0]) %[output:1df90618]
xlabel('Velocity (m/s)'); %[output:1df90618]
ylabel('Range (m)'); %[output:1df90618]
%%

%% CA-CFAR

numGuard = 2; % # of guard cells
numTrain = numGuard*2; % # of training cells
P_fa = 1e-5; % desired false alarm rate 
SNR_OFFSET = -5; % dB
RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

[RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
cfar_ranges_real=(cfar_ranges-1)*3;
figure %[output:755934af]
h=imagesc(V,R,RDM_mask); %[output:755934af]
xlabel('Velocity (m/s)') %[output:755934af]
ylabel('Range (m)') %[output:755934af]
title('CA-CFAR') %[output:755934af]
%%

%% Angle Estimation - FFT

rangeFFT = fft(RDC_plus(:,1:numChirps_new,:),numADC);

angleFFT = fftshift(fft(rangeFFT,length(ang_phi),3),3);
range_az = squeeze(sum(angleFFT,2)); % range-azimuth map

figure %[output:4a415ee3]
colormap(jet) %[output:4a415ee3]
imagesc(ang_phi,R,20*log10(abs(range_az)./max(abs(range_az(:)))));  %[output:4a415ee3]
xlabel('Azimuth Angle') %[output:4a415ee3]
ylabel('Range (m)') %[output:4a415ee3]
title('FFT Range-Angle Map') %[output:4a415ee3]
set(gca,'clim', [-35, 0]) %[output:4a415ee3]

doas = zeros(K,length(ang_phi)); % direction of arrivals
figure %[output:47d289c1]
hold on; grid on; %[output:47d289c1]
for i = 1:K
    doas(i,:) = fftshift(fft(rangeFFT(cfar_ranges(i),cfar_dopps(i),:),length(ang_phi)));
    plot(ang_phi,10*log10(abs(doas(i,:)))) %[output:47d289c1]
end
xlabel('Azimuth Angle') %[output:47d289c1]
ylabel('dB') %[output:47d289c1]

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
figure; %[output:0b98aa48]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(1,:,:))))); %[output:0b98aa48]
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value); %[output:0b98aa48]
xlabel('方位角');ylabel('仰角'); %[output:0b98aa48]
zlabel('空间谱/db'); %[output:0b98aa48]
grid; %[output:0b98aa48]
% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value); %[output:9ca53d11]
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

figure; %[output:831569ce]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(2,:,:))))); %[output:831569ce]
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value); %[output:831569ce]
xlabel('方位角');ylabel('仰角'); %[output:831569ce]
zlabel('空间谱/db'); %[output:831569ce]
grid; %[output:831569ce]


% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value); %[output:4bf89ee4]
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
figure; %[output:19555f9e]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi_sorted,ang_theta,10*log10(abs(spectrum_shifted))); %[output:19555f9e]
 
xlabel('方位角');ylabel('仰角'); %[output:19555f9e]
zlabel('空间谱/db'); %[output:19555f9e]
grid; %[output:19555f9e]


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
if size(peaks_info, 1) >= 1 %[output:group:9c49baf6]
    main_peak = peaks_info(1, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ... %[output:206c3f38]
            ang_theta(main_peak(1)), ang_phi_sorted(main_peak(2))); %[output:206c3f38]
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(main_peak(1)), mod(ang_phi_sorted(main_peak(2))+360/N_num/numRX+180, 360) - 180);
end %[output:group:9c49baf6]

if size(peaks_info, 1) >= 2 %[output:group:3081a954]
    second_peak = peaks_info(2, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ... %[output:2c7eea0e]
            ang_theta(second_peak(1)), ang_phi_sorted(second_peak(2))); %[output:2c7eea0e]
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(second_peak(1)), mod(ang_phi_sorted(second_peak(2))+360/N_num/numRX+180, 360) - 180);
else
    disp('未找到明显的次峰值。');
end %[output:group:3081a954]
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
if size(peaks_info, 1) >= 1 %[output:group:105bcb42]
    fprintf('精确仰角: %.3f°, 精确方位角: %.3f°\n', peaks_info(1,4), peaks_info(1,5)); %[output:3a6511ac]
end %[output:group:105bcb42]

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
%   data: {"layout":"onright","rightPanelPercent":29.1}
%---
%[output:8be1bd6b]
%   data: {"dataType":"not_yet_implemented_variable","outputData":{"columns":"1","name":"ans","rows":"1","value":"119"},"version":0}
%---
%[output:7d33c344]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnU+IV9cVx6+rZArVYmQSM06UIi4t3Ti4kWQbF6UFzSYJLhRBKm4MFRSSgBanGUGYIIiuYjeZLLrSbUI3kmwKlm5MFrY2ThozoSFQ29WU85Lz886dd987993\/v\/d9MMy\/8+47f+79\/M5999+W9fX1dYULHoAH4IGReWAL4DeyiMNceAAeaDwA+KEiwAPwwCg9APiNMuwwGh6ABwA\/1AF4AB4YpQcAv1GGHUbDA\/AA4Ic6AA\/AA6P0AOA3yrDDaHgAHgD8UAfgAXhglB4A\/EYZdhgND8ADgB\/qADwAD4zSA4DfKMMOo+EBeADwQx2AB+CBUXoA8Btl2GE0PAAPAH6oA\/AAPDBKDwB+oww7jIYH4AHAD3UAHoAHRukBwG+UYYfR\/\/3vfxV9\/exnP4MzRuoBwG+kgR+72QS+r776Sr3wwgvq2Wef7XbHO+8o9e67P8i8\/PIP399+++nPY3dmpfYDfpUGDmr7eUAEv08+UeqVV+wP+vhjANAvDFnvBvyyuh8PD+0BaXe2F3594GPFcQRO6BAmKw\/wS+ZqPCiFB3qh9qMSvXKU8REA+y7q\/lK3GFd1HgD8qgsZFO7yQC\/UQsOPykP2V2WlBPyqDBuUtnkgGPy2bJE7GfCT+6ogScCvoGBAFX8PJIcfjf7SwAeu6jwA+FUXMiicpNurT2\/peiDgV22FBPyqDR0Ub\/NAsMyPCpd0fTHdpdqKCPhVGzooHh1+fdNdAL6qKyHgV3X4oLzpgaCZHxVOAOTVHTz1BdNbpqLiAX5TEUYYwR4IDj+4dmo9APhNbWjHaRjgN864D7Ea8BviNdxTrAf64Pfvf\/+70f1f\/\/qXWltbU3Nzc+r555\/v39ygWIuh2FAPAH5DPYf7ivSACT9e68vfSWnexeXx48dq69atan19vfkbbW\/Vu8NLkVZDqSEeAPyGeA33FOsBgtz9+\/fV7Oxss18fX7xvH3\/XIUkylBHS3wDBYkMbXDHAL7hLUWAOD+jd2W+++Ubt2rVLbdu2rYFZWzbX1j2mvwGCOaKX55mAXx6\/F\/NU6RZQxSj8oyKsN0OP\/kxZHf+9b5PSrneDgGBp0Y6jD+AXx6\/VlNo3QFCSIQw6E3gMPvoutUciBwiWFP3wugB+4X1aVYkSCOQyyDZYwV1ZaXe2TX8XuwHBXDUg7nMBv7j+Lb50FwikMEaS3XXpIbVHKqc\/CxBMUQPSPQPwS+frIp80BAIhDRmS3eWCHz8XEAxZA\/KVBfjl830RT84BP9\/sLjf8AMEiqq63EoCftwvrLiAV\/Ezg6e\/tQk4sltojlZNEN0cmWOsovcSfqWQAv1SeLvQ5ISGgmxgzuysl8zP1SAnBWHErtJpGUQvwi+LWegoN2YhyAc8clJAcRh7S7hwQjKl\/PbXXT1PAz89\/1d\/t04i6Bit4GVlqB0ntkcr56B8zE0yhv4\/tNdwL+NUQpYg6ujaiErK7Uru9Nr1iQNA1bhGrULVFA37Vhi6M4n2NKPRUlDBa20vps4fvlMqF1DckBHPoH9IXJZQF+JUQhYw6tDWi0rO72jK\/rneC9D96ReD6mgDw8280gJ+\/D6sugRsRTzeh3+mKNRUltrOkUJDKxdRXzwRdIViC\/jF9k6JswC+Flwt8hrkF1I4dO5odjbkRFqiySCUpFKRyood6Cg2BYEn6e5qf7XbAz9H1\/A7MtZvi+Jgo4m3dWcrwvv76a7Vv376p2MVYCgWpXJRAWAp1gWCJ+qf0VYhnAX6OXqyp0kmmotRkjyRUUnukcpJnhpaRQLBk\/UP7I1Z5gJ+jZ0uvdK6DFaXb4xieoPv5uT47hjzFk2OqD4xMW9xi+K6vTMCvz0PG\/0urdF3ZnW0Ld92k0uxxDMcmcak9UjlffULdb0KQYitZyRLq+dNYzqjh99lnn6lPP\/1UnT59WhzbEhqNa3bXZVwJ9oidLxCU2iOVEzwyqQhD8H\/\/+5+ir2l5V5vUiT8+bPTwe+ONN9StW7fUgQMHRP7P1Wj4YB16Pl2hpqLkskfk7AFCUnukcgNUSHILnTv8xRdfKB6lr3EALomjOh4C+BUKv5DZHTK\/zR6oHX6sP30I8gfikMnSuQGU8\/mAX0HwSwU8vPOTH3SUs3G6fGjZBkZK1b8EvZLB76OPPlIXLlyY2Hz8+HH11ltvbfDBe++9p27evGmVefLkiTp\/\/ry6ffv2RObixYvqyJEjk9\/pHd6bb745+X3\/\/v3q+vXravv27Zv8Te\/8cnZ72wYrzMO1Y1eS2jMg0z9Se6Rysf0\/tHyb\/oCg3KNJ4Efgu3btmrpx44bau3ev+vbbb9XJkyeb92wMQJJZWVmZgIreZ5w4cUKdOnVqAjeC4+rqqrp06ZKamZlpBisIdB988IFaWFho3oHQPZcvX25+Z1iSO\/ge3TU54Jcju3PJIORVp0xJKdSkcmVa2Z+56hCkrjF9qIbcMbtUv7joFR1+DKCDBw9uytDOnTvXAHFubq7J6EwZAuKDBw8aQDIwz54924CNLwLinj17mrJJ\/u7duxtAR0BcXFxsvszsLwX8fKeiuARziGztEEDm90In1ABBe6uIDj\/bowlKBD\/K0ghKlAmaYKPMbmlpqckGCX4sT9kjXzrw3n\/\/\/ebPenfaBk2SiwW\/0rI7ZH6bPVA79F31BwQ314Fs8JOAjWQ4O1xbW5uAUM\/guLt89epVdeXKlU3ZI8Pv6NGjGzLPkPArPbsD\/J56QN\/QgeoU9TpoQ4fauoSu8GMPAIJP60IW+HFXeOfOnU2WpmeBelZXMvx4nh03JnJp6sGKId3cod3EEM9KUYYJBdsHE+ny+PFjtXXrVrW+vt7Ar6b3YkPhBwhmhl\/b4EZbl7Y0+BHoaFY9Ty595plnJg2mtsyBq4BvI0oBNJdnkD33799Xs7Ozk\/lvbR9Mut30f55EXgsEQ8WNyqnNdpf60CWbPPPTgcZZXqmZny1rGOMWUKEqXIxy9Dh999136ptvvlG7du1S27Ztm6yEkWS8NYEgFPz0D8HUEOS45VqdkhR+5tQUdrxtUELyXnDogMfy8nIz4EFf+vK2vsGK0JUuBgxcyqzVHlucuEG98EL3KGiX3TVAMFbcUtoeywZp\/U8GP5qScufOnclcP11B23QYHWwk0zYiTOXSRe8ObVNd2rrUBD8eHaafaTI0BYOurnWzuQMmDaxUrhZ7pINKUnskciYIuEss9W1MOYn+Ps9PAcHYNvTZnwR+vHKDJyO3KWVOhLZNcqZMjVds2CY588TovknODEDS79ChQ41afSl47oD1BdT1\/yXb05eFt9kqtUcqR8\/QQcB1pK+euMbBVd5Ff9eydfmYEExlg83+6PBjiD169KhVBwZi29I1cwkcd4\/v3bs3Kctc3mYuoyt5eZtPpQx1b+4KqNth22Zfz8T77JbaI5WzgSA3BIfo3+e7rv\/HgGBqG0z7osPPx+Gx7401yTm23iHLz10Bh2R3fY1Ussmnr92519D66j+0DoWEYC4b2HbAr6BdXYZWSJ\/7UlfA0LAzbZfaI5Xr822uScOh9O+zz\/b\/EBDMbQPgB\/hF3w49NvDMrmmKzM+EQmoI5gYH2+8Dwdw2AH6AX3D4pYRd7syv7fkp5svlBkcIu3PbAPgBft7wo0qsf1HD0A9PSjkyKm1QUrmc3cIQ7zaH6j\/0PpdMMHYM+mwA\/AC\/QfDLmd2FgEKqhucCg77GOqR771JmSFmJ3aliYLML8AP8RPDryu4kR2SGbFg1wS\/Eu7E2e3ODQxrPLgjmtgHwA\/ys8Cs1u6sRfqEhmBscUvh12U3\/kwxOuT5LKg\/4AX6TCkiVxvburqTsrmb4hYJgbfBrs3vLli1Nfdu9e3eW\/RQBv5HDj85\/pS\/anou+6KpxX0K9cUmyiVLgIXk3VnO31\/ZBRXZTvaNdeHIdvA74jQx+5rs72p\/w+++\/V\/Pz85MtoKTdhhLlpFCTyqWy0RWCpek\/xE+5bQD8RgC\/rnd31J2VZEpDKneOe6QNSiqX2gYpBEvV38VfuW0A\/KYQfi4js7kroEtjkchK7ZHKSZ4ZQ6YPgqXrL\/FJbhsAvymB39CR2dwVUNJIXGSk9kjlXJ4dQ1aHIL+PpXeytegfYnAqhl+pTMCvUvi5ZHclV8DQFVsKBalcaP2GlmdCkF5X0N\/6dqwe+rwU9+WOAeBXEfyGZneA32YP5G54Q+HCH3o0UkpnldA5OHT0Zo1X7hgAfgXDj2HHFZ4qeNcW+0MaQO4KOETnEDCv3W7Sn06p4ylKfOxmTacI5o4B4FcY\/GJkdyFgERpSscqTNiipXCw9fcvV9ecuMX841nL+cO4YAH6Z4Zcadmajy10BfSEw1J7a7W7TP\/Wegr6xyx2D5PDjszpee+01tbCwMPFf2\/kc9E\/9jI62cz7MMzz4UCMuuMQzPHIDT6+0uSugbwMC\/DYf0dk3TSa0z4eWl7vuJYef7SQ328HlumPp3tXVVXXp0iU1MzOjbKe3Xb58uQFr3+ltqc7wKAl2Q2ExtIKnvk\/aoKRyqfWXPk+if+kQlNgg9ccQuWTwMzM78xhL\/YDy7du3b7LFdrA5AXHPnj3qyJEj1nN7FxcXFX2Z5caCX9s0FDKoxDWzuSvgkEob4h1m7Xa76F8qBF1sCF1PqLwk8GNw0frRY8eOqTNnzijOztgoOnLywYMHzeHjbZctM9QPKudDyPUybNCkZ4SEX8nZXQhYxKh8Mcrsa1AcJ5oqsra2pubm5pqpIjWNkpLf+uxs821pEBxiQ8g6kwR+usJ8jq8Ov7Z3eeb7PltmSPBbWVlRV69eVVeuXFEHDx5sskC+GH5Hjx7d8Hdf+HEWF3MaSshA28rKXQFD22ja05aFM+geP36stm7dqtbX1xv41TJKOhR+7OtSIJi77hUBPz0z5Pd5FCi9S1sC\/ChroF1QCOA7duxo5liV2JV1AUruCuiiq0SW7KH5b7Ozs012xJcZJ91ukklx6JBEf6lMiLjlhmAIG6T+apMrAn42A3Tgff7552ppaUldv359w7u7mJlfW9ZAGzDSHmQvvfTSBHw+Ach9b+4KGMp+vTtLKx927do12aKrrUvbZnduGLj4ImTcctkd0gYX37Fs8fA7d+6cunHjRvN+JgX8+t7d5Q7YkCB33VOrPfzBxPEiG3nBP\/2vb81rl925YOAS2xhxS213DBtcfFgN\/MgoAiG9K6T1jHwNHfBYXl5uBjzoi36m+YDcTepaQpY7YC7BlcjWZE\/fB5PLuzCJ3alhIImX\/t4u1j6MqeyWxMDFJ66yRcCvbRCEDKFu74cfftjM66NBkZMnT6qzZ89umBxN7wXpohFeHYQ0D5Au2ygxQ4++UxmHDh1q5PvOmM0dMNcA98mXbE\/ba4e+tc1Se6RyDNTS3gm66N9XB2z\/jw3BFDZ02V4E\/EhBHWL0e9tKEJIhWPF7P9sk51OnTjUju6VMch5a+VLcl7sCmjZKsrsQ3fghdseGgUu8h+jvUr4uq9vNCUJfkiB5Vkob2vQpBn4MwJs3b070NCdCty2BM5e3UfZ34cKFSRklLm+TVIxUMrkr4JDsLhf8+LklQDBH3EJDMIcNet1JDr9UjVrynJCTnCXPK1EmRwX0ze5yw68ECOaIm+53fRMFygKHZIK5bQD8Mu\/qkhuIqSqgCby+d3dD\/SK1Ryon0SNHJhhSf4mNNhmfnWRy2wD4AX5RTm+Lmd2VkvmZeqSEYG5wtL2r5ZhLV8vktqEVfl1Lwsjovk0IfD5JUt6Lbu+wNaJdWQD9z5x7xy\/JU8RW2qCkckN0TgHBmPoPsZnv4VFx0q8Pgrlt6ITfvXv31OHDhydbSLGBgN9XvZNofSpQynt9KmDXYMWQd0Ah7JbaI5Xz0SkmBFPoH9v23DZY4fe73\/1O\/eY3v1F\/+MMfGh\/QKgueXAz4jRd+ubqz0oYobVBSOelz+7rioecJptTfxwddHwC5beiEHwGQtvw5f\/68un379mRXZcBvPPDryu540MKncYS+V9qgpHIh9QuZCebQ38cXbbZTebFWqUh07YUfZ3s8f466wb\/61a8U7Z1nbjIgeWBJMnjn1\/7Or\/Tsri\/LkjSonPDQQcDvQ11fE+TU36cN67bTJiH0++7du7PspyiGHxnMy9AePXrUrIUF\/J71qQdF3MuNiHc+od\/pijUVJbbRUihI5WLq6wPBEvT38Q3pTxvK0g5J+\/btKx9+ZCwvGXv48CHg92y98DO3gKL9Cfnwa9csxKcRhL5XCgWpXGj92sobAsGS9B\/qo9w2YJ7fiOb5tXVnKcP7+uuvs336Dm04tvukDUoqF1q\/vi47D4z0dYdL1N\/VV7ltAPymGH5dgxX6NvySd2SuFTuXvLRBSeVy2CHJBEvWX+qz3DYAflMGP9fBitwVUNpQpHJSe6Ry0ufGkrOtoa1F\/75MN+cHL+BXOfy6sjvJVJRpaER6A5PaI5WLBTXXck0IUmxzgsNVf9u7zpw2AH4Vws81uyv50zdEIxoD\/NhGhiAdpEVfuUZKQ8Qt9wcQ4FcJ\/PQ1k1TxQk1FyV0BQzSiMcGPbaVpInyKII3S1zhCn7vuAX6Fwi9kdofMb7MHcjc8X+iz\/vQhSD\/3jQ77Pi\/G\/bljAPgVBL9UwBuSKcWo\/DHKlDYoqVwMHUOUaeofYnPREHq5lJE7BoBfRvi1DVakPgQ9dwV0aSwSWak9UjnJM3PI2PSvCYK5Y5Acfm0HE3HloQOK9DM8jh8\/3pzKxhffS5ss8GWe4cGHGvH\/SzvDI0d2h27v9HZ7becT++ywnArmo4MfA848nIg2TlhZWZksmeN1xHwSGwWE7l1dXZ3sL2g7vY3O9l1YWCji9DbfqSixK2LuChjaPqk9UrnQ+oUqT6p\/yRCU2hDKZ2Y5yTI\/8+Q1HX6c0R08eLA5cpIvAuKDBw+a7I\/vbzu3d8+ePc19tnN7FxcXFX1t3759g\/2xdnUpLbtD5vfUA3psaFcRGinlTR1iNbIY5bqCo0QIutoQ2o9J4Mfgmp+fV8eOHVNnzpxRnJ2RQTaw6fsGksy5c+ea+3ibLbpXBx5ts0WX3lW2lU1yoeBXenY3ZvjVHJsYcSsJgqOAnx5E7s7q8KO\/tYGN4Ed\/p12k19bW1NLS0qadZLi7fPXqVXXlyhVlZo9d55H4wI\/n2eU8ryLEJ2HuChjCBr0Msuf+\/ftqdnZ2MgWE\/p96ICm0XWZ5vnErAYK+Nvj6OEnmNy3wowpDs+p5cukzzzzTNCrJMjLfQMW6P3cFDGWX3p398ssvm5jQLuQcn1DPKaWcUHGjckJvsS\/1USgbpM\/L9s6PH1xT5mfrMo1xC6ihFSzWfRybtsybwJerQceyN3Tm11Zeap8Bfj\/uEF1Kt7dvsCJ3wEI3rprs6YtNCQ06dHxs5cWKW8pMMJYN0hgU0e3NMeCxvLzcDHjQF\/1M8wEpGHR1rZvNHTBpYKVyJdsTarAiZYOW+t1XLnbcUvgstg19Pi4Cfl1TXe7evdvM6yOZkydPqrapLmQkjfDaprq0ZZUMPfpO8wcPHTrU+KpvgXjugPUF1PX\/pdnjmt252JuiQbvo4yObKm4xfZbKBpufi4AfKUfgunbt2uR8YNskZ4IVH5xkm+TME6MZqlQ+AXRmZmaDH3xGe20z630qdI57c1fAUNmdi+9iNmgXPXxkU8cths9S22D6uxj4tS1dM5e3mROlyRhzeRsfscmGlra8zafCx7g3RwWMmd25+ChGg3Z5vo9sjriRviF9lssG9nty+PkEPPS9yPzaz+0N7WcqzwReqP0IQ+gaskGH0EdSRm5whPBZbhsAv4y7ukgqeWyZWBWwlOzOxX8hGrTL83xkY8XNVScfn+W2AfAD\/IKdBVEj8Noau0+DdoXHUPnc4DD1HuKz3DYAfoDfYPh1DVb0jZoPbfQp7xvSoFPplxscNjtdfJbbBsAP8HOC37Rkdy6QcmnQLuX6yOYGR5\/uEp\/ltgHwA\/w64ZdjKkpfw8r1f0mDTqVbbnBI7ezyWW4bAD\/AbxP8xpjdSRtz6OkeLs\/VZXODw1XvNghSGTi319WTgeQx1eXpVBfe0JMqKV0lTUUJFO7gxeTMBGuDHztf9xltJku\/7969O8uGssj8Rpr5mVtAUUX8+c9\/3lTCaRisCE66jgJzQLBW+OkQpLOHv\/vuu2wHrwN+I4KfrTtLexSur683n8IMvxq3dk8JvLZn6RCk\/9OHSKwPktrhx68P0O3NVGunvdvrOhUlRwaTKfRRH5sCgoCffwiR+U1Z5hdisAIQ9G9Y5sBI6EwQ8POPEeBXOfxiTkUBBP0bWCwIAn7+sQH8KoRfiOzOpeoAgi7essuG7A4Dfv4xAfwqgR+fr5BzKgog6N\/guAT99DSXgZFpOXcYAx7h6tKgkkoe8Eid3bk4EBB08Va3rASCJXzwhbP4aUm5s1dkfgVlfiUDr29qB6bI+OFBhyDNudy2bdtkD0QeLNG\/+z2tjLsBv4xxyJ35tQ1W1Hi4NjJB\/0rMH3z\/+Mc\/1Pfff98UuGvXrgaCseYK+mvtVwLg5+c\/r7tzwK+27M7FwYCg3Ft9czAl3WH508qUBPwyxiUF\/GJORcnous5HA4Lt7hnywTfNEAT8MrbgWPAbUskzuiHao8cOwZAffNMIQcAvWtPrLzgU\/EJW8n6t65MYEwRjf\/DpEKx9kAnwy9iWfeDHWz5xZZ\/WEbmQ4ZlWCNqAx3UkpA+5rGmAIOAXo2YIy3SFH1fyv\/3tb4qmIzz33HPq+eefn+x9J3zs6MVqh2BJo\/Q1QxDwy4iCPvjZKjm2gAoTtJDLvcJoZC8ldnfWV\/8aIQj4+Ubd4\/42+LlU8tozGA\/XBb21RAjW+h63JggCfkGbkVthy8vL6v3331f0ff\/+\/c1mnnS5buEOCLr53SadG4IuH3xhLI5XSg11EvCLF\/\/Okhl8JETgO3HihDpw4IDXbPoaKlwmdzs9NiUEcwxWODnDU7jkOgn4eQZ36O3U5eWLQEi\/E\/xOnz7dfPe5Sq5wPnalvjcGBEsarEjpzxLrJOCXsgZ0PIvgBwgWEowWNXwm+U5Td9Y3QiVBEPDzjWbg+wHBwA4NXJwEgrUOVgR2VWdxJUBwNPD76KOP1IULFyYBOX78uHrrrbc2BOi9995TN2\/etMo8efJEnT9\/Xt2+fXsic\/HiRXXkyJHJ759++ql68803J7\/T+7zr16+r7du3O9UtQNDJXcmFTQiyAph07haKnBAcBfwIfNeuXVM3btxQe\/fuVd9++606efJk826NAUgyKysrE1B98cUXzSDEqVOnJnAjOK6urqpLly6pmZkZxaD74IMP1MLCguJ7Ll++3PzOsKTqwPe4VQ3VvAtEd9jVa2nkCXR07us\/\/\/nP5oE7duzApPOBrs8BwamHHwPo4MGDmzK0c+fONUCcm5trMjpThoD44MGDBpAMzLNnzzZg44uAuGfPnqZskr979+4G0BEQFxcXmy\/X7E+vR4DgwFYV8LauwQp6DGd9LtvCB1RvKopKCcGph5+tRhCUCH6UpRGUKBM0wUaZ3dLSUpMNEvxYnrJHvnTg0Zw9uvTutA2aQ2sqIDjUc8Pucx2skLwTHKbJuO5KAcHRwk8CNpLh7HBtbW0CQj2D4+7y1atX1ZUrVzZljwy\/o0ePbsg8fasyIOjrwfb7Qw1WAIJh4hMTgqOEH3eFd+7c2WRpehaoZ3Ulw4+rFiDo38hcszuXJ9a03MvFrtSyMSA4Svi1DW60dWlrgB8gOKwZmsBzXVLo+lRA0NVj9sycfEng8t1PcHTw04HGWV7NmZ9ZRZAJdndnc09FAQTzQ5BmT9BF7+hpyhutqiKQpr6SHl1pTk1hY22DEpL3gqkHPKQBig1B0qP0Uc2Y3VlpHGxygKCvB3+4X9IdprZAbZm+87JSmuZGXzRzw3c56VBLksGPpqTcuXNnMtdPV9g2HUYHG8m0jQhTuXTRu0PbVJe2LvVQh7neNyYIhhqscPWxjzwg6OO9p\/eaEKS2yNkdS\/32t79tfqRMr4QrCfx45QZPRm4z3JwIbZvkTDDhFRu2Sc48MTrEJOdQQZpWCJac3bnETpLBuJQ3RlnqzpIf9VVa5AeCXinA0+MSHX4MsUePHrXWBwZi29I1cwkcd4\/v3bs3Kctc3mYuoxu6vC1W5Z0GCKYerIgVi7ZyAUG5t7k7y\/NrGXT0nbqz\/G4vxE5Jcq3kktHhJ1dlXJI6BEN9OsbYAop04+5s7sGKlDUEEGz3tj5YUWp3VlpPAD+ppyLJxYAgqeo7yXdaurO+YRs7BEsdrPCNK90P+IXwYoAyckOwxsGKAG4XFzEmCE5TdtcVYMBPXP3TCKaEILI795hOIwR9sztzGznzPXzKrehcIgr4uXgroWwsCP79739Xjx8\/VnT85k9\/+lM1Ozu74cCmhCZW\/ajaIRgquyPwffjhh5OdlErZik5SuQA\/iZcyy+iHLblOG7Bld7QP3vr6emNZ6ZOlM7u\/8\/E1QdAEnu9EY9\/5uSm2okO3t+TW46CbFIIu3VnfgREH9adatEQIhsrubIGzLUvV5W0yJazMQuZXYZM0IUhzqvbt2zeZkkIm8UYBlNVJLkBQ4qV+mdwQjA083QO8\/JT25KRVVDz\/Vn\/npy9RLWErOl1\/wK+\/PhcpoQOQFHz99dcns+ilwGszDMu9woQ7FQS7Bitir6rgBQX6QgKzKwz4halPKEXzgH7uMFUwnmXv+k7Q5lRAMEx1iwHBlNldlxfMJaksqwPv888\/L2oTYmR+Yep1caVI3wm6KA4IunjLLusDQd+pKGEs2FyKuS+nDr9Sd2AH\/GLVhkLKBQQLCUSLGlIIlpLddXnS1qXV9+yk+0s4e6fNDrzzc2wn\/E7jtdde23CKHBVT0rnDpI8OQZrWEGKBOTJBxwpjEW+DIJ1kWNqed13W2vbh1DNCur\/UregAP8e6bNueq9Rzh2NBUJrBOLp3VOK2LaDog+rWrVtV+MLcQ7PtwDBqMyVuRQf4CauYuZ2Wvjdh12R4Z90XAAAGaklEQVTPUs4dBgSFgY4s1tWd5UGsEBl6ZDM2FG8ubzP37Sx1KzrAT1BLOHjz8\/Pq2LFj6syZM815w3x4em3b8MfoDiMTbK9IOaeiCKr2qEUAP8fw89pFHX61HsAECDoGXyhew2CF0JSpFgP8HMM7TfBj02PvLu17xKFjiJKLlzoVJbkjKnsg4OcYsGmEHyDoWAl+HEmnu9q2cHddWUEDAqurq5OdUajcUreBcvdUuXcAfo6xmWb4AYLdlYG6s6GnovBgweHDhzfAzwSi7bAufv1S0mFdjk0qmzjg5+j6NvjVNuAhNdm1O\/zOO0q9++4Ppb\/88g\/f33776c\/0e00DI7Hf3emjoDr8bPWJgJh7Gyhp3alBDvBzjFIb\/Hz3NSMVSj13mHTrg+Annyj1yit2R3788UYAlgzB2MDTvcRnTtPf9G5vydtAOTaXosUBP8fwtMGPipj2c4dtEPzPfw50go\/d++O+qZu8nTsTbBusSHG4tr4E7E9\/+tMG+JW8E4pjcylaHPBzDI8NfmM5d9iE4MOHt9STJwd6vUjdX+oW266UEEyZ3bXZa\/YU2t7vLS0tqevXr6vS9sDrDXRFAoBfRcEqTVXKmujdngR+pLst+9PtigHB0qaimEvCAL88NRvwy+P3qXnqli2bTZmZ+awViBL4cWm+EMyd3dkC3PY+D\/DL0xwAvzx+n5qnmvAj8M3Pv6HoXeDa2ukJBClDpIEP10sKwdKyO5udvPtx2\/9ffPFFRTu70FXqNlCu8StZHvArOToV6KZPb2F1CYDPPbesfvKTzyYQXFg4MAh+XZkggyLEROOcrjYzv66pLqRnyTMDcvrR9dmAn6vHIL\/JA21dXxLSIRhqP0ECBWWDf\/zjHyd6hCo7V2jbVniUug1ULh\/FeC7gF8OrIyuzb57f4uJn6q9\/\/WF1hCuouqai1LoFlFk92uBX6jZQ01S1Ab9pimZGWwiAvLqDfqbLnN7SN1ma1S91sCKje\/HoCB4A\/CI4FUV2e6ANgjSxN\/S6WcQBHujyAOCH+pHFA+a5w6REipUVWYzFQ4v0AOBXZFimXyn93GF6D9h1ta2eMbdKxxZQ019nQlsI+IX2KMoL6oG2rZrM7Z3ogdgCKqjbR1EY4DeKMNdrpG2HEx12BEjb8YjYAqre2MfWHPCL7eGRlW+e5LV\/\/\/5NC\/RDuERfH\/vll19iRUQIp46sDMBvZAGPaa65403M3YV5Lzxa7YAtoGJGdXrLBvymN7bJLTN3KyEFCIiLi4vNl749k49yJmQBPx9vjvdewG+8sQ9uuZ6NceG2dapDH87Z5M6dO5s1rnQBfkO9Oe77AL9xxz+Y9bat\/Bl+R48eVUeOHPF+XttSMMDP262jLADwG2XYwxudAn7mYn+2AmdehI\/nGEoE\/MYQ5QQ2xoQfZ4\/z8\/Mbjnfs61rr3XDb+8i2ffMSuAuPKMADgF8BQZgGFWLBj8FHPjLPtND9hi2gpqEWpbUB8Evr76l+WowBj66dj\/U5hNgCaqqrVhTjAL8obh1noehajjPutVoN+NUauQL15vl3p06dakZ2Y05yLtB8qFSZBwC\/ygJWurpmNzXW8rbS\/QD9yvcA4Fd+jKAhPAAPRPAA4NfhVJo8S1Mh6JSwvXv3TiRDT9yNEFcUCQ\/AAz0eAPw6HMSQo802eSkVidugiNoGD8AD9XgA8OuJFb3DWllZ2TDHrG2JVT0hh6bwADxAHgD8euqBOYKJLi8aToke4IGmixcvTtZQc9395S9\/2boypkQ7UuoE+Am8rWd69+7da30PKCgGIvBAVA9QPb1z507zjnpubk6dP39ePXz4MMpmslENSVQ44CdwtH5mxJ\/\/\/Ge1urqKT1KB3yCS1gP6Guhf\/OIX6ve\/\/70yD3pKq1HZTwP8BPHRJ+vSJ+nZs2fVwsKC4E6IwANpPaAfI3D8+PENA3VpNSn\/aYCfMEb8TgWTdoUOg1gWD\/AH9V\/+8pdNU7SyKFTwQwE\/YXBs016Et0MMHkjiAX2FzeHDh\/F6psPrgJ+wSprnRghvgxg8kMwDXEdfffVV9etf\/1qdOHFC8TrrZEpU9CDATxisth1LhLdCDB6I7gHu7uqju1Rnr127hu6vxfuAX0+1xLu+6O0WDwjgAZrmcvPmzQ2ju3j\/1+1YwC9AxUMR8AA8UJ8HAL\/6YgaN4QF4IIAHAL8ATkQR8AA8UJ8H\/g\/HYJ1A6ldl5AAAAABJRU5ErkJggg==","height":174,"width":232}}
%---
%[output:32c807f5]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tXV+IVscVH6FBfWi0Nm3XpCYptTYPwZKXGF+a5DEV2qbghj4ooSiCJGiKslsUmkAsu1VaiqFUlD7EQrA+pBRi06eYvBhLIUF8CfhgYjHaB1kDZbV92HLu7nzOd3funXNmzvy5954Pll39Zu7M+Z0zv3vmnPmzYmFhYUHJRxAQBASBgSGwQshvYBoXcQUBQaBCQMhPDEEQEAQGiYCQ3yDVLkILAoKAkJ\/YgCAgCAwSASG\/QapdhBYEBAEhP7EBQUAQGCQCQn6DVLsILQgIAkJ+YgOCgCAwSASE\/AapdhFaEBAEhPzEBgQBQWCQCAj5DVLtIrQgIAgI+YkNCAKCwCAREPIbpNpFaEFAEBDyExsQBASBQSIg5DdItYvQgoAgIOQnNiAICAKDREDIb5Bq76fQc3NzlWDXrl1T8\/Pz6uGHH1Zr165Vq1at6qfAIlUQAkJ+QfBJ5ZwIaLLTv6EvQHR3795Vn3zyifrWt76l7rvvvur\/hARzaqrMtoX8ytSL9MqCgCa5O3fuKPjRZKc9OyA4+Ny+fVtdvnxZPf7442rlypUK6unymgTFGxQTE\/ITGygWAU1yJtlBZzXJ6d91AUzyW7NmzehrIEFNoOINFqv2ZB0T8ksGtTSEQcA2lXWRHZb8dDkgU+0NCglitNLPMkJ+\/dRrZ6RqitsBKekfqjBNnl\/9OUKCVGT7VV7Ir1\/6LF6atridL9lRPT8XCWpvsHgwpYNBCAj5BcEnlV0I+MbtXM9t+x7r+bWRIHwnU+IQLZRfV8ivfB11qodmcsJcgkKN24UI7Ut+ZpuSHAnRQDfqCvl1Q09F9zJG3C5EYA7y0+1LXDBEE2XXFfIrWz9F9i5F3C5EcE7yExIM0UTZdYX8ytZPEb3LEbcLETwG+dlIEP4PpvNN6w1DZJC68REQ8ouPcedaKCFuFwJaTPJrIkFJjoRoLE9dIb88uBfXatviYq4lKKmETkF+khxJpc147Qj5xcO26CeXHrcLAS81+UlcMERb+eoK+eXDPmnLXYvbhYCTi\/yEBEO0lr6ukF96zJO02PW4XQhIuclPkiMh2ktXV8gvHdbRW+pT3C4ErFLIT5IjIVqMX1fILz7G0VoobXFxNEGJDy6N\/OrJEe2VS4aYqFjm4kJ+zIDGfJwtbgenFsOBnbLe7B7yJZOfxAVjjhDas4X8aHglLY2J20GZGzduqImJCbmrYkk7XSA\/IcGkQ8namJBffh2M9YAatxPyW67ALpGfJEfyDUAhv3zYVy2Hxu2E\/PpBfiYJ6sMU4P8kLhhvgAr5xcPW+mRb3M7cQUHdJyrk1y\/yM6WRE2XiDk4hv7j4VreG6ZvDYpxvJ+TXX\/IzvcGbN29Wt9J9\/etfl2s4mcaskB8TkOZjqHG7kC4I+fWf\/EBC0POnn35aTYMXFhZkOhwyaJbqCvkxgBgatwvpQlfI79VXlXrttUVJn3lm8fcvf3nv7xAM6nW7mPBwyW\/qWceK5S5iF2rt3wv5eeDHHbfz6MKoSunkd\/68Us8+2yzhe+\/xE2Dfyc+8cF2O2\/cfPVnJ7+jRo+rUqVNjvd+2bZs6cuSIWr16tb9UzDVLPhSgJPKzeXdAfq7PwoKrBO37IZGfRkaSIzQbgdLZyG9+fl4dOnRIbd26VW3fvp3e88g1UsbtQkRJTX42gnv66XtTWh9ZYPoLz+X6DJH8hATp1pON\/G7duqX27NmjDhw4oLZs2ULvOXONnHG7EFFSkZ9r+hoiA9Tl9P6GTH5NJCh3ES+30Gzkd+XKFTU7O1v9rFu3LnTskOuXFLcjd96oEIP8fKevIXII+bWj56tnczoMLcii6Xs4ZyO\/s2fPqsOHD49pPGa8r+S4XQhp+A4KW5uxvbsmOSH7C4mP+sc3Qyyenx1pSY6M45KN\/CDZce7cOXXy5Em1cePGqlcXL15UZ86cYUt4tMXtqDspQggqZl0u8stFfIBNnfxcfXFliIX83F4kjA2wnSF7gtnIz6ae0DhgV+N2IeTIRX6wHAWTmQ3pq6uuXv+H6UfbNFnIz4X04vdDzxAXSX6Tk5OoDHDbJTyg3L54d22m3Cfyww3ZxVJtGWIhPwqS4ySox80Qxk4nyO\/48ePqH\/\/4h3r55ZfVpk2bxvbLmiQ3BIXVzZqL\/FasoA2YEko3eX9Cfn7aGVpyJAv56TV+69evVwcPHhxpCqa9U1NT1Y+OA8KXQHyaAP\/+97+PPLohkp2Q3z0EhPz8SA5TawjJkSzkp5Mbx44dUydOnBgtdYEM8NWrV8cIUSsKCHDHjh3q9OnT6sknn8TobxBluDw\/M7PaBeCaMsTQd98scclyc+mZKmOf44LZyE8T4M6dO0f62LVrl5X4tPcn5LfcdDkHBWbqW09KwL8xCQrqoHOVt5FfaJbY1WbO7zn17CNHH0kwK\/lRlCCenx0tzkHhQx65ssQm6XJliSn2mLosp55D+l6PC3b54iwhvxBLKKAu96AAAtRHT2mPri2zmov8fKHn3kfs2w9qPW49U9u3xZr1WkH4rovrBYX8Qq0gc\/3cgwIzVc4M0bLmObfSpZItt57b5NQk2LVF00J+qaw3Uju5B4WQXyTF1h6bW88YKbsWFxTyw2i14DK5B0VfssSlZ4hz65kyBLpCgkJ+FK0WWLaEQYHx\/ihJiZgwc+8jjtlX89kl6Jkqa+nJESE\/qkYLK1\/CoKBmiUtIklDIuIQYYQl69jV96LsmwpKSI0J+vhotpF4pg4KSJS6B\/CjqKyFDXIqeKbjZypY0JRbyC9Vm5vpdHBSYaXJmWIvLEHdRz206BHngLmK4hvPRRx\/Nom4hvyyw8zXaxUEh5EfXfxf17JIyt0xCfi4NFf59bgPygacvGWIf2X3rdFHPLllzyyTk59JQ4d\/nNiBfeDDeHyUp4dsPTL22QxQw9TnKdFXPrqnvjRs31MTERLVDJPVHyC814sztdXVQdClD7Do2n1ml1sd1Vc9CfgzWIQcb2EHs8qDoQoa4BOIDzXdZz03DP7dM4vkxEHPOR+Q2oFSyY6bJIX2pT7FLWN5iytNHPeeWScgvZMQUUDe3AcWEQC+Ohd\/r109Ea6oU767kKWIM8HPbrpBfDK0mfGZuA+IWtem60ZmZVWp21j8oDp7c++8v9lYf1bV\/\/5yamrpTXYuQI+BOwa5vei5hKi\/kR7HAAst2fVBQrhvFTH0p09eSdhu4TKvrerbJl1smIT+X1RX+fW4DosLTdt0oeF9tHpgrQ6yJb\/\/+2+qBBy6rxx9\/XK1Zs8bZxToJ6oM5nRUTFuianjHQ5JZJyA+jpYLL5DYgFzRm3A7+1h998x71Bj5bhtjWh+PHL6sdOzagyE\/XN0kQ\/q+k04lL17PLDsTz80FoqY4sdbGDV9qg0GQHvdVeHvztS3ZQ13bWHubSpLm52yTyMxEu7erG0vQcMJRHVXPLJJ4fhxYzPiO3AZkkZ5KdnsK6prJt0LmmuS7YOZarlBIXLEHPLryp3+eWSciPqrHCyucwoJC4XRN8vt6dSx1cZ\/HlJsEcenZhG\/p9bpmE\/EI1mLl+CgPijtuZkIV6dy74ucivKS6Y6urGFHp2Ycn9fW6ZhPy4NZr4eTEMKEbcLiR25wtpzAMJUidHYujZF1euerllEvLj0mSm53AZUNPiYhCLErezTV+ffvreXcApYYpJfqYcKZIjXHpOib+rrdwyCfm5NFT4974GxB23iz199VFD6m1rMeOCvnr2wS1VndwyCfml0nSkdrAGxBm3i5Wc4ITIZ50fV\/sxSBCrZy4ZUjwnt0xCfim0HLGNJgOKEbcr0buzbWd75ZXb6vJl\/A6PWOqpxwVDkiO5iSIGRrllEvKLodWEzzQNCP6Gj21xMSVulyM54QNZ07T29u0yyE\/LxJEcyU0UPvpx1cktk5CfS0MFfw8kd\/fuXXXlyhX1wAMPqJUrV46SE1Sy02J2xbuDqXfTJwb52ab6sIhae55YMwGdaa+csn0uN1Fg5aOUyy2TkB9FW5nL2uJ2QH7w841vfKP6oXy6ELvzSVpwkp\/rZeDTP9ARNS6YmygodoUtm1smIT+spjKUw8TtfAzINaAziDryoPSe3ZCtab7k5\/syCFlIjSVBHz3n0COlzdwyCflRtJWgLHW9ncuAfAd0AlFHTfh6T1zT3tCXQQhRN8UF68kRl55T6ourrdwyCflxadLzOZTDPG1NNBlQ6ID2FKexmu0kZQ7SsDXY5vnFehmEeH+mDNrb13Zhni2Y85pHbnvQU\/+cMgn5xdBqyzNtcTszOUE9306T3x\/+MDE65h2C8Jgjn2KITjlJOUb78Ewb+cV+GXCRX50IdYJkxYoVVZzwkUceKf7IfaxexfNDItXV8\/wwcTskBNZz7XJtHbP1mXv6isWlXi6Wd9fUn9jb6MCGbt68qf71r3+pb37zm1Viq\/Q7RzC6E\/LDoKSU6hL5UeN2Lghiey2u9uvfl+DdQZ9Sk1wu8tNTxE8\/\/bQivYWFhaJOmabajy4v5IdErmTyC43bmRCUMqCb1FKCd1fayyAFJiZRgG70dBj+1rfPdc0bFPLrIPlxx+0AgtIGtHh3OMNMQXxtyYEUJ8rgkKCXGjT5Xbx4Ue3cuXOE2ubNm9WJEyfUunXrliGZ0\/OzkZ1+45q\/seoX786NlLwMxjFyEQV2vaAb+XQlXDLF7km2bC9sydq9e7eamZlRW7ZsUfPz8+rQoUOVvEeOHFGrV68ekz01+XHH7cS7azZleRm4hzmWKLpEgliZ3Oj4lchGfmfPnlUXLlwYIzogxNnZ2eqn7v3FJj\/OuB2oQga02yDFu7NjZLOdqak76rHHbqiJiQlUprdOgnIX8XKss5Hf0aNHq94cPHhw1Ktbt26pPXv2qAMHDlTeoPnhJr8YcbsSvTvok7nuL9bCYhfVycvAhZA77vvWWzfUj3+8FkV+ujWTBOH\/KIcpuHscVmKQnp+e4m7dulVt3759GflNTk6O\/T8UCCU\/7rhdF7w76GOqgHzTMBDvDu\/dYRamz8\/fIZGf2XppyREhv4jk1xa3o+6kMI2otAG9f\/+c+vjjtVUXOQ4G8H2fi3fnRi7Udjg891LigkJ+jOTHHbcrzbuzLT+Znr6jcu6PLHGqDzjBFrePPlpTsREHYbhpbXmJWC8Drq10uUmwWPLT8TfbFBTUDMtUjh071rg0pc1YuKa9bZfwQPt98u6apq+pDSjWgPYhl6a1iL5HWvn0wVYn1Ltz9YOL\/JrigiHH7bv6bn6f2nbrfWtMeGjyu3Tpktq2bduy5Sch5AedoCQ8jh8\/XsX84Af+hvWAAJz+aJILIbtSprXUxcUxDKik6yebBlNbLDMl+aV+GcTcR5w6ORLDdink20p+U1NT6ic\/+Yn69a9\/XT3z5MmTauPGjdXfoeTXtNRlenq6Wvun24G2NOnBbyDN73\/\/+1UfuMiuDtizz+Y5FcUnOcFpQLE9Foph6rLUlwHU4ya\/kl4GMcnP1E+K5Ain7frYlpP8gAAfeuihagHyO++8o15\/\/fUqExtKfnqR8969e6vnlbTIOTb5+QzoJuX6GlBqj8XHOH1eBpzkV+LLwBcTH\/yhTsy4oK\/t+spCmvYC8cGP9sLAWzt8+HA1Df7Rj36k3njjDa+Yn+6Efp7+dynb21as4IJ3+XO4jZdqQCUOaM6XgS\/5deFl4LPOj8uSY5Ag1Xa5ZNHPQXl+5hRUe2zXr1+vYm9Ne3G5Oxq6zo\/SHw7y4x7QPp5fFwY098uASn5deRmUkNWve4I69OQbfuoc+YHAeop67dq1XpKfSRoU0tRlYwxoCvl1ZUC3XT\/pg7uu0xTz6\/LLIDdR1PXBkRzJLVO27W1U407p+UHfMN5fKu+uDasuD2iqDWDL18mvDy+DGEQx9LuIhfwaRpRrwKT07mxddPUPSxSc5eRlgEPTx3Y4yc9lOz79q0+JMXuIOWXCIT9eSsivBTUwktdeWywg28bazct3wPgY7RBfBr5E4TszCFlIjU2O+MrEZTNCflxIRniO6w0docnWR6a8flKm+uMIUIki1HY4tgTW44Kl3UUs5JeaQVraq7+lMad8xOh+CdPXLnh3KV8GbeTn6925bCfE+zOfDX3XRAj\/X8pdxEJ+LgtI8H3oW5qzi7mnr1qWWAPaB6sSXgY28ottN1zkVyfCUu4iFvLzGQ2MdWIbcFNXSxjQ0LeSSM6G1VBfBrG30QGZw13EkJnftGmT9xmFIUNRyC8EPYa6sbfSlTqgc5F+yS+Doe0hpsYxGYbb2COE\/LgRJT4vJvmJd4dTRm7vrrQXAaCWAhMhP5x9Bh9jj2wmeTHMYmqfTqUwXle\/ShvUpb4MciW2mvSXynaE\/FwjaOn71Ds8kN0KLhZKfjCgwYg+\/HBV1ReOJQo+Qknszo2avAzGMRLyc9tMVaKv5Beyjxje0E89lfcY+9IG9BNP3K7sRR9h\/8ort9VvfrN4nH3Kj7wM3GgL+bkx6jX5gXAY76\/p+smUBlT6gD5+\/HJ1Tef999+vvvSlLyk4jRyyiWvWrFEbNmyofsf+lPYyKGWqb8M9pe3a2peER+zRgHi+a8C0xWBSGJCrfwgRWYvUvTuY6v\/sZ5+phx9+eFk7QH5w+hD8ho8mQQ4iLP1lkCp256vcFLbb1jchP1\/NMdfz3UfMbUClD2jw7n74w\/sr9G1k16aWzz77rCJC+IR4g6W9DEr27tr0wW271CEp5EdFrLDyXAZU2oC2eXdc5\/+Z3qCLBEt\/GZTu3Qn5MRBGXxMeodD4kF9JA7pOcjB9feWVL6q4HRATx\/S0CeM2EizpZaD3EJeQ1Q+1V7O+j+1yti+eHyeaGZ5FMaCSBjRABVNYID8guBRk16Sen\/\/8tvrtbxeTIfUpZGqVlpDYSiUzxXZj9EnILwaqCZ+JNaCcxGfz7sCb8Ynb+UJb0taxJhlyJ7Z8sfWth7Vd3+e76gn5uRAq\/HusAcXcRtcGkfbuIMuakux0n3KSfhMuPgkKrJ4LN9ex7uWWScivS9Zi6SvWgGKTn21x8f79i1PamHE7E5KSYpk+3l2bKWL13CVzzi2TkF+XrCWA\/DALqX2hMBcXpyS7vnl3Qn6+FuhXT8jPD7diamHfnhzkR1lcHAugPnt3Qn6xrMb+XCG\/tHizt4Ylv5A9xNDpkMXFHEKXGruD5TIp9hFj9cyBdapn5JZJyC+VpiO1QzEgjPdn8+7gcIBUcTuAqWveHdfOEfH8Ig2ShscK+aXFm701Cvm5vKc33\/xMPf304tavlGTXl9gdZecI1RAoeqY+O1f53DIJ+eXSPFO7WAOCgQk\/f\/3rF+qPf1xcdqKnaznOAOyad0dRVwwSxOqZ0s\/cZXPLJOSX2wIC228yIE128Hi9mR\/+zrXerlTvLub1k7YTZaiHMWjcchNFoJlaq+eWScgvhlYTPtM0oH\/\/+99VyzayyzWVLeYuYnW+wua8eqb6\/ctnzqtX31v8O\/anToKuwxRs\/clNFDEwyi2TkF8MrSZ6JgTa4fPPf\/5TrVixQq1du1Z97Wtfy7pPtkQP7z31rHpmifzGVJPhSBTQ2RdffEE+ZDU3UcQw6dwyCfnF0GqkZ+qprB48upmJiYlqQH3lK1+p7j8FEoTfuT6uxEqsfi3bNqZeVa+q19qbi3EzN0JAalwwN1EgRCIXyS2TkB9ZZekqUON2YExzc3PVhUbw0SSYmghjb6WzaWCZE4ftRI5sjyEAlgRzE0UMq88tk5BfDK0GPFNPZUPjdkCC8AOf1N4glnd8YEIfCkDpRCbvz5TflRzJTRQ+unLVyS2TkJ9LQ5G\/t5Ed5\/l2pjeYigQxi6l9YCWF6CidKID8NB7a268ft79y5Up148YNBSGO1J68j64wdYT8MCj16OpKW9xOkx1A4bsUwgVjShKk8I6t32jvrk1obCegMWDV+se2EBGmyLpzLsAZvje9wdWrV6v77rtPfec73xHyY8AWHiGeHxOQTY+hxu0id6eKB+q4IHgQ2hvkbDdkHzHJu2vrNLYTdfJzZWvYOohDHHQFNvTJJ5+o\/\/znP+qRRx5R3\/72t7PswMH1GF9KPD8kVl26w4MrboeExqtYPTnCPSXGOF6w\/GS07g4ys7DujtOzwnTCJDMX8WmkI06TdZxW\/4Ym9Uvqf\/\/7n7p58yZ5mYyXgSSoJOSHBLlk8osdt0NC5F0sRnLExSNJ1t45O\/HeONlikySMGWIgAPPHJDv4GzL29U\/Mu4i9jcij4qDJ7+jRo+rUqVNjsG3btk0dOXJEQYzD\/JREfrnidh72RarCHRe03kWceu0d5UJkLPkBqp7en43sTJKzkV2bElOcKEMyIkLhwZLf\/Py8OnTokNq6davavn27E7Kc5Ne0uDj3PlknaJ4FuElw1A0suTB6ViQIMNNkj6mvbSqrSU5PaUn9tBTGrhcMbYez\/mDJ79atW2rPnj3qwIEDasuWLU5MU5NfF+J2TtACC9gWTVM9k7EuYMkvwLMKEhlLfk0ZYmjcSLTceeqpqjtz+\/ZV02tNdDGXqnSJBAdLfleuXFGzs7PVz7p165w2G5v8uh63cwIYUIAtOYIll1zk55khrnbUnD+vVj33XDPKibPEdRKEe5FjLaPyNa3Bkt\/Zs2fV4cOHx3BrivdBIW7y62vcztcQsfWCkiNY8su59g7Rxzt\/+5saeXVzc2rVhx+qiZ\/+1A2hZ5zQ\/eDmEhwnyoS031Z3sOQHyY5z586pkydPqo0bN1YYXbx4UZ05cyZKwmNocTuywRIX9XrFBT09K\/CqFEyZmz6cXpWjrRtvvTUiPh0CWPv885Xn5\/zkimUuday05Mhgyc9mKG1xQB\/Pry1uV9oUwDlwYhUIJBYyCSI8q2rHhbnNo434NC6cXtX58+rOL35RPRm8uiput3+\/ujM1ZY\/blR7LrNlOKXFBIT9DMZr8Jicnl2WAMeQncTsiQ7qIj0As6OSIq826F4cllkCvKmgJCobQCVgStehdPDcJ9p78ILGxe\/dudf369UpJDz744NhU19RcE\/kdP368ivnBz+nTp9WTTz5ZVdNkZ55vl2KfrLe1lVYxArGgkiOFrL1rW4JCympjyS9nLLPF9lwnysQy296Tnw04vcZv\/fr16uDBg6MiQH5TU1PVj44Dwpc7duyoiA8+sAD6scceq7b46E9f19vFMrrRc7HkBxU8ppV6DzEYuff2OSyxIPrYtnUsaL1dF2KZCGNKnRwZJPmBHiC5cezYMXXixInRUhfIAF+9enWMEMHre+ONN0aq++53v6t27txZeX8St0NYdFsRRmJpa4YcFzQfhu2jxauqSHdmRqnXFk9z1hlaiN1BTJHk3bmgxvQzdyzTJYPxfYrkyGDJTxMgEJn+7Nq1a4z44P+1x6fL6CkwkB\/8vPzyywSVStExBDADFiowTde8SJDgVVVLUJb2ykL2tXX5CWeGGDAqNJYZavEx44KDJj9fxQAhahKEZ2gS1LFA3+cWVY+49MSr7wRiGTvzjjrQa51DJ0d0PQRJjy1B+fhjVS0\/cX08pvKtjywklukS2+f7GCQo5OejCaOOOS3uBQkGEgsZTgSxxFp6or0053H7Dkzm3n57bPtYtSaw9HV3GNy1MrlJmmwk9ypwJkd6QX4QvzOnr5s3bx6L5bVhrZMfL7zwAmqPb9OzTG+wsyToIr4Yg8HVZqKlJ01T4tESlHffVWt\/97sKAb32TjUtb8GSHzwsF7Fgya9tH3EAiYVW5UiOdJ789FKWmZmZirw0mQG4tqOp6qDrY63efPPNIPLTz+00CWIHbeCatmWGX9B0DbxAOLDz7t276r\/\/\/a\/68pe\/rOD+itFuCsv5dsvkwRJLE\/mVHHIIZa0I9bt6F3HwMfaQob1w4cIY0WEOLdBr+i5dulSpg4v8bCQI\/\/fSSy9V5Fp0XBBLfl3wWAh9tC1BgfMcwbtYWFiojmwn3UWMJb\/cR9hj+smdmIlAfvqR1Lhg5z0\/8NzgU1+v13ZclSY+WJ\/34osvqn379intOXLrBjxBmJbr5TJFT4kxgyHG1JcCOraPyAxxNd1tOfLJlhxxrsnzSeS4pv8xcHe12SHiM00IS4KdJr+mA0nbtqnVx1l92kwZh9SyxU+JQ4kFAIk9ZfMgFq4jn0gnymCwNMkF63XnDDlQDT5zeVdyRMhvaftbLM\/Ppv9iSdCDWEbypfQiEMSiM7BAWNxHPqHWC1LxwJIfYTqfmXuKab7Uu4iDYn5d8\/zq1lBfLwhxweyLphHEMrb0BIRyDXTuKRvy2KfR1ZiRjnxykiAlkYPBnRvHYugpXUdKuot40OTXlBzJGhd0EZktDoT1WhinbHN\/+YtaNTtbQaiXnsC2sTvT06Njn0ZDCts\/T6+qToJedxFjya\/QpSfp6Cu8JVibC4eRQJITPjoJCY5HyoRkEPlBx30SHiZ8KWN+GLWZJ8hkI0GKxwJCRSYXaKILRz6hTpRpMoKQkAPGsAZcBsYUfMw9+np7KqzA0N93jvyalrpMT09XGVzzdBab\/ksjP5s3mI0EsQMG67UQPatijnyCfhMTOaTkiMYZg2NHM7BYU+Iop1dY6GPotHdnengc7YQ+I9jz0+S1d+\/e6gBS6iLnUsmvUySIGbQgEHH5CZyCYt445n3rWIhX5RMGMEaFMy5ojqDAtkIHY1fr28gOZIEYOnyyx9EbgA0mP3hu\/TKi+vY2m3eo+1M6+dlIUL\/JUrvpjYPDk1zuvPtuuhvHMARd96pcZKQBQWxRQ5MgNeTQVcYK7LdtKqvJrvjNBEuys5BfII6dq17kYQoIcjFvHQPiS3rjmIvIEiVyyCfKdM4643TYFbejJip8zgPQW2FNCdtufHQhIeTnQqjl+6LWCyKXn4A41Vax5567d0BAGwaMGeJqSc7SwaKjU1fanh8xkROUHAmwma5UjTmV9TkPoGlZXQieQn4h6C3VLYYEKbeORSQWBkgXH4HwZkdtIaa+Tf3ySo6wCVnGgzTZQW\/MrGyMuJ3PeQBtNzv6Iijk54ucpV7apW\/AAAAIIklEQVSORdPeS1ASEUsQvNg+Mq29Q8cFg4Qqp3KuuJ3P8jjMYSlUZIX8qIghysc8aXp0TDvcJTs3N+oN6cgnileFzBBXHYEprL5vF4GTs4hnIsf5XEeBvpKgJjvbEpRUSQrfXWH1pCqoMCTeVw2BBTgzSD7REOCYEke5dcyXWHwSFyHoYry\/SGvvup4ciRm381WpL\/mBt3ju3Lmxa28haXLmzBnUuaG2\/gr5+WqRWI9CgprsTC+PZb1dvc9UYnERn34+5\/vU1WYk4jOh6lJypG0qW8J6O1\/ysw230DigkB+RxEKL10kQlgjAlGPTpk33bh5baoQ8laV2jkos2CQJZ4YYZKJmiak4EMqz3EVMaM9VlHsJiqu90O9jkN\/k5GS1wYL6EfKjIsZQvn4XMTwSFobv3r272tjNep+sq78UYsGSH7TJ6f25ZMjwfa64YFe2jrWpxCfh0eb5CfllGAC+TdbvIobYhQ5CF72PGDNNjjH19QU6Qb3YJFhi3C4UVup5ANpbXL9+\/bIT46emphT8uM4QsPVZPL9QTTLWp8QFGZvFPwpLfkxLT\/Ady1+SMzlSetwuFG2f8wDAQTh27NjYrZBAolevXh0jRErfhPwoaCUqWywJ+maIE+FWQjPou4iNznYtbseBs+s8ANvUuL4lbteuXd7EBzII+XFocukZPvsV25qPuV7QW2yM95cgA+vd\/4QVm6bEfYjbJYQxWlNCfkzQ+uxXpDRdzGEK1AwxRcielv3ggw\/U+++\/X0n3pz\/9aSRljK1jPYUwilhCfkyw+uxX9Gm6iCkxJUPsI2QP6jTF7fS+2dOnTyc9sr0HkLKLIOTHBClX+h7bnSJIENvZAZSjbB3bsWNHRXwlLDoegGoaRRTyY9A+58JNandyHKZA7WMfy\/dxCUof9dQmk5Afg8Zzkp\/ufpHJEQZsS3pE35eglIR1ir4I+TGgXAL5mWLIlJhBqUqNbhVrunWMenoxT6\/kKVwICPkxIFka+dm8waJ3jjDogOMRlLhdaHvaZt55553Ro15\/\/XWvPaqhfRlqfSE\/Js2nTnhQum07TEGC7araUmhuLdSYpliCAvby+eefj45j0mtE4SJvOOhCPvEREPJjwpi6X5GpWdJjJC5on8qmIDtTUU1HMQEhPvroo+L9kazav7CQnz92YzV99isyNe31mGIWTXv1Hl+pxK1jYCvT09NqZmZmbEN+2xWveImlJBYBIT8sUohyrv2KiEckL9K35EgXto7ZNumD4sF+\/vznP49t3k9uEANqUMhvQMpuE7WrJJjy1jEuUxHy40Iy7DlCfmH49a62bdF0qsttsGDmunUM2z9XOSE\/F0JpvhfyS4Nz51rRHpVe45ZzqUyJcbsQhQr5haDHV1fIjw\/L3j4p9ZQ45xKUFEqUhEcKlN1tCPm5MZISSwjEIsEuxu1CjKJtqQs89+DBgyGPl7pIBIT8kEBJsXsIcBym0LepLNU+YE0f4HjixAm1bt26arH1zp07lSxypiLpX17Izx+7wdekLJpOuXWsC4rR3t+lS5dG3ZXtbWk1J+SXFu\/etmZbNJ1r61hvQRbBWBEQ8mOFc7gPs91FDGhAllgO7hyuXZQsuZBfydrpUN\/qdxFD14EQhfg6pMSBdVXIb2AKF3EFAUFgEQEhP7EEQUAQGCQCQn4DUnvIvcL68M0XXnhBzpsbkM30WVQhvz5r15At9F5hWJd26tQp1Do0n1OKdf+uX78+phFZ9zYQA80gppBfBtBzNOl7r3B9PRqGjHxOKW7a75oDK2lzGAgI+Q1Dz8rnmH1NfBs2bFAvvvii2rdvX3UAZ9sx676nFMtBngMxxILEFPIrSBmxusJxwVJ92tzUV99N+3KEeyzty3ObEBDyG4BtpCQ\/n+OabFu9QC2YKfYA1CciRkJAyC8SsCU9tnTy017lD37wg9GJJpJdLsmC+tkXIb9+6nVMqtLJr0kFEgccgHFmFFHILyP4MZquLxl58MEH1cmTJ9Xbb79dNWeeFdeUnLD1Cxvz85n2tpGfXOgTw0rkmYCAkN9A7CD0XuE6+dnidBCj++pXv8p2LaPcZjYQ48wkppBfJuBTNxt6rzDW8\/M5pRi8RbjHFjzUjRs3jqAB8rt69aqcbJzaWAbSnpDfQBQNYrruFW6LsWHJD9qhnlJsi0kCiU5NTVU\/JiEOSF0iamQEhPwiAzzEx7tOKbZ5h\/UtcTpWKcQ3RAtKI7OQn1LV\/Qm2aZcepJOTk2r79u1pNCKtCAKCQBIEhPyUUprk4OBNMxvaRIpJNCONCAKCQFQEhPyW4LVlFusb9KNqQh4uCAgCSREQ8luCu54NlSlvUjuUxhwI6GSVecObttknnnhCHTlyRK1evVpwJCAg5GeAZXp6cKWgLQ5IwFaKCgKsCIB9njt3rloS9NBDD6lDhw6pa9euje7+ZW1sAA8T8jOUbF4c\/cEHH6jPP\/9c3qgDGARdEdE8Yux73\/ue+tWvfiWHPwQoT8jPAE8vt4D\/gjfqgQMH5Mj2AOOSqvwImFcR7Nq1SxaAB0As5FcDT8dWNm\/eLNOJAMOSqnEQ0C\/ojz76aNmOmDgt9vepQn413TYte+mvCYhkXULA3KWzbds2CcsEKE\/IrwYeZRtXAO5SVRAgI2Cee\/j888+r3bt3q71798oCfDKSixWE\/CzT3gsXLsgb1dOgpFocBPR018zughf4+9\/\/Xqa\/npAL+S0BJ7E+TwuSakkQsF0dKvG\/MOiF\/MLwk9qCgCDQUQSE\/DqqOOm2ICAIhCEg5BeGn9QWBASBjiLwfxTG6X3ZziYaAAAAAElFTkSuQmCC","height":174,"width":232}}
%---
%[output:945b79d4]
%   data: {"dataType":"text","outputData":{"text":"仰角 (弧度): 0.8737\n","truncated":false}}
%---
%[output:3f910af1]
%   data: {"dataType":"text","outputData":{"text":"方向角 (弧度): 1.2216\n","truncated":false}}
%---
%[output:6e18f158]
%   data: {"dataType":"text","outputData":{"text":"仰角 (度): 50.0613\n","truncated":false}}
%---
%[output:99974e03]
%   data: {"dataType":"text","outputData":{"text":"方向角 (度): 69.9917\n","truncated":false}}
%---
%[output:1df90618]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tfX+MXcd13llYSf3oFDa4CiRSfdACIiA3TsgYDkgvnNguXNuJJaB\/uCRbFGIYlMQ2jAUGKAmy4DVQwCLKBZcAAzFsCLJww0WQkIoLBBCJWK5R24pB04ghi4nTuJQBWrJEpfJj\/SPmqrHdV5y7nLfzZmfunJk5M3fue3OBhS2+mTPnfHPmmzO\/Z4bD4RDKVxAoCBQEpgyBmUJ+U1bjxdyCQEGgRqCQX3GEgkBBYCoRKOQ3ldVejC4IFAQK+RUfKAgUBKYSgUJ+U1ntxeiCQEGgkF\/xgYJAQWAqESjkN5XVXowuCBQECvkVHygIFASmEoFCflNZ7cXogkBBoJBf8YGCQEFgKhGYKPK7fv067NmzZ1SRW7duhXPnzsHGjRunsnKL0QWBgoAZgYkhv5deegn2798PJ06cgB07dsDKygocO3astvz48ePQ6\/WKHxQECgIFgRECE0N+zzzzDFy7dm2M6JAQFxcX678S\/RWvLwgUBGQEJob8Tp48Wdt1+PDhkX137tyBhYUFOHToUB0Nlq8gUBAoCAgEJoL8xBB3fn4edu7cuY78du3aNfbvpfoLAgWBgsDUkt9Xv\/rVUvsFgYJABghs3769FS2mjvyQ9J5++mko5NeKv5VCCwJjCCDxLS8vt4LKVJLfE088AYPBVrh790FW0DdseB1mZ29Ekc2qqCSMT2fcTnQnlppjckc67\/0E3P3U\/2oo8+0A8H3D702\/xTFDj\/XPAsA\/3Cvw5wDg7+MU7inVzT+wPb2uKQl3Wqys+3chG8mvjehvIsgPUaUueGDEh+T3yisfgZWVBzxdwpxtdvbFmlTDZadrCHw6c8F5HwD8pFFYrfMfHIaVnc9xFZpEzhrWDwPAmwDw1nv\/m6R4QyHvAIDvGRXg8Y\/1dvZ6fwf9\/nN15FfIL6D+TVtdjh49Wu\/927JlSy09jPzUCtQ1UlPvh6U3\/RZg\/FRllTqFaxXA\/FMZW+\/Tgak+NgcAtzK2UafaLwLAX1t1LuRnhYiWQGxyPnDgQL2ya9rkHEZ+NF26lep+APhut1QW2n6uAvgwkp89UmzXwNBOz4VE1bSxyDPcbwr5MXolRn9VVY0k6o63FfJjBDyZqH8CAN9ZX9qnK4DfokR+hvzJ9A8pqMu6N9tdyC\/ELzzy0smP2lu79MoeCk98lub5pkbzn64AnqSQ38SD6GBgeMTmUFhj0kJ+XEgS5dDJjyhwlGxye+hVE7ntCyA9gfl\/qgD+Qwj5lY7L1cs50xfy40STICse+amFc5EFA0kQcAlPQo2Uw0saSThYAfxeCPkx6mIUZao\/eZ4y9Zyl6puU8in+TJGzBlQhvxT+J5WxnvxS9P6xJp0Tg5dNcfdWE49UAItIfvIKqUrCgnzkNG6NNBuztYq0bYt\/+YX8EnvWOPk92ri\/iUe1fOZYeOzJSMq\/rQD+S+6RX0q8kOixM9csDqVUg1hWr3cf9PufLvv8iHgFJ0s37A1WNSMBlOi4ieRtQybb7wYodlcAl1Tya3uaoOsRplrXlOkMYTMlbRn2ttaw18gP9wJipXX9a7Oxe5IWF+Ra8pOFU0g7VJkypeGLYBn2+iLnmS8s8kvRmGTDWiYXT4z9s5nwfS8AfGW92I9WAJ9ta9grdzo5EWBoZ+g\/h+da74X8XBELTB9GfpTCQ52PUoYuTWpiFkMd25wmHit8iWCUh\/6tkp8wyUNvAhr+SdSh5zsB4G\/9xY1yIiniQtPXGWStiijkxwYlTRAv+XE7vm3OhLs8GmbZptpWAby4JF0M4HNJwLRjip01fuaLDWLVfyG\/WMga5PKSH0X51I2LGmlRdE+VRgy1HIdcNflxDHtVzBz1GIMpJG8I3lgufs234YSU4J9XHh2sTeUU8vNH1CunnvyQoNBpsBfU3UfmVVTHMpkarSBvlcQ5hvcuHYMmKn64Avi2HPmFQG4bvsuyfSJMzN8WMVJwiaWbWW4hP0q9MKaJH\/k1kUKIg9mGxIwgdUXUjgrgOkfk1xWDY+nJNS9o02+8bRTys+HF\/Ht88mNWeJ04lwglti5ty8cbfGKTn0t02jYe3Sq\/kF\/i+sqX\/EKGkVyEmEtDJw4r31IB\/DQ2+fk4qE5\/lzrq4rytO06F\/NwxC8oRh\/x+mbAF4FcA4C+DdO9OZiJ5jQzynQ74BACc6Q4sRk0FMeJiAHaC9luQAcRQ1WcvqMsUCu1WZp9KKOTng1pAnjjkp1OIQgC+jT4AgOCsuUSHAPBIBfCt1JGfC3FQwOaWRykzVpomIl4fIBTyi1UPBrnh5EchNRejKFGji7xpSnsUAE4QDM7pBAZB3SlJUsgPHzy8cwcWFhbgxo0bY9X+1FNP1e9x4Cfe5Lhy5coojfw7\/uP169dhz549o9\/t19iLF7S65m3cBCzbL0ciMSJT1B23FTHsR5utAAbUyM83Yp004vSNNHV3ACKmlM3Rej8q5IeHn156CdRX1lQ6wqcpb9++DcePH4derzciuosXL8KOHTtqGfv3769fasP\/zvMBIxOZTMcEN3sXU+\/zo5Ife+lF4DoE3BbtCvndi9iWlpbg3LlzsHEjPn49\/onI8NChQzWxiQ8JcW5uro4OTU9XLi4uAv4JueHD3uLzfgiIyMtl1dNS0nsqgK9Ryc8ULftGQjYUdB2db8Tum8+mY7u\/F\/IDqInr1q1bcPjwYW1tmCJDmfDOnFld9ZNl6EgzPvnFGCpyOKnPqqBcrkt+twjA27qPVwCfoZKfdylKRtvwOXTDcFf9R61z2Q49eU89+enm8tDb5Pk8nMvTRYZIfpcvX4bTp0\/DqVOnYH5+fjRHiDIE+e3atWv073TyC3HC3HrqWNGNK6GEEoNSXivk52pzqvQqKWNnhfNxf69RIA\/\/nHryEwTV7\/dH83lYW\/KQth3yS+W0XSknpDOIZOPvVAC\/nzry47RFngJgnA4IUtFwd2KQTF3mOej1rkO\/\/1y5xl6FRya8mzdvskd+g8FWuHv3QVhZeUB5AAc14V6AsA2VsExKGi4PdBnCcpUZQp4qMdyLXLzJr+3Ix4QF53RBG3VM9xWM+jZseB1mZ28U8tORH64Anz9\/HgaDATv5YXlIgIPBNnqNlZR5IVA\/Wv6nTJd1pjYtxhYajumNkE6KhuHs7Is16YlveXkZtm\/fTsvMmGpmOBwOGeWxicLIT5AfCtVthQlZ8BgMtsNggK+36b68e002kEeCMOp8k2fv3Uhm\/EYEn6sAPpx62DsJvhE6ysBIHL\/verkiRn34TX3kp+7PE2gi+V26dKmeB8RFEdwErdvqgulxhde01UUlTfqCB6VeOYcplPJip6ESFjUdVV+cZ8Lr0ZGAHb5XKoC+jfxchri2eTeOyMrBvuhJXbDhV2bqFzwQUlzcECSG\/ytWgHfv3j3a14dpkLjEXkBxmkPd5HzgwIF6ZTfPTc78DkSXyElY8Q670+0BmB3+OxjM\/IFLFoe0nHhRip2+Y46F\/O75BZLbhQsXRl4iSE38g+4InHq8DaO\/qsI73lY\/+\/E2XOwI\/bgXR1AfWwQSqvNk5P\/48J3wmRn1cZ7QId1kYNNshUrs7US0hfwS+5p92GsiHuoQoU3i0p2\/9DlDS7VVrrwYkZJOj7Vy3hiehp+fOQ0Atzy8iENf17qmELMP9h7mR88i7CjX2EeHmlqAnfyokjAdRwNyKa8pbU66cNnUJOd+GD7\/2zDza08TD9en0CnHMnR+QSHh+LaUyC8+xmMlNJNfSPifU48deyGmqfGka1jHhgDHZxI70NQVR+1U3bfuFPJL7Ey8kV9i5YOLsxET1dFNiiDpogy\/LRBm8\/R6v2\/4QfjyzBcAIKTT0pXKtZ2FW69gB7AIcKl\/1yH\/+qIL+cWuT0V+OPm5OEgq4yiOaCO+2Lr6EIrAWr+63Lza23YkHjv6VusrtV9SfK7Zpwr5xW5z7ORnUzi1E9r0afq9C5FJwxaQlQqgZ9vnF4JPqry+HRMlXx7bknRIFvJL5V\/3ykkf+bUVgaSOPHwrUkSElIaslPFXFcAvTQL5+WLX7XyF\/BLXXzj5JVZYW1yu0aUHgYXA+XwF8GuTTH5tdZwhlULPW8iPjhVLyskgvyYopuiJzP9RAfwz3Oenu7OOxV08hZg6J\/HvLp1EzhF82LRJIT9P9\/LNNvnk54tMB\/P9cQXwryc58utgnTioXMjPASyOpLzk1zQscendOSzzkZGrjsSV4U9XAL8Vk\/xyjrp86ps7D6WeygkPbtS95fGSn7caLWQMJboc5hmVxnayAjhMJb9Q+1uoMmORXZoLNONeIr\/EPhWP\/LgdMjXZqDv0Y1zYwFzZn6oAPmkiP8QPIzfuDdfMNtTiUte1agO37zZhtEaGhfxi+FKDzHjkZyo0fDNoYogCikvZiADgSAWwqCO\/psd7AsxrzJrTEDnHKHc9PoX8YvmiQW568rMZyH2Pm24eJrQxtB2ZGDBsjPxsuDf97mqva3qKbok7EnDdDI2dOq6ymy6gtXf62ZKfuCzUVE3qfXuU6swhTTrys1e+HQ8XGbbIw4VkXcq1WxEtxcEK4PfUyC+m7rqORUd8MciQgqLP06DuFxJQNKGkyYr85AtD9+3bZ3xEXH5rV3dhKMXwttLEI79YTpQ6AkhZM67RhqLbRyuAz1IXPGx2mQhrkvG3YWL7PQybbMgPiQ9vRsabkDdu3GizevS7bz5yAcwJ18jvN2BlRTzEYiskbDOnWbrrcDSQLGxman+nRjGUbQ8uChDKXagAzoWQX8woEW11rV8XfLqfNhvySwGl7m0OUa56jb0aecrRpsijXmOvDtXdr7EP68lSYBhWhkoopsbvSrI+wy2TJQ518JsVwB\/+VwD4ThgsziRFIGajRpxYqYU4YBeImDm7DZu1XQRTRX6C4NT5Qnx74\/Lly6PHicSLbuIxIgQa896+fbt+za3X64HpAaMTJ07Ujx6lecAoB2cTbmib84vm7S29OXIfwMLRwMjPhklTQ6biTU1n0yXH38P8fyrIT318SCY\/QVLz8\/P1q2viQ0K8detWPe8o8uuerpybm6vzmZ6uXFxcBPwTQ\/l4c36\/DgB\/rvFQ29DK9ntbTm\/rwal6hTWQxlI+XgF8JmTYS7UhRbpYUyspdPcrY+LJTxBXv9+HvXv3wsGDB0FEZwiZidgwsltaWqqjQUzD\/Wj5K698BFZWOF5v86v47uXKsHHWw95JIT9Oj4gVbfLOYU48+clVqnugHP9NR2xIfvjv58+fh8FgMCJCeTFGDJdPnz4Np06dAjV6FMS6a9euUVQZL\/KjOK9YFIi1MkzRgZpGjvy4okBq2cR0j1UAV1zJL2IkSlSblgw7G9xH13RjTRf8yGxttuSneydXNsNni0t3yS\/mJDWtKYSlMpEXpSfXkQUlH2pMTedp3SMVwLd05Bcr8vHU05pN1Tdk5ZwDc5cOwn\/aJkvyMy0WWOvQkiAn8hsMtsLduw92dOgb+9wtRwMK9RZC\/nqfn+k+P5cGTChropJwTmF8EADwESndh50u+tL3lB\/fCb3eF2HDhtdhdvYGLC8vw\/bt25MjPDMcDodqqbrhIodmOZEf2oMEOBhs4zCtyDAioEad4r\/lfxdk6xixPVoBfFOO\/FyH5x0h+WDvymvkMjv7VzA7+\/WRVVmRn2kFNrQOdOTX1oKHO\/H5h\/d63Fz30rmin8N8kCsZOdqYbLVXJUk1qsyLXADEsDlk+OxYFw7Ph+JwF78sIz9UzLQQ4QqJnF5Hfk1bXa5du1bv68M0CwsLoNvqgvJxO4xpq4u6mBJ3wSP2cDQE\/chENFKNo5MQ5GIhlfdVAF92XfBIhYNrXVH1oqZzLT99+izn\/BCGVAseWBYS19mzZ+uV3S1bttTEu3\/\/flA3OSNx4dYXXPE1bXIWeeybnF2Ot3E6BsdQi4NgTDZx6MeJV4OsD1QAX6SSXyhmofm5MFGjuRj1xTknaLY7S\/JLueCB0OiOrqnH23RkrB5vQxLFs8nicz\/exuWgseWowy7uIW5HoosdFcB1JL9Y+vrItUX\/lHnN3IbRcfw5S\/KLteARB0I3qbzD3vcCwFfcFNCmpkQVoT08R4OyNWwXKCgkYJG3rQJ4kRr5uejmmpbBFtcirem5XvGzdQA2vzRjkyX5xVrwsNZXggS85MetsM3RuMuLJY87EjXoadznF8subrmUTo+jTNu2nxC\/s9nQwTc8dIsTHNXQtoy8yU+Hjs1xqYjaemiqHN90EVYek0V+tgbuiwlXvl8FgL\/wENauT2QZ+dkWOxBlnxMeHrXDnqV75CcgiEAezui6RAgJhoLvqgC+4TrslaNSl9utZbDSLAiYq8dUD7JtNmLjis79p0KyJD\/nNtGhDGnIL5SoXEgmJfhtN3rF1pr8lhrekeDAxkYiHGW0KSO0k\/In0UJ+ieudTn4phzq5kl1o5ZiIQ21wnvavO+HRpK9Lh+QfzYQiFi9\/fiSeDfn5Xkfvmy9eJTdLppOfi4aejdelCO+0sU+SUBWL0Pgah71qnchRq2u0E0F3KmwTnC4b8kOMywNGOXsa19aFnG206aZE4w9XAN92nfPTlZE7uXEtetnwbfqdcjOQm55ZkZ9senm6MsRR2sirOp7vSYCUw\/1AnEabnAPleGc33d7dJFDeGyoiUJchuZCd82iDBmi25EdTv3up\/Ia9rsMkgYvPxuLcyMc3KmrKJxoudQHFQA5skZ\/sxz5EZGoHPvXfvTblq3EhP1\/kPPPRyc+2isXd8\/puu\/AEwpjNbejiVnrT3W8UScr8pXPkx11nFJ27ngbrDPcQ\/oTdkEJ+7JA2C6STn5Bji8R8IyMfw30jUGpZseVT9SCme1sF8COOOT9KeU2dUy4dl84OnX9SI24KLv5pCvn5Y+eVc5z8HmLo0bo0tHEZ0qWMklz0kqr9oQrgVZn8bB2Vl8swZUrVsbS5MOa2RaiQH5NrUcW4R34pSCDmUJOKDEe6lFEwAHid8MjNTg7\/yjnyNONdyI\/DFx1kuJOfg3Br0ljDDY6ogkOGFQDeBE6bnHmLXpXWZqTJVV\/cHRaFzFfLLOQXwycbZDaTX5NDUSo1sTHOxU2CDZLR76kAvhZzzq+bEZWzW7SUoZBfYuCbyc9z7impDbGix6RGWAojRjWtDXtTYzUp0yLjuGVNfvKJj8cee6y+JRlvT1YfB0\/tCiHltTvsFZoTG3eIodOQdyzy8114ihkNm2Q3dWAx9XFxCttWL5RFjYz19vZ6\/w\/6\/T\/K6+lKNAtPeODjP\/iuxgsvvADiMaFXX3113fsaLpC2nTYN+YXOo3Q1usMIBYn99TTVvG6116XYmNGULxG76N\/9tFlGfupNzurLaLqX0ihVIeTu3r0bduzYMcpiuj9QfqND986H+oaHeiRvct\/woKDdlKYp8uQc+keOYsgnPGydSdPCRcxFjdBOMtQPOPLrnnKQo0Yz9lmSn\/qGh478Ll++PHpJjQrhyZMn4cKFC3Dx4sUx8qM8k4l5b9++XT9l2ev1jK+3nThxopZtf73tI7Cy8kCD6ikds2l\/FBcZuUY6lKE5RSaVPDyIcrYCGMgLHgLHlHXX5P1U21EGVz1TW2Pb6e6HXu8b0O8\/l9ewVxDHpk2btG\/iqkRkg1GN7FTyw4htaWnJSKamh81Rj7m5Odi5c6fx3d7FxUXAP3zuEr80w14VEY+GbQN19HtTVEOZtzEVRCE2spJxEpIjP7l4WxTIqWrT8Nf2EDqnHj6y4pNxlpGfmPPbs2dPHaW9\/PLLozm\/Z599tl74UAnMBK8grn6\/D3v37oWDBw+CiM5EHowsb926VROt7jNFhnJEeubMmTqrLENHmu2Qn4\/zTVMeW5RpiKDeXgF8P8ZWl1Di5+rsQvXI24eyJT+ETTcXt3nz5tHj4q7Q6h5F0s3loVx5Ps8UGSL54fD79OnTcOrUqXWr0LonOLtDfm0M3XSN1hQp2QjL1Ts8ItCkZ3up9oRE29Qy1HQpfIWfiLMmP9+qMOXTkZ8cGYr5PMwvD2ljkN9gsBXu3n0QVlYeqXebT9+XcvjHha5CuF4nPLiiMtmmGDJDMWu6wbvN87+rdiHxbdjwOszO3shrzi8UdhfyM6WVCe\/mzZvaOcGQyA\/LRQIcDLZJKuToxLFqQ5bLMb9DjQRFBMEQSax7upKqgw1Tl0gqFDvqXjmbzj6\/uyzI+MhX86zWz+zsizXpiW95eRm2b9\/OUYCTjJnhcDhUc1CerhR5qHN\/mN7lLWB5n+FgMGAnv\/XEp68oGpput1msykRHeDPyy2M07TubyvvR8qZOjoGUowDqQsiyAi6+ydV5NAOAUR9+2UZ+Ys+cupcOoy2x4DE7O+u04dmX\/BAo3HCNCyVbtmBlrn7dWfDgblC5D1m5ImjL\/FnSOT+KTdz1HINFQyNVPp2ynPNTt7qo5spbXXD1V5z+wP13TZ+O\/EyEiOR76dKlel8f6rOwsACHDh0a2x+IeuCHK7y6jde6VeLuLHjYnCxNL23Twv93CplYpD9QAfydvNrbRsNOPXT0R5yWM9Qe+oJPluSnWyWVgRNzbefOnYPPf\/7z9Yor\/n+xl84EsonoZBLDvLqTIJgGiUuUIyJTMewWsg8cOFDv+wvf5ExzFf9UnORla\/SmskIiFZfhlD9KjTnXkV+kcoxiuSJwLjkh9sc8kqfv6LIkP9fIL5T8sMrE6Q9Rfepcom4e0jQkFzJox9t851JUR0Miwb\/vST+EkAvVkTlJlFpm7HQ2Mr9XftDZ3tg2+MqnR07NJYRE1in8dnXFN7sTHggqZc4PyeXYsWMgToL4VnfKfHzDXi7SFNbHIjEhN5b80NoLGGZNJPmF4pl7\/rX6zpb8EEJdtCWiKZzfQ+LDT96flzv0YeQXc2ggkMthCNRUixmRaPJhLzUismFk+x3xJ0a\/xqqilGHKrIsa+ac5er0fQb\/\/36Zjn18OxBhGfrEs4HcsHk2pjZ2nNGcp6y42SNE5OWupZMgcUy\/z\/EZBWUd+XjhknomP\/HTDtZB5ltTA+Tlsai2by\/sEAKye6dZ\/TaccbJbo8KFEU3I+kz\/gvB6S4N\/alMjsd17iLuSXuHrDyY9rQjqx4cbi1Abq846JC+lTCERWVvfQ+T0Zb6kAfhrjYoPQupk0H7HhQa3\/8dMs2ZKf6cIBAYNuJdUGUQ6\/h5OfbEXIcNWVBGzoqb2y7dgUdaFBRDIpI0XivOc\/qgD+b47kZ6srrt\/\/FQD8CZcwRQ7VP6nEh+LH6zVb8lP31UVCOLnYNfJ7HFZWfl55tFyQmaniqQTAOzxIDlJQgY8DwLP3JMSKgO7hy0J+oTr61HVIp6mrHGpHJvJSiU0ti4oVDZMsyc+2yTmobbSceTzye\/je2dqQuaGWDVpXvEtP3JbuxMjOqt4+ALhgTQXQBUwIZtRJYi3qICHj2yvpbjgq5Eetc6Z0vMNeJqW0YqhRZkwdUst2tDnqnB8XQSOGtkgoFTlz2hTuG1mSn+2ER7jZ7UnoDvmFYkRtUBwNgloWc+Sy7kqrUMxyyM9RHznYgTo0D6+zJD9UWz07mwucoXqsJz\/X+ZJQDUr+VQR8550k\/Fjm\/Ep9rCHg2xZMQ\/GmTvE+6PVeze94G+U+v7LaG9JobL27bZjkUjYDyZCLs60w2wSZJtTlf5eGxu+qAL4Rstrr29hNdrhEwDYsfH6nTBu0reOaXdlGfj7QdyFPvsNeiuO6IsxJoi5lc9rS0FG8pwL4Wgj5UW0yLYi52CnbQV01peqnpgs9GkctV7bfvcxCflScmdLlS35tERUTsG2ISX62l8tI7q0usl4ukZ0LeXPZ3oHIzzb0LcPeJmdwcSqXtPwOmJ9El6HoUQA4EWhC251OrK0rVFhsUzA6OSLKcyHa9XKyjfzU25rFu7qmN3SpULedLt\/IT0YmNiGq84Ah84JhDWDN6qZGaMCDdcEjFPOQ+c62CbidVpkl+ambnOUr5fEqK9sj4+1ASSu1G+RHs4WWyiWSoknUp\/KJIELKA4Boj5YH6qXNrutgdKSHnQli+R1GJUTdtEGy5rnATpAfRnuLi4v1H15Vr\/43Yy1FF9Ud8nOfQF4FLxXZYVmh0VJIdb8V4JFDAN8SCx6hw8c2iCHE\/i7m7cDZXrHJeX5+vn4PAyPBI0eO1H\/4epor+YkX30R17du3r350SP7Ua+zVNLqLFtRr7MXeRCGXdo29zolChoFddEpXnZuIIvZKpqQr6yZnn06Da8jvij93eu5OjIZllpEfQis\/UoTRHpLT3NxcTYa6l9JM1YFpz549C+fPn6+JUwyp8ZFiQYBqWepjRChbnoPEobfpASN83nLHjh2MDxhxEKEpiguNVrgbQcfktX6NfYkWzR5jnwPNlvxUwhHPR964cQOoK71qBCmAkh8kf+ihh+rr8EWUKdLI84qCMHVPVzYRsi5CpQ97ZcfW9fAmQuPuRWMTkkzAHWvM3nN+TVMK1GhOTedS7y3Mjza6UcJoXZqayZr8YjU7ecUYo0rdm7xIkEtLS\/VTlUh++T5a\/l4A+EosqDKSSyUFF5VpwyO9xJ8DeNfvak545EYsVDxUfG0dke13arnUdPz1P5XkRyE2OTocDAYjIpTfBhbD5dOnT8OpU6fWRY+6q7nokR\/VKVIPXUMIg2qTmi6kTJdGQyEuSd7bKoAfpTjh4YtbjHw+deFSBzF01svsLPnhu7qPP\/649aFy1Wz1xhjTvsHY5DcYbIW7dx+ElZUHGmrbd8U11IEoJOBaRgyZrjowp496pRVFV475YEo5lDQ+pEiRGycNEt+GDa\/D7OyNfF5vk1dmdXN7IppCSHBIKkdiFJh0ixu6IW1s8kNdkQAHg20GtTl6y9RDE0oNpE7jMhfmqNsjlbTVhZqXo16pZYWki4ibl1qIG\/pz+GWns7Mv1qQnvuXlZcAF0NTfzHA4HIpCZcLBlVl1hVUmRnWbCUVxVT7maTPyMxMfxRquNIUg7UjihPx3NQ3vEAAs2bM7pbBFyKHkGXM0gZEo+hPeyJysK5E7AAAUPklEQVTvh1EfftlEfroLTOVV1pdffhmqqoLNmzePtq24wGu6H9C0kkuZF5S33Jw5s\/qEobx\/UCebf87PBQXXtDkNq1x1j50eGzk+XWkiv1CSiq1\/G\/JdOtqYJL1qezZzfrrFAUGIP\/jBD+D5558H3eZkShViBHn16lUtaZq2w8jEJrbZ6La6CMLT7T3URZX+5NdmY4oxn0O1RzQYteGYiNmlgVG8pykNx8UGoTqU\/KsINNW7\/rde7\/9Av\/9s+3N+TXvpLly4AD7DXIREnNzABRLcfKz71I3Qpk3OSFxintG0yfnAgQP1RmxBqlje8ePHATdG47ee\/ChzK5Q0uTcCKtnlboek30crgM92YbV3EvyH3y+yi\/x00dXt27fHCIQKgyCx1157TZtFEKLu6JoaZequ2FIJWT1GZz\/e9iisvuyF80nlW3vlLOZ9c4w4R93q0nZn0Xb5jPVkENUJ8hNDy\/hwxC\/Bfdibat5NHhrYhrm230Nw5Gp0uqGObR5JnDQg2vdoBfDNkMhPJfmOkD65ernqklygU8JCfk5whSd2J7\/wMt0k5DREarPxEDqd91UAXw4hP7ea6WbqlHOwbggV8nPDKzh1\/uTnaiLVuW1bOFzLzSC99gEj9ZyqiCJN0ST++08A4HsZGMSlAvXUUbuRbnbkhxcXUD7q5QYUWSnT8JGfHBXZhnOxLSRESWQVOGWRC3VMeI\/gxi42oHYCjkVlm7xNe+03tthhewf0et\/M7+lKu+LdTaEnP3Qk\/N70MCwG8aW+ZcPDbGsWJFE8DYBRVaSv9SutItmVrVjeKZlsIr9s8WZWbI38fgNWVnDIMykfRqL450PgHcVAS34TOLw3Vk\/s6I+KJTXduCGF\/BK3O75hL5fiLsNM1wUIrjkd13IFNj7lOzSkZO\/2ctV1V+TwRngmqwv5JfaH\/MgvMQCTVNzDFcC3u7raGxK1EbcCZV7XhfwSV1Bc8vONkHzzxQQvR50Uez9eAXzGRH4+UacNzxDCssnG338RAP6akJA6z5w3SRbyI1Q1ZxI9+bkMPTm1mQRZrkMkRlJlfcAI68JhyO1cdSYfo25L0RXoir1remcjnTIU8nOCKzxx3MgvXL80EkRUhJEBLpCE39HGo7cj+Vg3OXNGaqYo6oMA8AUe86dMSiG\/xBXOT37UIQiXoXn13lxWmeU0DN1Y5vxcCTIkUouPll8JXNG4GzaF\/PxqyzsXP\/l5q9JixtRzQUjYSDLMF0h8oAL4YqoFj7Y6HVdybtGt6qLpwUAhv8R15UZ+IT1iLk7LPZ9JlReCHZGcva6xT+xw8CsA8JeaQl3J1NWfZPxlPHXkRMSbGbpe7x3Q7z\/d\/n1+zHZlK26c\/PD9jryv\/OYDkkpaPiW6NkxRRghBAsCHKoDP2yI\/x3lEH\/PX5XElNpZCG4TQo7E1IfFxK5Ff7HpX5Nsjv5AGaXJ6WWaIfC6w4ju2XVMqDg3vIj9WAVyxkZ9dk3xTyB1WG3Xm26nR8hXyS+x5dvLzUYjbMZuIIbeowgcvpjxsW124609nH3UPHxM2ycTQiE6nTiG\/ZJW0WpCe\/CbhIoGUQMYcQjvY8S8qgD\/TRX7+DdKhdABQ58pSleum5XjqFETfpN8aZlNBfur18pQr6hE++Zp63VX36jX24l0PAb39GvumB8tDHEzNSxni+czLcOpIkeXbuH3yETok0pyfahdH5OxjTxO+ucuj+AYlzfipm4knP\/VxIvEWBz5SLJ6ZNL3dK8OpviFsesDoxIkT9UNJ9AeMbJVGIS6bjJDfOUiRo8GH2BApb7IHjAhEPGYiN5lFwq9lsRNNfqZnKdXHy+U3ejdu3LiuSppelpubm6tfazM9Xbm4uAj4J+TGmfNr2YuyLj4i8e6uAC7hsLeJbHTbOESHwnH+VzcF0HaHiQ4REfcxf\/OfAplo8jO1STXSQ+K6devW2IPjcl5TZDj5j5brEHSNQjiYsc1IpoFIfrMC+EPTai+18VPTceDYRRkcIw+93VNJfnKkh+\/pHjt2DK5cuTKGkDyfZ4oMkfwuX74Mp0+fhlOnTsH8\/HwdBYpP9xD7ZER+1JXDGJtXqT19AlJZqADOTcpWFx+SQYzxS3U2m3exZOrITwyFN23aVEd6gqD6\/f7Y28A4xyeGtN0jvxyGPXJf0mbkRiFgz0bcGPl1IcqydWI6ssnNt5pwbh6lTB35iWjt3Llzo3k4HXwy4d28eROWlpZAzRMS+Q0GW+Hu3QdhZYW64uvidNToKNcGStHfN7LDfPiuB8N1+433+eWKrU0vFz9TZWG94RfjJTreDhSJb8OG12F29sZ0HG9TFzqa3EBOOxgM2MkPy0YCHAzwiNu0fW63b4SjE6m836kAfl8Me12GjSEEE46GWQLHAkxM\/ZpkU3V\/K8zOXq9JT3zLy8uAuz9SfzPD4XCYolB1a4qtTJn8MO3Ro0cBt7Fs2YIgr34hCx7dIz5KNGZDlfv3mDrhkBBvgWk4e90Y+ZkilUhEzA0tWR5vREYudiyh2ygAoz78Nmx4E2ZnvzTZkR\/O3129ehXOnz8\/Rl4IAK7k7t+\/vyY23J8nPiS\/S5cu1fOAOE+4sLAAhw4dGkuDcvHDuUPTVheVNP0WPGI2cpu7UaKUJuejzLnZdMj092MVwPGuL3j41M9kEPjEz\/khQV24cAEuXrw4Rlxyc5JJDP9dLIrs3r17lAfTIHGJeT\/TJucDBw7UK758m5xdGn4bvbBt0txFf460XBgQSGFs2Mu7EsmBRFoZroTYxpap8Y58oslPRHWvvfaa1g9kQhQkKRKqZClWhW\/cWJsrUI+3qcfo7Mfb3oexZ1ofJZVGifZColEugiIZEy\/RpyqAT4ZEfogDRs3Ml6x6Wyzq3TaMlDsGKomF+Iu3QY0ZJ5r84kAWJtVv2BtWZvzcMVf5OLR3WYxwKO\/fVwCnQsiPUpaNiCgyYqQJjXRTkSHqiav76zuYQn4x\/KJBpp38TD1pCmfJtaFxVBLahn8hl8cqDX4i5vxcsVWj9l8GgK+7ClHSyz5PmG4ILE1kL+THBCRVjJ38qJImNV2HhsRPVwBPxo78QuuZSk654h6LDN8Kvd63od9\/brJXe0PdhzM\/L\/k13DKsVTqWIzUhRJk\/5EQ4oaz\/XAH8dhP5hRCKiDIRP\/xj2JQdFRp5aqEbdV4iv6gOsV44jfxcScrWyHTyqJtCEwMUrTjXBkmYyB+Rn2t9+Rjpqr9aRmh95zwlYvN\/Pd6F\/Hz8MCAPjfwCCgjK2qaDxyQQ0wtmTWARtvCcrwD224a9bWIa5AwemVPMS7uo1Uz4hfxcsGRIqye\/0F6dQbEiQkGAUCfLFcATNvKbBmB\/HQD+\/J6hMTsxXiwL+fHiaZXmFvlRw3lCQ7VqpibgkMkhw1lxQwauh8ulOvlcBfBhQX6um3yFmr4YcURZoUNhrrppR04hv8S4u5GfUK47vek4nL4N21YpvkRjk+v4+xj5OebNJjm1g42tcHo9er0fQ7\/\/J2W1N3bVCvl+5BdDO47IIYZeqWVSOxbNHOALFcC7y7A3bY3xzaGWyC9tzRmerkysBKk4wmonSY4tUapybHp4\/P4\/K4B\/ykF+VAL20LFkMSJQyC+xc+QT+SU2fBKLu10BbKKQn4iyXY\/ZpR8KNj\/GlKoSKZ0BTqngh0fX\/L5Cfn64eefqDvmlaHix5gR9qsdDlx9XAD+jI7\/Qc68++qfI42uXB7a1Ob75dFis73gK+aXwGamMOORH7Sn9e8nEMGVe3OpQ\/W3D34UfzZwm6Mo3T7VWWNfnbF2JlOLjhKqQkvR6b0K\/\/0xZ8HCDzT+1O\/m5OglVt38OAP\/dkjhGo6Xq55pOF6lyRg7r9Xl0+C\/hmzN\/qvyQK2aCPDo8x2p0CbUjMO0GmKL7\/FybT4r0evLj79VS2DLtZXxg+KvwxZm\/SACDSqhcpJ5iaiMBPGNF0LEpw97EdeMe+VEVlCs9BzK1RUB0J6UiMJ7OdwOvLtJGPPFiAfE+7aruC8P74dxMrItIQ4gph\/r3q7X4udb8rpBffLTHSohHftyGUFcmXUnG55wtxTZXPSgym9OcH96C\/TM4jJymL8WwOUUZAIX8EvutmfxskVJsRTmihbZtiI2RLP8++PGwgp+Z+Y+BhaZp6GtKYj1jVPmdQL3V7LEjeaq6FD1W0xTyo2LKlC6fyG+aiKrpOJx\/xPg0fBKehE8xeQZVjDosd63HTI4GQvur1YX8qD7HlC4f8mMyaGLEuJIIwA+HJ+AfzxxNgABOQeAnR2shc4I6lSkRk6+pHKMK37LN+Qr58WPaKNGd\/CiOw+W4lLKEebY5QdvvsYF3veWaos844UznnB8Fp26kKeSXuJ7cyS+xgqPiXIgQM1GGMb7RijzU85XBj+O\/Gc7BH83c4hfcWYmp5y91QNH9tpBfYkezk5\/78EuYgJW5YcPrMBhsa7DKlTy4okpUaf12HJrOmNd1s7fcCPwxBdAfi0Kc3\/ndg\/DlmS8k9iD\/4vAKpw0b\/sbiH77yQzBWy1yru1X\/+N8wGPyS5AM8PlnIz7euPfMJ8hsMtsLdu9igQ76NAHBnJAAb5OzsDaDLfgsA\/DREgeC863XuAcA\/NOj1s\/d+1xU9jkewcgYBQuc3P7EX3jjyGgC8HQC+TywuVMcm+5tVcPcPoknkZL8AAH9DTo0J\/XW2+7aQvby8DNu3b3fSiyPxzHA4HHII6pKMJ554or7aqnwFgYJAuwgg6T355JOF\/FJVQyG+VEiXcgoCzQi0EfEJjaYy8isOWRAoCBQECvkVHygIFASmEoFCflNZ7cXogkBBoJBf8YGCQEFgKhEo5Beh2p955hmoqmoked++fXD48OEIJfGLXFlZgWPHjsHu3bthx44d\/AUEShT6XblyZSTpqaeegp07dwZKTpP9+vXrcOnSJTh+\/Dj0eritKM\/vzp07sLCwADdu3KgV3Lx5M5w\/fx62bMGz2JPxFfJjrkckvrNnz44cRTgRrmp1gQBPnjwJFy5cgIsXL2ZJfqjf7du3R+SBZLJnz55s9ZXd66WXXoL9+\/fDu9\/97qzJT+ezql8zN5tWxBXyY4RdRCXz8\/NjkQg20KNHj2bdc6o9fY7kJ3Q8dOjQGDEjIc7NzWUd\/cmjgcceeyxr8kN\/XVpagnPnzsHGjbgpHED49qZNmzrRiVOadSE\/CkqBabDHR\/I7ceJElsMGQSr9fh\/27t0LBw8erHXNbdhrwhGJ5dq1a9kSiiA+7FC+9KUvjUWuga6VNDt2Mvh1YQRDAaaQHwWlwDS6njRQZLTsYmiWI\/mZcERyuXz58likEg2gQMHqsD1QXLLsplFNMgUiFFTILwKossiuDRcK+cV1iK6SXxemblxrrpCfK2KO6bsUlaBphfwcK9gxeRfJT0yL7Nq1K+t5VceqgEJ+rog5pO9ib1nIz6GCPZJ2jfy6NnJxqZJCfi5oSWnVvXzqCl6OWzBsOuce+XV1wUN2sS6RnyA+1D\/3fYk+zbiQnw9qljzo4FevXs16a4vJhJwjv6atLmhPF1Yhu0J+wg8+9rGPdQJXn2ZcyM8HtYY8uW8StpmbM\/mh7ogvXkkm9qDlGGE3YdwF8uvKZmybL9t+L+RnQ8jhd+E0r72Gtwuv\/3LcOKxqmTv5qZuxUf8uHW\/rAvmJDlznw7lv0HZormXBwwWskrYgUBCYHARK5Dc5dVksKQgUBBwQKOTnAFZJWhAoCEwOAoX8JqcuiyUFgYKAAwKF\/BzAKkkLAgWByUGgkN\/k1GWxpCBQEHBAoJCfA1glaUGgIDA5CBTym5y6tFpCvY1X3UjcJDjW+WVVLh61wmu28MF5jqvUua9owv2Ri4uL9Z+4ANSEG+5VPHLkSP3HYYu14ksCLQKF\/KbIMSi3c1DSyJDFIj+1WrjL4b4AFfXDi0qpR+y6dMfjpDaRQn6TWrMGu2xRnesVXNykZKoOznJi3KyNuL7\/\/e8n337NHXlOmRuzmFvIjwXG7ghpOr6mi\/rE2Vlhoe72GvV9Et0La7qjffIxqia5L7zwwthreHjV\/htvvAG69yQo5K0eMRN2P\/744\/Diiy+CeBlO6HTmzJn6USf8tm7duu7GaMyPR+zwxT4c8urs173gV6K\/dttNIb928W+ldNP5UjW6kt+ewPc8dFccqXnk90DENUiCQOUzuHIEik844nOZMpmpcnW6ya\/kIZCUu+d0BC+fFxYkLZ\/TFnqbrniSh7y6NKapBNcphlacZYILLeQ3wZXrMoRUicP05KYghQMHDtS3+upISfeehkx2qBe+CSvfDGwjOxPJyjIolzLo0uhsNRGdLrLEf8MP8XAhNApZT6F7JjO5kF8yqPMpSNfo1Hkw01VRal6VlChR5WAwsL61ayNDRFOdv6QMeXVzhzrCopKfbhVaDOd1Q2TVC7pwy0s+nsurSSE\/Xjw7I02dbzI9Bm4ySMxhyWTy0EMP1cNX\/NSbf+V0XOQnR3FINOrQWae7brtPCPmhDsvLy\/XTpDh8F586V2oiwkJ+7TWZQn7tYd9qyfJq44c+9KF6GCo\/Bk5dXW0z8pOjM+p7w9yRnzzkNVWoIOn7779\/3WJJIb\/2mkEhv\/awb71ksddt27Zt9SqnHK1RJ+kpCxHqEFU356fOxVGGvShHpMNtJj\/84Q+tb000zfnJ84eUYS9GetSN17qIs8z5tdsECvm1i3+rpcurnLrbkNXVXt08G8dqL24P0Q275S00psUMmw0qwE2rva7kh7LlLS743y6LJy6LI606yoQWXshvQiuWapZtkYBjn9\/mzZu1jznJ+\/zUOTGVVOW9c+qeQOqxPYGJerrDd87v5s2b2lMduqv2dde\/l31+VC+Nk66QXxxci9SECLjOm1G2xMRWv5zwiI2wXX4hPztGJUXGCJies7SpzH2211ae+nuJ+lwR409fyI8f0yIxAQKuc32qSm1GXuVWlwQOQiji\/wMTSxxfTxLFawAAAABJRU5ErkJggg==","height":174,"width":232}}
%---
%[output:755934af]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAHVRJREFUeF7tnU9oXcUXx083gdhufqkLmxJ4hYCuKm4SgyIEF4LtSkmy0Jb+oCEQkSKkNJDoQhvIowkGDIGQgJCni6S4swUXEvAPtS4s7UpNwUfBRtBXQVsK3bwf5+q837zpzLv3vnfvvHPv\/V4IYjN35sz3nPvJmTtzZw7U6\/U64YICUAAKFEyBA4BfwTyO7kIBKBAoAPghEKAAFCikAoBfId2OTkMBKAD4IQagABQopAKAXyHdjk5DASgA+CEGYivw7bff0osvvth03yeffEJvvPGGta6HDx\/SO++8Q+vr6\/T+++\/Tu+++G7tNvQ518yuvvEKffvopHT58uFGfzTa9MbP9KLaF1Tk1NUUffvgh9fb2xu4XbuieAoBf97TPXMsKFN999x1tb2\/T008\/HfThp59+oomJCXr++eetEFC\/f+aZZ+jHH39sujeKCAo+Jrg++OADeu+99+ibb76hF154IahKldX\/rVUbUWxrVae6\/\/XXX28L6lH6jzLpKAD4paNrLmtl2DAIzGxLB+D58+cfywC5fKVSCeDw1ltvka2MSzAFF9c9pk1x4RfFtrA6uY5Lly7FhnougyRDnQL8MuSsbpoaBiHOCrkMD0EHBgYaptZqtQCGnJkx\/FoB1Na\/sPI\/\/\/wzPfHEE8GQk9sOA5XeRlTbwupk+L355ptNGWg3fYW2oykA+EXTqfClwgDgEsi8L049apjNdUd9pxan\/qi2hdXJgP7ss8+Q+WXsKQH8MuawbpnbztDOBi8z22rVnzhlVT2tJif0d4ZxbGsFP9f7yG75Ce1GVwDwi65VoUu2Az\/XUDlqptQJ\/MImPOLYFjbb22qmu9BBI7zzgJ9wB0kxL2zo53pfx7OxrktBQ4Ho5s2bjaL8u9deey1YIpPGsFfNFIfZxr+39T1shluK32CHWwHAD9ERSQGVhZ06dcq5nk+fnOBK9YkOvRE15KxWq9aZY72smo21zTBzOTODiwLpVhmlzTZXnQBgpNARWwjwE+saeYZFWeqi1ruFDZOjQMoGN1MVcwgdpd64trWqU830trt4W56Xi2MR4FccX3fc07BFzk899VSQyfHSk7Dhqsq+SqVS6Exu2CJn\/Z1bGPyizCCbtv3www\/BFy2294j6FyJh7xk7dgAqSFSBXMHv+vXrdPr06YZAx48fDz6p6uvrS1S0olemsh1dhzgAUveFZWB6\/QpIX3zxReOfW33e5gJRGBxttv3xxx9O+HF5Zdtvv\/2G5S4ZejhyA7\/bt2\/T5OQkLS4u0vDwMPFf5Lm5ucAVCwsL+O4yQ0EJU6GADwVyA7\/Lly\/TtWvXmkDHQCyXy8EPsj8f4YQ2oEB2FMgN\/PjbSr74G1B13bt3j3jHjZmZmSAbxAUFoAAUUArkAn5qiDsyMkJjY2OPwW98fLzp3+F+KAAFoADghxiAAlCgkAoUFn7ff\/99IR2OTkMBaQoMDQ11xaTCwY+h99FHHxHg15V4Q6NQoEkBBh\/v9diNKxfwY+GiTngw9PgTrYM0Sj1USlTzR1SlB7SbSt2JGqpVBpvTUvbxeqF1syZKD4ZfN7K\/3MDPtdRldnY2WPs3ODgYKK\/g9x\/6L\/XQscQj\/z7tBlBNo+7Ejf23QticlrKP1wut\/6\/JI\/qF\/qSPg8wP8OsgBtUi5+np6WBm17XIOW34ddAF3AoFCqUA4Jeguzn7m5+fb9Ro+7wN8EtQcFQFBTpQAPDrQLx2bgX82lEN90CB5BUA\/JLXtGWNgJ9nwdEcFHAoAPh5Dg3Az7PgaA4KAH4yYgDwk+EHWAEFkPl5jgHAz7PgaA4KIPOTEQOAnww\/wAoogMzPcwwAfp4FR3NQAJmfjBgA\/GT4AVZAAWR+nmMA8PMsOJqDAsj8ZMQA4CfDD7ACCiDz8xwDgJ9nwdEcFEDmJyMGAD8ZfoAVUACZn+cYAPw8C47moAAyPxkxAPjJ8AOsgALI\/DzHAODnWXA0BwWQ+cmIAcBPhh9gBRRA5uc5BgA\/z4KjOSiAzE9GDAB+MvwAK6AAMj8iunfvHk1NTdGtW7eaIuLixYvBeRx8qTM5rly50iij\/57\/8fr163T69OnG77GNPR4wKCBXAcCPiPjwIfOUNdNlfDTl\/v4+LSwsUG9vbwN0W1tbNDw8HNQxOTkZnNTG\/48DjOQGPSyDAqwA4Pdvxra0tETr6+vU19f3WGSozHBmZiYAm7oYiKVSKcgOXUdXlstl4h9VL4a9ePCggAwFAD+iAFzVapXOnz9v9YorM9SBt7q6Gtyr12GDJuAnI\/BhBRQoPPxs7\/I4LPT3efwuz5YZMvx2dnZoZWWFlpeXaWRkpPGOkOtQ8BsfH2\/8O+CHhw4KyFCg8PBTgBoYGGi8z2PX6EPaNOB3kEaph0rUQ8dkRAKsgAIFUoDB94iq9IB2qVKp0NDQkPfeH6jX63XvrUZoUAfe3t5e4pkfm8AAPESjEaxBESgABZJS4D7tBtBTF+BnKMvw4xngjY0NqtVqicMP4EsqlFEPFIinAGd9fCHzc+imw4+L2JbCYMIjXtChNBSQpEDh3\/mZ6\/OUcxh+29vbwXtAnhThRdC2pS5cnmd4XUtdTGhiwkNS+MOWIitQePix83lyQ0GM\/6tmgCcmJhrr+rgMg0utBVRfc5iLnKenp4OZXSxyLvJjhb5nQQHA718vMdw2NzcbPlNQU\/9g+wTO\/LyNs7\/5+flGHfi8LQuPAGwsqgKAn2fPY9jrWXA0BwUcCgB+nkMD8PMsOJqDAoCfjBgA\/GT4AVZAAWR+nmMA8PMsOJqDAsj8ZMQA4CfDD7ACCiDz8xwDgJ9nwdEcFEDmJyMGAD8ZfoAVUACZn+cYAPw8C47moAAyPxkxAPjJ8AOsgALI\/DzHAODnWXA0BwWQ+cmIAcBPhh9gBRRA5uc5BgA\/z4KjOSiAzE9GDAB+MvwAK6CA2MzPPADcdJW560pWXAn4ZcVTsDPvCoiCn75t1NmzZ51HSeonrtm2jZLsNMBPsndgW5EUEAM\/Bh\/vj8f74dkODnc5pd37uuVkwK9byqNdKNCsgBj4FcUxgF9RPI1+SlegUPCzbU+vHGTu5GwOu22Hm5s7OZvvKbGTs\/Twh31FVqBQ8FOAMydLePv5nZ2dxvkc6lAjdR4HBwjfu7+\/3zjY3HWGx+LiYnDuB87wKPJjhb5nQYFCwM88f0OHn4LUyMhIcPCQuhiI1Wo1mHRR99tObyuVSsF9rtPbyuUy8Y96j4lhbxYeC9hYBAVyDz8FroGBATpz5gydO3eOVHbGDnaBjTO7paWlIBvkMji3twiPA\/pYJAXEws92WprumHaWuNjO6OV\/s4FNP7S8Vqs1QKjPRKvh8srKCi0vL5OZPao+jI+PN7JKZH5FerzQV8kKiISf631Zp0JKgt9BGqUeKlEPHeu0W7gfCkCBmAow+B5RlR7QLlUqFRoaGopZQ+fFD9Tr9bpZjS1j6rwpIknw4\/4wAA\/RaBJdQx1QAApEVOA+7QbQU5co+LkmISL2zVlMEvwAvk69ifuhQHsKcNbHl8jMjw1zvYtrr7v\/3GWDHyY8OlEU90KB7Cog8p0fy+lrwqPVUpdr164F6\/q4zNTUFNmWurCtvBzGtdTFnEzBhEd2HxZYni8FRMLP54QHu5PBtba2RhsbGzQ4ONjIEM1FzgwuXvrCM76uRc7qHixyzteDgt7kTwGR8PM54cEutX26Zn7eZstEzc\/bGKK8MYO68Hlb\/h4Y9Cg\/CoiEX1oTHhLchmGvBC\/ABijAEx6\/0J\/0saylLq7JiTw4DPDLgxfRhzwoIBJ+YZMdLHw7X3hIcBjgJ8ELsAEKCM788uocwC+vnkW\/sqaAyMwvayLGsRfwi6MWykKB9BQQA792t6Nv9770JG1dM+DXLeXRLhRoVkAM\/NgsHGCE8IQCUMCXAqLgp3caR1f6CgG0AwWKqYBY+OXVHRj25tWz6FfWFAD8PHsM8PMsOJqDAg4FAD\/PoQH4eRYczUEBwE9GDAB+MvwAK6AAMj\/PMQD4eRYczUEBZH4yYgDwk+EHWAEFkPl5jgHAz7PgaA4KIPOTEQOAnww\/wAooIDrz07\/4OHHiRLBRKG8gap6PmyU3An5Z8hZszbMCYuGnHxp+48YNUudp\/PrrrzQ5OUn6FvNRHaQ2SZ2YmKDh4eHGba4ttPSdmm27PZs7OZtfpWAn56ieQTko4F8BkfAzd3I2DweyHRYURbpLly7R5uYmbW1tNcEvyklxfO\/+\/n5woFFvb6\/zDI\/FxcWgbpzhEcUjKAMFuqeASPiZZ3jY4Lezs9M4TChMPjOzM+HHGdvS0pKzPtfxlgzEUqlEY2NjztPbyuUy8Q8fesQXhr1h3sLvoYAfBUTCT2VNR44csR4LaWZhraRS4BoYGKAzZ87QuXPnSGVn6j6Ga7VaDdqyXa7MUIfy6upqcKtehw2agJ+fwEYrUCBMAZHwY6P1oyHv3LnTeOf3+eefBxMfZvYW1lH+ve3Qctu7PC6rv89zZYYMP85AV1ZWaHl5+bGJGNspdIBfFE+hDBRIXwGx8OOu2yYi+vv7G+frxpXHBj89M1Tv87hefUibBvwO0ij1UIl66FjcbqA8FIACHSrA4HtEVXpAu\/JOb+uwb87hK88Um8NeW2EdeHt7e9Z3gp1kftwmA\/AQjabRVdQJBaCAQ4H7tBtAT12VSoWGhoa863WgXq\/XfbVqy\/xcbetLbWq1WuLwA\/h8eR3tQIFmBTjr40tk5hfl6ErVnTjv\/tqFH7c1OzsbZIyDg4MNJTHhgccKCmRXAbHv\/NSEh7mQmIGjJjwOHz4ca8GzDX4uIHL729vbwbo+nhSZmpqimZmZpvWB\/F6QL57hta09tM0SY8Ijuw8LLM+XAiLhZy51MSXXl7rw7K\/6+oMXH7e6XKDTIcb3274E4TIMrvX19WDNnj4bzYuaVd3qyxMscs7Xg4Le5E8BkfCzLRHRpVcTDQyiL7\/8MlhuoqDUDvz4HvX1h2s4bRuKu7JSVQc+b8vfA4Me5UcBkfCLm\/lFhZ8Et2HYK8ELsAEK8ITHL\/QnfSxvqUuUd36cWc3NzZH6EiQLDgX8suAl2FgEBcTCj8W3DTXVUJLf7zH4+NIXJ0t3GuAn3UOwrygKiIZfHp0A+OXRq+hTFhUA\/Dx7DfDzLDiagwIOBQA\/z6EB+HkWHM1BgazBz7XbiuqHbRlJFrwM+GXBS7CxCAqIzfzMRcV5cQbglxdPoh9ZV0Ak\/MIWOWdZdMAvy96D7XlSAPDz7E3Az7PgaA4KZOmdX9gXHln2JuCXZe\/B9jwpIDLzY4HNjQPyIjrglxdPoh9ZV0Ak\/KLs54fZ3qyHHuyHAt1VQCT8uitJuq0j80tXX9QOBaIqAPhFVSqhcoBfQkKiGijQoQJi4Rc29MWwt0PP43YoUHAFxMLP3K1ZHSruOkA8K35E5pcVT8HOvCsgEn7mImf9PA3eyop3clYwjOIgde6HKnv27Nng3A39MndyNsvYPrczd3JWM9SqXuzkHMU7KAMFuqNAJuDH2V65XA5++PwM8\/9bScfgW1tbaxx0rsDK53QqAOrb4qv6+XxfdR4H169nogxg1xke6kxgnOHRnYBGq1AgqgIi4afAMTIyQmNjY8GmphcuXAh++OjIqPAz61Gi6GfyHj16NNgUVbWlyujZpQKm7fS2UqkU2Og6vU2HNteNYW\/U0EQ5KJCuAiLhx102szHOvFqBJo5M+ntDzvRsx1IyIJeWloKDkRh+OLc3jsIoCwXkKyAWfuZQU52de+vWLep0pjcK2PTssFarNUDIsNSzQz48aWVlhZaXlx\/LHm0bNCDzk\/9QwMJiKCAafmm4wPxu2DV7DPiloT7qhAJyFCgc\/GyTG7YhbdrwO0ij1EMl6qFjcqIBlkCBgijA4HtEVXpAu\/KOrgzzwdbWFp08eTKY\/Y166UDjiRO+upX5cdsMwEM0GtV8lIMCUCABBe7TbgA9dVUqFeLVH76vA\/V6va43qq\/Js73bU+\/R+B6ejIgKP9cuMa6Z3CjvBfUZ3tXV1aAb+vpBW93qnR\/A5zvU0B4U+EcBzvr++a+gzM\/MzMy1dToYzQXGrRzL9Vy9erWx1k8v61oOo4NNTbbYlroo4LmWuphDakx44BGEAjIUEPPOz7aBqZ453blzh+bn56m\/v98KMZec6ssNHiYPDw9bi5kLoXkobFvkzOBS2aZrkbNaGI1FzjICHFZAAZcCYuBnWxaiAPLXX3\/R119\/TbbP0lq5VkHs7t271mIKiLZP18y2bBstmNmn+RkdPm\/DgwcF5CogDn62oeXm5ibFGebKlRtfeEj2DWwrlgKZgN\/+\/j4tLCwQf1Ob9Qvv\/LLuQdifFwUyAT81qZAH0QG\/PHgRfciDAoCfZy8Cfp4FR3NQwKEA4Oc5NAA\/z4KjOSiQFfjxxgVRrk43N4jSRhplAL80VEWdUCC+AmIyv\/imZ\/MOwC+bfoPV+VMA8PPsU8DPs+BoDgpIH\/YWxUOAX1E8jX5KVwCZn2cPAX6eBUdzUACZn4wYAPxk+AFWQAFkfp5jAPDzLDiagwLI\/GTEAOAnww+wAgog8\/McA4CfZ8HRHBRA5icjBgA\/GX6AFVAAmZ\/nGAD8PAuO5qAAMj8ZMQD4yfADrIAChcj8zB2Wo+zSzKGhb6Bq2+3Z3GBVbW2vwgo7OeMBgwJyFcg9\/MzzOdR29HxUnTppzXV8pe428zAl1xkei4uLwVkhOMNDbtDDMijACuQafq6T2cxT4vRjKm1HYbqOt2QglkolGhsbI9fpbeVymfhH1YthLx48KCBDgVzDzyWxmekxuKrVatOZu\/q9rsywk3N7\/0P\/pR46JiMKYAUUKKAChYSfnunxuSBzc3N05cqVJvfr7\/NcmSHDb2dnh1ZWVmh5eZlGRkaCLFBdthPpkPkV8ClDl0UqUDj4mecDK0ANDAw0HZKkD2kBP5GxC6OgQEcKFA5+KltTh4+71NOBt7e3R0tLS40Dy9U9nWR+B2mUeqiEoW9H4YuboUB7CjD4HlGVHtAuVSoV4glQ39eBer1e99WoOdHRql29bK1WSxx+3DYD8BCN+uo+2oECUICI7tNuAD115R5+5tKUsCjQ4cdlZ2dniZexDA4ONm7tZMID4AvzAH4PBdJRgLM+vgqR+fH7u6tXr9LGxkYTvFgAnsmdnJwMwMbr89TF8Nve3g7eA\/J7wqmpKZqZmWkqw\/XyxesFXUtdTGhiwiOdgEatUCCuArl\/58eA2tzcpK2trSZw6ULpEON\/V5MiExMTjXu4DINLvSt0LXKenp4OZnyxyDluKKI8FPCrQK7hp7K6u3fvWlXVgaggqQqasFSzwvrRmubnbeZndPi8zW8wozUoEEeBXMMvjhC+ymLY60tptAMFWisA+HmOEMDPs+BoDgo4FAD8PIcG4OdZcDQHBQA\/GTEA+MnwA6yAAsj8PMcA4OdZcDQHBZD5yYgBwE+GH2AFFEDm5zkGAD\/PgqM5KIDMT0YMAH4y\/AAroAAyP88xAPh5FhzNQQFkfjJiAPCT4QdYAQWQ+XmOAcDPs+BoDgog85MRA4CfDD\/ACiiAzM9zDAB+ngVHc1AAmZ+MGAD8ZPgBVkABZH6eYwDw8yw4moMCyPxkxADgJ8MPsAIKIPPzHAOAn2fB0RwUQOYnIwYAPxl+gBVQAJmf5xgA\/DwLjuagADI\/GTEA+MnwA6yAAsj8PMcA4OdZcDQHBZD5yYgBwE+GH2AFFEDm5zkG0oQfO5NPoT9Eo5571X5zsLl97eLeCa2bFQP84kZQh+UV\/A7SKPVQqcPaTGdW6QHtUhp1J2qoVhnDGjanpS7io5WyKvYqlQoNDQ35cYLWyoF6vV733mqXGzx16hQxBHFBASjQXQUYem+\/\/Tbg58sNAJ8vpdEOFGitQDcyPmVRITM\/BCQUgAJQAPBDDEABKFBIBQC\/QrodnYYCUADwQwxAAShQSAUAvxTcfvnyZZqfn2\/UfPbsWTp\/\/nwKLSVf5cOHD2lubo4mJiZoeHg4+QY6rFHZd+XKlUZNFy9epLGxsQ5r9nP79evXaXt7mxYWFqi3t9dPo220cu\/ePZqamqJbt24Fd\/f399PGxgYNDg62UZvMWwC\/hP3C4FtbW2sEigointXKAgAvXbpEm5ubtLW1JRJ+bN\/+\/n4DHgyT06dPi7VXD6\/bt2\/T5OQkPffcc6LhZ4tZM64Tfmy6Uh3gl6DsKisZGRlpykT4AZ2dnRX9l9P8Sy8RfsrGmZmZJjAzEEulkujsTx8NnDhxQjT8OF6XlpZofX2d+vr6gidExfaRI0cy8Uc8ymMN+EVRqcMy\/Bef4be4uChy2KCgMjAwQGfOnKFz584Ftkob9rp0ZLBcu3ZNLFAU+PgPyldffdWUuXYYWl5v5z8yfGVhBBNFGMAvikodlrH9Je2wytRuV0MzifBz6chw2dnZacpUUhOow4rNYXuH1Xm73TWq8WZACg0BfimIqleZteEC4JduQGQVfll4dRPXc4BfXMVils9SVsJdA\/xiOjhm8SzCT70WGR8fF\/1eNaYrCPCLq1iM8ln8awn4xXBwG0WzBr+sjVziuATwi6OWVtZcy2fO4ElcghFms\/TML6sTHnqIZQl+Cnxsv\/R1ie08xoBfO6qF3MMBfvXqVdFLW1xdkJz5tVrqwv3JwixkVuCn4uDVV1\/NhK7tPMaAXzuqtbhH+iLhsO5Khh\/bzvrylmRqDZrEDLuVxlmAX1YWY4fFctjvAb8whWL8XgXN3bt3rXdJXDhsGiodfuZibLY\/S5+3ZQF+6g+4LYilL9CO8bhiwiOOWCgLBaBAfhRA5pcfX6InUAAKxFAA8IshFopCASiQHwUAv\/z4Ej2BAlAghgKAXwyxUBQKQIH8KAD45ceX6AkUgAIxFAD8YoiV9aJRN6Q019K16ndan\/CZ9fLXBrzTDJ+5nMRuwknvUsJLhMrlcvCj9sBz6cbLdS5cuBD8JNGXrMdlt+wH\/LqlfBfajfKBepQyuulpwc+UJ+l2kt4DkO3jvfqifmWSpW3OuhCqXpoE\/LzILKeRsKwu7i40SUPJpVSS7aSxuSzr+tJLL0XeADbpzFNOhGXHEsAvO75KxNJWX3DYsj71+Zhq3LaBg7lFv+2QIdvXLfqXBK3qvXHjRtOBULzb9O+\/\/062LdWjwNv8ykL1++TJk3Tz5k1ShyMpm1ZXV4NzTfg6fvz4Y5um8v38lQkfWsVDXlv\/bYdYIftLJKTbrgTwa1u67N7o+sTKzK707dd5S3vbLh\/mPfqW+GonEAVQ\/TM0PQPlU8z4xDgdZma9Ntv0g6LYG1G2X7IBXv9kTkFa\/1RR2e3a5UQf8trKuF4lxH3FkN2Ik2k54CfTL6laZRtCmuBwnTqnoDA9PR1sbGmDkm1LeR123Dk+FlHfHDMMdi7I6nVE+S7ZVsbWVxfobJkl\/xtfrEccoEWBdaqBUPDKAb8CBoDtoTPfg7l2SzHvNaEUJaus1Wqhx02GwZDdZr6\/jDLktYHfBqyo8LPNQqvhvG2IbIZbFjY6yOsjAvjl1bMh\/TLfN7nOw3VVo95h6TA5evRoMHzly9z8Ui+XFPz0LI5BYw6dbbbblvt0Aj+2oVKpBKfz6YeQm+9KXSAE\/Lr3AAJ+3dO+qy3rs40vv\/xyMAzVz8ONOrvazcxPz86iHrmZdOanD3ldDlWQfvLJJx+bLAH8uvcYAH7d077rLau1bs8++2wwy6lna1Ff0keZiDCHqLZ3fua7uCjDXq5HleNlJn\/\/\/Xfoduut3vnp7w+jDHs504u68NqWceKdX3cfAcCvu\/p3tXV9ltO2Iag522t7z5bEbC8vD7ENu\/UlNK7JjLA+mAK3mu2NCz+uW1\/iwv8fZ\/IkzuRIVwMlp40Dfjl1bNRuhU0SJLHOr7+\/33qeib7Oz3wnZkJVXztnrgmM+tme0sT8uqPdd357e3vWrzpsu03bdkDGOr+oUZpOOcAvHV1Rq0cF4r43i7IkJm3z8YVH2gqH1w\/4hWuEEoIVcJ3oFmZy0t\/2hrVn\/h5ZX1zFki8P+CWvKWr0oEDcd32mSd3MvLCri4cAidAE4BdBJBSBAlAgfwr8D8+HSAUgfqchAAAAAElFTkSuQmCC","height":174,"width":232}}
%---
%[output:4a415ee3]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnV9oXdWex3\/l+jDJXF96CvbPTSdoxD5IqiKpQYUZHK6OFRSdNA\/aUjUhlzpaxZQGGrVqwyQ0waCxTmmEIdGHxivch7aMwxVmUOlU7thpcGCkVUJLmwqecqnX5D44nOG3m9+ZdVbXOmfvc\/ZeZ+29vxvKvSbr7\/f325\/81p+91qpSqVQiPFAACkCBnCmwCvDLmcXRXSgABQIFAD84AhSAArlUAPDLpdnRaSgABQA\/+AAUgAK5VADwy6XZ0WkoAAUAv4R9YHl5mV566SU6fPiwsabNmzfT0aNH6bbbbqMvvviC7rvvPmuL3njjDXrllVfozTffpFdffbVqywcGBuitt96ilpaW69LVyv\/BBx\/Qk08+mbAybooXTVWdk6q5WCwGum3fvr0h\/aScy5cvl33D1GZJd++99wZ+gSeaAoBfNL0ipxb4cUYbjKRQeVE\/\/\/xzYocO+zDMLl68WLN8KY\/Tc10ffvghFQqFimr4Z0899RRFbUPYtrpOx33lvnz\/\/ff0xBNPJAqJuOHHu9BWr15NU1NT19lJ6vrkk09I\/ii61jbt9QF+CVswbfCL0t6EpWu4eBVG3333nRX4DVe0UkDc8Ovs7KSrV68GkaT+x5D\/eH355ZfE8EPkV58FAb\/6dAudKwpMfIj8pL0LCwsVkaEaaUjn1YhD8m3YsIFuvvnmIHrkxzTc1MvicvjRo1d9GqDaUN5kEI5iDx48GAwd+ent7aU9e\/ZUDEl1QMp0woMPPnhdZPzNN98EZZw5cyYoj6cyvvrqK+I+87DTBD\/TtEetqFodzrKWDG59WMujiK6urmAKRIefRO+iiW4D6cfrr79O7733XgBQfvIWQQJ+oTFWX8K0wU9evPb29vIw2vRSywskMFFfcpkzNIFU8qlDUJmDVOGmD7+j6MiW0tPzz3julR91+kEFsUDJpIGAWO8bA1CgoetUrZxqAFTht23bNtq\/f3\/F0Jd\/f+jQIXr66aepr6+vAn6s2+zsbAW49WkOFeJ6n\/MURQJ+9TEtdK5qCx76X+RqCx7V\/irHOecnIFJfTtMLxQKo9drgwn167rnnyhP3pvlGHRJLS0tBdKa\/iDpwqxnBFEWrkSAvMPFjWzRQ+9za2moEpw5yHX6m+kQ325yr3qbBwcGgbnXoy3k5Mn\/ooYcqdLINu6Wd7777bqCpTUfdVqGdPKUJAb+EDRclYnE57LWtFtdaFdWHfhKtCfxkCCiyqi\/Uxo0bgxdZT6NrxENJXvXWoyN1aF1rddMEWVPUaQOGCj\/ui2kVVwenWtbjjz9uBCaXVQsyerncFhn6sgavvfYaPfvss7RmzRrjHwnRXl\/VFz11GEr6KH9cEn5tnBQP+CUss6\/w0yMPGWaatrmY5uhky43M01WDn4Bs06ZN1pdVjSIFfjbTcBS8a9euoCyZr1LnrHRA6+Wo83nV4FdrvlCHsQl+YbY46e3T4cf9kaHvDz\/8QO+\/\/z7xfJ0pQlbn++QPGZfPc5Vq5PfCCy\/Q22+\/HWyxkieuBZuEX6nYigf8YpPSXFBa4MettwHQtjXGNOw1RX4Cv7vuuit05KcOlaOayDbclKiL2yOQDwM\/ibD0\/XthIj9djzB9sZXLf3B4uMvzsTx8NUHStKhjG\/YKDBH5hbEK0kRWIE3wU+cnZYhka78+Txcm8uMXNsqcnw021TYR2+bwxHB6fyR60stUASrDdS5DXSwJM+enLz7IHxnTz\/UITJ3zZN1+\/PFHuvHGG4Ool\/dn6n21TZvoi0eY87umNCK\/yDiLliFN8OOeyYuxdu3a8oohv3gff\/xxxdcG+gptWPhVW+1VF3VMm62rbc4Wq4SZN1XLlqF4NfipX9+YVnttUaRptTfMvJoJ4NIvVSNb5KeupJtWduVnrJl8XWSySzRPT1\/qTMHv1KlTtGPHjrIVeJMoz7nwLvlmPWmDnzo0VLee6JPn\/MLzI\/NitsUME4xMc4hiH3UhI+o+v7Baq9CQucNa8FP\/MOj7\/O6\/\/\/5g\/tE0hG50n5\/oIWXzf8uGZxMk9flOnt\/kuT2e45NIUtJw33nLjPQnS581hnnfMwO\/c+fOUX9\/P42OjtKWLVuCfV779u0LNBgZGTF+4xpGIKRJXgEBhEAk+RrjqcEEpHhKTrYU22pvsrX6V3pm4PfRRx\/RyZMnK0DHQBwbGwv+NTP688\/szWuRaehabYGieS39\/5ptEWWYYbgP7dfbAPhdUyQz8OPhFz\/8xYE8V65cIR668UZRjgbxNF8B0zDQ9ClZ81ta2QJ9qM6\/jfq5nS99AvwyBD8Z4nZ3d1NPT8918ONPhNSf++KEaAcUgALNUyATkR\/g1zwHQs1QIK0K5BZ+fBwQHigABZqvAJ9O04wnd\/Bj6L3zzjvBWWh4oAAUaK4CDD7e8N2MJxPwY+HCLngw9HhPV7HYSUtLa5uheSx1trZepkJhPvX9qEcM9D0bdhc7MvyaEf1lBn62rS5DQ0PB3r+Ojo7gPRP4Xbjwa1pevqmed8+bPIXCmQDgae9HPYKi7+m3e0vL99TW9q9B5Af41fMWrOSRTc68a51Xdm2bnLMEvwbkQlYo0HQFAL8YTcDR3\/DwcLlE0+dtgF+MgqMoKNCAAoBfA+LVkxXwq0c15IEC8SsA+MWvadUSAT\/HgqM6KGBRAPBz7BqAn2PBUR0UAPz88AHAzw87oBVQAJGfYx8A\/BwLjuqgACI\/P3wA8PPDDmgFFEDk59gHAD\/HgqM6KIDIzw8fAPz8sANaAQUQ+Tn2AcDPseCoDgog8vPDBwA\/P+yAVkABRH6OfQDwcyw4qoMCiPz88AHAzw87oBVQAJGfYx8A\/BwLjuqgACI\/P3wA8PPDDmgFFEDk59gHAD\/HgqM6KIDIzw8fAPz8sANaAQUQ+Tn2AcDPseCoDgog8vPDBwA\/P+yAVkABRH5EdOXKFRoYGKD5+fkKjzhw4EBwHwc\/cifH8ePHy2nU3\/MPT506RTt27Cj\/HsfY4wWDAv4qAPgREV8+pN+yppuMr6ZcXFykkZERamlpKYNuZmaGtmzZEpTR398f3NTG\/40LjPx1erQMCrACgN9KxDY+Pk6HDx+m1atXX+cZEhkODg4GYJOHgdje3h5Eh7arK8fGxoj\/SbkY9uLFgwJ+KAD4EQXgWlhYoD179hitYosMVeBNTU0FedUyTNAE\/PxwfLQCCuQefqa5PHYLdT6P5\/JMkSHDb25ujiYnJ2liYoK6u7vLc4RchsBv27Zt5Z8DfnjpoIAfCuQefgKotra28nwem0Yd0iYBv2Kxk5aW0n\/rvR9ujFZAgWgKMPhaWy9ToTBPs7Oz1NXVFa2AGFKvKpVKpRjKib0IFXhnz56NPfLjBjMAi8XNsbcdBUIBKGBXoFA4E0BPHsBP04rhxyvAR44coWKxGDv8AD68nlCgOQpw1McPIj+L\/ir8OIlpKwwWPJrjvKgVCsShQO7n\/PT9eSIqw+\/o0aPBPCAvivAmaNNWF07PK7y2rS46NLHgEYfbogwo0LgCuYcfS8iLGwIx\/l9ZAe7t7S3v6+M0DC7ZCyhfc+ibnHft2hWs7GKTc+POiRKgQJIKAH4r6jLcpqeny1oL1OQHpk\/g9M\/bOPobHh4ul4HP25J03buJ6A4iukxEx5KsCGVnVAHAz7FhszvsvYGIfo6o5iNEhTuIin8moska+XcS0T8T0WP0i9ImOkDDdDt9TT\/QGpqif6D\/\/OXdRD99TfTy7UT\/QkT\/fYCIfkO0eQ3RmdFrZT87TPTXRLRARPyjnzhN2Oeelfb9IWwGpPNcAcDPsYGyCz\/HQgbVbVqJ\/P7YjMpRZ8oVAPwcGxDwcyw4qoMCFgUAP8euAfg5FhzVQQHAzw8fAPz8sANaAQUQ+Tn2AcDPseCoDgog8vPDBwA\/P+yAVkABRH6OfQDwcyw4qoMCiPz88AHAzw87oBVQAJGfYx8A\/BwLjuqgACI\/P3wA8PPDDmgFFEDk59gHAD\/HgqM6KIDIzw8fAPz8sANaAQW8jfz0C8B1U+mnrqTFlIBfWiyFdmZdAa\/gpx4b1dfXZ71KUr1xzXRslM9GA\/x8tg7alicFvIEfg4\/Px+Pz8EwXh9uMUm++ZhkZ8GuW8qgXClQq4A388mIYwC8vlkY\/fVcgV\/AzHU8vBtJPctaH3abLzfWTnPV5Spzk7Lv7o315ViBX8BPA6YslfPz83Nxc+X4OudRI7uNgB+G8i4uL5YvNbXd4jI6OBvd+4A6PPL9W6HsaFMgF\/PT7N1T4CaS6u7uDi4fkYSAuLCwEiy6S33R7W3t7e5DPdnvb2NgY8T+Zx8SwNw2vBdqYBwUyDz8BV1tbG+3cuZN2795NEp2xgW1g48hufHw8iAY5De7tzcPrgD7mSQFv4We6LU01TD1bXEx39PLPTGBTLy0vFotlEKor0TJcnpycpImJCdKjR+nDtm3bylElIr88vV7oq88KeAk\/23xZo0L6BL9isZOWltbS8vJNjXYL+aEAFIioAIOvtfUyFQrzNDs7S11dXRFLaDz5qlKpVNKLMUVMjVdF5BP8uD8MwGJxcxxdQxlQAAqEVKBQOBNATx6v4GdbhAjZN2syn+AH8DVqTeSHAvUpwFEfP15Gftww21xcfd29lssEPyx4NKIo8kKB9Crg5Zwfy+lqwaPaVpeTJ08G+\/o4zcDAAJm2unBbeTuMbauLvpiCBY\/0vixoebYU8BJ+Lhc82JwMrkOHDtGRI0eoo6OjHCHqm5wZXLz1hVd8bZucJQ82OWfrRUFvsqeAl\/BzueDBJjV9uqZ\/3maKRPXP2xiifDCDPPi8LXsvDHqUHQW8hF9SCx4+mA3DXh+sgDZAASIv4WdbnMiCwQC\/LFgRfciCAl7Cr9ZiBwtfzxcePhgM8PPBCmgDFPA48suqcQC\/rFoW\/UqbAl5GfmkTMUp7Ab8oaiEtFEhOAW\/gV+9x9PXmS07S6iUDfs1SHvVCgUoFvIEfNwsXGME9oQAUcKWAV\/BTO42rK125AOqBAvlUwFv4ZdUcGPZm1bLoV9oUAPwcWwzwcyw4qoMCFgUAP8euAfg5FhzVQQHAzw8fAPz8sANaAQUQ+Tn2AcDPseCoDgog8vPDBwA\/P+yAVkABRH6OfQDwcyw4qoMCiPz88AHAzw87oBVQwOvIT\/3iY+vWrcFBoXyAqH4\/bprMCPilyVpoa5YV8BZ+6qXhp0+fJrlP4+LFi9Tf30\/qEfNhDSSHpPb29tKWLVvK2WxHaKknNZtOe9ZPcta\/SsFJzmEtg3RQwL0CXsJPP8lZvxzIdFlQGOkOHjxI09PTNDMzUwG\/MDfFcd7FxcXgQqOWlhbrHR6jo6NB2bjDI4xFkAYKNE8BL+Gn3+Fhgt\/c3Fz5MqFa8umRnQ4\/jtjGx8et5dmut2Qgtre3U09Pj\/X2trGxMeJ\/fOkRPxj21rIWfg8F3CjgJfwkalq3bp3xWkg9CqsmlYCrra2Ndu7cSbt37yaJziQfw3VhYSGoy\/TYIkMVylNTU0FWtQwTNAE\/N46NWqBALQW8hB83Wr0a8vz58+U5v2PHjgULH3r0Vquj\/HvTpeWmuTxOq87n2SJDhh9HoJOTkzQxMXHdQozpFjrAL4ylkAYKJK+At\/DjrpsWItavX1++XzeqPCb4qZGhzOdxueqQNgn4FYudtLS0lpaXb4raDaSHAlCgQQUYfK2tl6lQmKfZ2Vnq6upqsMTo2VeVSqVS9Gz15TDBz1aSCryzZ88a5wQbify4XgZgsbi5vs4gFxSAAnUpUCicCaAnD+CnyahutSkWi7HDD+Cry2+RCQo0rABHffx4GfmFubpSFIgy9xc18hsaGgqG2Pzw\/+eFko6OjrL4WPBo2A9RABRomgLezvnJgoe+kZiBIwsehUIh0oZnE\/xsQOT6jx49Guzr40WRgYEBGhwcrNgfyPOC\/PAKr2nvoWmVGAseTfN1VAwFKhTwEn76VhfdZupWF179la8\/ePNxtccGOhVinN\/0JQinYXAdPnw42LOnrkbzpmYpW748wSZnvGlQwG8FvISfaYuIKqMsNDCIPv3002C7iUCpHvhxHvn6wzacNg3FbVGplIHP2\/x2frQu3wp4Cb+okV9Y+Plgagx7fbAC2gAFiLyEHxsmzJwfR1b79u0j+RIkDQYF\/NJgJbQxDwp4Cz8W3zTUlKEkz+8x+PhRNyf7bjTAz3cLoX15UcBr+GXRCIBfFq2KPqVRAcDPsdUAP8eCozooYFEA8HPsGoCfY8FRHRRIG\/xsp61IP0zbSNJgZcAvDVZCG\/OggLeRn76pOCvGAPyyYkn0I+0KeAm\/Wpuc0yw64Jdm66HtWVIA8HNsTcDPseCoDgqkac6v1hceabYm4Jdm66HtWVLAy8iPBdYPDsiK6IBfViyJfqRdAS\/hF+Y8P6z2pt310H4o0FwFvIRfcyVJtnZEfsnqi9KhQFgFAL+wSsWUDvCLSUgUAwUaVMBb+NUa+mLY26DlkR0K5FwBb+Gnn9Ysl4rbLhBPix0R+aXFUmhn1hXwEn76Jmf1Pg0+yopPchYYhjGQ3Pshafv6+oJ7N9RHP8lZT2P63E4\/yVlWqKVcnOQcxjpIAwWao0Aq4MfR3tjYWPCP78\/Q\/7uadAy+Q4cOlS86F7DyJcUCQPVYfCm\/v7+f5D4OLl+NRBnAtjs8+IY3vtMDd3g0x6FRKxQIq4CX8BNwdHd3U09PT3Co6d69e4N\/fHVkWPjp5Ygo6p28GzZsCA5FlbokjRpdCjBNt7e1t7cHbbTd3qZCm8vGsDesayIdFEhWAS\/hx13WozGOvKqBJopM6rwhR3qmaykZkOPj48HFSAw\/3NsbRWGkhQL+K+At\/PShptydOz8\/T42u9IYBmxodFovFMggZlmp0yJcnTU5O0sTExHXRo+mABkR+\/r8UaGE+FPAafkmYQP9u2LZ6DPgloT7KhAL+KJA7+JkWN0xD2qThVyx20tLSWlpevskfb0BLoEBOFGDwtbZepkJhnmZnZ4kXQF0\/q0qlUqmeSmdmZuiRRx4JVn\/DPirQeOGEn2ZFflw3A7BY3By2+UgHBaBADAoUCmcC6MnjDfzUPXmmuT2ZR+OG82JEWPjZTomxreSGmRdUV3inpqYCLdX9g6ayZc4P4IvBi1EEFKhDAY76+PEq8tMjM31vnQpGfYNxNQ24nBMnTpT3+qlpbdthVLDJYotpq4sAz7bVRR9SY8GjDm9FFiiQgALezPmZDjBVI6fz58\/T8PAwrV+\/3ggxmzby5QYPk3nzsenRN0LzUNi0yZnBJdGmbZOzbIzGJucEvBVFQoEYFfAGfqZtIQKQq1ev0meffUamz9KqaSEQu3TpkjGZANH06Zpel+mgBT361D+jw+dtMXoqioICMSvgHfxMQ8vp6WmKMsyNWaNYi8OwN1Y5URgUqFuBVMBvcXGRRkZGiL+pTfsD+KXdgmh\/VhRIBfxkUSELogN+WbAi+pAFBQA\/x1YE\/BwLjuqggEUBwM+xawB+jgVHdVAgLfDjgwvCPI0ebhCmjiTSAH5JqIoyoUB0BbyJ\/KI3PZ05AL902g2tzp4CgJ9jmwJ+jgVHdVDA92FvXiwE+OXF0uin7wog8nNsIcDPseCoDgog8vPDBwA\/P+yAVkABRH6OfQDwcyw4qoMCiPz88AHAzw87oBVQAJGfYx8A\/BwLjuqgACI\/P3wA8PPDDmgFFEDk59gHAD\/HgqM6KIDIzw8fAPz8sANaAQVyEfnpJyyHOaWZXUM9QNV02rN+wKocbS9uhZOc8YJBAX8VyDz89Ps55Dh6vqdTblqzXV+pmk2\/TMl2h8fo6GhwVwju8PDX6dEyKMAKZBp+tpvZ9Fvi1GsqTVdh2q63ZCC2t7dTT08P2W5vGxsbI\/4n5WLYixcPCvihQKbhZ5NYj\/QYXAsLCxV37qp5bZFhI\/f2Xrjwa1pevskPL0AroEAOFcgl\/NRIj+8F2bdvHx0\/frzC\/Op8ni0yZPjNzc3R5OQkTUxMUHd3dxAFymO6kQ6RXw7fMnTZSwVyBz\/9fmABVFtbW8UlSeqQFvDz0nfRKCjQkAK5g59Ea3L5uE09FXhnz56l8fHx8oXlkqeRyK9Y7KSlpbUY+jbkvsgMBepTgMHX2nqZCoV5mp2dJV4Adf2sKpVKJVeV6gsd1epV0xaLxdjhx3UzAIvFza66j3qgABQgokLhTAA9eTIPP31rSi0vUOHHaYeGhoi3sXR0dJSzNrLgAfDVsgB+DwWSUYCjPn5yEfnx\/N2JEyfoyJEjFfBiAXglt7+\/PwAb78+Th+F39OjRYB6Q5wkHBgZocHCwIg2Xyw\/vF7RtddGhiQWPZBwapUKBqApkfs6PATU9PU0zMzMV4FKFUiHGP5dFkd7e3nIeTsPgkrlC2ybnXbt2BSu+2OQc1RWRHgq4VSDT8JOo7tKlS0ZVVSAKJCWhDktZFVav1tQ\/b9M\/o8PnbW6dGbVBgSgKZBp+UYRwlRbDXldKox4oUF0BwM+xhwB+jgVHdVDAogDg59g1AD\/HgqM6KAD4+eEDgJ8fdkAroAAiP8c+APg5FhzVQQFEfn74AODnhx3QCiiAyM+xDwB+jgVHdVAAkZ8fPgD4+WEHtAIKIPJz7AOAn2PBUR0UQOTnhw8Afn7YAa2AAoj8HPsA4OdYcFQHBRD5+eEDgJ8fdkAroAAiP8c+APg5FhzVQQFEfn74AODnhx3QCiiAyM+xDwB+jgVHdVAAkZ8fPgD4+WEHtAIKIPJz7AOAn2PBUR0UQOTnhw8AfnHYgS+Ruo+I\/kREvw1Z4CYieoyIfiai3xPRf4XMh2RZVQCRn2PLuoHfDSsvedTO3bMClK8jZPwVET1F9IsbiP6XofIfIfMOEw0Q0dqVLJ\/8johq1fsropM76Yl7fkv30ef0J\/olTdKLVFz1W6IHfkM0SETTRPTxAaJ\/HCa6g4j+7g9Ez91NT059QI\/R7+hnuoF+T39L7\/9bH9HfHAjR1r8nenYTEbPzf4jo\/csrlYTIyrD9y9uJfvozEf3TirZh8nGau4mI89XSxFTeX6zkDVtXPtMBfo7t7gZ+jjvltLobiApD1wK\/PxLRv3MEd6x2C\/5qmOgRJfD7Ngz4aheLFOlVAPBzbDvAz7HgFdWtWaEfUxNP3hUA\/Bx7AODnWHBUBwUsCgB+jl0jK\/Bjx+Eb74vFzY4VbH516Hs27A74OX6XBH7FYictLfFsfzofBl+hME9p70c96qPv2bC72HF2dpa6urrqcYWG8qwqlUqlhkpIYebt27cTQxAPFIACzVWAoff8888Dfq7MAPC5Uhr1QIHqCjQj4pMW5TLyg0NCASgABQA\/+AAUgAK5VADwy6XZ0WkoAAUAP\/gAFIACuVQA8EuJ2Q8ePEjt7e3U09NT0eJTp07Rjh07yj\/r7Oykw4cP0+rVq8s\/47zT0\/zR7bWnr6+P9uzZk5KeVzZzeXmZ9u3bR8ePHy\/\/4sCBA9fpksrOrTT63LlzNDY2FvxT7Rim72H8Ic3axNl2wC9ONRMq66OPPqLh4WHSX3J+Sfr7+2l0dJS2bNlC8nJwM0ZGRqilpYU479zcXBmIkmfXrl2pBAaDfHFxsdw\/edlnZmYCDdL+XLlyhQYG+MQJMv4Rq9b3MP6Qdn3ibD\/gF6eaMZel\/6XX4cdgO3nyZBkEXL0aNTD8OErq7u6uAB3nW1hYSF30J2AYHBysAJ0tKo7ZHIkXp0ZtegQfpu+1\/EGNIhPvTAoqAPw8NZKA78KFC0FkNzQ0RNu2bauAGL\/0\/KhDWPUlufXWW4MoQocFv2Tj4+PXRRaeSlFuFoOddWA9Ojr4TMFrj+ml970vevsEfPwHjh81Wpc\/arX6PjU1VdUfshAZx2lXwC9ONRMqS4Cmwk\/gqEd1ato777zTCAt+0fhFOnLkSAVEEmp+bMXaoK0P7WOrsEkFmfpTq++Tk5M0MTFxXZRv8p0mdcu7agE\/70xyfYMAv2ua1AKAvtCTAtMamwj4ubEc4OdG54ZqAfwAv1rgR+QX\/RUD\/KJr5jwH4Af4AX7xv3aAX\/yaRipRwDY\/P1\/Op2\/bsM3bYMHjmmRZWPBQncY07A2z2IMFj0ivHgF+0fRqSmob\/GxbG2RVcMOGDdatLvoWmaZ0LGKl1bZ7cFFp3bity2CCX5i+1\/IHdYU8ovSZTA74pcCsNvjpG5Ztm5wPHTpUXtnNwiZnPpJMFjeytslZIll9qwv\/nCP9an0P4w8pcHdnTQT8nEldf0XVtivI1x9Sur451vRJVJo\/bzNNE2Tt8zbb1p0wfa\/lD\/V7YfZyAn7Zsyl6BAWgQAgFAL8QIiEJFIAC2VMA8MueTdEjKAAFQigA+IUQCUmgABTIngKAX\/Zsih5BASgQQgHAL4RISAIFoED2FAD8smdTa4\/UbS\/1bndxfUoIt5mPsOK7lmWTbhynuMieuEuXLlHSB6GK7uvWrcvMRuwsvDaAXxasGLIP8sLfcsst9O2336biSCvT8VtxwI\/L4M3fhUKBNm7cWHEgbEg5QycD\/EJL5TQh4OdU7uZWJtDgw035E7g0HGWfBPxUGPG9KOoXMElYCPBLQtXGywT8GtcwFSXIcLWrqysYeumfSnEnTF8QSOfWr18fRIp8FDqfDi0HqwpQn3nmGXrxxRfLWvBQkqMqvmOEh5b8qF9iVPteWWB0+vTp4O4SeWSobqsz7Jce6idxctq16CJ1qe3jI\/\/lAijTBVG6btxOfuS+Df7\/fJ2APuzVv8aodyoiFQ7oYSMBPw+NkkST9G9gw34TK0Plhx9+OICmDi15gbdu3VoeOsptcSoo9PrCwI\/n+GyRH0NRrTNsf1hbHfy1\/hDInKC0ua2trdxXXR8pn2Ep7TPBj+s8ceJEeepB\/+OUhA+gzEoFAL8ceITpwIMwL5vpZTfBTx82mkCk52sUfnqdYRdiTAc7VGuvHhHq843VwCmQ1OFnO1wiCsBz4LaJdxHwS1zi5ldge9n06ENvabUXWx\/2qkfIu4CffupJGJhz\/2ShQ72\/JAzkRRtefqDaAAAByUlEQVQVfnI7nj6c1f\/Y6PAztUGddtAvqmq+B2WzBYBfNu1a0Sv90nK9y6a5srAvqO2+Cb5IXd1CEnfkVw\/8qs1psiYyr8nD7TCRqcx\/6tGhDH1tc376XF8Ye+TATZ13EfBzLrnbCqtFROr1mKbIzQRF07BXB5GvkV+1YWWtuU018pMhtxwWW0\/kZzqvz61noDbAL+M+YIvgpNs6EEwT+KpEScNPH4qH3epSa9hrmvesNszn36mr2ib4cYTYyJwfb95W79IV7fWfZ9xFm9Y9wK9p0idfcZgXXp3v4u0YIyMjQcP4f3lOS3\/igp8MDU0nE6vDTxMQqh3zbhqCcl1hTrBWLw5\/4IEHQsGv2mqvbF0x7fPTIR\/GVsl7TL5qAPwybO+wq4cSHe7evZv27t1rVYTn8GRfXKMLHlyJfso0bw159NFHaf\/+\/eUtIGoa2Tpy7NgxijrnZ4rQ9I6qAHr55ZeDfYv64kO1BRO5hEr2+XH5vD3ItskZ+\/ya+\/IBfs3VH7VnUAGBXXd3N\/X09GSwh9noEuCXDTuiF01SwBRR1ppnbVJTUa2mAOAHl4ACDShguiDK9AlcA1Uga0IKAH4JCYtioQAU8FuB\/wNQZDZubsk93wAAAABJRU5ErkJggg==","height":174,"width":232}}
%---
%[output:47d289c1]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQ2sVtWZ75dTO8C06SgykQ8rKpQ7ls70ojOcwtzRqomTgq03VpjUApMLEiIcx1pBcA6NNZUKBaZ2BA0RTC7SNgVrMkYwNalWc0fmOCNk7NjJnYB628rHHYr0ZjrI9IOb\/zr+X5+zztp7rb3f\/fW+77MTApyz197r87efr\/Wsc86ePXvW6KU9oD2gPdBjPXCOwq\/HRlybqz2gPWB7QOGnE0F7QHugJ3tA4deTw66N1h7QHlD46RzQHtAe6MkeUPj15LBro7UHtAcUfjoHtAe0B3qyBxR+PTns2mjtAe0BhZ\/OAe0B7YGe7AGFX08OuzZae0B7QOGnc0B7QHugJ3ugdPgdOnTILF261Bw5cmRYB+\/cudP09fXZnw0ODppFixa1fv+Hf\/iHZtu2bWbs2LE9OSjaaO0B7YHye6B0+AFsmzZtSoQZ4bh+\/XoLw9OnT5uBgQHb8nXr1pkxY8aU3wv6Bu0B7YGe64HS4bdnzx6zf\/\/+RJD5fg8gbtiwwf5R6a\/n5qQ2WHugkh4oHX4bN240l1xyiZk3b563Qfg9rlWrVrV+f\/LkSbNs2TKzcuXKlmpcSW\/oS7QHtAd6pgdKhR8h9uqrr3rtfVRxZ82aNQyOLDd\/\/vxEaPbMCGlDtQe0B0rpgVLhR3venDlzWpIdgffnf\/7nBo4N2PeywO\/ll18e5hgppVf0odoD2gOV9cDo0aMre5d8UanwS2oR7XwAH5waMfAD9B566CFD+C1YsMAsXLiwlk6r8qVnzpwx+POhD32oytfW+i5tc63dX9nLMc4A3+TJkyt7ZyPgt3v3bvPggw+azZs3R8MPsOvv77d2QEiNvXC988475vjx4+bCCy+0E6UXLm1z74zzqVOnzPjx42uZ27VJfoAfYvl27Nhh13PI4QGJD\/B7\/PHHzcyZM3uBAbaNAMGxY8dqmyB1dLS2uXfgV+fcLhV+iPFbs2aNefTRR83UqVNb6whq75tvvmmBlxTqgnKI\/WM5hV89X0eFXzU9oMCvHvilws\/nzYUnd\/Xq1fYPwEanyPLly61nNynIWeGn8KsGQ\/W8ReHXZfDDNCLM9u7da2fVxIkTvZLg2rVrW7POt71N4afwqwdL1bxV4deF8Ctq6ij8FH5FzaUmPkfhp\/BLnJcKP4VfE6FVVJ0Ufgo\/hZ\/TA3Uuih8cftvW5pNTzi9qjUc9p842R1WwhJu0zQo\/hV9D4AfwXfPwQfPJKeeZ55dfUcJyT36kgqB6EFQ6wO++rO5xLtXbW2SHqtpbrdp7zcMHzA8On7JD+PzyGZVKf3UviiLnbeyztM3VA1\/hFzs7a7qvrkUB+F095Xxz37NvmHuvv9R8+c8urawH6mpzZQ30vEjbrPBTtbchau85dz1nJb77vveGhaDCr1w0KvwUfgq\/BsCP9j7A7weHTpkXDr9dqd1PQVA9CMpFu\/\/pdY+zqr11jHqGd9YxQQi\/s5uvNV\/+3htW9cW\/q7rqaHNVbUt6j7a5euAr\/Oqe9YH317EoADxKexKEoa76j9deMr8zfXbotuDv62hzsFIl36BtVvip2tsAtRfOjiEv7xVGqsCheL+f7d5sxkyf1TYAFQTVg6BktnsfX\/c4q+RXx6hneGcdE4Tw23v1OxZkdH6kwQ\/gI\/w+fN93M7Rw5K11tLmtChdQWNtcPfAVfgVM3DIfUceiWP3X3zT9P3\/SnH5tv5n2xBELv1C4y0\/u\/awFJQB4wfy77J+8Vx1tzlvXosppmxV+qvY2QO3tW7HVPH58na0JIDZucEYQfv9680Rz0X1PWPgBggq\/bFhU+Cn8FH4NgN+uBdeZq6ecZ2sCkN184prUWD84On56780WfpAW8f92VF8FQfUgyIbqYu6ue5xV7S1mHEt7Sh0TBFIcJTeAbNGFA1Hwg4pM2x\/+nfeqo81561pUOW1z9cBX+BU1e0t6TtWLAt7diaunt6Q4wGzZtU\/b1iUlOJDAU\/jlmwhVj3O+WhZbqu42K\/yKHc\/Cn1b1BHFVWMDsW0v\/LnWXB+6hqivL5435q7rNhQ9ajgdqm1XyU5tfzTY\/wgtqK\/+9Y\/ZG8\/LoyxMlP3h6cdHOR+eHwi+eggo\/hZ\/Cr2b4SbW1Hfi1E+6iIKgeBPGYLu7OusdZ1d7ixrKUJ1U9QQC\/Z57aZxbs+r5tD6Q4SH5fe2tC4v5exvj98MrFNhHCLY\/+SVuxflW3uZSBy\/hQbXP1wFf4ZZykVd9e9aIAyF44fGoY\/B763ZvMlvNuSoQfvcO4D0kQECP4qc\/MyR3rV3Wbqx5T3\/u0zQo\/VXtrVnt99rtY+CEYGmnvl7y0SuGXkagKP4Wfwq8B8INz47P33G9rAhji\/zefuNYmN8W5Hu5WN0h+hxdtN3NeGN2CH4Kk8wY6KwiqB0FGVhdye93jXKnaywPMJ0yYYFatWtXqwMHBQbNo0aLW\/\/XQ8vfmVtUTBCCD7c4HP0APaq17qJGEHwD53a9+ydx90VGFXwZEVD3OGapW2q11t7lS+O3Zs8esXbvW3HrrrS34HTp0yCxdutSsX7\/e9PX1GQISPb5u3TozZswY2\/l6gFE1BxjRwbHhi59vSX6wAS68cMDcPemodXzgQnJTZnHB1jbpFLl9Sb\/CLyMy6gZBxuoWcnvdba4MfoTckSNHhsEPQNy\/f\/8w0OHeDRs22D9jx45V+B07ZsaPrxd+g6MuN7f\/\/ElD+9\/\/\/j8L7LhASvyDVx6z8GMsIOCHe\/Nucat7URSysjM+RNtcvapfCfwozX384x83\/\/RP\/2Sk2rtx40Y7TaQafPLkSbNs2TKzcuVKKw2q5FcN\/GRcHyU\/hr5gDGa+8y8WcAAgs77g\/\/g5JMPfmT7LBkJ\/94G1FogKv3gCKvy6FH6Q7nbv3m2+8Y1vmE2bNrXgRyjOmjXLzJs3rzVTCL\/58+e3fq5qbz7JL0tqecIPzguEquCKhZ\/c\/4t8gPD4xsLPzQCtIKgeBPGYLu7Ouse5dMlPSnFwZAwMDLQFv\/7+\/pY0iOd1+4UJcvz4cXPhhRea0aOzLQrAbN\/Xh2LuIK2tuXrIXpd04f4TDyywe3l5L8D0\/LPP2SKQ8HBByoPkB6mPPwf8cMQlyiFIevrufnPxrteDw8N34kbe306bgy9s6A293ObJkydnnttFDGPp8JNqrevtzSP5sdELFiwwCxcuLKIPGv2MM2fOmBMnTphx48aZUaNGZarrr5\/dYfAH1\/uuX2L\/pF2\/OXzA\/OqRfvPIvL3mjr7z7a3yGSwLkMKmJ+H3F7OfMJ+YNMaW2\/\/iS+bKp1aac2\/bYn5ryhWp7\/zlIyvs788ePtiqYzttztRBDbq5V9uMdk+bNq374IcQFqi527Zts46LIuBHya8XpD6szXYkgn974PN2twYkNjgm5t65NnW5\/+KprVbNfWzh91uHlPNnKEjo4e++M0NSIKXB\/zJ5l1l91QRbDpLilJ23mnH37AoeZoTno40AL0CJPcHttLlBPMtUlV5t86lTpypz5rkDUqrkB6lv+\/bt3knAWL4dO4YkE3V4+NdKO3YRhK0AVACS3K\/rexOOqwT4INFB7UUZXEx0IOFHiY8eYPwO8EOMHw45wrOwvxeZnUOZXZgBRmaAbqfNmYjToJu1zdlMOkUMXanwcyvoC3JOCnVZs2aNjf2bOnWqfYw6PLI7PBizd\/dFxyzEkhwQPJ6y\/9STVqI7tHB7C350gkj42fEYfblJgp9MiJoGP1\/uQNRRQVA9CIqASdZn1D3OtcOP8X\/Lly+3nl0Nch4+hfJOEBm2Qvgd2fCalczcC5Iadm4QfpDYeJ+EHx0dsfALpbXisyFpUupU+GX\/yGWFTlPuzzu3i6p\/7fBDQ7jzg43S7W3vDW\/eCUKwwAuL83exC0Oqs3ICEX6M3csDP7zn3j+71EKTkl8s\/KAyw3aI92eBX5YwnqIWTFnPyTvOZdWniufW3eZK4ddOh\/a62vuhn71uPaKxR0LSVgfgYZ8t4Cfj9+RY4JDyHxw+1QpcToIfIMWdHbAl4qLXF1IhbX74OdNcpdWXMYTcOofYQLwbjo9jgV0tRaTLb2c+Fl22bhAU3Z6Y59XdZoVfzCjVeA8myJG\/22dDUHCFpClWlWCh\/Y7JBxi83AT4MXcg4IekCXSSxMAPZeEkwRUbTF3jMAZfXTcIghUs4Ya626zwK2FQi3ykhB+cDLFJQgkWSnFuthYXfghQnvr4rdaJ8dm\/+orX5pdF8sPZv6G6so7YFwyVGafGAe4f+MyKoOSH9oyZPssCMMarXOSYlPGsukFQRptCz6y7zQq\/0AjV\/HtMEEBi8K13LJjgjWWK+bSqZYHfOXc9ZyWvrPADjLnTA9IbMr3wioXf1346wfzzlYst\/KCaA5gh+EmVF2VipeGahzL19XWDoI6+qbvNCr8aRh0qaaztTsIPkk5swgDAD2ChFMekpMzTJ5udBX4\/\/uBWK23B5pcGP57rkdZO1hHwQ0IEAjML\/NCXCKeJ7c8ahjvqlXWDIKqSBd9Ud5sVfgUPaOhxdETEqmqYID9ecJmFzZwb59idEzFlGeAs4QcYPrRjy4gqAn5wVkDtxHvS1N4s8MOL0rI5s44AF4KqAT9kgP69e76ZqvZS8oMajvCcLIlT0f+8mgTMukEQmrdl\/L7uNiv8yhjVhGdy0UKCC4GBjyD8oFYObl1hvahJXlv5WoKFsHMPJnIlvzzwQworeGghAbpqr3sWiK9LUEeUg6oL+CEVFkJeQvDjIekX\/\/sKC78suQPpKIn5gFQ4NTSwO2PSjiLGRuFXRC9GPoPw4x7ZGC+lm2MvJoQE1SFYAExc7pGUrDJ3dyTBj8\/C35C0KPkBWjysqF34bfjiLdbBEgs\/6SV+Ytxz0aYA+fGB6h7T\/+wnN+1W5JBH31a3FBRd0QJvrLvNCr82BjPrgiCAmAw0RvqQwcqwi5UNP0CNMGLX4J0EXFHwY7vk+yjRhSQ\/6SXG7hWYAmJA5jpKYvqffUCJsSznSt0gaGMZ5C5ad5sVfjmHjosBKmzsKWVctIi9i93478IvzXHBpkiwuJKfDGDG\/a7kVxT8CLKkvkmCH8ohr19akLN0lBB+MSCTdYr9iKCP5Ba\/mPfkmVJ1gyBPndstU3ebFX7vTm4MZCgDiWtTY5xZjNSBsjL8hDFtIaO73KkBu1iMLS0pHT2e5e7vJfwQpkJVuQjJr0z4SUcJ+jX2Q8K+g80U\/R8LMrYF8wN9G\/uxo7kBf4fGuW4QtAuyPOXrbnPPw09+1WMhlld9kh5YqGoxIRruTo0k25204cGehedjvy1UZS7CWPhRWmxH7U2DH7zL3MsrU2Eh7RbqHZL8JKRRx9gPibQVIj4xFIQtVV7kRWQWm6zzBM8JgdYFAeZY1g9yHgBJbSHLxz8L2JPqpfCLHLGy9vZSGgAwYu05hB8WQRb1SS5aLHJcISlCqsqQ\/ELwwz5dgAVe2DT4QeK773tv2DpgX+8vFo+2Qcb0KrtSblabX1o9XfgxOLoK+CHcZ8t5QwcwxR6sTjWbcY0hkElo4t8xc0uCAIldMRa4YkGLezEvGUOaBWR57Jl5hAZ3qSv8GgA\/LAjsnIiVBKiKQmrJIkFID2xILXSlDtrq+G7foqAKS6lKwo+TlWovkxnAY9su\/ODAYaAy6034cS8x1T63juhDwo91RAbo\/3fBZd4Mvz5bYewHiP2PEJ2PvfJYdHwgywF+SOyQBX5UlUNSvgQBMnBz33Lsu9DvqCeuLHZoCbHYjz\/ekUdoUPhFws69rSzJjxMbwIgNlpXexthF5DohQhKchJ\/cqREDP8a++eDHtFaEH98TK\/md6DtopQsZ6pIEPwYUy8UoE6ciPq8I+MVspSMc6ND57le\/FB0fKOMRYV+MhUSWuSXhh6B2QC\/L1r28YTycTzjmAGsgpIm4H2VIzyGwq9qbE3osVgb8JJD2Xf1OtDG7HfgRSGkQk13lblNjOV9iUpmUFGDxwY8B0ry3HfhBqgMMQvCT6hvfy\/i8vPCT5WLg50qM6MeYrYJuuVj7oiyHj2roXYQfUpfhBD1AGmMYq43ktUNzLme1Z2YBu8KvwfDDQqJEE2NjwcLBIoYE8AcHdgQnNu0x+JJnhZ+7Tc1VX2W3ok4vHH7bqnRQ45EthQ4PliP8YHeTVx7JLy\/8UMf+nz9pVTt3ZwjaG6P2Svhl8YBT8nvmb\/dFnS3MfuP7YkArx1tKyGlzy4UfTSqxdsm8YTyEX57YU5uH8aKjdhxjkm2o2psTgmVIfpSiZLbjGPjJUAt8nfGcUDn5LgApDWKyi7LAD6osrlj40d6HMoSfhArrwSBnwPSJcc+31N4s8KPtioDOCz9pb6WtkDtD0lQ2F2J0roRsam65GNBK+KE\/2bdp7yL83v\/iNw1OtEM5SMexqqj0ZMMumUU1B\/hwf2zIkJRq4YSLDTJX+DUMflgEWNRM9R5aDK7tiGElMfDjuyT8Qu+jevHC5iX2bFOp3rgePcIPnl5cvoOIIPnBBnfNwwft9jQ4O\/A3258FfogHZDIEn8ODNj\/UBX38qRvntDzMhJ+UTtm3r83fYmbMvMLr8ADooELKesbAz4VmLPzcj1bMu9AOWY65CmPhh6M\/5YcmNLfwvrxhPDICIatKHwt2VXtzQq9Mm58bRoKJEIJREvxC5dx4PVcNdbuHINv23A1WNXTh50tuwDI7j6+z5\/VK+LHeNGwjKUAV8GMgONoALysvSJDyYyClTKS0+uVVnx8GP\/eEOQnNGM+5C7\/Y4zU5bnxfzLsIP9m+kEeakh8OcUfuxqzwo4aAj1BeJ1yojhwj94MQu25U8ssJwTLUXh\/8QuqCawBHc2J2C7jvIoySMrTIWDgJv7RygB8yMi95aWUi\/Nj9dFLw7A6osJjURUt+hB\/eh\/g6ABcXAe2T\/PLAL2R6cCEWe8iSdG5BYs\/iqMIHiO0LgcWFHz5csNtmSWEmPdkxkQtuBALaGuO5dfsy1DaV\/HJCr2zJT4aRxAyiO2Fiz6htIvwAehxZyUOJ0NdFwc8NvZDwk4COhZ97vKYr+ZUNP0rRWeHHciGwSPj9zYlLbU5F2PvgJAtpFa42ktWTTSdcrD0zj9DgW\/4a5BwJxTIkP9eZEJqgqKprAOfEC0mMSfDzlXMDgWMlP2ZkhuQHqDNJqFQpWx+T0ZfbRbX6r7\/VOrWtE+DH4zWzws+V4PKOm9zdkzZ1GaLEMcD\/ESTty6SN5xAE\/7lydiuhbKxWkTeMx1VfY1V6dy7HrBuFXyTofLdVBb+0CZoXfoASUkFJKTNt8RF+d086asMxII2dXjfbOjzSyjUVfqgzt5Xh3zgvhKp5rOTnOnN8YTxpjgEf\/GLCVlyI5fXSh6QqCT+ZWSeLNkKpnc6ckKPEZ89E2VDYSlaw96zae\/r0aTMwMGD27t1r+8B3IPng4KBZtGhRq4+qOrQcg3jziWtb+etCExQVdL+W+FnaIpJn4rono2FiwwHhSgMx8JPlaOuDCoukpLAThSQ\/1BuB0nklPyYkSPL2EhKQfPBvfFTQ1yH4YQxeNJeaP\/kf\/cMcHu3AD2XpAZfQjIGfDGtCWEeat10uclerCElVZcAvpC67ElysSu\/TmND22N0h7KeuV3s3btxojh49atatW2fGjBlj8H9Icdu2bTNjx441hw4dMkuXLjXr1683fX19hrBEB7EM\/l2W5OdLppk2iO7XMhZ+sKu552MkqUK0b1Hy+4vZT9hgZSn5EX4EJaSpWPjRCQH4cZsXJ2SszS8r\/PB8mXYe\/\/ftDCH8Zl0120z8kzmtNhPwPGFOhtakSWPsH6jL+PhIU0Aa\/FCOaqcctyzwc+dWmlQFEPB8ZplcIgbQecN42oFf1nXTc2qvCzZ0wMmTJ82yZcvMypUrLez27Nlj9u\/fPwx0KLdhwwb7B4AsA36u44JSXUjs98EvzebBnRRZ4YddEAwoToMfQcmwFZmOPsnmB3AiVq4q+KFPsVMB8AOEYMhPgx+8pLhH2kNd+Mm2hWIf6dH2wS9pBwUldoybm+A1JrRDxs8hRX9IqkqCX4w9zdVG8jrhYu2ZvraF1k3Pwc\/XYBeIkARxrVo1FJjrA2ST4OezHaWpy4AfwMR4PZkoNE3yS4OfLOcmKMgDP25RQj+XIfkB0HhHLPyY0QSQBuR+cOiUlWppK3RV+hj48eMjoZk2biH4pR0i5TogssIv69a9dsJ4pB06iz3TlfxC3naFnzFW7aUajA6BPXDWrFlm3rx5I+A3f\/781s+LVnt9Xtu0rzPz3\/lsR+3Az3ecJO1bSZJfUfDDAt73t\/ta+zObBj\/mEKRKXzT80nZrSIndJ\/mlefeTtArML19CCvS7lPyyws\/9IGeJYfTBL81WmNa2kIPFBWDX2\/zYYKi3a9eutf\/duXPnMPte1fBLSqYZgh+2hCWFWmBSuLZC2pvSJD+oeK53TcIP6sTWK74yzObHyQ7VUAYro2\/TTmFDOR44jj3JgB\/aDNWP0lbZkp+MKeSeUjgSeLGO\/D\/qE6vS+xatlLzd94Xgx7yIWeCHsfuP1\/bbeeLLWJMGv7d2rjO\/fnbHiCS0vrklQZI1jAfzEtJo0r7xJPjJRLnSLhlS6X1SH4GfdlZLUrmifl55Gnva\/CDV3XDDDZklv\/7+fgtOXPAKZ70w8J967F9aadQR4PnM4o\/ax2AQsan89ftesZNDXixH+CF4dc3VE1rlkH0Xp475yjzZd8pM391vbUf3Lccuh6FnI2kl4Hfz9iFPOK9PPfYj+89dH3zGYJ8n4Ic60uHBclickI7gwYRdC9cziy83l917pYXi3C\/ieMn32sEkmagH2oE9tGivCz+E1cgL+eX4HtSJG+\/lu+CAYD+iLBYsUjNZNVOovT74sR\/ZJwQx\/i\/hJ9+H58pyqCOywbj7nccMvGRWXzXBLH78uhH9v\/fr97fOCfaN219e8IZZceBLI8rhXXg\/dqK4F94nU\/SzL9kfvrmFZ6x\/4aj5yEtbzR+\/9i0LPzknMQfcOSLfy\/kg5+QTt8616bB8dcT8wnyhKUDOE7QNH8Vrrh\/yzMsLbXPDr+RYJ7VtxIPe\/QEkv+PHj5vJkye35nbSvWX8vHL4oRGQAnfv3m0efPBBs3nz5kxqLzthwYIFZuHChZn75O\/fOm0+9+TR1mHXcCZ8+6ahDLgA2K8e6TcH7vy++cSkMcOe\/bknj5i\/f+sdCw0Yzs+9fom5o28ILPhao+z7b9s6rAzfdXjOKftcQOc7q+a27uE+zv92\/w77s28Mvm36Lhpt\/8b7sfl\/\/4svmb+e\/iXz7ZsmmFGjRtn7UO7s4YOt56ANqBsu3HfF14cW+p23XDusHSxH+L3ymU227n2TRreeB9i8cftlw9qBwFvADxAG\/FAG9\/FdAO0\/Tr+l1Y+yL993\/RLz4ODbNjcd6vk\/XxpKz44L5WQ\/yrb98qLp5v0\/fc2+h5d8n1sOdUR74CWW16UPvW6+MPN8c9ueuSP6BO3An9\/eNHReBi+Om4SfHDe8C+3CH185CT\/2JefWubdtMb81ZehcFfd9SEKLLW3oZ85J1A9zgHPELYe5gq2C2BUi++R\/rV1i+8KtI8pzLtOZI+dJUj+yT9La5ls3wxrq\/OfMmTMGf6ZNm9Z78EO4y44dQws\/1uFByS+P1If3UIKT2Y75lU37OvNrSfjhi0p1DZIQpEYcvMML0s4np55npcyf3WJaCSqRoIAXykGtnXPngJUAbxoc2vcKNa9v0hiz6sw++3uf5Celo8cWft9sePGolTggZeK5rpSJ5yZJfvjKowxtbGmSH+2QgFKM5PcP028xf\/\/Tdyz8IKHCZir3+8p+lHX0wU++zy0HiQWSLNPms4\/TJD\/fuMk5ggS3iJlkkDmfiX5EX829c8iMw8udW7Ic55avjpD64NAh\/NBPnJNJdcQ7Wc43JyH5YVx9p8ahT3ARflIbQT+ib91yrsbka1seye\/UqVPe7D1p0Czqd6VKfgheXrNmjXn00UfN1KlTW3WW4S1PP\/20N9QF5RD7x3JFOTzS9oimebtklhXXduTaPGgsp6E+KVEovXQM\/aCUA\/jJ\/a8+m5+EH1LTY\/HIMBLf+bu0p0GtgvcZExxwZVIDwo958jhgMmOIhJ+0L7oprdiX8NjCW5sEPzcch3U8Z8oMK40CBF97a8i8wPehbaiztBWijoAqLqhtOJwJqaRgpyXE3D5xx42xfXiGLOfaQZO89LTxpmWp9nmJOSezbt1zA+hjPNmso4SfPK0vaUdJzPEDoaBqF1pd7fBgwLJ0aLhxfgx9Wb58ufXslh3kHJPtOMlwDpgxbbucaEnwcw8HchcRy3FSSBVPbgELwQ\/1\/cBjQyo5Qdpk+DHO0OfwYJ8QfoDFnBdGW6mIITlJ8GM\/yiQKsG3Fwo\/OEXx48DGRkp\/8IISC0wk\/eYwA6gawJKUiYywi7ovdtyzDcdy+TIpA8O0bj4Gfe\/xA2hEJsZJZV8MPneBub5s4caJXEqQnGGXK3N5GCc6XTDMtxx73zYbgJ7+saEtaolAXfljUkL5wxcCP6iPhJ50JsfBDHQBy\/F2W5IdwCoCZam8M\/KAuw\/hP+IXADrAkwU9K3jLOUn605Lih7xFnmZTgNQ0skDjTErUmwQ8ggHMFly8Jrc9LnBZAn+TJZjvlB0HCLymo2s3A7duTnRb76ANi18Mv9isQuq8otddN+Jl2zoWsEyaaVLvkIpKR8e3AT6p4afDjoiVE8sAPtjequyH4cSM7VNs8am8e+MF5ceVTK1vwC4E9D\/xkAgA5bsw5KNvqSn6YG25oU15AYE7CxvuxPf2Z4UdtxP3YtQM\/X3KPNKEhTapNW9cKvxD13v19FfBLG8QY+AFCWOhQmXhlkfwAM27+Z4ICOEJe+fQm88CNl7c8Yj74yRhEvDsk+bnwA8DxLpRzbX51wo99kgV+lGA5BkmSn9wZAomNtsUY+MXEZ\/qkI19wtISfuwUvzQ4ttRGfPdMXH0j1NatKr\/CLBFUZtxUFPww+rrSUSq74TqkgJPkey4EAAAAgAElEQVRh0cHmJmPTsJCSJAhCjOorFzrPEMbfgClCGNLghwUFuxgN5iH4oX48bB11QHs\/9sqO0uFHSZV\/+2x37BNKfriXkiPnlQ\/slPzQl8ygzNhHeTgTxpCxjwQLnws7FsukjRudMvSIjhucYc0UUJVxpWWp9mXxAcQQiwjJLwl+rh1aJrSAKSbGmYN2twu\/tLap2lsG+drM6sKtaTBkY4KGUr1jEBH4C5hBIsCFRRGCH5uOLz7K8nCgGPihLAJ0sZAYhoOfMRauU+GH+qMvAFt+HMqGHzzhBBk+IBg3ZERmJpp24SdttVLKjEnRnwQ\/hPEgKLws+NE5QkjffdGxVhhPjErvHpEQK9V2lNort6HNnTvXbkm7\/\/77W\/n47Jft3e1pJXHO+9h2JD+5zQxAStojSrUXExrwYwJOem2TvIauBCHhJ72USd5ehmgQfm7mkyySnzwwSNol0TZKLJT8uLOjTMnPBz\/aGEOSHwJ8YZfMItWy\/Qz\/gbrMn4XgJ22ulPxgF3RNAUleejdEyQeINPhxZ44vY02SNlKVM6fr4cedF8y1hyQE27dvN4Agc+v50lRVAcE88AOUMFFlFmHUNQQ\/tscNmQjZjlgOE3zL797U2uPJn6fBD6oaFjokv1j4oX7YESAB0XT4oS\/oaEmDH8J7\/vhH32oLfgCXNAW4\/e9+tDhuUmL3nTDXDvyk9Av7LmNB3a17jGFMikDgBz0r\/ND\/IZWe9kypWrsaUyzYO0LyY0jKhAkTWrst3Bg8NsRNUNpk+EHdcQ3gsdmOCb9Yw7mE3w+vWGIDiNMWn7T5YcJBElt04YDdBSFz3iVJfhJ+sOUgkLgb4ZelbVSpIfmF+j8JfkxMgPHknmYkk+UVG6LkAkImbXA\/rEnwozaCdkFdv\/nENdac4gugjwnjceHnxiIy8B4fKLQdUjc1n9DxAz6ptiPgJxMOML2U72dojCshNhV+qBcGE4sHxybyioVfVq8hng\/4QH3F5Ma7AaSkRcTFhzKQThHEC\/hh94WUPpPgh8X1BwceG9rtkAN+BGyZai+kO1y0+eWR\/LK0TcJP9j\/emyb5cdwAF5x7mzZurqPKF5+J8XR3vUj4uXNLwg8nt8mEFDKMR74LbUqS\/GQYD50cnFNS8vPBD+3jnOX44f9SY3LbxiMZMBfd5BJJfGhMqEs3ww+DKT15MQk\/JbBcr6HrWZMSBO13O2ZvMki5TsD4JIii4Ef7ZBbJj7sQ0M4Q\/NB\/kABw6HjWOD8uVtreqN7jmTFqL9sG9Z42yrQwHsIPHwZIcIRYFvhJyTtN8qOU7YvPzAu\/NE822pCkjSTZM2E\/\/NpPx7cccFLyy6LSo1ya0ECwo09iz\/JQ+EWKjXlsfpT88sIP5QlNSC5Ji0\/CD4sPwGsXfnLzf0jyywoILJQk+LmqGofHTbAae4YHFjOgSWkWMLnpnq9Y72uZ8GOi1lj40cnlmh1i4AfAwFbrSkdJ+5bZp3I7I5JJIKlAO\/CTXtu0MB5IcOiXMuAXOsJVLneFX8nw4yTIKvkRPiznHuwt48XQBKom+DJDfc0CP+a8w4Rk0oGs8KMhHXVBqIcrCeDn0tubBD+5tUoOTafAD7Y+xL0xUSsl75DkVwT88C6MIT9GIfjJORkLPze0JmkLni8CgUHcWeBHyR1tgxMG8aC+kwFl7KMvk4xvmSv8KoIft4BhELHwmdE4aYIyHAMTVBrAuYhi4UdpMU3tDWU+SZP8UDfAtgj4oW1JKksd8PvN4YO2bVCVcbEv08CeBX55P1quzY+Sn+ulT5pb7ofVfrAiJD8GcNMOl7Z7KCn8inNZBoP7nDnSccY4TemES1s3HQu\/V199NQpHvuQDUQVz3pRX7eUEx8ZwbgGLHURM8iSvYRr88PVHOdj8qoIfuxXGZtQ7i+SHumLHS5PghwXOnS70hMdItZScs6Tolw4nSOyQHHmlqb0yJZgvRCkNfmgTpOyYVGRJu1ekKSYUxoP+gO0vqzPHl1gi7fiBjlR7czKpsmLtwi8p7ZObF06K79z6BW+ZG7ISAz8sIn5BQ5Kfu\/k\/Vu2l5KfwGzqfhPCDNI3\/ox9xTKdP7eWHkfDjRwvjRkm6DPhRq8iasUYGcGcN48E783iyKdXGnr2SJadfY9TeyiiW80V54SdPssLX2d0gHwM\/nHCWZjh31Se5iPLCT6Z9SlJ7pXRUJ\/x8X3t5ylceh0e7kl+V8GN8phuiFDKpEH6xSRuKgF9WT7bCLyewiixWNfwwMZHCPMvxjpiciNqHlEjbEyGWVfKjdJC2t7fJ8OMHwfX2oj0Ixg15e4uAHyQ+2f++jDVQKTE22FMrP1pp40abH9XerPCTcwshRHnglyWGEe3DRx6SXxb4yV1HXS35Mc6v22x+Ickvaf8rE29mgR8mGRZ31fBjfB+AU4fNL8nOA7C0Cz+0Df2KDwtsgCFPtlR788CPYIqFH0+m8wWnJ0l+En6uSSUt1CVvAHe78HP3VqfZ\/LpC7U3a2obdHQ8\/\/PCIbMxFSnm+Z7Uj+UHKyLpBXmYdxoKLPdvWhR8neprkl7b5P0byI\/zwDuwMqMLhwczCUG+TIvrrgh8PScoCP\/QzogD++colVuKXknfofJKknTlJ8JNzK0u6Ls4laBVZJT+fdBpjz0Q5ajNcl10NP98+XwmkTtnbizpD8ssDP6m25oGfNGpjMWWFH76gVA1nXzXbm8+PqmHV8Asdos25UhT8IPXRAREj+Un4ETSMj\/MdziQldgImK\/yQyEKeTJcWwJ0VfnTCsRy3NabFMEo7NGMY83iyJfxk+EuSrbzjJb+kPb2c1J20t7fT4McJ1lT4yZiwkLRfNPxipFqovRJ+UvJOs\/lRYpcqKT5a+HmM5JcHfgBslnRdSfBLA7svjCfGCcfAe0i1lPx6An7dKvkhzsn9OifZ\/FzJLyZkQkoQWSU\/JF6Q+1\/lFrCQ5IcJiqsKtbcq+CFuTSZ1ZVaTkErfLvxoVyMEY+Anj+WMSdSaNyEF7Zmu5BeSauHM4a4jhPFwboaccDhylAH0MCGwnE+qpXTa8ZIfFhLO3F20aJFNZMosL\/h5p9n8YFBmsLHv65wGP5SDeI+Yqlj4YWIAYlXCL1btpZcSYGFiVoxpliDnpsOPiVplxhqaHWIkv7rhlxac7sJPBtCHPNk++MWAvSfhh0Xh8wDLxKYhtafI3+d1eNQNP0qQIZtfXskPW5wgzeIKSUfM1YaF0gnwu3vS0ei2uVmq88KPql2Vkh8l9xh7JmIJuXsl5rB5GcaTx56JDytjJmOS0HaF5FckuIp4Vjvww6KXXzBpu0iS\/Kg+ccApEaTtFKD6ISU\/eEUnrp4edHhkgZ8MIG4y\/JhHDnVkAHBsnB8koCrhR4m9CvjhXTIBQlb4cS5K9TUk+eF9Cr\/hJDrn7NmzZ4uAU9nPKBJ+0nYRgh9VC6aq4tc5yWvoqr1Z4UfbXZrNT+G3y+aWkwk\/25X88sAPtjfkekRcpwukNG+vhF+efcsKv2JoUzr8XNV54sSJI2IEaV9kk3xJE4qCH+Alxfey4EcJIhZ+8Cgm5bxzHR6dCD8cechMOnKHh9v\/0i4pJT+ABieOhRwerqNEmh3SpCMJPwAJH8iQt7du+MV6srl7xSf5ubGInFsy0xAzwMSovYz\/jMFTV+\/tJfhmzpzZOhfEdZi4ByLR04zO46FJVhp6+WWzcOFC8\/jjjxs8L\/aC6iXV3rzww0Tv\/\/mTiWfbcpsUFxHrR\/ilGZf5uzrgB1Ud2UVis7pkcXhItbdO+PGM5jLhR20izSNKsOfdugeww6TiOmVC3t60MJ40+DEAW+EXSxtxHyS6TZs2GZ4Gh1+5YTSA4f79+4eBDkDcsGGD\/TN27NihMI6K4ece76jwO2DHoRPgJxO10pMNaa5o+DGvHtRemd4qlKKf0ilzFcK0YtdGRBJavIuRB2m7h\/A8fpAVfn54la72+l6LHSK4Vq1aZeS\/eS8lxpUrV5q+vr5C4JfktUo721aeudsJ8Auphj5vbxWSHw++qUrtbQd+tLnGqL1Fwc\/dt+zak117ZhXwk1suVfLLIfn5ilDymzVrlrnhhhvMwMCAwb9lLKFvh0m7kh9j4dwDdLoZfu4i6mb4SVthEfBDyAa89GnmCgm\/LICQdklf0oYQ\/ORh80lbJ32Sny+GMUntlfDDs\/B\/pLEPSbVq80sBJVThNWvWWKfHpEmTMsOvv7+\/JQ3CMRK6PvXYj1rZazlBr7n+WvOLp7ZaNei+5TcN8xr+2wOfb6kfaZIf0o7LC4fPyFAX\/u71+14xl917ZWsRPbP4o61iMC6feGBB63cyHAT1wu9QRzg87v2zS83o0aNtWZbD72SoCyS\/vxz3pm0bFphbR0zc5599zj7zyb5TZvrufltnafP7vXu+6e1S9CMuWf9Q38Pmt+HFo60jDyn5yba5\/Y86sv6uw0O2DUc9Sm+vLLfrg\/vMH7\/2LVs9V+2NGTf0ybh7dgXHTcLvnCkzzOBb7xg5t+AgWHP1hFY3yTq6u1dkxhq3jnJOyuzNrtrrjg3nJNVe1PHs4YPmtflb7NgzQYGso5yT6Acm8cW9uI\/rJqltmO9yXNLmCBwex48fN5MnT27N7dCcKvL3laq9rkQnpcBYyY+NX7BggXWAhK7PPXnE\/NFr37JxVZygs66abX797A4LgTtvudZ8YtKY1mN++cgKO0Fcr+HWK75iHR6Y4Cj3xu2XDXv1f66c3YLfv5672T4Dk+fc27aYK75+nZ1o\/zj9FvPtmya2yv3m8AHzq0eGJiF+B6cAyuH\/qBd+h3d9bMbHzN1\/Ot6MGjXKlmU5\/O4Tk0abFQe+ZH+OOgIQaNtvb3ppRNeg3OeeHDrv4\/CcU\/b5+Pcrn95klh9Ya+9\/\/21bvV2KfsQl6x\/q+28Mvm0efPlt84WZ59vnu2175TObDMZCXmwb+v8vL3jD2zb87ts3TRg2bmgz\/uB3En6PzNtr3x0zbsxZFztuEn5oA8rJuYWP1h1957eaJ+tI+KHMP3z0Fju3OO\/cuSXnpDwnBP135VMr7Xsx9u7YcE668GM5ZgySdZRzCxWX8PtC3\/mtdZPUtgN3fn\/YuKTNkTNnzhj8mTZtWnfDz7dfOA\/8KPnFSH3oeJ\/kJ132IcmPGZzxtfyj177Z8vamSRA\/\/uBWKz1mkSCQ7Vh6RKV0dOUnrjRf\/fTveyU\/CT9p87t41+veeTdmYAiKP7vFWMkSdfzneVvMinfhV6Tkt\/6Fo\/aMitVXTbDPR58AGJDa8G5IV246LEoeLvxce6Yr+UEiYRiMVHsfW\/h9+246PJLGDc+X3vaskh\/hJ+cWpECoirxkHdEnq87sa52HzLH3SeyQ\/Fh\/CT9KcNw9FJL8GH4lJb8PfGbFsDpKrcKFn1R7k9qWVfI7deqUGT9+fPfCLyl8BZ1btsMDp47x0BbXLuPbRyltR1I1CR3sLUNdJPxibUcu\/GQsXGycnwTEtCeGJDX3Oueu5+yPfrF4tE2ZVbXDA\/BDgDLe7dsKxTgzd4dHjDPHhZ9sG+DhO5MY4+YLNZLjlmQX80l+MbFwaBucQIAJxgOmCwk\/n83PB7+0rZMYY9fb68ae+vLyJR0\/4Nr8kuIz1eYnVhzj+ObMmdOK9ZMLMinUBXbB9evXm6lTp9rb8zo8AD\/3uD5O0CLhB48cJ6jCb2iEGecnvb1Nhh8BBGjGws89nyQtgF5+WKuEnxvAnadtCr80Bd7zO4JvxowZw+L45K1uxuiig5xd+IXSPuWV\/Ag\/SBdJi8jNC8evMyZn3ZIf95cmZWXmub154vyqgh+y76AvmbQhq+QnzQ6f\/auvtLy9aZJf2uFMabtXuD0vq+TnerJ50lySVCudcNR8FH5DBCrV4QGVdvv27V5kyuwwkP7Wrh0yuOMqcntbUfALBct2Ovykbco3YHXAT2aeiVF7OxF+vsPg09ReH\/wANWRpdsfQ3XXUDvz4UWGoSxLYsU5Cc4nzq6u3t2UUFFNvL0rtlZIfBuqTU88bETJB25G0+cXAD9uBIN25kh8SQSJjL6QEV3KS2Y6TtoBVYfMLTdimwS9rDGOMza8Oya8o+H34vu+OWD8Kv3QClSr5NQV+iKFiBmcJP59TIGljfTvwQ9xTEjzahR9sTFB9cOVxeEAagDMnFJtVJPwGt66wsYo+FVs6PNIkvzT4QaVkgDJUVu7tDTk82oWfm64rRu1Fv1495XwbisWPbh7JLwl+qJObLi2P2itttfi3Sn5F0i3wrHYkP8IPr5AJP7sRfgiGRkgJFr3vcr29gJ9v4bhli4Zf0nBL+BFiLtjhqArBj9I2Pg5M2uCDvPT2div83IxBCr8KbH5FslHhd\/mwOD956LeU\/DZ88fOp3Z4XfjgICmc6hNRj+fIkby8kv7Tr9iX9VlpJgp\/PrgSpCe9jOUrb+BhgCxiuNOlImisgLeVxeOSR\/NCvuADrPJJfWjaedsN4pDNHJb8iaZbxWQq\/YuEHyQmLI1byyzhc9va88IOUiSShSfALxTD6Ep1mgR+0A0jOUnWW0E+LhYvNVejWUZpbsqi9Cr88M1MlP9MktZfqGXdBMBCYe3sfuLFz4Ye2QPpi20KSnw9+UNVQHpBIgh\/L+eCRBX4AH\/5QcooNdYmV\/HyBwE2FH00BKvnlh2zbJbtd8usV+MHpEFLN88IvaZIhDKlJ8Ev76PpUehlAz1CXmJ05Rai9Cr+20dX+AzoRfrQdwf4EFdCqgmKvJ\/5Pb28e+HGbVB6bX5VqL96V5VzXXoYf9mQzew9XTdPgl7YzSuP82mfdiCc0HX5QWxjL98S45616JuGX1CUSfgx3cFWMpDi\/IuBHFa+EIWvZ\/IqAH6QlqoZJam8Zkp+vf\/La\/FAuyRPPtvngh3bRUXWi76DtB27Bi3XmyKQNsd5emCdkIlXuB1f4lbFaUp7ZDvwQR0XpqKxQF3lehcJvaCDZJ50GPyZcwIcpL\/yynF+LvoqFn0xIgXeUBT+aJyT8MI6o57jBGSNOz2P9VfIrAYxFwY\/xb0mG81CQc5LtqC74uR7RkD2NEgTV3jIlP4RxXPPwQRuTl0XtZV\/KtlUp+XUa\/BAs7ovrbCeG0YUfzrHm7iSYJRBC5CaTxdpR+DUEflx82FhPya8d+KXFuBUJPwTjwgZIdQfJQH3eXqofCMnAFeNMUPgNn5wSEJTYJfzS0m4xIYXvQPYqJb+88GNIj4QYVfo0+PmWt0p+JUCPj8wj+ck4s06Cn1w4gJUPfphsF\/\/7imGxcPLrnDQUnQA\/frSqlvx6DX5pYCf8CLXQ3FL4NRh+dCbkkfwYVpAm+XHBIjFluza\/EPzQzT5AhCaoNJw3We1tCvxC+487WfJL2svtSn603b5w+O3UY0sVfg2GH\/LVYUtYFvhRXVb45RvYvDa\/psAvqdUx3t4mq70P7diSOKAKv3xzvdRSedReafPLAj\/mhWsy\/CjFSdUwJoW4qr3DpykcMUxF5qq9nQA\/JO3A5cuQk+TwUPgNjWxXp7RqF34ES17JTzoukhaSL84vRu114RfrtZXww1c+aeEU8SXrBMmvSPjBgQBHVdL5JGl9mjXUJTaGlOeTcKcG6vjZe+5Xya\/b4UdAQIKj5BfaI4pJSMmP8IvJeeez+cUE4xJ+2BDPzB5Vwa8IwKU9w4Uf7o1Jn1Wl2lsE\/AgWjBuuMuFHW21W+CFMJSlu0R1D3CcjB+A4jLX5xdid+T7N5By5AvOovUXCL7RoXfhBqgqVQf2qhl\/S5v\/IYch0W9Hwyyuppu3t7UT4oR8+8Ng7IwKN3cGRbcty9squBdeZH16xuLUHm2m30hLeqsMj09LIdnMnwS\/LRPPBT0qMvlAX35c6Vu3tZPhlmzHv3Z0VfiGJ3XV4+CS\/0DPctsSqvcxYg3nhpsUqCn55Etfm2Xqokl\/kjO4F+MF47e79VPgNBXBnhYmcVp0Av5OvPGfefuV5M2nRgPcAb2mrlfbeNGksr+Sn8IuEUlW3NR1+nJCI88sr+fkmcifDD30CexHiI9MA5M6hJJtf3rmWllDAp\/aGQFuG5BeSghR+eUc\/uVxXe3urtPnlHRqp9lYFP9Q1C6Dzti1W+iobfmn1zyMdKfxG9qiqvUWskoRn5JX8+DhO2Lze3hjnRZ7mA35pITFlSH7IcpPlLI487fLZn\/CzmH4sWvIrGn6UaO979g0DSb8Im1+nS35p0nVS\/4faXMS8S3tGpZLf4OCg+c53vmPWrVtnxowZ06oXfr5o0aLW\/4s8tLwT4Je2G0Dh177NT+G3OJOkn8fmlwdUPQO\/Q4cOmaVLl5oZM2YMgx9\/vn79etPX12dOnz5tBgYGbF9KSNYl+aXlS8sz4G4ZSH5Vwi8mbKGIdvWa5If2wuOOnRZQAZOODs0rBTXd5pdnzvQE\/Pbs2WPWrl1r+2fu3LnDoIbf7d+\/fwQQN2zYYPBn7NixtlyV8IMIz4wp3Qa\/PJO0iDJ1OjyqkPxiVXqF33s90PXwI\/h27txpXnzxRXP06NFhoNu4caPtjVWrVrV65eTJk2bZsmVm5cqVVhpU+A3P5+eTHmPj\/IoAWZ5nZIEfFsU9f\/sv5sY\/usQeH4kr5IHNUyeUyePwQDmZvxE2v6bCD1szf3jFksptvDHj0fXwk50A0En4UcWdNWuWmTdv3gj4zZ8\/v\/XzqiQ\/Gm59qbpjBjTrPVh8VJd8ZYu2+WWtX1H3Z4XfsWPHzPjx482PF1xWKvww3sg2jQO6s3jA64CfDKOKDXJOSnRa1Li28xyF38CAaRL8OJgxk6udgY8tq\/ArF36U4uzfzsl6aWPkHsheheSHPeqsY8z8zLsdMHZutnufwi8j\/Pr7+1uqMLzCsRcmwokHFphx9+yyuyh+8dRWg5Oykq4xAy+ZZxZfPuycgth3FXkf6oFMzjgzwT3SkO+BdATp8QOfWVHkqwt91r898Hn7vN+755vB52JRHD9+3Fx44YXm\/9760Ua2DfDb8OJRAyAteWlldNuSGi\/b7BtnzAO8a83VE+wjmjI\/g4OZcgPbPHny5MS53c7zQ2UrDXUpQu1lgxYsWGAWLlwYal\/r9785fMD86pF+c+5tW8zZwwfNr5\/dYX5701BKJ9916UOvm2\/fNMF8YtJ7ITnRLyvwRtRj8eXvM3f\/6XgzatQo75P\/c+Vs877rl9g\/Tb1++cgQmN9\/29ZgFc+cOWNOnDhhxo0bZ859YZc5Z8oM81tTrgiWq\/KGbwy+bR58+W37YVp+YMiZF9O2pDrKNvvGGfMA77qj73z7iKbMz3b6HG3Gn2nTpvUe\/NBxWR0elPyySH14j5T8AD97BkaHSH63\/dffMV\/99O+nSn6Q+rKGV7QzcbOWzSv5JUm7Wd9f9P3rXzhqEOS8+qoJZsW78IuRapPqESP54V1Uez\/12I\/M6qsn1K6ZtNOvaPOpU6esbbeOca5V8kPHJYW6rFmzxiD2b+rUqbZ\/q3J4cDCh1nxy6nm1T64Ymx9AjpyDvmy+7UzOIsvmdXjUsShi2i1z3GVpWxr86OTxtRnzQNr8YurY9Ht62uaHwWGQ8\/Lly61nt+4g56ZNmBj4Na3OvvpkAUTdiyKmPxV+Mb2Ufk\/d41y75Efpj0HQ+H+d29vaH9Jin9At8Muy97PuRREzggq\/mF5S+LXfSwWrvagQUo2XFThbSIPffUi3wC9Lnyj8Ro\/oLlV7s8yguHsrlfziquS\/q0ibn8KvnZEov2wnwE\/ukc6i0qvN770eqHucexJ+TXYMuItDJb+RUlD5+M32BoVftv7i3Qq\/yH4rUvJT+EV2ek231b0osjZb4Ze1x4bur3ucVfLLN26VlVLJTyU\/TDa1+RW\/5HoGfui6UBKB4ru3\/Scq\/BR+Cr\/215HvCT0Fv3K6sNynKvyaD78iZkBIBVTJr4heHv4MhV\/xfVroEzHpscf4v1+ZnNig0Bc24GEhEDSgioVXIdRmhV\/hXW4UfsX3aaFPVPip5Kdqb6FLqvUwhV85\/VrYUxV+Cj+FX2HLadiDFH7l9GthT8VJWss+PkbV3sJ6tJkPUrW3+o+cwq+Za6FVq9CiaHj1c1VP26zb23JNnIyFFH4ZO6zq2xUE1UsEVY8x3hcaZ2yn++SUoUSm3XKF2lx2OxV+Zfdwm8+ve4K0Wf1cxbXNCvxcEydjIYVfxg6r+nYFgYKg6jlX1fvqntsKv6pGOud76p4gOavdVjFtswK\/rQkUWVjhF9lRdd2mIFAQ1DX3yn5v3XNb4Vf2CLf5\/LonSJvVz1Vc26zAzzVxMhZS+GXssKpvVxAoCKqec1W9r+65rfCraqRzvqfuCZKz2m0V0zYr8NuaQJGFFX6RHVXXbQoCBUFdc6\/s99Y9txV+ZY9wm8+ve4K0Wf1cxbXNCvxcEydjIYVfxg6r+nYFgYKg6jlX1fvqntsKv6pGOud76p4gOavdVjFtswK\/rQkUWbgR8BscHDSLFi1qVbmMQ8sj+6NxtykIFASNm5QFVajuuV07\/A4dOmSWLl1q1q9fb\/r6+szp06fNwMCA7d5169aZMWPG2H+3e3pbQeNV+WPqniCVNzhik38ddSr7nTrO1X\/kaoffnj17zP79+4eBDkDcsGGD\/TN27FiF37FjZvz48Wb06OonSNmL3vd8BYGOcxXzrnb4bdy40bZz1apVrfaePHnSLFu2zKxcudJKgyr5KfyqWAx1vkOBXz3wa4UfVdxZs2aZefPmjYDf\/PnzWz9XtVclvzrhVPa7FX4KPzvHKPkp\/IZsnd\/73vesZNwraq+2uXoQlA133\/PrHueOk\/z6+\/tbqnAdA1b1O+EJ37Jli+mldmubh0w93X5xnB9\/\/HEzc+bMypvbMfBDzyxcuNBKQnppD2gPdEcPAHq3335778EPwxfr8KDTozuGXFuhPaA9gB6oQ+Jjz9cq+aESSaEua9assbF\/U6dO1VmiPaA9oD1QeA\/UDj8GOS9fvtx6dpOCnAtvuT5Qe0B7oKd7oHb4Ufpbu3ZtayB829t6epS08doD2gOF9\/FxK5EAAAbrSURBVEAj4Fd4q\/SB2gPaA9oDgR5oFPzcBAdz584dtu0NbYlJggAnyvbt21tNv\/XWW4ftIGnqrEC9L7nkkmEB393eZjkWNHns3bu39eP7779\/RH80dfzS6uXbson7Y9ocM+eb0ie+9uzcuXNYeFpT2twY+HGA2VE+219MEgQ4UHbv3m22bdtm9wW7NsWmTBK3Hqg3VH93sXdzm90+APyPHj3a+uC5c6KpYxeqF4P2cR\/nJcuE2hwz\/qH3V\/V735r1jWFT2twY+PlCXtyBDyVBQAYYZIRxt8uh3JtvvtlI6c\/9Crrw68Y2+xajbz837kuShqta0O2+R0ptri07ps2h8Wfij3brWUR5rFdflIaEHea7u2\/fHeeq2twY+MUsiFBM4Ec+8hFvx2ICbtq0acRXt4gBb+cZBN9PfvITG9aDiSO39HFS4O+kxA+d1uak\/kpaOL6F0E6fV1mW4MMHDZfUSPD\/mDZjd0\/a+DPxR5XtyvouOYZvvfWWF5Dynqra3Gj4yckxadIkr1Qn9wHPmDHD27GYhADLo48+2ti4Qd9+5pjED53cZrmIkj5Qrhkj68Jryv2+doTa\/OCDD5rNmzeP0GR8c6Up7fTVQwotTWpzo+EnxWV0qk+lVfh1LvAVfn6NhKDsBvi5piuFX8RnypXWul0KUskvHQSuoyBiCjXqll6U\/LhmJ0yY0DLbKPwC09L9WuB2hZ8\/36GqvY1iXGJlehF+rlcXndPT8KOE8+qrr7YmiowDSgtN6VSHR6jN6IgkO06ntjkrkmKM\/zzPJeuzm3C\/D34xba7K+F90H2HeIgOTK7E3qc2Nsvkx1s0NiuTAhJIgJDlFOsFjmAS\/bm6zXHBpYR+ut7PohVrF83zwi2lzaPyblviDbfrwhz88YoOC\/MjLIyrwc\/mRr6rNjYFfUpCvnJgxSRDwnIcffrjl2e2UIOck+HVzm13ouNJCtwQ5o51JXutQm2PGvwp4x7wjLZhblm9KmxsBP59aKDtLBv4Skvy9Gzjq2zrTCdvb0sIXurXN7oLyzYNu2d6WBL+YNofGPwZMVdzj1lO+U67TprS5EfCrYmD0HdoD2gPaA7IHFH46H7QHtAd6sgcUfj057Npo7QHtAYWfzgHtAe2BnuwBhV9PDrs2WntAe0Dhp3NAe0B7oCd7QOHXk8OujdYe0B5Q+HXJHJDxjXnjGqtOlYQ6I48hDqPnToUiUlgxMPjIkSMmabdQUcPu27xf1LP1OeX2gMKv3P6t7Olc8FOmTDGHDx9udO5Cdoovz2IR8OMunwsuuMBcfPHF3m1WRQ2Mwq+onqz+OQq\/6vu8lDcSGtgzicStPAe5lJcV9NAy4CdhhMOg5FbHgqo97DEKvzJ6tZpnKvyq6edS30J1debMmTZvmi+jRtoWwokTJ1pJEedB4HwFptInUBcvXmy+8IUvtNoAVRJS1dKlSw1US1xyG1pakgbC6ODBg\/bAJl5U1ZPeGbvNTe4HZop\/9gvfJeuHs1140p\/vvGi331BPXDxoCf9Gkl2Zsw4\/c7d65TVFlDpxevzhCr8umABuAoDYhABUlefMmWOh6UKLC1geIcpjQSUo3PfFwA82viTJD1CU74xtD4bSBX\/oQ0CboC8bids\/fD5gyfr54Id37tu3r2V6cD9OXTDluqIJCr8OH0bfcYExi8232H3wc9VGH4jccu3Cz31nrCPGl8Enrb6uROjaG9PAyZRNLvySsghlAXiHT8mOqb7Cr2OGyl\/RpMXmSh9u6bSF7aq9MiFlFfBzTzmLgTlVzSRwyvxyaXDmu3kMqqvOuh8bF35uSjWfqj1v3ntZuTt8+nV09RV+HT18Q2oebVa+pvhsZbELNCn1+qJFi4aFkBQt+eWBXygtGu2aULdjJFPaP13pkKpvks0vLa2Taxvt8KnX8dVX+HXwEKZJRPJMYJ\/k5oOiT+11QdRUyS9NrQzZNjkF5EeBWcHzSH5un3XwFOvqqiv8Onh4kyQ4NskFgs+AL5tfNvxcVTw21CWk9vrsnmlqPn4nvdo++EFCbMfmh+BteaC471CuDp56XVF1hV+HDmPMgpdODYRjrFu3zrYWf\/sOAyoKflQN5QE2BLFUP31ASDvrwqeC4l0xRxXw\/ZB4r7vuuij4pXl7Gbrii\/NzIR8zVh06DTu62gq\/Dh2+WO8hpcM77rjDrF69OrG1CPlgXFy7Dg+8xD1OAKEhN954o\/nyl7\/cCgGR9zB05OmnnzZZbX5JJ4XJxkoA3XXXXTZuke1Mkvzw86Q4P\/wO4UFJQc4a59f8haXwa\/4YaQ0b1ANJ50c3qIpalcgeUPhFdpTe1ns94JMoQ3bW3uulzm2xwq9zx05rXnIP+E4C9G2BK7ka+viSeuD\/A\/nNzMRzOK4BAAAAAElFTkSuQmCC","height":174,"width":232}}
%---
%[output:0b98aa48]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQmUVdWZ779bVUzBAkVBq5iKKcwoKkVjt4bEfuklOARZQK\/4YGmE1JMuNN0BoQPar1\/ABgNvQWCpPDAOvKSboju0MZi0tsHGKE+MD6XbKIodIgIKImEqpqp7e3373H1r31NnHu45557\/Xesuirr77OG\/9\/3V9+1vD5lcLpcjvKAAFIACKVMgA\/ilrMfRXCgABYQCgB8GAhSAAqlUAPBLZbej0VAACgB+GANQAAqkUgHAL5XdjkZDASgA+GEMQAEokEoFAL9UdjsaDQWgAOCHMQAFoEAqFQD8UtntaDQUgAKAH8YAFIACqVQA8Etlt6PRUAAKAH4YA1AACqRSAcAvld2ORkMBKAD4YQxAASiQSgUAv1R2OxoNBaAA4IcxAAWgQCoVAPxS2e1oNBSAAoAfxgAUgAKpVADwS2W3o9FQAAoAfhgDUAAKpFIBwC+V3Y5GQwEoAPhhDEABKJBKBQC\/VHY7Gg0FoADghzEABaBAKhUA\/FLZ7Wg0FIACgB\/GABSAAqlUAPBLZbej0VAACgB+GANQAAqkUgHAL5XdjkZDASgA+GEMQAEokEoFAL9UdjsaDQWgAOCHMQAFoEAqFQD8UtntaDQUgAKAH8YAFIACqVQA8Etlt6PRUAAKAH4YA1AACqRSAcAvld2ORkMBKAD4YQxAASiQSgUAv1R2OxoNBaAA4IcxAAWgQCoVAPxS2e1oNBSAAoAfxgAUgAKpVADwS2W3o9FQAAoAfhgDUAAKpFKBksPv7NmztHjxYqqpqaEFCxYURH\/jjTdo1qxZhf+PGTOG1q9fTz169Ehlx6DRUAAKhKtAyeG3ZcsWWrJkCc2ePbsAv3379tGcOXNo+fLlNH78eJKA5KYvW7aMunTpEq4KyB0KQIHUKVBS+EnIHTp0qAh+DMSdO3cWgY7TrlixQrxh\/aVuXKLBUCB0BUoGP2nNXX311fTOO+8Uub0\/+MEPRENVN\/iLL76ghoYGmj9\/vrAG8YICUAAKBKlAyeDH1l1TUxOtWbOGVq5cWYCfhOKECRNo2rRphbZJ+E2fPr3o90E2HnlBASiQXgVKAj\/ViuNAhhrwAPzSO\/jQcigQpQIlgZ\/q1uqjvYBflN2PsqFAehUIHX68hIXdXLlsBfBL72BDy6FAnBQIHX5s9W3cuNGwzXIt35NPPuk44DFz5sxCXvPmzaP6+vo46Ym6QAEokBAFQoefXgejRc5mS10WLVok1v4NHjy4kM2uXbto7dq1hf9v2rQpIVKjmlAACsRJgVjAT67\/mzt3rojsOlnkzBBkK5DhB+svTkMKdYECyVAgFvBjqeTODymb3fY2wC8ZAwy1hAJxVaDk8AtKCMAvKCWRDxRIpwKAXzr7Ha2GAqlXAPBL\/RCAAFAgnQoAfunsd7QaCqReAcAv9UMAAkCBdCoA+KWz39FqKJB6BQC\/1A8BCAAF0qkA4JfOfkeroUDqFQD8Uj8EIAAUSKcCgF86+x2t9qXAF0R0XMmB75i5jIhw14wvWUv8MOBXYsFRXJIVOEtEhwoN2LVrD9XXj1EaVAsAJqh7Ab8EdRaqGqUCbeBbu\/bHtG7djwuVaWy8i+bNuyv\/\/0FRVhJlu1AA8HMhFpKmWQG2+M6SHnxSkTYAsvuLu6aTMFIAvyT0EuoYAwU0+A0dOsm0Lnv3vgDrLwY95bQKlvCTFw\/t2bOnkN\/kyZNjcZE4TnVx2sVIF4wCHxHP8c2cucg0u02blufnAOH6BqN5uLmYwo\/v3pg1axYtXbq06OpIPnfvscceow0bNhSdsBxuNdvnDviVWvG0l\/eREMDe8uOILwc+8Iq7AobwMzpqXm0I38tx+PDhSC3AcoPfN7\/5zbiPlVTXb+rUr9LUqV9zMOcH+CVloBjCz+7CcHkBubyRLYrGliP8nnnmJ1FIiTIdKtChg2b9WUd7sdzFoZyRJ4PlF3kXaBVgyw\/wi0lnmFQjkzlLVVVY5xfvXnJeO9M5P\/2lQjJLzPk5F9dNSsDPjVrRpWUAfvDBazRixAgR\/dVeWN4SXY94L7kAP6PIrlW2dhcMea+Ssyfh9jrTCamCV+Dv\/u77NPWW\/jSwx3LqVPs0VXTB3dHBqxx+jljnF77GjkqA5edIplgkYvh94+ZPachV2wG\/WPSIt0oAft50C\/wpwC9wSUPLkOF3501HaVj3V6li+Hqq6AzLLzSxQ8wY8AtRXDdZA35u1Io2rbD8Jn1Co7JvUWbYesp0uy7aCqF0Twpgzs+TbME\/BPgFr2lYOTL8pvzZQRqZ+Q3gF5bIJcgX0d4SiOykCMDPiUrxSCMsv2tP0KherwB+8egST7XAOj9PsgX\/EOAXvKZh5VgEv97fpkzvb4dVFPINUQHs8AhRXDdZA35u1Io2LcPvjhuO0cjK31DFwHuoov+90VYIpXtSAJafJ9mCfygJ8Fuz5n\/Ttm3P0yOPPErXXz+unQhbt\/4TPf74OtPPrVTb+nuiGb9qSzGtjujHX23\/xF3bibbsb\/v95q8RTenvvD9OXiC6\/UWi148SPXwN0ZKxzp+VKUW098bPaWjL21Qx8G6qHHSP+0zwROQK4FSXyLtAq0AS4Hf8+HH67ncfEPVdtWoNXXYZ72zQXh99tI8WLvwu3XjjV+iBB\/7Klap6oMmHb+hJ9LOvE3XrSKRCS5+5GwCqZfmB3203HKFRuT2An6uejldinOcXk\/5IAvxYqt\/85k363vcepMmTbytAzgqKdvLuOko0+UWiExeIJMSkFdi9I9G2rxPV9yQy+p0EWd+uRK9OJqrtal7aoTNEN24jOnCmLQ3gZ9c75f15u6Uu06dPLzq\/L67Nx\/a26HpGur\/33ddIU6ZMJf7\/q6\/+G61YsYoGDRrsqmJLdxP9r7eJVCvPKAMJOjWdBCenl5A0elYFrPq5X\/j1o53UdUADdR40x1WbkTgeCrSz\/PjggiVLloja6Q8yjUeVtVoAftH1hrT0Tp48Qddeez298sqvSILQTa1UV5bn+Pgl5\/PUOT99OjkXqFpzVq6vhF+3DkT\/MJHoz1\/RLEA\/8Lv92uPUq\/NLVD3wPqoeONdNs5E2JgpYur1xBiHgF+0IknN8J0+epHHj6umhh\/6WOnfu7KpSVvN4nJEEoBP4WYGM4ffUXqLH\/4RIBaYf+N157Snq0fkXgJ+rHo9XYsfb2+IGQsAv2oEkrb9PPjngyerj2uvhJ6036QrLOb9h3dsitKpF6AVkXp7RK83R3gnjPqWRHXYAftEOQ1+lO4afWgpOcvalueHDSQl4yMrLeT\/+f7du3TzN96nwU+fy9IC6f2T84Hf1kJ\/RmF7N1H3gXOox4C+CHxDIMXQFLOHHlxjt2LGDFixYEHpF3BYAy8+tYsGllxFfnucbM+ZqscSlW7fu7Za\/2JUYJPycLncJyvL7o8G\/pJ61x+iqukaqqWu0ayo+j6ECRfDji4nq6+tp\/\/79xFFfvrIS8CtNryXF8pNzfUOHDivM88nFzeryF6eqGS1XMQKUn2ivWpeg4Dd2yEvUp+YI4Oe0o2OYzhB+zz33nKjqHXfcIaKq9957LzU0NAgY6l9RnegMy6\/0o+ncuXP0\/e\/\/De3d+36Rmyt\/\/+abu1zv7lB3dsgAhH7Oz806P6P1gGHAb8yQX1KfmmNUW9co3nglT4FMc3NzbvHixVRTUyNqf9NNN9H48ePpiSeeoC5dutCRI0cE\/HjZCy+Befnll4Vl2NjYSPzcjBkzRPpSvwC\/UitOYj2f2fY2PwudzXZ4mC130bdcdXlLCb+ufc\/QgH730eB+DaXvDJToWwFh+cl7erdt29Yuw9mzZ\/uGn\/5+kNra2naXnstL0mUF7CxKwM9337vKwGhnhz4DJ2nMCpXWnvzcbBmK3d7eUsHvihGvUF3tH6h793H0xyOfcKUlEsdDgXZur7T8uHoy4OHH8pPg47lEGTjR3wAnb4pbvny5sCIljLkOy5YtExao\/gX4xWMApbEWvNSl\/7CXqGefM\/Sl7hNo4sh1aZQh8W0OHX4M0JUrV5J6wbmEG7vaDESG4c6dO4tAx0BcsWKFePfo0QPwS\/xQK58GMPz6DPtXyvSvpL7Vo+hPR\/ywfBqXopa0g9\/GjRuLmh+E22ukJ0eW+cXwU3+WaaXFOH\/+fMM5RVh+KRqlMWsqw+\/y4f9GVf0rqbZ6NN02fGXMaojqOFFAwE+6nYcOtd1GLx8OA37S8pswYQLdeuutInDCP0+bNq1QZwk\/s4MWAD8n3Ys0YSjA8Ds15n2qq\/2C+lWPBPzCELkEeRaivQwfjuLynB8HG5qamqiurq6w1CXIaC+7wosWLRJBj969ewN+CTnPrwTjMRFFMPyqRr1JmX6dBPzuGvo\/E1FvVLJYAdN1frt376a77767sNQlqHV+eotOtQJh+f0E4zMBCjD8zoz+LXXrm6X+1SPovw\/9mwTUGlXUK2AIv+bmZrrlllsK0d6gtrfpAx1cGcBP65Kk7PDAV4hIwi\/brwsNu2Qw3Tv0e5AlgQq43tsr1+PxXKAbKFotX3ET8Jg5c2ZBZp7327Rpk9iSl\/QX4JecHmT4fTZmP1HfjlRJOfrhdVjnl5zea6tpAX4STk52bBit3bNqvAyoTJo0yRCYZktdeF6Q1\/4NHtx2OjADb+3ataI4wC+JQy75dWb4HRhzkLr0zdF56kjrr9PGI17JUqBoh8eBAwfoW9\/6VsHlnTVrlm1r7E57luAbO3as6YJlmWbu3Lki4otFzrayI0GECjD8PhxzlFr6dqWOdIGeuW5VhLVB0V4VEPBjV3bz5s1FcDL6nZdC2KXVrx2U+UyePLlQpnpYKn+O7W1e1A7vGSdb15ykMaqh0YnOZkdU2W1vM1PA6AIjs+sx7VRk+H0w5nOivp3pIlXR5uuW2T2Cz2OogOXVlXogxqn+WOdX+t4I42ADp1dSOk1npIrZBUac1u7iJKP8GH6vj26lTN8OVEWt9Pz1D5e+M1CibwU8neTsu9QAMgD8AhDRZRZWJ7d4vbBcPdKKrb2ba9pObVbB5OfqSqMzA\/Xlurn4nOG3Z8wpOtXnMsrlMvTiuEUulUTyOCgA+MWhFxK01MXItQ3iwnIVdPKEF\/U+Xq+Hmarurt\/7P+RQYfj9amQHEe3NZHK0fdx3YzKKUA03Cgj46Y+ccpOBXcDDTV5u0sLyc6NWsGlV93fUqNHigNPDhw+7PsbeqFZGsHJye5vTY+xlmaor7PYWN4bfa6OydKr3FVSZaaEfDptKY7v1CVZk5Ba6AgX4LVy4kPitLiuxK93u5BW75\/18Dvj5Uc\/fs6r7O2HCDbRly2bXJzgb1UB1RVVL0An83AJMWpLyhjg+Ldrpi+H30ogv0fk+1VSRaaXHht9B13ardfo40sVEAV\/wY4tR7vk1OnYqzDYCfmGqa5+3dH85pZe7O4xK0B9oKq25oOGnRoy9RHw1+HWh07U9qSLTQutHTKbru19lLxpSxEoBX\/CD5RdcXyZxhwe7v++883Yg7q6qZJj39qrg8xLp5Xoy\/H4+9HI616cbVVCWNo76OuAX3FehZDkZws9qbR7XTG5tA\/yC6yfAr01LL\/f2OpnzCwJ8En4vDLuMztRqh+z+aPTNNK77lcENBuRUEgVM4cfHWfFuCz3gjPbhlqSmukLg9kahenGZfi0\/N+6s12ivrHFQ4JPw++cv96IzHPCgVvof\/UbT\/f2HR98hqIErBQA\/V3KFlzitlp9R4CHoqyvNAilee5Pd3q1fvpJO1vYiohzN6zeCvlM31Gt2eC4iBQC\/iITXF5tW+BltO5PaBHF15bDubYumzbrabaSY4feTwX2oufcVVEE5eqD\/UPpLwC8m3yTn1QD8nGsVasq0wo9FDXJvr34nCOc\/+UWiExfMu88L\/H40cAD9ofZKylCO\/nrAQHpwQNvJQ6EOFGQemAKAX2BS+ssoifDz1+LkPs2W3\/8ZMISO1FxFnSpa6HsDBtBfDxiU3AaltOaG29vsor2sVVQ7O2Q\/IeCR0hEbg2Yz\/NYPHExHr+ot5vweGtifHhpYF4OaoQpuFCjAL6gjrNwU7ict4OdHPTzrRwGG3+P9RtNnHPDIZOkrl11K268f5SdLPBuBAoXz\/Pjg0tWrV9PEiROpS5cuEVTFXZGAnzu9kDo4BRh+T\/QbTZ9e1ZtymVaa2KMbbR83IrgCkFNJFMgcO3Ys52Vfb0lqZ1EI4Bd1D6S3fOH29r2ajl7Zl1rZ8rv8EvpVPZa6JG1EFM35yePjt23b5rgddicuO87IZULAz6VgSB6YAgy\/p2uvE\/BrqchSjnJ0bvI1geWPjEqjAM7zK43OtqUg2msrUWwSSPgd71UnLD9+nbwNc36x6SCHFbGFHwdCPv74Y7HVLU4vWH5x6o101YXh9\/dX\/hEdu7IftWRaiShDX9yB7W1JGwUF+KkHmtbW1tKGDRuIj6lqaGgQbVq\/fr34v3y5ueoyDFEAvzBURZ5OFGD4NV3xxwJ+uUxOvD+b8mUnjyJNjBQogp8+8MHr\/RgyN998M33wwQdFt7sBfsH2ohu399y5c+L05Dff3GVZiUceeZSuv36c44rqL\/pRj5HnTKy2oslCnOyWsLqBzWsZ6i4Rozqo+3tVQbweZrq55w10sscgaq1sZcOP\/vGmWrqhZ\/xXSTgeDClIaAo\/dnd5+cuzzz4r3N6dO3cCfiEOiKjhZ3bDmQpAr2CSsjm5gc1rGSpQjeCnPyhV1skr\/Lb0uJFO9hhIrVXs9hJtmVhDE3p1DnGEIOugFTCEn3R3p0+fLub6+E5dwC9o6YvzcwM\/q5rIE5bHjaunhx76W+rc2dkXUn\/D2SfNbXti7aw5o9vRjOoY9A1sZtaovr52VqHbnhWnulR\/hf7QawBlMzmiTI7+ctSl9J3R3d1mhfQRKmAIv61btwp3V87zAX7h91AQ8JPg69Onr6vTlfWHhy4ZW3zYgNVR726ugPR6Jp9ZGWbWqh5+avucHHpq19sMv3++ZCKduHwgZSs1+DH4HhjTze5RfB4jBdrBb968efT000\/TjBkzaPz48ZpJD8sv9C7zCz95fWS3bt1dgc+sYU6gZnYtpFGeTg4uNQKTVRkSft06EP3DRKI\/f4XowBkiPfzMIOnl\/g5um4Bf16\/SycsGUo7hR0T3X11ND1wD+IX+RQmwAAE\/Dl40NTXRr3\/9a8Mb3AC\/ABU3ycoP\/ORtaidPnqAVK1bRoEH+jleymz+TTTA6dNRMKSfws5qrM5qbY6g9tZfo8T8pDsbo8zELdnBdvdzjwfB7rvPX6FSPQZStyIqAx\/3XVNP9Y6vDHygoITAFMs3NzbnFixcT7+qQS1z011cy\/JYsWWJYKAdEpIUYWK0cZISlLppIauTXbXTXzkLjz\/URX\/mMGcyChJ+bMoxcd1kXI3dbDYC4dYUZfj\/r+Kd06tJBlKvKiqUu46\/sRP\/31ralYA6GMJJErEDmww8\/zM2ZM4cefPBB+ulPfwrLL6IO8Wr5yQvE77uvkaZMmRpY7VXwGFlHTtxitTJeLD83ZVjBzw7ybt1fcXtb1X+jk5cOplyFNudXX9ORNt1+aWD6I6PwFTANeNx0002Y8wtf\/0IJXuC3des\/0eOPrwvs3lx9c42is3qX18wy1OflBH56C0xaZ07KKDX8tmX+jE53H0LZqlbKZQT\/6L25V5RwxKAovwoYBjwOHjxIjz76qNjlwS4w5vz8ymz\/vFv4ychuEBeGqwEBFUBm8HPjjqotdxPtdVtGqeH3i+wkOt19MGUrs5r1Rzn6bePl9h2NFLFRwHSRM+\/uOHz4sFjY\/POf\/xzr\/ELuMjfw87qWz6wJKjiki8tpb3+R6PWj7YMCTkBjBE436\/yclKG2xyy9mfvuJlij143d3l+0TKbT3YYQ8akuHPHNEVU\/9yZ9+84aumZ8Lxpbzze74RVnBUzhJ\/f68kJnfmGRc7jd6AZ+cp7PrkZu5gGtIqJ6d9TMUlTrYwQ6Jzs8ZB5OynACP05j1Ta3832cn4Df+dvoTPchwt\/N5SO+1c+\/QVWHvhDVuqa+F90zbyQgaDdII\/zccm8vb3FbtGgR8fzfqVOnsL0txI6KGn7cNLu9vbL5VnOBTtJY7e118rxRN9hZikZr\/ex2rph1N8Pvl2fvoNOXDCXivb3C7SXq\/P8\/pM679xY9BgiG+KXxmbXlqS69e\/cmXgbDL3Z\/1ePtcbCBT+V1j7uBX7AlIze3CjD8\/uX0N+h09VABvlxlVri9nd\/eK95GL0DQrcrhp7c9z2\/fvn30+uuviz2+cbrbA+v8wh8cKMFYAYbfiyem0Jmuw8R8n4AfEVV9eoy6vvSqpWyAYHxGlS384lPV4poAfnHtmfKvF8Pv5eN30ukvDRN7eznaywudqz77nLq+bA0\/qQ4gGP04Afyi7wNRA7i9MekIB9Vg+G0\/dic1dx5OrZU5DYAZDvxmqOuWf3SQQ3GSexpH0j3zcAy+a+F8PgD4+RQwqMcBv6CUDD8fht+Oz6bS2S7DqbWCqJXX+omFzhnq8tMtnisAa9CzdJ4eBPw8yRb8Q4Bf8JqGlSPD77XDU+lsp+GUrSBqqdIsP4Zf1WvbqeLzo76KBgR9yef44djAT54cLWtudyUm5vwc9zESBqwAw2\/nJ3fSuY4jKFtJ2rtCbPKgir3vUubDdwMpERAMREbTTGIBP44o8+EKy5cvF\/uJ5TIarrV+iY1sCeAX7sBA7uYKMPz+3+\/vpPMdh1OuMlOAH+\/vpQ\/epdxHwcBP1gAQDGc0xgJ+RnuHGYgrVqwQb\/XWOMAvnIGAXJ0rwPB74z+n0IUOwylbmaFcJVGuIqNlcOwItb71ivPMXKQEBF2I5SBpLODH+4j5tWDBgkKV5fa6+fPnG54XWC6W31Nr\/0O0+aWdj9Azz\/zEQZchSdQKCPjt+wZdrNIsPwYfA5DdXn5f3O496OG0bYgQO1XKPF3k8JMu7oQJE4ouRlf3FhtdmJ5U+EnYPbWu2DXqc91zgJ\/\/8VySHBh+u\/beQRcr8\/Cr5HUuecsvl6PzO9wvd\/FacViDXpXjAFUup21MjOhV7vDbvesIvf3GEaGuHniq5IBfRAPQQ7ECfu\/dThcrhhFVVgjrj+HHX6RMLkcX\/n0HZU\/6i\/i6rRYg6FYxwM+9Yg6eMLPurB4F\/BwIG5MkAn7\/cSu1ZIZTrqpCAx9HezNs\/eWo5eP3qOXge5HUFhB0LjssP+damab0Ajt9ZoBfAB1RoiwE\/PZMEvCjijbLjy8y4tuMWj75LV089H6JamNcDCBoL3\/k8OMqugl4rF27VtwpzFdszpw5kzZt2kT19fX2LQ0whXRld+86Sm\/v0lxavy\/Az6+CpXtewO\/tW6glN5SoqlKz\/CortIAHEWVPHaXzH75WugpZlAQImosTC\/iZLXXhswR57Z96m5wMdMgmlQp+QVh3dm7vsGHDY\/GFQSWsFXj\/\/ffo5JnvaPBj6PGbAx7s9oqIb46a9zwfOxkRIS7ukljATy5ynjt3roj4Wi1yZstv3bp1hVaw1cdWYNDWn4QdF2QVqAhqhHeq\/jygrCo07yu\/8qKw\/iKg3JOdDU\/MacdPZTK8NqWCKFMpfs5UVBJlqrTfZyook+lAJH4nP8v\/W8mWXgW1ZBl8le3hJ0y\/HJ3\/3U5qbdZOdY7bC9ag1iOxgB9XRH83sNn2Nrb81Jd0g\/1CMAxXNrpBn6GMmHxvw6AW1I80sB+dHIWSNV2kFhroGH5VlKnokAddhzwMGX4d8r+vokwl\/8ywqxLvjABfZd7tZYgWL3e5eORDunhsXwzabF6FtEMwNvDzM0oYiCoEJQid5MkWXiksOyd1CT4NINheU9ZE24grV3kJyy\/DcKsiqmDrLw9D8XMegPwzvys7EFVpVmEBgMIazLu9wuTOUeuZL+j8wTeD79IQckwrBMsCfnI8qBDk3zU2NgqX2OpV3vCTLVchKOemspTLsQuYRmtQApA5Jd1gtt4k+CQENSBmKjrlwZgHJFuBeguwIj\/vl3d72fVt\/t2\/hoCq8LJMGwQDhZ\/Zflw5h7dt27ZCzy1durRoR4fbU13shoA6N2jnErPL+9TadwOL3NrVLbrPJQTzbhq7xTmGYGsKIaj+QVCtwDwA2cITFmDHPADZ2uOfGXxtEGTXV1iADD851cBAzeXo4vH\/pIsnfx9dd3ssOS0QDAx+cjsa671+\/fqiwwjUO4D5HhAJumeffVbs2\/VyqovTfpXuMFuFgKDeEmT3L784N5ejHAcDUjc3aGQF6txgMfenga8IgAxItgDZBS7AL69xNksXT+yni6c+djpUY5munCPEgcBPtdr0gQqzAwoYiHV1dcL683Kqi9uRop8XtIoQp8kS1NgntifkLRc2AtPlEmuRX+0lpwK0YIg29yfe\/HMlu78qADtq7q+YK5SWn8woS9TaSmc\/e6MsrOpytAZ9w0+Cj91YfjU1NRVZfmzVGa3XU4Enl664OdXFLfxkekDQTDm9S6xN3KfFJTYCoFzyooEv7\/IKC1B7k\/ydcHslQOUBB1kitv7OfEKtZz8tRJi9jtu4PFdOEPQNP7VTGGh6+DEcV65c2c4VlmlXr15Nq1atIrenuvgdDG4gyGWlIzAi7EBlmUybNZjLtkiTxq\/0MX2+zf1VLUChB8OuAEA579c5bxUyBHmeMG89K63jPxwtZw5R67nP8suO5Nxi8oNM5QDB1MLPyBLk39lFiNPjEutBWHAMy8aKaU9hFYAMKi0SLAFYwdaenP8TVh+7wVpAhK1ELeChvLJZymUv0PkT2iEHbWsvAcE4\/AVMPfysIMjBGLOdI4Bg+XyBi7+IagCEo+CSf9quj4rKTiLowSDkOUARAMm0wU\/DHy+k5uARv1up5dwRar3Auz3ynworUdiXCmDjgAPvdUiiJQj4GfQ3lsmYfQnkl1e\/e0RaSN6\/PPF6sniDYMH6E1vd8nN+Ivgh5\/\/y8CvsqFHAxsGj1vPUcvYQ5XJy6sBomU3yXWFuNUNwbH3PdvcQ83eKX3J+XxoVYWxNdToPn7\/lAAAMpUlEQVSWQodf3AIeToXhdG7mBdNlCRbMoYI71+YiulE4aWm1+T\/h\/lZ2ElagZvVJ+KntyWnLhnKtAnrZi6ep9eKJosivNk+oWIpltuB8SP279OtdmwuiMPD4zR6VhGFZw89qqQurwhFeN6e6RPF1AQSjUD2OZbLFxtZfp\/yyl\/wCaAExCbK2emuRcp73ayViALacoWzrOd3SlzYrsNx23BynH9G3G28VgtjttIqit0O3\/LhRvKaPASIXP5stcnZyqksUIsky3UCQn0lPhDjKXill2RJ+0t3NH4ZQCHa0d5c18GnWX671AuWy5xX3t5R1D7+s07Sd5jU2ioLumTcq\/AJ9llAS+Enrb8+ePYXq6re3OT3VxWd7A3ncLQTT6RIHInXMMmG3t4oypG1xk8dhiUhv0XwfV5tdXp4L5QXjefc3e5FyuYukLRsqjzm+kfVnhCs7cnxz4MfKhd35gcIv7MrGLX+3BykAgnHrQbf1ycOPT3rJH4elLYTWLXER68PzR4gV5v3YAmQQtiR64fgF+h01NN5KF2g\/NczTXFqrl9G+frmtVT4Xxd5\/Lhvws+s9B58bQRDLZBwIl6gk+SUwPOdH2o6OjNgSqM71qRDULD9h3wm3V5v\/S9rWQQk7L66s0aHE+ikvzjeqvf+AX8BfQCyTCVjQ2GTXBj+x6JkBKC2+IgCqLm9+i6A4PTo54AvKlTVb6aHCjgHZ0NBA8+fPF1Fg+SrF3n\/AL6Qvl5t5QbjDRp2grinUPi++Yrq0c2Zt29ektccAlJZesdsrTsfRapyf99N+jms0V1p3ImBhc\/5lEF8XdXXHwYMHI9v7D\/gF0ZsWebiBIGeDCHGbmEb7ZbVP1ahq21l8BeCE0qe6Pc+iGupi7zzsNEq3wa9wYnRpYW0lgerKRhGoUG9rjHLvP+AXyhelfaZuIQhrUGqoP4o\/Dz9b8GjWV5u1GAR8jOpiNIBUIAdRrv9BGpQr67cm+rM7AT+\/iiboeUDQT2fp4aO4m0YwLBxMIMsMej+yLsDhp2kBP+s2Khtw8YbZyQBITU2N2NzAL8CvFMrHrAxEiP10iH4+sP1SE6vc4zr35kcRFXZjx\/eK5Zo7fVQX8PPT42XwrNt7iOEOm3V6GwDN1921WYBlMHQoLq6sEy31u7zkM1Hu\/cecn5OeCzGNeg8xuwDqqRc4aj9E4ROYdRxdWTsZ5e6uvn370rJly4jv8FFfUe79B\/zsei+Cz93MC8rL1sv37uEIOiAmRTLsxtZrLmwUUVm\/MlhdaqbmHdXef8DPbw+H+LwbCHI1eJnM7l1HU3AFZ4iiR5i1CrtSrbkLs7n6\/fpqWepFZ1Ht\/Qf8PPS+DNcfOnSo6Gl1z2KQ9xC7hSDmBT10akSPlHqBcUTNjGWxgJ+HbjELz6uTuHPmzKHly5eLLTtGexw9FOvqcFXOHxD0onI4zzDkztB2kXkHGiCOfkqiKxuOOtHkCvh50N3o8FU1m7DvIcZpMh46LcJH2IXlk1DkqcZRnl4coQyxKxrw89Al6qZro8fV7Tvyc7OolofiC48wBBEh9qNgOM\/KQz2dHvsUTi2Qq50CgJ+dQrrPjSZnOYmc75MubqnvIXZ7mszbbxwhRIhddr5JchmoEJf3xHSBcTAtLa9cAD+X\/SmDHZMmTSps0ZHAmzFjBnEUa\/HixSW\/hF02w21wBAcpuBwA+eTSuiuHqKw3BZL\/FOAXUB\/KeT4GHy\/mLLXlp2+GWwgiOGI+ENiyYxeWgxRwZQP6wsQgG8AvoE5g+DU1NdHq1atp1apVkcPPqyUICGrKhe3KOjm6PaChiWxMFAD8AhoaEn58Q92TTz4pcpUnV\/DPYQQ83FQdlqC1Wqp1VwpX1u7odjd9i7TeFAD8XOrG0dVFixbRhg0baPDgwYWnGX779++P\/T3EWCbT1uFRzdtZ7Wetq6ujadOmuRyVSO5FAcDPpWpG0VwezAsXLhRvBqIMisT5HuI0LpMJ25V1OpScnGSiPwDAad5I51wBwM+5VoWU+vma2tpaQ0twyZIlhWfUvYweigz1ETfLZLgiSYoQywXGDL5S3E\/hpKPsDvDkqZMePXo4yQppfCgA+PkQr9weLYd5QXWBcVzX3AF+8fjmAH7x6IdY1SJJEIyLK+umAwE\/N2qFlxbwC0\/bxOccRwiWOiobRicCfmGo6j5PwM+9Zql7Qg9BPlzTav4s6LWC7MqOr6+nctk+hoBHPL5CgF88+iERtfCyTMbLHmLVlY1ToCKoTnJydHtQZSEfcwUAP4wOTwoEHSGOY1TWkzAOH7I7ut1hNkjmQwHAz4d4eJSIIcgWIb+lO8z\/6l+cjl98QVNX+qr4me+nuKa+pzjw1eiZctbXydHt5dz+OLQN8ItDL5RBHYyCIzyxL8Eom9jY2Ch+jMuauzKQHk3wqADg51E4PNZeAf0dxDIFW3WbNm2CZFAgVgoAfrHqjmRXRr2DmFsiXV0c257sfi3X2gN+5dqzaBcUgAKWCgB+GCBQAAqkUgHAL5Xd7q7RQd5B7K5kpIYC4SkA+IWnbVnkLI\/ncnoHsR6UkydPFsf6q0c0OYEpr4PbuHFjQcPZs2cXHQ5bFuKiEZEqAPhFKn\/8C3dzB7GEmv4mO26lBKATmKqnYvPRTvrzEeOvGmqYBAUAvyT0UoR1dHMHsVFaPezsYMoWotHtd+pJ2RHKgaLLSAHAr4w6M+imBHEHsX4fqx1MhwwZQg0NDTR\/\/nyx80O+zE5CCbrNyC89CgB+6elr1y0NAn7qCSa9e\/e2vdN47Nix4o4UnmNU70gxuzvFdaPwABTIKwD4YSiYKhAE\/NRbyrgguwvdAT8MyFIpkHr4waIwH2p+4afX1kl+gF+pvvooJ5HwM\/sSue1OmY8ajXSbR7mnt5ujU+flVC30gQ7+DPAr99GSrPYlEn76qyLVL9a2bdtILrWw6wqOIKo3rOnTG93KZpdnuX1uFJ1966236N5776Xm5mbKZDKiyarmVktT7GCKgEe5jaD4tieR8FMjfy+\/\/HIBYEuXLhUXPusXyLL8+kWy+jVpahfBImxTw+0dxPIPitkfILOlLjLIYRYUMXouvl+r6GsWZXRcfn9qampivTA9cfDTu05GXwq9daH\/v\/xCT5o0ybBz1El6XB5NpLeQze4glunkHyEjBDiBKefz2GOPFe5CxiJn9zCVS4z4OLEFCxY4ykA+M336dGFEyJeRp2WVoZVh4agiJUqUOPg5WTRrBT\/5\/KFDh1xJDBfYWi6jk4nVJ1Qg2sFUfym8keXuqvNSmtit9WcW\/OO+XbNmDe3YsaPwB8lMUrtxEKe+TBz8pEsr3So3lh+fIiyXWuzfv1\/0n\/pX0ewvnNltWyn9TqHZMVHAbs7arpp6C93K43HiytpNF8XNIkwU\/FSrzQv8VNCZTbwvXLiQ+K0usAX87L5G+DwOCpi5rU7qpu7E+fjjjy0DgWp+6sEV\/J164YUXDK1Do+i\/k3qFmSZR8GNxT58+TadOnaIZM2aI7U9uLD89\/NRTQ+xEhttrpxA+j1oBP\/DTz7O6bYtRkNFJHlGe1pMY+LFJzTd\/3XXXXbRy5cpA4FdXV+doYheWn5NhjDRRK2B2H7BdveRzn3\/+ecFqs4qu693bPXv2iO8k78dmCOq3JpodSmHkfdnVNcjPEwM\/2WgpfBCWH+AX5FBCXlEr4DSYp1+GJOcOVe\/G6g++WfTd7BnAL6CREQT8rHYaYM4voI5CNiVXwM5DMZp3k7+7++676fnnny9YbVa7qMyiwk7hqwoDt9fFMLGCH5vfmzdvpurqarrkkksKkVy9ee0kHK+vEub8XHQSkkaigP4QWH0ljODIIOMlLFOmTGl3mo6Z62vmrnL+K1asEG8+hFa+YPkFNByM4Ce3qMm\/IjJkf+DAAWIg6sFl9hcSS10C6iRkU3IF7JaZcIWsDvEw+k4Y\/c5qXhHwC7nbVfjxTgNet6ffRmM3kWr2Fw3wC7nzkH1oCjhZQ2e16NkIdEaur1UeZp\/B8guo2\/WWn1G2VvCzWg4A+AXUScimpAo4sfqk5cdR2fXr1xe5pfyZmTekX\/hs9d3ysugac34uhopf+FmF8FX4cZXmzJlDchtclJ3kQh4kTaECVouLVTmsxr4Z\/FRrjr8f\/J2QN\/nppTYDIyy\/FA5KNBkKhKmANAR2797teM+t2UEHdpFibodVQMVqLhDwC3MUIG8okEIF2NLatWuXoRtr5IIa3aEsZZPwe\/jhh+npp58mPhfTyUsGExnATU1NpnXhvfT602Xs5uadlO8nTeIWOftpLJ6FAlDAWAEnlp+ZdnY7S1TLT90GF\/XyMcAP3wYoAAVSqcB\/AVtJzVABqw82AAAAAElFTkSuQmCC","height":174,"width":232}}
%---
%[output:9ca53d11]
%   data: {"dataType":"text","outputData":{"text":"最高点坐标: (Theta: 30.20, Phi: 60.10), 强度: 37.07 dB\n","truncated":false}}
%---
%[output:831569ce]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQuQVdWZ779zugFbaIhcMcPTVlFUFEGxGZyR8XFj3YB3MsQL5OpANEI6kEYmsQnUgM7EgEMrpOBCqVzQoK1TQ3MjN5O0t8o7akISKdtkUCI+MSGiMIJgBGwe3X3O1Lf2Wees3r3fr7P32f9d1dWPs57\/tfavv7W+9cjk8\/k84YECUAAKpEyBDOCXshZHdaEAFBAKAH7oCFAACqRSAcAvlc2OSkMBKAD4oQ9AASiQSgUAv1Q2OyoNBaAA4Ic+AAWgQCoVAPxS2eyoNBSAAoAf+gAUgAKpVADwS2Wzo9JQAAoAfugDUAAKpFIBwC+VzY5KQwEoAPihD0ABKJBKBQC\/VDY7Kg0FoADghz4ABaBAKhUA\/FLZ7Kg0FIACgB\/6ABSAAqlUAPBLZbOj0lAACgB+6ANQAAqkUgHAL5XNjkpDASgA+KEPQAEokEoFAL9UNjsqDQWgAOCHPgAFoEAqFQD8UtnsqDQUgAKAH\/oAFIACqVQA8Etls6PSUAAKAH7oA1AACqRSAcAvlc2OSkMBKAD4oQ9AASiQSgUAv1Q2OyoNBaAA4Ic+AAWgQCoVAPxS2eyoNBSAAoAf+gAUgAKpVADwS2Wzo9JQAAoAfugDUAAKpFIBwC+VzY5KQwEoAPihD0ABKJBKBQC\/VDY7Kg0FoADghz4ABaBAKhUA\/FLZ7Kg0FIACgB\/6ABSAAqlUAPBLZbOj0lAACgB+6ANQAAqkUgHAL5XNjkpDASgA+KEPQAEokEoFIoffyZMnadmyZTR06FBavHhxUfRXXnmF5syZU\/x93LhxtHHjRho8eHAqGwaVhgJQIFwFIofftm3baPny5TR37twi\/Pbu3Uvz5s2jVatW0aRJk0gCkqu+cuVKqqmpCVcFpA4FoEDqFIgUfhJyBw4c6AE\/BuLOnTt7gI7DNjc3iy9Yf6nrl6gwFAhdgcjgJ625q666il5\/\/fUew96HH35YVFQdBh89epQaGhqoqalJWIN4oAAUgAJBKhAZ\/Ni6a21tpXXr1tHq1auL8JNQnDx5Ms2YMaNYNwm\/mTNn9vh7kJVHWlAACqRXgUjgp1px7MhQHR6AX3o7H2oOBcqpQCTwU4e1em8v4FfO5kfeUCC9CoQOP17CwsNcuWwF8EtvZ0PNoUCcFAgdfmz1bd682bDOci3f448\/7tjhMXv27GJaCxcupPr6+jjpibJAASiQEAVCh59eB6NFzmZLXZYuXSrW\/o0ePbqYTHt7O61fv774e0tLS0KkRjGhABSIkwKxgJ9c\/7dgwQLh2XWyyJkhyFYgww\/WX5y6FMoCBZKhQCzgx1LJnR9SNrvtbYBfMjoYSgkF4qpA5PALSgjALyglkQ4USKcCgF862x21tlXgKBF9qoTi\/eXnEBH2mdtKl5AAgF9CGgrFjEqBk0R0QHGw7ab6+nFK5sMAwKiaIuR8AL+QBUbySVKgBL7165+hDRueKRa+sfEOWrjwjsLvFyWpUiiriQKAH7oGFCgqwBbfSdKDT35cAiAPf3HOZNI7DuCX9BZE+QNUQIPfmDFTTdN8553nYP0FqHg5kwL8yqk+8o6ZAu9Te\/tumj17qWm5WlpWFeYAMfSNWeO5Lg7g51oyRKhcBd4XVbO3\/Njjy44PPElWAPCLsPVuv\/32CHNDVm4VuO22G+m2225yMOcH+LnVNo7hAb8IW4Xh9+ST\/xxhjsjKrQJ9+mjWn7W3F8td3Ooax\/CAX4StAvhFKLbHrDKZk1RdjXV+HuVLVDTAL8LmAvwiFNtHVgzAd9\/9NV1++eXC+6s9WN7iQ9JYRgX8ImwWwC9CsX1m9fWv304tD\/SnM1W\/ppoL3vSZGqLHUQHAL8JWAfwiFNtnVgJ+3+9PnfmddNYlb\/hMDdHjqIAl\/OTFQ7t37y6Wfdq0abG4SDyJp7oAfnF8BYzLxPB78sEa6j7zCvUd8SPKnn1tcgqPkjpSwBR+fPfGnDlzaMWKFT2ujuRz9x555BHatGlTjxOWHeUWYCDAL0AxkVQvBRh+T9x3NmVzr1L1BZsBvwrsI4bwMzpqXq0738tx8ODBslqAgF8F9sYYVYnht2VpLfU58Rplxj5KNOjqGJUORQlCAUP42V0YLi8glzeyBVEQt2kAfm4VQ3g3CgjL7x\/Opn5Hfgf4uREuQWFh+UXYWJjzi1Bsn1kJ+N1\/NvU7Cvj5lDK20U3n\/PSXCskaYM7Pe1sCft61izomw++Rxf1p0MndsPyiFj+i\/IrwM\/LsWpXB7oKhsMuPYW\/YCrtPf926H1Jb20\/pwQcfookTe3tHt2\/\/MT366AbTz\/U5Hvic6Po2ov2fm5fl\/vFEyyeUPr\/jJaJt+0q\/b72JaPr57usi4df\/xOvUZ9xjmPNzL2HsY2CdX4RNVOmW36effkr33rtIKLpmzTo65xzeFaE977+\/l5YsuZeuv\/6vaNGi7zpS3Q38jp0h+uvniV4+3DtpLwCU8Ov36W6qufpRoi\/A4eGo0RIUCPCLsLEqHX4s5W9+8yr9\/d9\/j6ZN++9FyFlB0Yv80rob2Z\/ol9OIhvUn2v5HolkvEg3qS9R2C1H9ECKjcE7zY\/g91NSXzjvye6qeuI6y5yjmpdNEEC7WCgB+ETZPGuDHcsrh7\/z5jTR9+m3i91\/+8hfU3LyGLrpotC\/FJeQ4EdWik6C7bgjRv95CNLAvUfthomnPa9lJIDrNHPBzqlRyw2HOL8K2Swv8pKV37NhndPXVE+nnP3+RJAj9yK0Og2fUET1zo5aaOuRV\/66Gdzv0FfCb9wb1666ifhfOo4EXLvBTdMSNoQLw9kbYKGmBnzrHd+zYMbr22nq6777v01lnneVL7RW7iB54refQ1in89I4Ru4Iw\/B6Y+zqdnctS7YXzqRbws5MscZ9jnV+ETZYm+Enr78MP9wdi9ZlZd2HC77G7fk8nMp8DfhG+I1FmhR0eEaqdJvjJeT+Wd+DAgb7n+8zm+sKE3w\/m\/p6y+RN0zgXz6QsXfjvCnoKsolAAll8UKhfySAv8pMeX5\/nGjbtKLHEZOHBQr+UvbqSXQ17VwyvjhzXnx\/A7WvUZXTTq2zT4gkY3xUXYBCiAU10ibKQ0wE+u5xsz5tLiPJ9c3Kwuf3Eju9WQV6YThrd32j0f0phTh2hoXSMNrwP83LRZEsLiPL8IW6nS4Xfq1Cn6wQ\/+gd555+0ew1z591dfbXe8u0NtFtVra+a4CGOdH8Nv9OlPaNT5CwQA8VSWAr2WusycObPH+X1xrS62t8WvZay2t\/lZ6CzX6312pufaPlWBMHZ4fLXxj3Ru93EBvwtHfSt+gqNEvhToZfnxwQXLly8XieoPMvWVU8CRAb+ABfWZnNHODn2STsIYFcPIqjMrbpB7e7+06GMa0XmULhrVAPj57B9xjG457I0zCAG\/OHanyikTr\/Nj+J3X9RkNGTiB6q\/cVDmVQ02EAo63t8UNhIAfenCYCjD8blp0mLKUowsGXEY3jt0QZnZIuwwKOIafWjac5OytpSrd4eFNlXjGEpbfPR9TPkM0qnYs\/dfL\/1c8C4pSeVbAEn58idGOHTto8eLFnjMIKyIsv7CURbqsAMOv7rt9aGTuY6qrvZy+fNkPIUyFKdADfnwxUX19Pe3bt4\/Y68tXVgJ+wbU4LL\/gtAw7JQm\/L+aP0MjaK+irl64KO0ukH7EChvD7yU9+Iorxla98hdjCuvvuu6mhoUHAUP+U60RnWH4R95SUZcfwG3zvIKrL\/QedX3s5zbh0ZcoUqPzqZjo6OvLLli2joUOHitpOmTKFJk2aRI899hjV1NTQoUOHBPx42QsvgXnhhReEZdjY2Egcb9asWSJ81A\/gF7Xi6cqP4fdn9\/anDOXoygEX0J1jtOVfeCpHAWH5yXt629raetVs7ty5vuGnvx9k2LBhvS49l5ekywLYWZSAX+V0wjjWhOE34N5zqZY+pxo6QyuuaYljMVEmHwr0GvZKy4\/TlA4PP5afBB\/PJUrHif4GOHlT3KpVq4QVKWHMZVi5cqWwQPUP4Oej1RHVVgGGX\/V3h9MXMscpQ3lad81G2zgIkCwFQocfA3T16tWkXnAu4cZDbQYiw3Dnzp09QMdAbG5uFl+DBw8G\/JLVrxJfWoZf13cvoAGZDupLXfTYNesTXydUoKcCveC3efPmHiGCGPYaic6eZX4YfurPMqy0GJuamgznFGH5Bd+VnWw\/cxLGrmR2e3WD2KJml4ddGRl+R74zjgZmTlBN5jQ9dY3WX\/FUjgICfnLYeeDAgV41CwN+0vKbPHky3XrrrcJxwj\/PmDGjmL+En9lBC4BfOJ0wrMMJZGn111Gqd2sEdTiBVR5OVWP4\/ek7Y6k6001nZ07Sv1zzT06jIlxCFCh6exk+7MXlOT92NrS2tlJdXV1xqUuQ3l4eCi9dulQ4PYYPHw74xaizWJ2+4vbScaNqWVl1QR1LFYTlyPA7+p1xlCeifpkztHLMHLqyti5GLYWi+FXAdJ3frl276M477ywudQlqnZ\/eolOtQFh+fpszmPhGQ1svl47rS6MeRS8\/C\/r6Sbs8nCrE8Pvjor+gmmwH9c100j+NmU1X1Y50Gh3hEqCAIfw6Ojroy1\/+ctHbG9T2Nr2jg\/UB\/OLZS9Th7xVXXCkOKT148KDno+jVoeh3riB64l0i9Xy+II6it8vDjdIMv\/cW3khnV50QQ98fXjoL8HMjYALCut7bK9fj8VygGyhaLV9x4\/CYPXt2UVae92tpaRFb8pLwJGl7mzr8nTz5Otq2baunU5hlu8ihKN+ru+gK7TJxt\/Czu37SLg83fUTCr0\/2NB9+RBsu+xu6euBwN0kgbMwVKMJPwsnJjg2jtXtW9ZQOlalTpxoC02ypC88L8tq\/0aNH9wDe+vXasgPAL9zeJYe\/nIvX+zc4rhyKysuHPuwIHn5O8nCjFsPv3YU3iSh9smfo7uH1NHfERDdJIGzMFeixw2P\/\/v30jW98ozjknTNnjm3x7U57luCbMGGC6YJlGWbBggXC44tFzrayRxaAh7+vv\/5aIMNdOb9ntAzFybDXzPJTh7tWebgRjeH37\/OnU\/\/q45TNdNO8EdfQN0dc7SYJhI25AgJ+PJTdunVrDzgZ\/c1LXXhIq187KNOZNm1aMU\/1sFT+HNvbvKgdfBy\/8DNyQOhLed0Qon++kej2l4hePkzEQ+NnbtRCGYFNH99pHv96C9HAvs40kvCrznQJy69h5FX0rZHjnUVGqEQoYHl1pR6IcaoR1vlF0xpRwY\/BNP\/XRNv2ETEMJaiklci1bbuFqH5I73qHBb9Xv3UbVWVy1CfTRQtGXUHzR10ZjejIJRIFPJ3kHEnJbDIB\/KJpBb\/wMyql2e4Lp+v8nFxoFMQOj59\/8w4akD1NmUyO7jn\/Mvr2qLHRiI5cIlEA8ItEZi2TJHl7pSxRws\/pDo+o4Pf8vK9TNXVTbdUpmjToXHr6qikR9hZkFbYCAn76I6fcZGrn8HCTlpuwsPzcqOU9bJTwk6W026ERFfyem3uXtr0t2yngt238ZO9CImbsFCjCb8mSJcRf6rISu9LanbxiF9\/P54CfH\/UQ104Bdng8+41v0lnZTuqX6RLHWv3HjVPtouHzBCngC35sMco9v0bHToWpA+AXprpIm+G39a75lCGiAVWnxB7fz276EoSpIAV8wQ+Wn7uekMQ5P3c1rJzQDL\/WOxuJMl3C45vJ5OnUzYX1N5VTzVTXxBB+VmvzWC25tQ3wc9d3AD93epUztLD8Zt9LxDZf1WmiTI66v3R9OYuEvANWwBR+fJwV77bQA85oH27AZXKUHIa9jmRCII8KMPz+z982UVe2m\/KZbgHBl64dSzcMHugxRUSLmwKAX4QtAssvQrF9ZsXwa\/vaMurK5uhU1WnKZ3L0Yv1ldMPgWp8pI3pcFAD8ImwJwC9CsX1mxfD7f19bTmey3dQlLL8MLb\/kPLrvYu2KVzzJVwDwi7ANAb8IxfaZFcPv32b8o0jldFUn5TJ5WjpmCC275Is+U0b0uCgA+EXYEoBfhGL7zIrh9+L\/eEAMd3PZHHVncrRkzBDxhacyFDDc3mbn7eWql2tnh5S90h0ep06dEqcnv\/pqu2VPe\/DBh2jixGsd90Z1zytHkmfsDeuvJaG\/\/McoYbtDRY3SUE9qUdPUH0rgJ5x+e9ygvuaHIdgJxvDb8TcrKZ\/NU3dVN+UyOfqLc8+m7X+Jo+zttEvK50X4BXWEVVQVB\/w0pd3ATw8+2VYqAP3CzywPzksPNv02Nlke9VQX\/puTcFblVu8Jcdo\/JfyyuQx1V+UoV5Wj686toR9PwWnOTjWMe7jieX58cOnatWvphhtuoJqamriXW5zizEfap\/EYe3nC8rXX1tN9932fzjrrLEftJSFidKKynTWnjystRX3GK3YRPfAakWp1Gf3N6NQVoz27TsPJPGTdBvQh+uvntfMB9TB1IhbD7+VpzZTP5iifIWH98fPRjAudREeYBCiQOXLkSN7Lvt5y1y2t8JPgGzFipKvTlVXLSILO7PRkfduqQ1M7K0pC0uhMPvXODgkrOzA5CWdWDycHIJj1Y4bfK\/\/tYTHfl6vizW154fTY\/zVcX1nudz+o\/HvM+cnj49va2hynb3fisuOEXAZMI\/zk9ZEDBw5yBT4zaZ1ATYWm2Xycmr6V5WdklXGa\/PAhpvqhsR5qTsPJU6D9nOnH8Gu\/5WHKZ0lAj+f+KJOnf7n5PPrzLzqztF12aQSPWAGc5xeh4H68vfI2tWPHPqPm5jV00UWlS528VEGdR7Ma8hrBzC4\/szk6aTVandunAtBLOBXQKtzthvX6OjH8Xr15jQY9IspV58ThBn83biAtugq7POz6QBI+t4UfO0I++OADsdUtTk+aLD\/V8+vGwWHWXnqo6D2+Mp7TYbE+HwlM9e\/qHKA+fwlFPWgvHVSat+O0zMLx0fYSuDIffVwv8PvtlHWUr+LtbUT56m4Bv3vG19Ki8YBfnFjgtSxF+KkHmg4bNow2bdpEfExVQ0ODSHvjxo3id\/m4uerSa+Gs4qUJfvIC8fnzG2n69NsCk1OFkNHcm5NhsRn4VKBaQU3NVz8vec9YY6eF0fylnZfaC\/z+\/fp1lC8MeYUFmM3TwvED6J6rscUtsE5YxoR6wE\/v+OD1fgyZm2++md59990et7sBfu5bzcuwd\/v2H9Ojj27wdW+uVUmtnAJ6D6qZh9fOUvQKNTfw4zLoAbimnmjtHqL9n5esRqetxsPe165bL4a9ucJ8H3t+64f1paenlYwAp+khXPwUMIUfD3d5+ctTTz0lhr07d+4E\/Hy2n1v4Sc+unwvDZZHdXBrEcbwMed3cvWu0dMbIonMazqhpnNz8ZtakDL\/d9Y8Iyy\/XRxvyMvx4DPzOfOzy8PkqxCK6IfzkcHfmzJliro\/v1AX8\/LeXG\/h5XctnVkoVLHKoyWHN1sIZgUifttVta0br\/NR5OyNnhJFzxWk4I0gaLbtx2ooMv99ds1Fb51fFa\/3y4js\/by0412kyCBdjBQzht337djHclfN8gF8wLegGfnKezy5nN\/OAVvfb6tfvOVkmYrcoWV92pzs3nO4EUcOZ7SzxusWN4ffG+E1EDD8BQF7uoll+W75aS9eOqLZrGnwecwV6wW\/hwoW0ZcsWmjVrFk2aNEkUH\/ALphXLDT+uhd3eXllTJwuEzcK42dur9wybOSachHNaNyetKeB35ePCySEAKC2\/TJ4WTKqhBZP7OUkGYWKsgIAfOy9aW1vpV7\/6leENboBfMC3oBn7B5IhUvCrA8Ntz+ZYi+IiHvmz5ZUiAb\/51fb0mjXgxUSDT0dGRX7ZsGfGuDrnERX99JcNv+fLlhkVmh4i0EKOsU5qWukSpK\/LSFGD4vXnJk9oiZznkLXh9J46soif+J3Z5JL2vZN577738vHnz6Hvf+x49++yzsPxCbFFYfiGKG3DSDL+3L2wpLXVhALLDI6Pt+Hjy5uM0of68gHNFclEqYOrwmDJlCub8Am4JwC9gQUNMjuG3d9TTYo1fNx9sINb7kbAC+en\/\/C9p4vlZumvhWEAwxHYIM2lDh8dHH31EDz30kNjlwUNgzPkF0wSAXzA6RpEKw+8PI54R6\/sYfgxBseC5Oi9us+z3u7eo3xtviaKMrz+PJtQPobsWXhFF0ZBHQAqYLnLm3R0HDx4UC5t\/9rOfYZ1fAIIDfgGIGFESDL\/9X3xGWHvdVSTgx0dbCQgSUb89b1HfPW\/2Ks1djWNp\/CSGIYbEETWV52xM4Sf3+vJCZ36wyNmzxsWIgJ9\/DaNKgeF38L+o8GMAalYgz\/tlDx+mfjt+YVoctgYxJI6qtbzlY7m3l7e4LV26lHj+7\/jx49je5k1jwM+nbuWIzvD7+AtPUz6bKVp\/fLZfN5\/vl83zihfq+3+32RYNELSVqGwBLE91GT58OPEyGH54+Kseb4+DDdy3GSw\/95qVKwbD79CAp4kYdlW8r1eDoPiq0py+2Z0vER057KiIgKAjmSINZHue3969e+nll18We3zjdLcH1vlF2k9SlxnD73BNi4Afg4+tvnxVRpztJ2DI15i\/t4fye\/e40gYQdCVXqIFt4Rdq7j4SB\/x8iIeotgow\/I70ebIAvEzBAlS+cwpHD1H3b35um5ZRAEDQk2yBRgL8ApXTOjEMeyMU22dWAn5Z3t6WEQDkST7tuzb8ZYuQzb\/OF+3n\/eyKwh5iLJOxUyn4zwG\/4DU1TRHwi1Bsn1kJ+OWfELCjqmxx+CtgyCDMsssjT12v\/4Jyf3I272dXJFiDdgoF+zngF6yelqkBfhGK7TMrht\/Rzs3C8qNsVgCPqgvfxdxfRnh8uz54U3wF+QCCQappnlZs4CdPjpZFtbsSE3N+0XSQtOYi4HfqfxfhJyw+AULtix0gPBbOHTtMZ\/bsCEUmQDAUWYuJxgJ+7FHmwxVWrVol9hPLZTRcSv0SG1nypMLv0ksvC7dFkXogCrz99lt09MSjwuqjwlyfNvyV1l9WzAPyvN\/J9mcDydMsEUAwHHljAT+jvcMMxObmZvGl3hqXZPi9+ab58GjLhtJnu9o\/Dqe1KzLVLGUyBQ9Ehn\/mRXhVhe+FzzLVRMXPMpQp\/M4gE3EZcPyd5\/f458LvXflLSn8T4CtYfpms5vzgJ0\/UdfBt6jz4dujqAoLBShwL+PE+Yn4WL15crJ3cXtfU1GR4XmASLT+jptvVfogWzX5JDKHEQEq8yHnK58WVOcG2dkWmpugmoFetgS+r\/SxAmOW\/aT\/L30vfs5TJcniGG4fJUEZYe1oaAngMPgFGCUflZyLqPPA2dX78TqTqwkPsX+6yw08OcSdPntzjYnR1b7HRhelJhd+P1r8hWu1HG8wWx7Jlwi+0NqYCBO07ufYPg4RW4ucCADPZPorFxxDsU7AIqwXoNCuR4ddHwE+DYAGE\/HcBwIIlKL4r8JNeYC1j6vjdz+wLGkIIWIPeRQX8vGvnKCZbdq+9csgGeGZJSRBqLzYsQSvJe2pVHP4WgacNhUtAZAAyDAuwZKtPAK4wZC5AsGQFFqy\/ohXI1iFv+9Cs885D71HnJ+856hNhBAIE3asK+LnXzDaGvXVnm4QuQGF+SbNv3EZOWXiGYJbyee2aSW0YzGCT84CqBagNiaWlWLQGqzgOw02bK9SsQtUCLFiBDL\/CdEXn4feo88jesmsNCDpvAsDPuVamISXsdrUfptfaNSsPTzkV0E0dSAhSaS5Qg55+PrC6AMMCFBmYBWtQwFMFoIBiYT6Q\/x\/lctTx\/v8vZ6V75A0I2jdF2eHHRXTj8Fi\/fr24U5iv2Jw9eza1tLRQfX29fU0DDOFvKBtgQZCUjQKaFSjnTrUhruoUKTlCik4RORcoh8sCenIuUJsDFHOyPB8ov8uh79H3qfNPf4hVqwCC5s0RC\/iZLXXhswR57Z96m5x0dMgqRQW\/4IeysXpHKrgwvR1IGgT7FLzA0unRt+gMEbDj4bJ0mBQdInJesOAckfATsxF5Ab7Oz\/bFVkt4iHs2TSzgJxc5L1iwQHh8rRY5s+W3YcOGYi3Y6mMrMGjrT1p35l7Z2PZxFMxEAeEEEZwqzAeydUdy3k\/9XvIKl4bHmke45BDRfhZLY9gKzGlpdnwYzm6PIBsV1qCmZizgxwXR3w1str2NLT\/1kcPgICDI1h3m7YJ8zeKYlsVQWM4D6q2+4tpBZQ2hsiRGQFDDKnUe+4C6TuyPY8V7lSntEIwN\/Pz0FgaiCkEJQidpMvBg3TlRqpLCSAAqVqBY86c5PLRF0tp8YGl5TMkZUlwgXdwhUpj\/K1h\/Jw+9kiix0grBioCf7GkqBPlvjY2NYkhs9QB+iXpPAy5sTytQDIXkULiKF0gXdolIEBoti5FLYZSF1l0dH1FXx8GAyxp+cmmDYKDwM9uPK+fw2traii24YsWKHjs63J7qYtcV1LlBuyExz+\/9aP0eLFOxE7UiPzezApX1gD3gV\/AWC2dIwVlS+C7WYIplmHk69enuxK7JTAsEA4Of3I7GTb9x48YehxGodwDzPSASdE899ZTYt+vlVBen76EcDrNVCAg6VS2N4dTF0eKGDkMrUPMAq4ujteUzYjG0POYln6fu04ep6xQfcprsRemV7CEOBH6q1aZ3VJgdUMBArKurE9afl1Nd3L6e+nlBKw8xLEG36lZKeG1ZTGkrIf8uD0MoeIAVK7DHYQliP7a2x1jYfvkcdZ7gHR+cHnuCkw3BSrQGfcNPgo+Hsfy0trb2sPzYqjNar6cCTy5dcXOqi9fXDRD0qlx645WOyZJr\/\/T7hDVACsuPIVhY9Nx95gh1n\/mseFAFIBivPuQbfmp1GGh6+DEcV69e3WsoLMOuXbuW1qxZQ25PdfEroxsIcl7pdYwUTuz0K3ji42vnAIovcQiCPBVGd3ZgYa+vtP66z3xKua4ThdpXjpaVYAmmFn7yXXTrIU7fkFh\/sow2oZ\/ORxsGZzK84Fk5KKF4gGpp2Fsa\/nZT9+kjlM93VaRkSYZg6uFnBUGAchKkAAAMZklEQVR2xpjtHEkXBNUtYhr80nvEljIPKK1AcUq0PE1ansBTsPLyecrnOinXdZzy+e6KBCBXKokQBPwMuiOWyZi9o3oIlib302UNqgAsDXvFfJ\/i9Ch5P7o1AOZOERW21lUqBRmCE+qH9LqHmN8pfuT8vjQqwtia6lTb0OEXN4eHU2E4nJt5wfRagoVJfmEQ5irCs+msj0gAZsX+YO0OkMKJL8q8n0iroAsPffO5M6mZNri4fg\/9qn1rUU4GHn\/xiErCsKLhZ7XUhVVhD6+bU12cdcxgQwGCTixBFYI8LOYhXqXPDUoAahafgCA\/wgJUH178nKc85Yjy3SnRhuhTeoK+2XirEMJup1Wwb6yz1EK3\/LgYvKaPASIXP5stcnZyqouzaoUTyg0EuQTp8RCrw2EJQTnnVenWoNwhwvUuneysXWmuPYw9+ZNmBVbmP4Yz9AdqaLyVztA+alioQS\/OTyTwk9bf7t285Ud79NvbnJ7qEgcx3UIwvUNibi0JAX7pFQjEoSEDK4MCQFFlneWnq3clrPeT0o2t\/1wMZcdO6gj8WLnAmsckoUDhF3Zh45Y+lslYtYi8UlLeRKeGrVRvcWlZUAn6pXpXAvSkdXeCXnI0lDXa1y+3tUplyrH3X\/yPypf+HceNLYkpjxEEsUxGbT5jEFYCDBLTST0W1M9Q1uhQYv2UFxerXHv\/AT+PncIsGpbJOBHUwCHgJBrCRKJAUENZs5UeKuwYkA0NDdTU1CS8wPKJYu8\/4BdSd3IzL5iuOcGQBEeynhUQQ9jGxsgcFerqjo8++qhse\/8BP89dxllENxDkFNPjIXamH0IFr4AcynLK5XBUqLc1lnPvP+AXfN8yTNEtBGENRtQwptnoh+YyYDLXLgY1lPXbKvqzOwE\/v4omKD4gGFVjmcGLV6IYfWYevneJ4++t9uOoCKuFpANk6NChYnMDP4BfWGrHOF14iMNqnNKx9MY5KDtRRAALC89iH27cPNXlHso6aU29Vxfwc6JaBYdxew8xhsP2nUE7YcUMaoW\/G1l\/xcXInIfx8Fbe+RuHrXtxGcrat0jvXV4yTjn3\/mPOz0nLhRhGvYeYhwDqqRc4at+v8NaLjp2kXoKdORCdpBNEGOmVdbrAOIg8\/aYhd3eNHDmSVq5cSXyHj\/qUc+8\/4Oe3dUOI72ZekC3B1145hLuHHbWD03m9eDg1eCg7of48cVbehEnnJW77mNWlZmpzlWvvP+Dn6KUpTyA3EOQS8jKZXe2HcQVneZorkFzlUDZJ1p1ZxfX79dVw6kVn5dr7D\/h56LLSXX\/gwIEesdU9i0HeQ+wWgpgX9NCoZYoS9QLjMlUzltkCfh6axcw9r07izps3j1atWiW27BjtcfSQravDVTl9QNCLyuHE4SHs5\/SSSLwPXUB8H67V\/u9wSoFUVQUAPw\/9wejwVTWZsO8hxmkyHhqtjFHYuuNz7uSpxuU8vbiMMsQua8DPQ5Oom66Noqvbd+TnZl4tD9kXozAE4SH2o2A4cSXsOPUkHOoZjgrxTxXwc9lGRpOznISc75ND3KjvIXZ7mgw8xC4b3iJ40r2ywSmRrJQAP5ftJZ0dU6dOLW7RkcCbNWsWsRdr2bJlkV\/CLqvh1jkCD7HLDlAInsQ1d95qWrmxAL+A2lbO8zH4eDFn1JafvhpuIQjniHVHgFc2oBclRskAfgE1BsOvtbWV1q5dS2vWrCk7\/LxagoCgplzYQ1knR7cH1DWRjIkCgF9AXUPCj2+oe\/zxx0Wq8uQK\/jkMh4ebosMStFcryqGs3dHt9qVFCL8KAH4uFWTv6tKlS2nTpk00evToYmyG3759+2J\/DzGWyZQavFxDWav9rHV1dTRjxgyXvRLBvSgA+LlUzciby515yZIl4ouBKJ0icb6HOI3LZMIeyjrtSk5OMtEfAOA0bYRzrgDg51yrYkj9fM2wYcMMLcHly5cX46h7GT1kGWoUN8tkuCBJOmo\/yqGs00ayO8CTp04GDx7sNDmE86gA4OdRuEqMVgnzgupQNq4noQB+8Xh7AL94tEOsSpEkCDLsJtXXJ+rYJ8AvHt0d8ItHO8SyFHGFoNw+xnN4vE82aQ\/gF48WA\/zi0Q6xLoUegvX19ZbQCXqtIMOOD\/XkfOM6lHXTgHB4uFErvLCAX3jaVlzKUS2TUb2ySbXurBrfydHtFdd5YlghwC+GjZKEIgXtIY6jVzbMdrA7uj3MvJG2pgDgh57gSwGGIFuE\/CWHw\/xd\/3A4fviCpv50o\/iZHRWX13+eykM9nRzd7qthENlWAcDPViIEcKKAkXOEJ\/YlGGUajY2N4sckOiqc6IAwyVEA8EtOW8W+pPo7iGWB2RJsaWmJfflRwHQpAPilq71Dra16BzFnJIe6OLY9VNmRuEcFAD+PwiEaFIACyVYA8Et2+6H0UAAKeFQA8PMoXJqiBXkHcZp0Q13jrQDgF+\/2KXvp5PFcTu8g1oNy2rRp4lh\/9YgmJzDldXCbN28u1n\/u3Lk9DoctuzAoQOIVAPwS34ThVsDNHcQSavqb7LiEEoBOYKqeis1HO+nPRwy3xkg9LQoAfmlpaY\/1dHMHsVFYPezsYMoWotHtd+pJ2R6rgmhQoIcCgB86hKkCQdxBrN\/HagfTiy++mBoaGqipqUns\/JCP2UkoaD4o4FUBwM+rcimIFwT81BNMhg8fbnun8YQJE8QdKTzHqN6RYnZ3SgqaAVUMSQHALyRhKyHZIOCn3lLGmthd6A74VULPSUYdUg8\/WBTmHdUv\/PTaOkkP8EsGOCqhlImEn9lL5LZBZDqqN9JtGpUe3m6OTp2XU7XQOzr4M8Cv0ntLsuqXSPjpr4pUX6y2tjaSSy3smoI9iOoNa\/rwRrey2aVZaZ8beWd\/+9vf0t13300dHR2UyWRElVXNrZam2MEUDo9K60HxrU8i4ad6\/l544YUiwFasWCEufNYvkGX59Ytk9WvS1CaCRVhSw+0dxPIfitk\/ILOlLtLJYeYUMYoX39eq\/CUrp3dcvj9Dhw6N9cL0xMFPP3Qyein01oX+d\/lCT5061bBx1El6XB5NpLeQze4gluHkPyEjBDiBKafzyCOPFO9CxiJn9zCVS4z4OLHFixc7SkDGmTlzpjAi5GM00rJK0MqwcFSQiAIlDn5OFs1awU\/GP3DggCuJMQS2lsvoZGI1hgpEO5jqL4U3stxdNV5KA7u1\/sycf9y269atox07dhT\/IZlJatcP4tSWiYOfHNLKYZUby49PEZZLLfbt2yfaT\/2vaPYfzuy2rZS+U6h2TBSwm7O2K6beQrca8TgZytpNF8XNIkwU\/FSrzQv8VNCZTbwvWbKE+EtdYAv42b1G+DwOCpgNW52UTd2J88EHH1g6AtX01IMr+J167rnnDK1DI++\/k3KFGSZR8GNxT5w4QcePH6dZs2aJ7U9uLD89\/NRTQ+xExrDXTiF8Xm4F\/MBPP8\/qti5GTkYnaZTztJ7EwI9Nar7564477qDVq1cHAr+6ujpHE7uw\/Jx0Y4QptwJm9wHblUvG++STT4pWm5V3XT+83b17t3gneT82Q1C\/NdHsUAqj0ZddWYP8PDHwk5WWwgdh+QF+QXYlpFVuBZw68\/TLkOTcoTq6sfqHb+Z9N4sD+AXUM4KAn9VOA8z5BdRQSCZyBexGKEbzbvJvd955J\/30pz8tWm1Wu6jMvMJO4asKg2Gvi25iBT82v7du3Uq1tbU0YMCAoidXb147ccfri4Q5PxeNhKBlUUB\/CKy+EEZwZJDxEpbp06f3Ok3HbOhrNlzl9Jubm8UXH0IrH1h+AXUHI\/jJLWryv4h02e\/fv58YiHpwmf2HxFKXgBoJyUSugN0yEy6Q1SEeRu+E0d+s5hUBv5CbXYUf7zTgdXv6bTR2E6lm\/9EAv5AbD8mHpoCTNXRWi56NQGc09LVKw+wzWH4BNbve8jNK1gp+VssBAL+AGgnJRKqAE6tPWn7sld24cWOPYSl\/ZjYa0i98tnq3vCy6xpyfi67iF35WLnwVflykefPmkdwGV85GciEPgqZQAavFxaocVn3fDH6qNcfvB78T8iY\/vdRmYITll8JOiSpDgTAVkIbArl27HO+5NTvowM5TzPWwcqhYzQUCfmH2AqQNBVKoAFta7e3thsNYoyGo0R3KUjYJv\/vvv5+2bNlCfC6mk0c6ExnAra2tpmXhvfT602Xs5uad5O8nTOIWOfupLOJCAShgrIATy89MO7udJarlp26DK\/fyMcAPbwMUgAKpVOA\/AQCMXVDGBFOHAAAAAElFTkSuQmCC","height":174,"width":232}}
%---
%[output:4bf89ee4]
%   data: {"dataType":"text","outputData":{"text":"最高点坐标: (Theta: 40.40, Phi: 70.00), 强度: 37.87 dB\n","truncated":false}}
%---
%[output:19555f9e]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAT8AAADvCAYAAACE79MwAAAAAXNSR0IArs4c6QAAIABJREFUeF7tfQ24FNWZ5lfdfYELXEWiZkAxJIPGn0REnEuYnTgas5kJmkkys8rOJDD4w6DkEpP1EpmBTWZ3ZB9IYAYDcb2DRlZmdyLJxmSUPE+ScaPGiFx\/QBKMEmMIRkwQEL383Xu7u\/b5vnO+qlN1q7qruqt\/66v79NO3u6tOnfOe029\/\/8eybdsGOQQBQUAQSBkClpBfymZchisICAKEgJCfLARBQBBIJQJCfqmcdhm0ICAICPnJGhAEBIFUIiDkl8ppl0ELAoKAkJ+sAUFAEEglAkJ+qZx2GbQgIAgI+ckaEAQEgVQiIOSXymmXQQsCgoCQn6wBQUAQSCUCQn6pnHYZtCAgCAj5yRoQBASBVCIg5JfKaZdBCwKCgJCfrAFBQBBIJQJCfqmcdhm0ICAICPnJGhAEBIFUIiDkl8ppl0ELAoKAkJ+sAUFAEEglAkJ+qZx2GbQgIAgI+ckaEAQEgVQiIOSXymmXQQsCgoCQn6wBQUAQSCUCQn6pnHYZtCAgCAj5yRoQBASBVCIg5JfKaZdBCwKCgJCfrAFBQBBIJQJCfqmcdhm0ICAICPnJGhAEBIFUIiDkl8ppl0ELAoKAkJ+sAUFAEEglAkJ+qZx2GbQgIAgI+ckaEAQEgVQiIOSXymmXQQsCgoCQn6wBQUAQSCUCQn6pnHYZtCAgCAj5yRoQBASBVCIg5JfKaZdBCwKCgJCfrAFBQBBIJQJCfqmcdhm0ICAICPnJGhAEBIFUIiDkl8ppl0ELAoKAkJ+sAUFAEEglAkJ+qZx2GbQgIAgI+ckaEAQEgVQiUDfy++Y3vwkrVqxwQL7\/\/vth1qxZzuvt27fD\/PnzndcXX3wx9PX1wcSJE1M5MTJoQUAQqC0CdSE\/JL69e\/fC0qVLaTRMdEyAL7\/8MixcuBBWrVpFhHjixAlYvnw5nbty5Uro7OysLQrSuiAgCKQOgZqT3+HDh2HRokXQ29vrkfS+8pWvENhIiEiO27Zt8xAdEuLq1avpIdJf6talDFgQqDkCNSc\/lPLWrFlTUoU1iZBHHEaaNUdEbiAICAKpQKDm5MdS3YIFC+DWW2+F\/fv3E7Cs8rKKO3v2bLj22msd0Jn8rrvuOs\/7qZgVGaQgIAjUHIGakx9Kdffccw9cffXVjlprSnXo2ED7npBfzedabiAICAIGAnUhv\/7+\/hFqL0uESHzo1BDyk3UpCAgC9USgLuT3+uuvj\/DaIvlt2bIF1q1bB2vXrhXyq+esy70EAUEAak5+QZ5cxJ3JD2P57r33XpoKDoXB\/8McHvPmzXOmbcmSJdDd3S3TKAgIAoJAbARqTn4YsrJs2TKK4Zs2bZrTQbQFskT48MMPB4a6BF2HKvT69euddjZv3hx70HKBICAICAI1Jz+E2B\/K4g9q5teLFy8mz26UIGckQZQCkfxE+pOFLAgIAnERqAv5sZrL6W2TJ0+GjRs3eiRBf\/pbufQ2Ib+4Uy3nCwKCgIlA3cgvadiF\/JJGVNoTBNKFgJBfuuZbRisICAIaASE\/WQqCgCCQSgSE\/FI57TJoQUAQEPKTNSAICAKpREDIL5XTLoMWBAQBIT9ZA4KAIJBKBIT8UjntMmhBQBAQ8pM1IAgIAqlEQMgvldMugxYEBAEhP1kDgoAgkEoEhPxSOe0yaEFAEBDykzUgCAgCqURAyC+V0y6DjoJA0f4JFN9aD7kJ34hyupzTYgiUJD+uprxr1y5nWOZGRI0cq1R1aST66bh3Pr8G8kP\/CKMLD4HVNTMdg07RKEPJD\/fbnT9\/Ptxxxx2erSOx7t5dd901oh5fvTET8qs34um7X2Hf9TB4yo9gTNe\/QiY7O30AtPmIA8mPKylPmjTJs68GY2GWoO\/s7GwIREJ+DYE9VTe1d98CJ37vB9DR+VnoGHtrqsaehsEGkl+5DcPNzYcmTpzYEJyE\/BoCe6puiuR3fOIjMKbjc5A9fXGqxp6GwYrkl4ZZljFWhACS38mxj0H2HbfAqFOWVNSGXNS8CITa\/PybCvEQxObXvJMpPUsWgfxzMyHfdRKy71gMHRM\/k2zj0lrDEXDIL8izW6p35TYYqvXIRO2tNcLSPpJfYdwgZM64RcivDZeDxPm14aTKkJJBIP\/ELLBPOwHQNRM6zrkvmUallaZBoO7kF+ZJ5tAaRqacZCmSX9OsobbsiH3sabB3LIHiqYr8clO\/3pbjTPOg6k5+vD\/vTTfd5ITR+Dcxl03L07wkm2Ps9rFnoLijB6BzGKzxMyHz3n9ujo5JLxJDoK42Pya5\/fv3g0l+SIjbtm2DlStXAscN4rmrV6+mR1A4jUh+ia0BaSgAgcE3n4bc7iVgjcoDnHqpkF8brpK6eXtZmps+fTo8\/\/zzYAZQY9A0HkuXLnUgZgdMb28vzJo1awT0Qn5tuBqbaEiD++6G4iv3QSZXgFGZLFh\/uL2JeiddSQKBusX5cWD0nXfeCWvWrHHIj0lx9uzZnjS6coHWQn5JTL+0EYbA4N4+OPHLjdCZy8CobAasP3pKwGozBOqS4WFKcejIWL58uZBfmy2kdhvO4Csb4dArX4MuqwPGWaMh86En2m2IqR9PXSQ\/U631e3tF8kv9GmxKAA49uxDQ7tcFo2AsjIHshx9vyn5KpypHoOZVXTCEBdXcvr4+clwI+VU+WXJl\/RD47XML4O0j\/XBGYRyMtkfD2I+I5Fc\/9Otzp5rX80Op75577gkcDcfy3XvvvZEdHuvXrwe09y1ZsgTmzZsHmzdvhu7u7vqgJXdJDQJMfhMKo6HT7oSxl22A7MRLUzP+NAx0RKjLdddd53E8JA1CUJBzWKjLsmXLYNWqVTBt2jSnG+zo4DeE\/JKeIWkPETj47E1w+K0n4ZRCB+QgCxNm3gM5Ib+2WhwjJD8OQsZR+guZJjHyIPLzF1GIEuSMUh8SoZBfErMibfgR+Mnj74cz8gXI2lkYX8zAxMvuhY7TLhOg2giBkmpvLYgwLL3NvBfiK+ltbbTKWnAoSH5ji0Xq+akFGyZdugk6TxPzSgtOZWiXI6e31YIIqwFS4vyqQU+uLYXA8Tf7YedPb4BjmQ44PT9EJDjhPZ+Bd07tEeDaCIHI5GeOWSo5t9EKkKGMQODokX7Y9rNF9P7Y4hB0FQpwxrt7YJKQX1utlpLkh2Eqjz\/+eOA+Ho1GQSS\/Rs9A+97\/wNs74Ee7l8AYewhOLZyAccU8TH7XZ+AsIb+2mnQP+WFYCoaN7N27F9Dri1tWCvm11XzLYCIggOS39ee9MLEwAKPsAnQVB+Hsdy2GqefIPh4R4GuZUwLJ77vf\/S4N4OMf\/zh5VG+88UZYtGgRkaH\/KOeYqBUSIvnVCllp98VXN8K2\/d8gyW90cRjG2oNw3pRF8J5zbhZw2ggB6\/jx4zbn2uK4Lr\/8cqqicvfdd1N5qQMHDhD5YdjLihUr4JFHHiHJsKenh3J0586dG1h1pdYYCfnVGuH0tv\/tF\/8WBo48BQOZsTDOPkES4JmnXAqz39eXXlDacOQk+XH4ydatW0cMEevuCfm14czLkEIReODF\/wr7BnbDWPskPcYXT8Lpp1wKV160QVBrIwRGqL0s+eEY2eEh5NdGMy5DKYvAt178O3jh6CvQaQ\/BePs4dNqDMKnrEviTC\/+p7LVyQusgIOTXOnMlPa0TAr3P3Ag5Kw9d9gkYhhy8s\/gmWFCEm7u\/X6ceyG3qgcAI8vMXIRC1tx7TIPdoJgR6nr0FJthHYcjKkbMD7X6jIA+3\/sFDzdRN6UuVCBD5mXtr+NsT8qsSYbm8pRB4aeBl+OqefwQLbMiATWrvGJL\/CvB3l21xxnLw7ecAbBtOP3VmS41POusi4Hh7sYw8enHR5ofhK1u2bIGpU6c6oS7i7ZVlkwYEkPxW7\/ka5CELnTBIJHgGHIECZOGm8\/4W3tN1PsHw6O4eePPtp+G9U\/4GLjz7xjRA03ZjDI3z27FjByxYsMAJdZE4v7abexlQAAIP7v8B\/O\/9P4ax1gkoQhbGwzHogDyMgmG45bzbYVrXeXTV17dfBaPsPPzeKdPhwxd+VbBsQQQCye\/48ePw0Y9+1PH2mruqNcsYJc6vWWaivfqx4qVNsHPgN5CBInRagzAGBmEcHAcLAP5k0sfgzyZ\/FH4z8FP4zs+XURA0HjfN+mF7gZCS0cTO7cXwl\/nz53v23W0EVkJ+jUC9\/e\/5X158AH42sBeyUIAOKw+jrSEYDUMk\/V0zaQ78xeQPw6sDP4N\/femLMMoehg67AJ+4YDWc1fX+9genzUbokB8HOkfJ2ODd2DAPuFFSoZBfm63EJhnOFf3\/BDlrmMgOxb0c5GGMNUxkeO2kDxH5Pbb\/W\/DD178Lo+1hIsC\/Ov+\/wZSu9zXJCKQbURHwZHi8+uqrcMMNNzgqL0p45Y5aVHsud0\/8XMgvCkpyTlwEZj3VBx2ZISjYWejMHCcv7xhrELJQhLmTr4D\/PPkK+P7+f4N\/f\/07MNoegqxlw5WT\/hyumPwXcW8l5zcYASI\/VGUfeOABWLlyJeXz4hH0XoP76rm9kF8zzUZ79OWZt34Lt7zwENhgwajMIGSsAtn+OqxhcnhcfMo5sPq9n4a1e74Krwy8qNXhAlw16RPwkckfbw8QUjSKkltX+gmxmXAR8mum2WiPvvS9ugPu2rcbstYwZKwi5DJDRH5ZK08ZH9O7psC686+D\/7Hnbnh54CUlFcIg\/Omkj8Gcyde0BwgpGkVFlZybAR8hv2aYhfbqw\/U\/fQSeffu3ADZALjukAp2tApFfFvJEiD\/uvhX+Yc9GwHhAPBElwou63g23nffZ9gIjBaMR8kvBJMsQoyEwb9dj0P\/WQSI6y7KVxJcZVpIgqM2Mtn\/gZrjmmX8gqa\/DKsAoGIILut4DK85TZe\/laB0EiPzYextUrLTcUMThUQ4h+bxVEDj70YchY9kwGm18GRXDl0XJDwqQyQyTJHj3RdfA8pc2ETmiJ7jTGoKibcH\/vey\/t8owpZ8aAYf8br\/9dsCHuUF4OZQwJ3j16tX0mDhxYrnTE\/1c1N5E4Ux9Yz85chg+saMfslaRQlswrxfj\/PA12f8sVIMBet51Pmzav42cICgNYhwgOki+d9ny1GPYagBURX4oMXLObyny80uWkydPho0bN3qIloOnGcBy5fGF\/FptqTV3f\/\/0uefgySOHKaQFpb9RVh5GWUq1RWcHSoBIdrMmnA4vHPsFDQYJEQkSJcK15\/8lOUTkaB0EqiK\/KJJfUEA0bn151113OQTIVWVWrVpFJfE54BphNMNvTFiF\/FpnkbVCTz\/0zE\/hybcOQgeWMNDkh5Iekt+YjJLyUAqcPeE0+NmxVyCDqrClPMF4rL\/gz+ESIb9WmGqnj4Hkh7u4+ev6maPCMleY2RGF\/FCiW7NmDfT19TmqMZPbpEmTqB0kw23btnmIrlzbQn4ttc6aurOPHn4brnxmN0lwFkl7eSI9ywIYrSVAlATx89GZQejMYa6vUoeR\/PD9G87qhoVnX9bU45TOeREIJT8sZ3XttdeOIDgkRjyqTWsz2wlqkyXG3t7ewA2ShPxkKSeFwGOHBuDK\/pcArCLYqN5mkPhsnd+rvbqk\/hZgbPYkjM2c1ESJBKhU4oVnz6SHHK2DQEPIjyU\/rCF4zTXX0C5w+D+SLR9Mfrh\/sPk+fy7k1zqLrNl7+pGnfgmPHh4AG4pQyOTBxkC\/DKu1ivTwMdoqwPjMIIzJYFqbkvjo2bJh0ZRL4OYp05t9qNI\/A4GGkB+qwsuWLSOb31lnnSXkJ0uyoQic+m+7oWjZULCK9MgjAVo2SYEWSoGA3l4lAZ6SPQmdSH7aBshB0LdMuRhuOefiho5Dbh4PgbqTn1+iM6VAkfziTZ6cXT0CTx48AZ94Yp9DfsOZAhTBhuFMnogQCRAyRbIFonp7eu6YDoNRr5EQ0RnyB6eeCf\/r4iuq75C0UDcE6kp+fkcHjlLIr25zLTcKQOA\/PfY6PHnwOBStIhQzLPkVYChTAKRBIsBMAWxLEeC43AkYg5kfJA26EuFlp54B\/2f6BwXjFkIgML2tnLcXxxc3s6NU+Eoch8f69eupnNWSJUtg3rx5sHnzZsC6gnIIAnER2HbgJMz9f78DO2NDgQjOVgSYKQL+5TNFRYJIjEh+qO5mB7UKXKRYwCwGQ5NqbMOv\/\/jP4nZBzm8gAg751bKEFcfxzZkzJ9BLHBbqgnZBjP0zs07Y0cGYCfk1cPW0+K3\/8ocHYNuBQUV6WZTw0M7HBFggVXjYKhAZ5q0iDGeGwc6dVDF\/GUV+LAEi+R248qMtjki6uu\/U88PCpevWrYMrrrjCqemXBBRMfDNmzAgNWOZzFi9eTJ7dKEHOKPUhEQr5JTFL6Wtj+28H4a++fxCKGZsqNiviQ9VXESFKgOj1zaMNUJPfYHYIhrKDAJaq9oJHR0ZlguDxvRmXwh+dVt80z\/TNXHIjtg4dOmRXktcbtQulVOirr77aIUSU\/lasWOE0K+ltURGW8ypB4NNbD8P23w0SwdlZW5EfEh9LfygJogdYkx+qvsdyQzCcHVbXIOGRHVDZ\/VDy+8HMS+CPT5tQSXfkmgYg4LH5scS1devWyF0pR1KRG4p5osT5xQRMTncQ6N8\/DPMeepPIDqU4RXxIdgBFJEL9Hkp87sOGox0nHfsf2QEzw9QmhsNgbb8vvecc+NLvnyNItwgCUs+vRSZKupkcAhduOOTY91jiQ\/JT6m+RCNCRAJEUs0oCHMoNA4bCIPFxSAyqyigFImH+\/e+fDV+adnZyHZWWaopAWfJDR8i+ffsCsyxq2rMyjYvk10j0W+fe963\/GXV2R\/8bcP2Si2D9r8dB\/2vDjpqrJD5X+kN7H2RA2f+ymgxR\/e0oQCGLxIcPZQtkAiTVGCW\/aZPhS+dObh1wUt5Th\/zMslNccgrLVC1apCrUmoUJ8HWcrS5rgbGQXy1QTa7NzZvepsbmLTgluUZLtLSz\/w1Fctt\/R0S3s\/\/AiLNPzjgPTs48V6m7SHr0MNVcrf6yGsx2QJQEs0XI5\/LKIaLDYtADjA\/0BiP9ffAd4+HfPzCtLuOVm1SPgIf8\/I4PdFYgyVx11VWwZ88ej7dWyK968Nu1hV07T8LSWw8AlUUBCy6eMQbm\/XUXXHzJ6ESHvKP\/ACDpbdrwc1+7Nti2TXY4Pk5efD6cvOR8gCw6KkCTn5cESdVl4uPQF+39LWSLUMgVtFNESYsYCoPxfygNYjxg1s7A2x+T\/XsTneQaNhZKflxc9P777ye1119ySsivhrPS4k1v\/vph+JevH0FPAEAmo54ti8ivWhIkwtt+wCfdIZvhLTKBJDh00QWA5Ie2O3ZwoOQHJAFqaQ\/VXSxtoEmPiDCryQ5DX3Iq\/o9sgkyS5A1GIlSZIHgc+vgFLT576el+IPmxussVVYKCkIX80rNI4o60d\/E+2LXjBEAmqwjJR4LY3vQZY+DTKA1OH1WyeSY7POm+DbvLdMVLgsXzLoDh8y+AQhagYISzEAlqdZcIUIe1oFRI4S5EgFq6w\/+J\/AzSI9VXf65VZ5QA8Xjwg1PgD09Xe1\/L0dwIBJLfgw8+SOou2\/mE\/Jp7Eputd7fd\/DLseu4EkZ5lZQGyOZL86H9DErRQLdYS4afnj3eIkAmvPNkFj9yacAbA6e+E4nkXQjGD4SugChcQAepwFi29OZJgDtVhlPa0A0QTIAU84\/85DHY2YgKp6otSfZkwsTe3XTARbrvwtGabEulPAAIjyA9zZjdt2gRz5851iogK+cnaiYPAh2c+DWCx1Jd1STCj\/8dnkwTpf4veKwweoVvlj+wl21zhxJuRbp055XTITDgTrNPOAHjHmURydsai5wISHj4jCZLUpp7d4GYVzqLUYCXp4TNJd0R+NhTJ3qcdJPgemTN1SIyWBLG\/t11wGnz+fRLoHGnSGnwSkR+qsFu2bIEnnngicAc3Ib8Gz1IL3X7nM0fgtpt+6hIekmCmg8jQkQRJDcb3A0gwi+cpiVCpy+r\/wsk3NUHyZ+i0UKSZQUkvi0SHBGpRsLJNrzFkRf3vkp+S\/JgAC0yCSGZaxTVJEMlPOTq0BKhthMpWiO1zPrCyJ37ufRPg8+8\/tYVmLL1dtY4fP25jJWXM6gjaVQ2h8aeemXChQwQ3Har3IaEu9UY82v12Pn0YPn\/jc2BZOaXmkuqLai8SVU4RHkmF6jMkMKUam68t\/VoTYNYlQSJDJD3z2TKIL4ufk\/fDI\/0RUTEJamlQBS8b9kC2C2rJj8kQHR+o+hZyGAhtZoSww0RLklYRPnDmGPjGfzwzGlhyVkMRsH7xi1\/YCxcuhC984Qvw7W9\/WyS\/hk5H69\/8c3\/9E0DpDwnNIb2MIkJFeAb56fcdEiRJ0CBBIjl2mGiJzyRC\/F+ry4oMDRJEAjReo6RnZxUpmlIg2QEtgHxOPaNjg6RFJDokQXKCoMOjoFRgkhSVXZCdJk6sIKrONsAr8yXLoxVWcqjD4\/LLLxebXyvMYJP08b4NP4dNX+N4OwssUETmlf5Q1WWJMIwE8RwmPKUW02vDLuhIfkh8AWTI6i+RH0qClLam\/kdiQ8mwqKVApf4q6Q8dI\/RMHl\/tBUYC1PF96PRgux+nv\/FrpQYr8nv5esnyaJJlWbIbgQ6P1157Db785S87++qKza8VprK+fbxvwwsUYIwHZlXgQd5bZBZ6xiMDFr72kCCSHqq\/wSRIajKrxiYJorcYX6N6bEqDrP4SERoqMavC9J5PIkTpj1LXvCTIITFIbESCOU2AKAVq5wfF\/rFHmMJltN0P39dOkH+Z8w6YNal0CE99Z0vuFoRAaJAzZne8\/vrrlNXx8MMPS5CzrB\/gPFk3BAXDVziwmLMqYpIgkR07RAxJkd5jx4h6dtVjU93VUqHHFhhAgvg5Sn5IhliACslPS4IoBeJrZf9T0p+SBNUzoCpMKrDrCTbzgVUBVLb\/2bBkxnhYctk4WTFNjkAo+ZkbDeEYJMOjyWeyBt1jktuxHdPIRubKqlsiASpCUUcpElSSoSv1aWcHOUf8JIik1mE4RrQ6XMomyI6QEbZA5SBBSY8cIUiA2jusVGQlAVJYTE6pwXkiQXRyKAmQpD1yhHBKnPtsqsI4\/nFP\/Qr+Q\/EYFVKY0S3OjxoszUSaLJnby1tMov1vYGBAcnsTgbx5G\/FmU7xgkNrIXNngUZiSIIBtq6wHhxw96WeK7FyboFaHo5Kglgo9NkG2DXpUYfYMK2nQYw\/UxEf2wJwiP8cWmEUnCJKgJkB0jNDDJUBOjVMeYBUPiP93\/OZNOPXbz9LYL+k+U0iwSZd8yaouvKcu9h3V385ON21H0tuadEZjdita+hiTGpKgIrTSRykSNGyCWHxAxwF6SZBjAzlMxgyYVk4TFScYoA47BGiow44k6KrDRIJmbCA7RzgjJGuRCkwEmDMIUMcFIslxfrCnSoxlg2VbMPF\/\/sCRhJkEZ3SfAdcvkcIH5VZPvT4vW88P99d48sknqZ6fSX716mDYfSTOr\/IZCC4OUK49Vm9RouNqKW7VlCiSIKvEIxwjRIJIdB2K2DgtzhMXiO9rmyCqvlp99gdLW+wQMZ0i5BFmO6HhISZJUAVLkyqsYwEpHhCzQbIWkZ8iQbT\/oeqrYvrMogh+VRg9HxP\/+fuBpgBFhEiCohKXW3G1\/rws+dW6A5W2L+QXHbnKyK5U+2zfw3PKESApvoF2wWASVASnnCBsE+T3kADVQxVM0FIgnkf5w1oS9IfHOISoYwVzOotEe4edzBD2DJMKbEER7X8oAZL663WEqMwRVfiU1V5yiGRQ8gMYv\/UpyL1+KGDsjKutVOKeC4kM5ag\/AkJ+9cc89I69N79CBvl5C8+E6ZdW7i00qxeHOyqSGDiSYBTy43t51WGvc0QTEvEpS4IqJMbNCOE4QfWeKwnqsBmdLucppuCowTpExu8VNojQplAaHQ+onx0CJPLjeEAjP5jj+5xnVQ5\/zHN7YMzOl3wgc9UZ98cDpWhUhxeIJJjEgozVhpBfLLhqd\/Lzzw5A782\/VNILWER+8xa+E6bPjEaCSHhhFYxr1+tKWw4mQa8kyMSK5\/rV4QASNEnSyRvWGSOmJMhkyHZADpLW7zue4JxShdXDlQDddDgMjVEkSBkeOjWOnCAA0Pn8Hhiz60UCKKqZQJwjla6nyq4T8qsMt8Svum3hC\/D8c0fdaig6WHj6peMDSbB+0l3iQzUaLEOCpDHjjwHSSYh3mAKotSToeI+5aIK2ERqB0WQT5PqCSHxMfmwXJMeHlgBJ\/WUSxNQ35QQxK8VwNgg\/qwwSG3K\/PQTjH\/lJAHhuKFCY1CwkWMs157bdNOTHlaO5a+W2xGw3m9+Hpj+mMhvIPsZqHdq8VKoXkuD7p1vw7JN7S8Tc1WfRJH0XFSjttSOitORKgm4gNZGgr0ACB0krjzF6gt3sEeUZVrGFKi1OZ5CY2SJa3aXPNSE6KjASIEqBRIq6OIIOiHZJUAVEO6Wy0LRYBDjlm98pA1VpIhQSTHqledtrCvJDjzIWV1i1ahXlE3MYDXbVH2LD3W8n8qNKKDc84yT++8M+3Np4GSgOH4XC4CEoFo7XdmXUvXW\/U4QURlIZnWrQRiC1S4KuTdB1kuj3dOaI40Fmp4iW\/hzPMEuGLP0xARoSIBIjpcTpeEByiDiFUt1agVwoAZ0e4771LYfUFZGTKBuCbOlYSiHC5BdkU5BfUO4wEuLq1avpgWX1\/Uc7kZ9ZCcXv4TRfKxJU2RTF\/PHUkKAnWBqlOw8JmsUSVMksriajyJCdIa5ThCRsJMIgKZBVYXrO6mwQHQqjvcCOFGjWCtRESPUBNb+N+vGjkDmo8p8NZUtRoEOGQYToOpFce6G6RjzEyZFgU5Af5hHjsXTpUmdknF7X29sbWC+wHcgP08fcncfcunec2M\/E53yZuQYe2QOVqlgYPAiFocPJrYimaclvDzQzRtgz7EpTngIKpPbqMlqcN+yxB3KQtEGCJP3ha1f1Vf\/7CNAJhuZ6gargiT5iAAAZWElEQVRiDFWE0RViUCKk2qgv7gZrT7l9R7zk500VLD0ZC3oupFAZOSpDoOHkxyru7NmzPRujm7nFGGDdLpJfcHEAMzdWFfDEklCsrnlCPbQN0K2NxyR4qA1JMEgVVuqwOjhbxEuCyr7Hkp5KoVP7h+D\/5mfaMZLtUFIgEiA\/0P6XY4eIekbbn1MmiytGsy1QB0gzAdKMvrQb4OUo5Odf3f6QmDBVWV0nJCjkVxkCNb4q+mY8\/u0XXRIcWRLKrJOnvZeoChdOQGHoTbALJ2s8qno3P1IKVD1AEtQ\/HJ68YeU0crNF8H9VOcZRhalAQo6Co+k8UoN1ylxO\/U+SHxOgtgeqtDgkQeX8cCpG66wQtgVy9lv+h1sSAKs0+TEW1\/dcJOlzMdAWyS8GWFFPrT7mziTCaCSovMRKCiwWTkJh+EibkWAYAZqzosve01tcUJW9w64TREmBo7Q0qMkPJUUkwCxLgywF6hqCJPWph0qF4zqBumI0V4lBpwjzcRFg+JFvRl02iZwnjpHoMAr5RcdqxJlY0BMPKuppF0kOSTajws2nVSqeUodNSdBxiFAIiD5HS0GFobegMPxWFSNstkujEqBZTFX\/eBieX1f1RXVXSYWcFcJSoKsCIyHqPUU8MYFYFVqToEN8XDRVy6RFgPzOR6H4lt\/pUXtchQTLY9xw8sMuxnF4rF+\/nvYUxi02582bB5s3b4bu7u7yI03wDNNu541Ri1r6KW5n\/LYvZetSRMh5rt48WLeasgX5k2+AXRyMe9MmPT8KASrJT2GAOPGPAuPF9j+tBnukwA5HAqQfGVSBHTugVnmNrBAPAXJuMJtwiwCFX++G\/Ktc3r\/+kCIJSjWZYNybgvzCQl2WLVtGsX\/Tpk1zes9eXn6jHuRXXo01JTSVlqVCFOLkvUb5YpiGcNfWNXK\/DLYDamkQe1IchMLwANjFoSg3avJzohIgk6AyByictFddh8QoqQ9tfJhCx5Igq8DaDugQoKv6ktprlMpSaXFcL1DDZwMUj7wBQy883hR4ik3QOw1NQX4c5Lx48WLy+JYKckbJb8OGDc4oUOpDKTBJ6Y\/V2U2k1sYhsrCqxkmToLZpORWUg+yCumqyrn7CsXFIfsX80TYgwbgEqPEgiZA96UryU5WlTQLU75EN0LUFekJhWOU1ttHEAGjeL4R+pnDaCzaceObBpiA\/7oSoxAqJpiA\/7Ih\/b+Cw9DaU\/MyD1eBqSRC9svet361tdsEhFirYNgqR+T23HKMW5dq43xOzr2zwx\/e8m4Qr9dwtJFrMHwN8tPYRHAsYHCun7YB6kyU3\/EXHBDqSn5YAyRPMKrCWAM0wGH+BVPYGszOEyc+2YeilJ6Bw9GDTQZ12Emwa8qtmZSAhmiTIRBilTSS9W+f9KOTU8DizaGrtyOujE2iU3vvPMSuhaPLzOEhYGlQqs10cpjQ5fG7dI6g4Av5I+d6nzBDNSFxB2imIygHRo7QDBD3BWhIMIkBdOos9vmZOsKMK872KNgzvfxGGD\/wi4g9n\/WcirSTYFuTHy8UkQXyvp6eHVOJSB9rz3N3Iws5sNRLkcWgCQDVPkyCJ+yQVKiIkOigOg104Abadr\/83L4E70niMw\/yB8TikzPNQivc4jTjkRUl+qowW2gBR+vNJgFwUgarDcNyfjgnkitHcn2IRCm+\/AYO\/3p7ASGvbBDtHLpmFTpL233gpUfILy8dlG97WrVud2bvjjjs8GR1xq7qUWwambbCcSuxVeUu1XD0JKvJhd2Ace2K5EZfrN6fEmXvpcjUVvbF3MU+xga1HgmGpcGxmcGMgVVogC4F6PxLKADHi\/cj7i69HKQLM4rPhBKHK0cb+wVwQwYkF1Co2SpjYhXwBTuz5QY2cYNWsC\/+1bjD1jFnvbPsq04mRH6ejIZx9fX2eYgTmHsC4DwgT3f333095u5VUdYk65awOo1RYHxKMtsFPY0gQiZdtfzpm0IkNdFPEkPwwUBrsQlSYG36eX\/pTjipzLqIQIIe+uMTnECA7RnQmiLuBki6dT9kgSIgcYqMrUpPTowDDB38J+SN7PVt7RrMfJwVtcJZIlFxi2nOkDcvtJ0J+ptTmd1SEFShAQpw6dSpJf5VUdYm7JPx2wVIe4molweh2vXp6h0f+yvM+GGYsnCsZodRSJK+wTSRYC2dN3FmMIN2GqL\/q7egE6Ki8WvpzpUDOGTb2C+GiCEh+SILmUbQBCkUYPvwK5N\/6tQ\/H6iR\/d8N419abJKL+ttqNBKsmPyY+VGPx2LJli0fyQ6kuKF7PJDwOXYlT1aXSSY5DgniPqDZBryQX37tr2qZciaUehOMlBAoIJq8wV1Emo2ALkGA51VfTn2P3M1PhcIxI8DoMhkJhlMqrHugI4ddGRRjeOhNjBpEAKSfYR36o+hYw2jkPJ17bpshvRAxopSQYZoap9NtR\/jpcm7jzHMYMtvpRNfmZACCh+ckPyXHNmjUjVGE+d926dbB27VqIW9WlWuDjkmAcadDtW1zy8i7m6BJktWiwVKSdIDpHGMOC2SmiJD\/8kha0TTDu2JLoY+k2Rv6ABPXRIEmzgrQTxqQIMEOEZ6q\/o3U2jSY\/yg\/WpEeZIGgbxPxhZT+lgyDDSOcC2Pk85Adeg\/yx\/fxBgiSo5s\/9AXZfV466WWWaB+O21g4e4tSSH09jXA9xdBKsfNl5wzQqlQoqvT8TMMcEom2QHTT8nUZbWlFJhJFjHyvtT9zrWAUsdZ0pJZpMpa9hBwiXwfKovsbOcRTyouIBUeqjVDguUsp7G9s22IWCIsDhQRh++xUqROuaEZKUBIPGHKUijHldvB+1VibB1JNfKRJEZ0xY5ki9SbD+JMOShDcsRqnFShJQ+5QhCaI0WIt0vrjEF\/X8ctkhnAWiUt9I9UWPr6XKYnkyRCgMpgOgA9XikSov4YI\/EAWU\/oagOPgmFE6+AUWKrTSJZiQJ1n\/Oo+I38rxWzCEW8guY79qEyVS+sHjT78YRDGeMGIUCDN2OSJBsZigJtgYJjtw0yZwfr+2PKueQ9Mcl8rk8liqQkOkYA4BqL2OC0h\/jgA4PlPqKKP3lSepDAsTsmpEhRbWWAqtZg9GvDZIG8TuFB9v3WahIOjU1ei8TTm8Lsvk1m8MjDjhx7IL1kQTj9L4W53LQtBsf6EqCKMfo9D9ShzlMJp4aVYteB7dZyllgkh8SnbkbHBeSVVJgJtcJFpIfSX3a3kdD1kRW1Fig5GdjHOUg2JRaeBSKBay0wz8WpnqqpUVPx\/2hO\/VDqtI7ndu9G57of8C5HAkPH6hRMRm2NfmVCnVBVNDDG6eqS6UTUc11cUgQ7xPNQ1xNj2p\/LYdRBG+4bdoFuWKKqfKhLKhsgm68XfOSIKLpho3QK6Xa0lYCSs31bCxFEiBmf4yCTMdYpfZ6gqfZOaRJjAkQM2nocZIkQCJCzKoxnS0+wvPOdL3tv5Wvs6PwIxiGX8Hf9FxDjZTLtKr8TpVfWXO1F7uGMX1IIBz8HBbkHKWqS+VDrf7KuCTYytJg2F66XjtVeeeIQp0dI\/ysJaPqpyShFnwOEPZ2s32PAsE5L1rb\/LKjIJMbD1ZujFM30O2MT+pDciO7KKq+SH5DFETukl\/8OMpmswcOwa9gCPbCkp4eQOJrRrLzL5a6kB9Lf7t27XLu709vi1rVJaHVXlUz6SDB8FQ+oi6Pba8UCRoZD3QhVrwu6GckhUbbCP0FEIzaf9hflPJIEtQFZGlD+RxkOsZDJjfWK\/HheXo8RE4k0WGMJI5XPUj11QSonoe1vTS+ZNxIAmSym9XdDRd2HyNVNsmyclV9QSNenCj5Rbxn25zWnGEyycJb2jEQpIaZQdNGDT1noyFNhkx6GAqiPcbelLT4ZBBv5AHkbsb9cQVo7DcWQDXqAFq5cST1eVRdujk7LFTQtCInTfBIevheEdVcJkB8D18ryc+zP3Ho5uYjR+naV+MhEPdsJrxFPdcA\/h9FugvK6+e0Vr5\/I3L\/8d5CfnFXQMD5QSTY+DCZBAbmNBEkBXpJLEgK8dYQZE+xETjNMXFmbBzbCTmOkMmDyaXqYZUKc3FL36vbcAl8twBqJtdFTg63TL4OZHb6hao9Z4zoFEH2hGsJ0LaHHQ+wkoKRBP21IqNnb9RSAiQVtkJVNqgosd\/khbA1KvdfyK\/qL5O3gThhMnhl+RL5CXewqub8X0gzRYylHn+4i5l5YO6zi8TCwdOGasxEqHRr3VtNIk5IjfYsm59HHle5GD9FeizV8f4fGZL2xlFQs7NdpnNPrbpTf5Sk50pjrtpL7xn2P\/W6dIhQaclbSZtJhhch2SWlyoZFephkhwS5aNEi6O3tJdWZj3rk\/gv5Rf7SxDuxertglEyFeH1K7my\/g8DMAAkjQS1JOeSmvcNGXUEvGTIJBWQoGCoz2dQcdVNVcfFWcwkate1s8xmKCeUAW5DJjlXhLLT7my+I2UPQeF8O79GOHQpwNsjNT3wRs2TCnU+K\/Ko5KlFlq7kfXmtGd7z22msNy\/0X8qt2JstcXwkJbsJy+k8fasLUMf9gTUnQrzLqL2ZoChxfi+eZRGiqx3w\/g3RMyZD40QIomiqjIgMVc8h98GVSBM4ZkyzW6sPMjlFgZUerUBdfep\/3ctdho5wcfulP\/xiwt5dK\/LHEx3a+6ggszhL2e2Ub4agwd2tsZO6\/kF+clVPFuXFJEPcC3rH9dxGqTFfRqcQuDQoVMRtn21epPVBKkyG3pqRDYr2IvXeJRaXjmQer5G6BV4rvc+L6OJg76FZm4j8HJRuByGzD44BvJj2SUrUXuE6e7iRV2Yigh57mr90p5Fctoi10fVwSxKG1TtB0SMUUZ36ikKBBc4EqMhOfS34uIZYjRa64oqRUJ4TF3OPX9Piajhh\/wQJSr41ir07hVJ\/Nj73ZdSS9RqiyUb6C7ACZNGkSJTfgIeQXBbk2Oyeuh7g1SVAT1AhbmbKFxTPUBzhbiOu8FWec0k5BqqoTrKz8uB4bnrkNaMm1xs4NPolVXYMMKZaRz2P1u3Z5z82gykb5evq9ukJ+UVBr43Mq2Ye4NSRBV6V0w0J8qmrF6p+pIvvUYB8ZurnHTMSaLLUkNyKtLXStaZVZe5hV+q4r+XmlQDevuVaFaZtJlY3y9fRnefE1jcz9F5tflJmr4TnmPsSoAphVL8olfZffazjZMIjKYIjiFOHQkLiG\/7BYuCDni47ZcwbhyzwpK+3pEziDw7EfmmEuxIiGtznueMI7wdIdV1COEmBc2XwlexVnd02ZMgVWrlwJuIePeTQy91\/IL9m5TqS1uHZBJEH2EAd1oJZBsNEHHBQjaKisWoqqrK9hsXs+aY86G2YrHDmSkQ4SP9HhNazO8vXJEV6rSXd+BEttamae26jcfyG\/6N\/eup8ZlwSxg+glvm\/DC\/TsfB2bqtqyTxL0oOoGNZeP1fNPhxlMHeQJNt7zh8v4m3KCp80PzL4ZwdcJrgokO9wvFzcKwqNVpLswCPz5+uZ55kZnjcr9F\/KrYPGyu37\/ftyPwT3MnMUk9yGuhgQxXKbaQNgKIIpwSbn0rWpVdpMMuTvRw2PcUl5BBBhheBFOaVavbISut8UpQn4VTGOYe56bqtU+xJWQIKrEt877UQWjrNclpUmwMjU4St85g6Y+mTRIdMdAzUMHvJt2P2tEgHEUZNJyjpBfBTMdVHzVbKbW+xDHrSaDfWt+D3F4Ca34KnAFk1qDS1iyw6aR7Ni5Vc6RVYOuSJMBCAj5VbAszKTroMvN9B3+PMyrVcHtnUvwyxTXQ9wKJDgSk+ScCNXgHeXaVvXKRhlbu50j5BdzRoOMs9gE2\/s4ir3e+xBXUk3mvg27Y45eTvcj0A5FPdM6q0J+MWee7Xlz5sxxUnSY8ObOnQvoxVq+fHndN2HnYcS1C7Zyqf2YU5fY6eiVZVUWn1vdK5sYMC3WkJBfQhPGdj4kPgzmrLfk5x+GkGBCEwtAVYtxf4o4FYyTu7u0VCsEhPwSQpa37Vy3bh2sXbu24eTXGEnQTGnjHlQbspLQBMVoph6qbJTS7TG6LKdWgICQXwWgBV1i7ll877330ilcuQL\/r4XDI07XayMJlgssbh0CrLcqW650e5y5lXMrQ0DILyZu6F1dtmwZbNy4EaZNm+ZcjeS3d+\/ept+HOG6YDNoEd24\/ADv634Cd\/QfUvhZBFY1L4Fi7WL2Yk2ec3khVtlQ+69SpU+Haa6+tfGByZWQEhPwiQ6VODPLm4mK+\/fbb6YGEyE6RZt6HuJIwGcohptS5g5FQq1VFk0g3951UD1U2ar+iVDLxFwCI2racFx0BIb\/oWDln+u01kydPDpQEV6xY4Vxj5jJWcMuaXhI3TAY7E5RD7HayOex89VZlo05SuQKefX19MHHixKjNyXkVIiDkVyFw7XhZXLvgSBJsLOn5A4ybNX1MyK85vj1Cfs0xD03Vi0pJ8LOUQ1zfbIxWLPsk5Nccy13IrznmoSl7UQkJ1jp9rllV2TgTKOQXB63anSvkVzts26ZlPwl2d3eXzGpIMmuEVdmLuo8B3rdZVdk4ky0Ojzho1e5cIb\/aYdt2LccNk0EAUBJ0w2SiQdJuRT39o45Suj0aUnJWNQgI+VWDXoqvjeshLiUNtoMqG3cplCvdHrc9OT8+AkJ+8TGTKwwEkARRIsQHq8P47D\/wPDxwg6ZxcCX9306qbNxFEaV0e9w25fx4CAj5xcNLzg5BIMg5goZ9Jka+rKenh\/6VSiiylBqNgJBfo2egje7v34OYh4aS4ObNm9topDKUdkBAyK8dZrFJxmDuQYxdYlVXyrY3yQRJNzwICPnJghAEBIFUIiDkl8ppl0ELAoKAkJ+sgbIIJLkHcdmbyQmCQJ0QEPKrE9Ctepu4exD7ifLqq6+msv5miaYoZIpxcPfcc48D20033eQpDtuqeEq\/mwcBIb\/mmYum7EmcPYiZ1Pw72eHAmACjkKlZFRtLO\/nrIzYlUNKplkNAyK\/lpqy+HY6zB3HQuX6yK0emKCEG7X5nVsquLwJyt3ZFQMivXWc2gXElsQexP4+1HJmee+65sGjRIujt7aUiBnyEVUJJYJjSREoREPJL6cRHGXYS5GdWMDnrrLPK7mk8Y8YM2iNl1apVnj1SwvZOiTIOOUcQCEJAyE\/WRSgCSZCfuUsZ3qjchu5CfrIg64VA6slPJIrwpVYt+fmxjdKekF+9vvpyn5Ykv7AvUdzp5HZMb2TcNtr9\/HI2OtMuZ2Lhd3TgZ0J+7b5aWmt8LUl+\/q0izS\/W1q1bgUMtyk0FehDNHdb85wftylauzXb7PMg7++yzz8KNN94Ix48fB8tSG5ebmJcKTSlHpuLwaLcV1LzjaUnyMz1\/jzzyiENgd9xxB2347A+QRfj9QbL+mDRzikQidNGIuwcx\/6CE\/QCFhbqwkyPMKRJ0XfN+rRrfs0Z6x\/n7M2nSpKYOTG858vOrTkFfCr904X\/NX+g5c+YETo5ppJfNowH8EnLYHsR8Hv8IBVFAFDLFdu666y5nL2QJco5PphxihOXEli5dGqkBvua6664jIYKPIE2rVIOlBItIHanTSS1HflGCZkuRH1+\/f\/\/+WBCLClwarqDKxOYVJiGWI1P\/pvBBknusyUvpyXGlvzDnH87tnXfeCY8\/\/rjzgxQGabl10Exz2XLkxyotq1VxJD+sIsyhFnv37qX5M38Vw37hwnbbSul3SobdJAiUs1mX66ZfQi+l8URRZcuZi5pNImwp8jOltkrIzyS6MMP77bffDviYNm2as3aE\/Mp9jeTzZkAgTG2N0jczE2ffvn0lHYFme2bhCvxOfe973wuUDoO8\/1H6VctzWor8ENyjR4\/CwMAAzJ07l9Kf4kh+fvIzq4aUA1nU3nIIyeeNRqAa8vPbWeOOJcjJGKWNRlbraRnyQ5Ead\/761Kc+BWvWrEmE\/KZOnRrJsCuSX5RlLOc0GoGw\/YDL9YuvO3jwoCO1lfKu+9XbXbt20XcS87GRBP2piWFFKYK0r3J9TfLzliE\/HjQDn4TkJ+SX5FKSthqNQFRnnj8MiW2HpnZT6gc\/zPsedo2QX0IrIwnyK5VpIDa\/hCZKmqk7AuU0lCC7G7+3YMECeOihhxyprVQWVZhXOCr5msCI2htjmZQiPxS\/H3jgAejq6oLx48c7nly\/eB3FHe\/vktj8YkySnNoQBPxFYP2dCCJHJDIMYfnkJz85oppOmOobpq5i+6tXr6YHFqHlQyS\/hJZDEPlxihr\/irDL\/tVXXwUkRD9xhf1CSqhLQpMkzdQdgXJhJtihUkU8gr4TQe+VsisK+dV42k3yw0wDjNvzp9GUM6SG\/aIJ+dV48qT5miEQJYauVNBzENEFqb6l2gj7TCS\/hKbdL\/kFNVuK\/EqFAwj5JTRJ0kxdEYgi9bHkh17Zvr4+j1qKn4VpQ\/7A51LfrUqCrsXmF2OpVEt+pVz4JvlhlxYuXAicBtfISYoBj5yaQgRKBRebcJRa+2HkZ0pz+P3A7wSGsgSVMgsjRpH8UrgoZciCQC0RYEFgx44dkXNuwwodlPMU4zhKOVRK2QKF\/Gq5CqRtQSCFCKCk1d\/fH6jGBqmgQXsoM2xMfl\/84hdh06ZNgHUxoxzsTEQC3rJlS2hfMJfeX12mnG0+yv2rOaflgpyrGaxcKwgIAsEIRJH8wrArl1liSn5mGlyjw8eE\/OTbIAgIAqlE4P8D0EiOjGz10WAAAAAASUVORK5CYII=","height":174,"width":232}}
%---
%[output:206c3f38]
%   data: {"dataType":"text","outputData":{"text":"仰角(theta)=40.30°，方位角(phi)=70.00°\n","truncated":false}}
%---
%[output:2c7eea0e]
%   data: {"dataType":"text","outputData":{"text":"仰角(theta)=30.30°，方位角(phi)=60.20°\n","truncated":false}}
%---
%[output:3a6511ac]
%   data: {"dataType":"text","outputData":{"text":"精确仰角: 40.270°, 精确方位角: 70.012°\n","truncated":false}}
%---
