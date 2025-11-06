clear; clc; close all
%% Radar parameters
c = physconst('LightSpeed'); %speed of light
BW = 50e6; %bandwidth
fc = 3000e6; % carrier frequency
numADC = 361; % # of adc samples
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
(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L) % 扩展的虚拟阵列 %[output:323f165b]
N_L=5;
Vmax = lambda/(T*4); % Max Unamb velocity m/s

d_y=lambda/2/2;
d_tx= lambda/2;
d_rx = numRX*d_tx; % dist. between rxs
% d_tx = 4*d_rx; % dist. between txs

tr_vel=-d_tx/2/T;
% tr_vel=0;
%%

%% Targets

r1_radial = 3000;
tar1_theta = 30;
tar1_phi= 60;
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

scatter3(tar1_loc(1,1),tar1_loc(1,2),tar1_loc(1,3),'r','filled') %[output:53416dfb]
hold on %[output:53416dfb]

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

scatter3(tar2_loc(1,1),tar2_loc(1,2),tar2_loc(1,3),'blue','filled') %[output:53416dfb]
xlabel('x'); %[output:53416dfb]
ylabel('y'); %[output:53416dfb]
zlabel('z'); %[output:53416dfb]
hold off %[output:53416dfb]

%%
tx_loc = cell(numTX,Ny);
tx_loc_t=cell(numTX,Ny);
for j=1:Ny
    for i = 1:numTX
        % tx_loc{i,j} = [(i-1)*d_tx (j-1)*d_y 0];
        tx_loc{i,j} = [(i-1)*d_tx-((numTX-1)*d_tx)/2 (j-1)*d_y-((Ny-1)*d_y)/2 0];

        tx_loc_t{i,j}=zeros(length(t),3);
        tx_loc_t{i,j}(:,1)=tr_vel*t+tx_loc{i,j}(1);
        tx_loc_t{i,j}(:,2)=tx_loc{i,j}(2);
        tx_loc_t{i,j}(:,3)=tx_loc{i,j}(3);
    
        scatter3(tx_loc{i,j}(1),tx_loc{i,j}(2),tx_loc{i,j}(3),'b','filled') %[output:2ba2173b]
       hold on
    end
end

rx_loc = cell(numRX,Ny);
rx_loc_t=cell(numRX,Ny);
for j = 1 : Ny
    for i = 1 : numRX
        % rx_loc{i,j} = [tx_loc{numTX,1}(1)+d_tx+(i-1)*d_rx (j-1)*d_y 0];
        rx_loc{i,j} = [(i-1)*d_rx-((numRX-1)*d_rx)/2 (j-1)*d_y-((Ny-1)*d_y)/2 0];
        rx_loc_t{i,j}=zeros(length(t),3);
        rx_loc_t{i,j}(:,1)=tr_vel*t+rx_loc{i,j}(1);
        rx_loc_t{i,j}(:,2)=rx_loc{i,j}(2);
        rx_loc_t{i,j}(:,3)=rx_loc{i,j}(3);
        scatter3(rx_loc{i,j}(1),rx_loc{i,j}(2),rx_loc{i,j}(3),'r','filled') %[output:2ba2173b]
    end
end
xlabel('x'); %[output:2ba2173b]
ylabel('y'); %[output:2ba2173b]
zlabel('z'); %[output:2ba2173b]
hold off %[output:2ba2173b]


% plane_loc=cell(numTX*numRX,Ny);
% for i =1:numTX
%     for k = 1:numRX
%         for j = 1:Ny
%             plane_loc{i+(k-1)*numTX,j}=rx_loc{k,j}+tx_loc{i,j};
%             scatter3(plane_loc{i+(k-1)*numTX,j}(1),plane_loc{i+(k-1)*numTX,j}(2),plane_loc{i+(k-1)*numTX,j}(3),'b','filled')
%             hold on
%         end
%     end
% end

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
fprintf('仰角 (弧度): %.4f\n', theta); %[output:32a8d39b]
fprintf('方向角 (弧度): %.4f\n', phi); %[output:2bad8e04]
fprintf('仰角 (度): %.4f\n', theta_deg); %[output:549f03d3]
fprintf('方向角 (度): %.4f\n', phi_deg); %[output:23fd0ce5]

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
for l = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            % disp(['Processing Channel: ' num2str(j) '/' num2str(numRX)]);
            for k = 1:numChirps*numCPI
                phase_t = phase(t_onePulse,fc);
                phase_1 = phase(t_onePulse-delays_tar1{i,j,l}(k*numADC),fc); % received
                phase_2 = phase(t_onePulse-delays_tar2{i,j,l}(k*numADC),fc);

                signal_t((k-1)*numADC+1:k*numADC) = exp(1j*phase_t);
                signal_1((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_1));
                signal_2((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_2));
            end
            % mixed{i,j,l} = signal_1 + signal_2;
            % mixed{i,j,l} = signal_1 ;
            mixed{i,j,l} =awgn(signal_1+ signal_2,snr,'measured');
        end
    end
end

% 假设变量已定义: numTX, numRX, Ny, numChirps, numCPI, numADC, fc, t_onePulse, delays_tar1, delays_tar2, snr

% numSamplesPerChirp = numADC;
% numChirpsTotal = numChirps * numCPI; % 总的chirp/CPI段数
% numSamplesTotal = numChirpsTotal * numSamplesPerChirp; % 每个通道的总样本数
% 
% mixed = cell(numTX, numRX, Ny); % 预分配结果单元数组
% 
% % 确保 t_onePulse 是一个行向量，以便进行广播/隐式扩展
% t_onePulse_row = reshape(t_onePulse, 1, numSamplesPerChirp);
% 
% % --- 外层循环仍然保留 ---
% for l = 1:Ny
%     for i = 1:numTX
%         for j = 1:numRX
%             % --- 向量化内部计算 ---
% 
%             % 获取当前通道对应的延迟向量 (假设长度为 numChirpsTotal)
%             % 注意：如果原始索引 k*numADC 有特殊含义，需要相应调整这里的索引
%             current_delays1 = delays_tar1{i, j, l}(1:numChirpsTotal); % 取前 numChirpsTotal 个延迟
%             current_delays2 = delays_tar2{i, j, l}(1:numChirpsTotal); % 取前 numChirpsTotal 个延迟
% 
%             % 将延迟向量变为列向量，以便与行向量 t_onePulse_row 进行隐式扩展
%             current_delays1_col = reshape(current_delays1, numChirpsTotal, 1);
%             current_delays2_col = reshape(current_delays2, numChirpsTotal, 1);
% 
%             % --- 计算信号 ---
%             % 利用隐式扩展 (implicit expansion, R2016b+) 或 bsxfun (旧版本)
%             % 计算每个 chirp 段的基准相位 (对所有 chirp 都相同)
%             % phase_t_matrix 的维度将是 [numChirpsTotal, numSamplesPerChirp]
%             phase_t_matrix = phase(t_onePulse_row, fc); % 实际上 phase_t 对每个chirp段内部点是一样的
% 
%             % 计算每个 chirp 段接收信号的相位 (考虑延迟)
%             % t_onePulse_row (1 x N_adc) - current_delays1_col (N_chirp x 1)
%             % 结果维度是 [numChirpsTotal, numSamplesPerChirp]
%             phase_1_matrix = phase(t_onePulse_row - current_delays1_col, fc);
%             phase_2_matrix = phase(t_onePulse_row - current_delays2_col, fc);
% 
%             % 计算相位差并得到复信号矩阵
%             % phase_t_matrix 会自动扩展以匹配 phase_1_matrix 和 phase_2_matrix 的维度
%             signal_1_matrix = exp(1j * (phase_t_matrix - phase_1_matrix));
%             signal_2_matrix = exp(1j * (phase_t_matrix - phase_2_matrix));
% 
%             % 将信号矩阵按列优先顺序（MATLAB默认）展开成长向量
%             % 需要转置使其按行填充，再reshape
%             signal_1 = reshape(signal_1_matrix.', numSamplesTotal, 1);
%             signal_2 = reshape(signal_2_matrix.', numSamplesTotal, 1);
% 
%             % --- 组合信号并添加噪声 ---
%             % 根据原始代码的注释，选择是加和还是只用 signal_1
%             % mixed_signal = signal_1 + signal_2;
%             mixed_signal = signal_1+signal_2;
% 
%             % 添加 AWGN 噪声
%             mixed{i, j, l} = awgn(mixed_signal, snr, 'measured');
% 
%         end % 结束 j 循环
%     end % 结束 i 循环
% end % 结束 l 循环

% --- 清理临时变量 (可选) ---
% clear t_onePulse_row phase_t_matrix phase_1_matrix phase_2_matrix signal_1_matrix signal_2_matrix signal_1 signal_2 mixed_signal current_delays1 current_delays2 current_delays1_col current_delays2_col numSamplesPerChirp numChirpsTotal numSamplesTotal;


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
% size(cat(3,mixed{:}))
% --- 假设变量已定义 ---
% numADC, numChirps, numCPI, numRX, numTX, Ny, N_L
% RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX*Ny);
% 
% % --- 参数计算 ---
% if mod(N_L, 2) == 0
%     error('N_L must be an odd integer for the original indexing logic.');
% end
% N_half = (N_L - 1) / 2; % Half window size excluding center
% 
% numChirpsOutPerCPI = numChirps - (N_L - 1); % Chirps per CPI in output
% numChannelsInPerNy = numRX * numTX; % Channels per Ny block in input
% numChannelsOutPerNy = numChannelsInPerNy + N_L - 1; % Channels per Ny block in output
% 
% numColsIn = numChirps * numCPI;
% numPagesIn = numChannelsInPerNy * Ny;
% numColsOut = numChirpsOutPerCPI * numCPI;
% numPagesOut = numChannelsOutPerNy * Ny;
% 
% % --- 预分配输出 ---
% RDC_plus = zeros(numADC, numColsOut, numPagesOut);
% 
% % --- 向量化 k 循环 ---
% k_vec = 1:numChirpsOutPerCPI; % Base vector for output chirp indices within a CPI block
% 
% for l = 1:Ny
%     % 计算当前 Ny 块在输入和输出页维度上的偏移量
%     page_offset_in = (l - 1) * numChannelsInPerNy;
%     page_offset_out = (l - 1) * numChannelsOutPerNy;
% 
%     % 输入 RDC 中当前 Ny 块的第一个和最后一个通道的页索引
%     page_idx_in_first = 1 + page_offset_in;
%     page_idx_in_last = numChannelsInPerNy + page_offset_in;
% 
%     for i = 1:numCPI
%         % 计算当前 CPI 块在输入和输出列维度上的偏移量
%         col_offset_in = (i - 1) * numChirps;
%         col_offset_out = (i - 1) * numChirpsOutPerCPI;
% 
%         % --- 1. 复制中心数据块 (向量化 k) ---
%         % 输入 RDC 的列索引 (绝对索引)
%         cols_in_center = ((N_half + 1):(numChirps - N_half)) + col_offset_in;
%         % 输入 RDC 的页索引 (当前 Ny 块的所有通道)
%         pages_in_center = (1:numChannelsInPerNy) + page_offset_in;
% 
%         % 输出 RDC_plus 的列索引 (绝对索引)
%         cols_out = k_vec + col_offset_out;
%         % 输出 RDC_plus 的页索引 (中心通道部分)
%         pages_out_center = ((N_half + 1):(N_half + numChannelsInPerNy)) + page_offset_out;
% 
%         % 执行复制
%         RDC_plus(:, cols_out, pages_out_center) = RDC(:, cols_in_center, pages_in_center);
% 
%         % --- 2. 填充边缘数据块 (循环 j, 向量化 k) ---
%         for j = 1:N_half
%             % --- 左侧填充 ---
%             % 输入 RDC 的列索引 (绝对索引, k 向量化)
%             cols_in_left = (k_vec + N_half + j) + col_offset_in;
%             % 输出 RDC_plus 的页索引 (左侧第 j 个填充通道)
%             page_out_left = (N_half + 1 - j) + page_offset_out;
% 
%             % 执行复制 (源: RDC特定列, 第一个通道; 目标: RDC_plus特定列, 左填充通道)
%             RDC_plus(:, cols_out, page_out_left) = RDC(:, cols_in_left, page_idx_in_first);
% 
%             % --- 右侧填充 ---
%             % 输入 RDC 的列索引 (绝对索引, k 向量化)
%             cols_in_right = (k_vec + N_half - j) + col_offset_in;
%             % 输出 RDC_plus 的页索引 (右侧第 j 个填充通道)
%             page_out_right = (N_half + numChannelsInPerNy + j) + page_offset_out;
% 
%             % 执行复制 (源: RDC特定列, 最后一个通道; 目标: RDC_plus特定列, 右填充通道)
%             RDC_plus(:, cols_out, page_out_right) = RDC(:, cols_in_right, page_idx_in_last);
%         end % 结束 j 循环
%     end % 结束 i 循环
% end % 结束 l 循环

% --- 清理临时变量 (可选) ---
% clear N_half numChirpsOutPerCPI numChannelsInPerNy numChannelsOutPerNy ...
%       numColsIn numPagesIn numColsOut numPagesOut k_vec l page_offset_in ...
%       page_offset_out page_idx_in_first page_idx_in_last i col_offset_in ...
%       col_offset_out cols_in_center pages_in_center cols_out pages_out_center ...
%       j cols_in_left page_out_left cols_in_right page_out_right;



RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX*Ny); % radar data cube

RDC_plus=zeros(numADC,(numChirps-(N_L-1))*numCPI,(numRX*numTX+N_L-1)*Ny);
for l = 1:Ny
    for i = 1:numCPI
        RDC_plus(:,(1:(numChirps-(N_L-1)))+(i-1)*(numChirps-(N_L-1)), ...
                    ((N_L-1)/2+1:(N_L-1)/2+1+numRX*numTX-1)+(l-1)*(numRX*numTX+N_L-1))=...
            RDC(:, (((N_L-1)/2+1):(numChirps-(N_L-1)/2))+(i-1)*(numChirps), ...
            (1:(numRX*numTX))+(l-1)*(numRX*numTX));
        for j = 1:(N_L-1)/2
            for k = 1:numChirps-(N_L-1)
                RDC_plus(:,k+(i-1)*(numChirps-(N_L-1)),(N_L-1)/2+1-j+(l-1)*(numRX*numTX+N_L-1))=RDC(:,(k+(N_L-1)/2+j)+(numChirps*(i-1)),1+(l-1)*(numRX*numTX));
                RDC_plus(:,k+(i-1)*(numChirps-(N_L-1)),(N_L-1)/2+1+numRX*numTX-1+j+(l-1)*(numRX*numTX+N_L-1))=RDC(:,(k+(N_L-1)/2-j)+(numChirps*(i-1)),numRX*numTX+(l-1)*(numRX*numTX));      
            end        
        end                            
    end
end

numChirps_new=(numChirps-(N_L-1));

DFmax = 1/2*PRF; % = Vmax/(c/fc/2); % Max Unamb Dopp Freq
dR = c/(2*BW); % range resol
Rmax = F*c/(2*slope); % TI's MIMO Radar doc
Rmax2 = c/2/PRF; % lecture 2.3
dV = lambda/(2*numChirps*T); % velocity resol, lambda/(2*framePeriod)

N_Dopp = numChirps_new; % length of doppler FFT
N_range = numADC; % length of range FFT
N_azimuth = (numRX*numTX+N_L-1)*Ny;
R = 0:dR:Rmax-dR; % range axis
V = linspace(-Vmax, Vmax, numChirps_new); % Velocity axis
ang_phi = 55:0.01:65; % angle axis
ang_theta=25:0.01:35;

ang_phi = -180:0.1:180; % angle axis
ang_theta=0:0.1:90;


RDMs = zeros(numADC,numChirps_new,(numRX*numTX+N_L-1)*Ny,numCPI);
for i = 1:numCPI
    RD_frame = RDC_plus(:,(i-1)*numChirps_new+1:i*numChirps_new,:);
    RDMs(:,:,:,i) = fftshift(fft2(RD_frame,N_range,N_Dopp),2);
    % RDMs(:,:,:,i) = RD_frame;
end




figure %[output:188fe135]
imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))))); %[output:188fe135]
colormap(jet(256)) %[output:188fe135]
% set(gca,'YDir','normal')
clim = get(gca,'clim');
caxis([clim(1)/2 0]) %[output:188fe135]
xlabel('Velocity (m/s)'); %[output:188fe135]
ylabel('Range (m)'); %[output:188fe135]
%%

%% CA-CFAR

numGuard = 2; % # of guard cells
numTrain = numGuard*2; % # of training cells
P_fa = 1e-5; % desired false alarm rate 
SNR_OFFSET = -5; % dB
RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

[RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
cfar_ranges_real=(cfar_ranges-1)*3;
figure %[output:4dfa0dad]
h=imagesc(V,R,RDM_mask); %[output:4dfa0dad]
xlabel('Velocity (m/s)') %[output:4dfa0dad]
ylabel('Range (m)') %[output:4dfa0dad]
title('CA-CFAR') %[output:4dfa0dad]
%%

%% Angle Estimation - FFT

rangeFFT = fft(RDC_plus(:,1:numChirps_new,:),numADC);

angleFFT = fftshift(fft(rangeFFT,length(ang_phi),3),3);
range_az = squeeze(sum(angleFFT,2)); % range-azimuth map

figure %[output:225cf498]
colormap(jet) %[output:225cf498]
imagesc(ang_phi,R,20*log10(abs(range_az)./max(abs(range_az(:)))));  %[output:225cf498]
xlabel('Azimuth Angle') %[output:225cf498]
ylabel('Range (m)') %[output:225cf498]
title('FFT Range-Angle Map') %[output:225cf498]
set(gca,'clim', [-35, 0]) %[output:225cf498]

doas = zeros(K,length(ang_phi)); % direction of arrivals
figure %[output:7f236b31]
hold on; grid on; %[output:7f236b31]
for i = 1:K
    doas(i,:) = fftshift(fft(rangeFFT(cfar_ranges(i),cfar_dopps(i),:),length(ang_phi)));
    plot(ang_phi,10*log10(abs(doas(i,:)))) %[output:7f236b31]
end
xlabel('Azimuth Angle') %[output:7f236b31]
ylabel('dB') %[output:7f236b31]

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
a1=zeros((numRX*numTX+N_L-1)*Ny,length(ang_theta)*length(ang_phi));
%%
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
nn = numRX * numTX + N_L - 1;
Ny_val = Ny;
N_theta = length(ang_theta);
N_phi = length(ang_phi);

% 生成所有角度组合 (i,j)
[theta_grid, phi_grid] = ndgrid(ang_theta, ang_phi);
theta_all = theta_grid(:);
phi_all = phi_grid(:);

% 计算列相关的项 (M = Nθ*Nφ)
col_term_A = sind(phi_all) .* sind(theta_all);
col_term_B = cosd(phi_all) .* sind(theta_all);

% 生成所有k和l的组合 (N = nn*Ny_val)
[k_mesh, l_mesh] = ndgrid(0:(nn-1), 0:(Ny_val-1));
k_vals = k_mesh(:);
l_vals = l_mesh(:);

% 计算行相关的项
row_term_A = d_y * 2 .* l_vals;
row_term_B = d_tx .* k_vals;

% 矩阵相乘计算相位
A_matrix = row_term_A * col_term_A.';

B_matrix = row_term_B * col_term_B.';
phase = (-1i * 2 * pi / lambda) .* (A_matrix + B_matrix);

% 指数运算得到最终的a1矩阵
a1 = exp(phase);

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
figure; %[output:964f43ab]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(1,:,:))))); %[output:964f43ab]
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value); %[output:964f43ab]
xlabel('方位角');ylabel('仰角'); %[output:964f43ab]
zlabel('空间谱/db'); %[output:964f43ab]
grid; %[output:964f43ab]
% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value); %[output:067e5866]
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

figure; %[output:866c1bb3]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(2,:,:))))); %[output:866c1bb3]
ax = gca;
chart = ax.Children(1);
datatip(chart,phi_max,theta_max,max_value); %[output:866c1bb3]
xlabel('方位角');ylabel('仰角'); %[output:866c1bb3]
zlabel('空间谱/db'); %[output:866c1bb3]
grid; %[output:866c1bb3]


% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value); %[output:4a06934c]
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
figure; %[output:2f8a7577]
% mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
mesh(ang_phi_sorted,ang_theta,10*log10(abs(spectrum_shifted))); %[output:2f8a7577]
 
xlabel('方位角');ylabel('仰角'); %[output:2f8a7577]
zlabel('空间谱/db'); %[output:2f8a7577]
grid; %[output:2f8a7577]


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
if size(peaks_info, 1) >= 1 %[output:group:02ee2734]
    main_peak = peaks_info(1, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ... %[output:450a373f]
            ang_theta(main_peak(1)), ang_phi_sorted(main_peak(2))); %[output:450a373f]
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(main_peak(1)), mod(ang_phi_sorted(main_peak(2))+360/N_num/numRX+180, 360) - 180);
end %[output:group:02ee2734]

if size(peaks_info, 1) >= 2 %[output:group:6e172733]
    second_peak = peaks_info(2, :);
    fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ... %[output:7a025eb0]
            ang_theta(second_peak(1)), ang_phi_sorted(second_peak(2))); %[output:7a025eb0]
        % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
        %     ang_theta(second_peak(1)), mod(ang_phi_sorted(second_peak(2))+360/N_num/numRX+180, 360) - 180);
else
    disp('未找到明显的次峰值。');
end %[output:group:6e172733]
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
if size(peaks_info, 1) >= 1 %[output:group:9bf69d9d]
    fprintf('精确仰角: %.3f°, 精确方位角: %.3f°\n', peaks_info(1,4), peaks_info(1,5)); %[output:906a5286]
end %[output:group:9bf69d9d]

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
%   data: {"layout":"onright","rightPanelPercent":18.6}
%---
%[output:323f165b]
%   data: {"dataType":"not_yet_implemented_variable","outputData":{"columns":"1","name":"ans","rows":"1","value":"119"},"version":0}
%---
%[output:53416dfb]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAFJJJREFUeF7tXT2sVUUQ3tcoJKJGNMpL1JeIdJjYqK8h0BgjhRXYCIWRvAYqIZBgAiS85BFpjMSEYCygMGBh408JHdGGhFIsiMqPCRoVE9HmmTmPuew995yzs7sz+\/PunOTmPjj7M\/vtfHd2Z+fMmVleXl42eikCigA7AjNKLnZMtUFFoEFAyaWKoAgIIaDkEgJWm1UElFyqA4qAEAJKLiFgtVlFQMmlOqAICCGg5BICtupmjx415tixlSFs3bryfeTIg7+rHlw64ZVc6bAuv6dLl4zZtq1fzosXlWAes6jk8gBrVRd1EQsHrzEHZDVQcpGhWuUFwWIBwVwXLA9h2aiXEwEllxOiKSlAJRfAodaLpBRKLhJMU1BoZoY+SCUXCSslFwmmKShEJRd4D8GxoZcTASWXE6IpKWC734eGrOQiK4SSiwzVFBSkWC91x5MVQclFhmoKCrrc8UosLyVQcnnBNQWFgWAYnYGueXW\/B028kisINq3UhcC9e\/cMfB5\/\/HEFSB+WrE8HSlZgkO327dvmmWeeMWvWrKkPXGaJ1XIxAyrdXMkKXLJs0vPS1b6SKwfqEX2WrMAlyxYBeXBVJVcwdHkqlqzAJcuWY7aUXDlQj+izZAUuWbYIyIOrKrmCoctTsWQFLlm2HLOl5MqBekSfJStwybJFQB5cVckVDF2eiiUrcMmy5ZgtJVcO1CP6lFbgP\/74o5EOvmdmZszDDz\/cHApTzq2kZYuALUtVJVcW2MM75VZgPJTGb5AMiAQfINc\/\/\/zTRF3Av10k45YtHKUyaiq5ypgHshQcCmxbJ+wYQ5a6QpegT6jjIhmHbGQgKiio5KpgkmwRQxR4yDqhlaLA4CJZiGyUfqFMyWFffWNQclFnt5ByVAX2tU4+w+sjGVU2n76wrGTbIfJQ6ii5KCgVVKZPybisk89Q2yQDKwiklgjcVXL5zIyWDULAVjL4Gy60UvD30N4pqENCJSTZn3\/+ae7evWs2btzI\/tiJkoswETmKfP\/99+a7774z+\/bty9E9W5+gYKDAP\/\/8s1m3bl3jJsc9k8\/eiU2gVkNA8h9\/\/LGR7bHHHnN6F33kUHL5oJWwLJBr165d5ty5c+aVV15J2HN8V+2907\/\/\/mvg8\/TTTzefki4kAFhPXKZSXPiUMSi5KChlKFMTuVx7J4Cv1AcS2wRweRd9VEHJ5YNWwrKlk8vHs1eykg05W\/CcDPeFvqkASh53nypPhbewNHK5rNNQqFHJSuaSzbZkviRztZ3wt5rclZKLDFVcQR\/rNNRTyUpGlS2EZNS242aJt7aSixfPUWsx1mm1k8seH\/zo4A8PLBX7lotKLiFFjW021bKQyzpNE7lwrC6SSZILfwh994EuvRS1XF988YX54IMPRjIcP37c7NixY\/Tv9v3t27ebxcVFs3bt2qYMnJns2bPH3Lx5s\/n37OysOXPmTHNISbmPHUmRS8o6TSO5XCSTJpeEB1aMXECcCxcumNOnT5snnnjC\/P7772ZhYcHs3LmzIRgSZ2lpybz66qsT9+FRh8OHD5sNGzaYAwcONNh\/+OGH5tatWw0B4Rq6jwSFcpzkSmGdpplcfSQDJ48EAaA\/KeKKkAuJMT8\/P2GpLl++3JDjq6++MtevXx8RBwYJURTnz59v7t+4ccOcOHGi+QA54QKCHjx4sPnANXQfrVssuXJYJyXXAwRwuQgH57WFVomQq085wJohuU6dOmXm5ubGyAfWDAlz7dq1EdHQCiFp33777aYLJGLXfbCGocvC3NZJyTWJwK+\/\/tqsdp588slqQquSkcte5u3du7dZ0rUtG4B36NAhA0vFK1eujIjYJg\/UgwuJ2nXf3tu5loWlWScl1yQCuHSD5SH8DVfpoVXJyAVLvpMnTzZ7MCBDbnKVbJ2UXP3kwsdZagitSkIu2yLBXqhvTyZhuT7++OPGoQEf+Pull14a++Wzo8qHlLqUe1Kbb47xScpGCa0KtWRScouTq+0VxEkEzx9c6AmEv9t7LrR06NBo77mG7uOeC0kF39Dnli1bmn65zzQ4lJPShpQiUPp2lZGUzdV2jCVzte0ad999UXLhOdbZs2cbd7t9wb0avIWhwErVk1IEDnklZaO2HUIyatu+GImRa4hYaKXggLi2cy5fgLnLSykCyllq3kLfcfuQzLdt6pyKkAsPjK9evTohB+x58GC59ggNKsic5SQUoe3cKTFvYei4KSQLbds1ryLkcnWa+r7LFZ9anpj+OBSh6+ih9LyFseMeIlls21n2XDFKxFlXyfUgiU1MMhuXFZBSUtAFrra7xgDtS4RWqeXiZHGCtnyUrG+5F5vMpo9kPrL5QsXdtj0GSNsN\/37++edJOfGpsiu5qEgVUm5IyXC5F2OdfIbZJlmNeQthDBBaBVm1Nm3apOTyUQAou5qXhSVEmiDJas1byG0VUT\/VcvkyNXN5UIQffvihCV5dXl5upCkld2GteQuVXBFKXbvlspd78OjFnTt3mgdG8d1ZEdCwVkUlrS1voZIrQg1qJFffcg+JJpGPPQLipmpbSV3eRZ\/+pAjA6Ylsj0eXhT4zLFi26+ypa7knqWSxw+uTjYNkkuOWalvJFatREfVDnBFSihAxjFFVl2x2EhrfCHZX2zHyS7Wt5IqZlYC6IYSyu5FShIChTFShyubK9NQlC7XtkHFItd1JLowNhJcW2I+EoOD2g4\/4OEjIoFLVybnnoi73qFhIKQK1\/6FyvrL5kMy3bZ\/xSLU9SC4IvLUDbZVctCmLtU6cCkyTmKdUqJJSSBbaNmVkUm0Pkuvdd981n332WeP6tfMFquWanLI+QsWGGrV7klIEihK6ysTKNkSy2LZz\/GANkmv\/\/v2N5YJ8F19\/\/bXBpJ6+5LJTotkpz2p+5GRouSf5lLOkkrnI47rPJVsXybjaTrmfc5ILnyBGIkBW3LfeestAajR8LmsIdNy\/ta1fjUlBJZd7LsXF+5JKRpWhrxy3bG3vIrQvcb7HLTfiQyYXVLDTS3ftxdqgg4XbvXu32bx5c3MLnjpGy1XLY\/6plntUxZZSBGr\/OZZXMAcQXIuRKdxv1JTC1ItcACxaIvh7yHLZyWTWr18\/ykeI5IJkMSUmBe1a7uV4iXcq68BBqhRW1Y5bxLAvruV3UnJxAo4WD5N9pk6tBv27XPElLPeomEspArX\/HJYL+sRx1xS3mOQQOWfewj5yST1IyKGkuRQ4VnZJ4rfbhn\/DHMK3b7RHe5xScq96cg0lBS1puUdVbClFoPafi\/h94+YgmRSmWcgFE6RJQcPUWUoRwqQZryUpm6vtGJK52g7FJhu5avEWhgIrVU9KETjklZSN2nYIyaht+2KUjVw1nnP5gitRXkoRUFZNCrqGbdqykQtGUHOEBtsMeDYkQS5NCnpPU6t56uGouMsVH9pujnoc5PI9y6MutThk68M0tu2hMcS23SdzEsuVQwntPpVcmhQU9aGLZHBPk4IGsnRaySV1ltdnBaQsAEw7d9v2GDQpaCCxoNq0kAuXe5oUlK4sgJkmBaXjNVFyNZOrhNAttAKaFHRc9XTPFUHaHFVBkUtLCooEBwvw999\/m0ceeaRJWgoRMBCaxHFxLwttmaTaVnJxzLxwG+3l3o0bN8yjjz5qnnrqqeb5phzX0H4O5OGK+7MdERJOB4n9HMqs5MqhmYQ+h5Z7+MIDIF1s0CpBlKaIr\/se63CRTMq6KLmoGtBTroY9V5fyunLAU8+fQuHj2s9xyKnkCp1F4Xqlkqsk5cUpkHLf41IR2\/e1uEouYZKENl8SubgI1YVFiIVI7b5vkwycHpQnipVcodovXC8nuUKWe7FwuEgmSXAf2Sn5CtWh4YNohrKpyVWK8tok+++\/\/5pXDpX2Ti+qJVPLlZg4djYq6Hp2dnYseSmKk4JcJWaJQnLBO73u3r1rHnrooYZg3O\/+5Zr2qUgKygWWZDuYXWrDhg2jfPbwdPOtW7fM4uKiWbt27ah7CXINLfcoewgJbFwW07VclJAppM1VnRQ0BJDUdcBqnThxovngyyD6MvtykculvKkxCN3P1Ugy8C6C3NUnBU2tJCH9QcLR8+fPj1kpO1ciZgqGtmPIVdpyj5PgNskAJ6rnLmS+QuvAeCFm8ZdffjEQvf7CCy+YqpOChgKRsh48xXz58uVOcs3Pz5sdO3YELQtDIhGkx81JqC5ZSyNZ11kbPhaCDhnOHwIpZ0m14U+c5JJWXl\/yhS73fPtpl89FMp8fNIklrZKrpQmx5JKMRAhR8tII7nMGlWO8nCRTcrVmsOs1Rl17rpKTgpa2n+siCSfJJH7QOEim5GrNPNVbCM4MJBi8mWXPnj0GXkebw11eovueanVCSJYytCqGZEqulhb4nHOhxxBJBuTat29fQzLpK9dy7+hRY44dWxnd1q0r30eOPPg7dNwukuUaL44nhGRKrg5toEZo2FVtSyZFspzLvUuXjNm2rZ86Fy\/GEwxat0lWYmiVD8mUXKE\/tT31OEnm4+1iHsZYcy5iYeHl5Tgp7OXeX3\/91YRWrVu3rsjQKgrJlFxx+tBbO5RkuZc\/XQMCiwUEc12wPIRlo8\/lGq9tyXyf1fKRI7TsEMmUXKGoEutRSCbh7SKKRypGJRc05rJeoWdtFEtBGoxQoS75oCuJ\/BzVHiILYd+EStmOD\/Aubtq0qYlrg6vkd3rNzNBR6SKXyzrRW1\/JuQHtwXfplkyTgvrMLENZINipU6ealtCF\/\/rrrzO0LNcElVzgPQTHBlychOoaWckkgzkG+T799FOze\/du8\/7777OlggMs1HL16DpYMPAmUpaLcnTxa9l2vw\/VfO21e+bLL1esClyuRDh+UnSXLoVkQCi48IcT5hg+EOjNfTSj5CJqTi0ko1ivzz+\/bd54YyVZZ+rD9NQkg3mDaB74hg9ce\/fubb7hrFPyUnJ5ols6yVzu+G+\/vTcilufQWYtLkqxtnVISygapaHL1PfyY6qV5Q9pk78mkDqN9tNn27gHBPvpoJZU0uuZD3O8+\/YeW5SJZyuUedazFkguItbCwYO7cuTOWFyPl614pINokg+WG9FLDlknaGUEZP1cZX5Lhcg\/3Trms09D4iyQXrJHBe7N58+ZG9qWlJbNx48bm75QvKvdRnFQkyxla5YNHaNkhkpWy3KOOrThy2Y+NrF+\/3hw6dGiMXJCEZm5ubuxJYztC\/tq1a4OP\/wMw1PQAVBDtctwkKyW0KgSLmDpIsjNnzjRvdUFnhKR3L0berrrFkcsWEkhjkwuJ136M3y535cqVwcf\/oX1qeoAYsGNItpqWe74Ydi33kFApl9y+ciu55ucbDFKQC8Gmkqz00CoOZetro7blHhULtVzGmD6LSAWRUq5NMgirspd80EbJoVWUMVLLdJ091bTco46zKnLBoGDPBdeBAwdGY2zvuU6ePGlOnz49ymdo7+Og0tB9OyUbFUSfcrt27RrtH955552RdzH1Ya6PzBxlV6t1GsKmOnKV6i2kKiAqWTtioLb9BGW8VEKhd9huE+I58QcSj2WuXr06KnL27NkmZAku132KrBJlqiNXaedcMZNSerSH79hCl3tdmbzsvttpyqH8hQsXRuRz3fcdB1f56sgFAy8hQoNrAqCdmklGtU5DeHUt9bF8V5SOvcx\/8cUXzcGDB5sPnoX2ZV7mnDNKW0WTizIAzjIwyfD4gX299957o\/2dK2eH675L1lpI1keokMhylzOpK8sX4IjnnS+\/\/PLEOwPs+3bmZRf+3PeVXPcRdU2yK9sUNHP48GFDeeuKaxJLI9nQci92r9i1X7L3W135KZE88L1ly5YJB5V933Z8uXDnvq\/kuo8oTvL+\/ftHG2UbbFeeRChLfesKdRJzkoxjuUcZZ3sPDXVc3l8lFwXZgsq0o0HaorneqgLlpcKqUpGMc7kXM7X2KuK5554btExquWKQTlS3yx1s77dcuelBTOnID26S5XyQcGhabXLBnqodX9recw3d1z1XIgINddN270JZ+0ytBHKh\/DEkS7Xco06pK+e\/yxvoui8dFDA0Tt1zDaBTYkCwLS6VZCU+SIjj6NrrAu7nzp1rLBa8ftd1juW6TyU6dzklF5Fcv\/3222DYFDSTK6yqi2S58kaEKGjbY7h9+\/axlxq6IjBc90Nk4qij5LqPoitmESZwyBso4S30nWA7OBjqpkrE4ivntJRXct2f6a61P+yz4IJNccpzrlDlwwcKoT53mrBQmaa5npLLmv22x\/D48eMTTzzDoyI3b95sas3Oznbm9+i7P82KNo1jV3JVMuuxoVWVDHNViankqmA6XUtS8Ki1r1I3+RXAzSaikosNSrmGXKFXGA1uS1Cqe1oOpfJaVnKVNycTErlCr9oHpa7HNHIerFYAN5uISi42KOUackWHtEN8XI9p5AwJkkOpvJaVXOXNyYREvuRyPaaR8zGMCuBmE1HJxQalXENKLjlsJVtWckmiy9S2K7i1vYdSy8UEfGQzSq5IAFNU9\/UW9j2b1pUKPIX809qHkquCmfc951JvYRmTquQqYx6cUvhGaOg5lxNS8QJKLnGI83SgERp5cLd7VXLlnwOVYJUioORapROrw8qPQBXk6trQA3R9\/58fVpVAETCmCnLBRHUlkOnKeaeTqgiUgkA15MIN+s6dO0cPMHYRrhRgVY58COC7BOw3oeCDsO0HYCWlrIZcAILtXoZ\/t9NHSwKlbdeFgK0rsH1YWFhoUh+kjKusilzw6wPptuAl1HDBI\/dLS0ud6afrUgWVlhsBe6Vz\/fr15k0y9gsRufvraq8qctkOjLm5ubF3NKUAS\/uoCwH7VVP2EjHVKKoil+3YePbZZ8feKJIKMO2nHgTQeoGuLC4uNglGU17VkcsOA8rxa5RycrSvOATs962ldGSg1NWRCx0bOdbQcVOttVMigN5B+AH+6aefzCeffDKWBi+FLNWRSw+OU6hF3X20dQT\/DaNKuTysjlyu92jVrRYqPQcCsBz85ptvOhO2vvnmm8nc8dWQy47y1r0WhwpqG9IIVEMuaSC0fUWAGwElFzei2p4icB8BJZeqgiIghMD\/sz9mwqzl1AoAAAAASUVORK5CYII=","height":117,"width":156}}
%---
%[output:2ba2173b]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAGWFJREFUeF7tXU2MVUUWLla0MYEOGAR\/oBcdZDExslHYOBo3\/izUhTqzkIWR4AIzkwhpJpKACUSILDQakw7oAkwMunA2OkvADepmCJtJjJnpoBFEMMCGRk168t1+51nvvrq3zqmfe+\/rd27y0j+vbtW5p853q853TlUtW1hYWDB6qQZUA8k1sEzBlVynWqFqoNCAgksNQTWQSQMKrkyK1WpVAwoutQHVQCYNKLgyKVarVQ0ouNQGVAOZNKDgyqRYrVY1oOBSG1ANZNKAgiuTYrVa1YCCS21ANZBJAwquTIrValUDCq4xsoH5+XmDD66ffvrJ\/P7772b16tVmcnLSTExMjJEmmnlUBVczem6lFRtM165d68sAMOG7n3\/+2axYscJQ7jaBTIGWprsUXGn02Ila6sAEAQEaAg7KXrp0yaxdu7aQHeCjUQ1ldDSL71IFV7wOW6sBgLhx40Yx+pRHpjKYykLa4LJHKvyfgKYgi+taBVec\/hq9uzwy3bp1y1y5csVMT0+b5cuXD4xMPsGqwEX3lUFGQPPVq9\/\/oQEFV4etwTfNo+kcpnZSP8kHLhfIaDTUKSPPaBRcPD01UsoHJttngkBcgLiED7lXp4wyM1BwyfSVtLQUTFy\/iSNkCLiqpow6krk1ruDiWGKiMrFg6gq4dMrIMwgFF09PQaVSg6lr4LLlIYYRz6ws46JmFFxBsHHflBtMXQaXThmHbULBFQGucjoRqHHEnOCDELMmZfEk4sT6TRREziGjTX6QLsbNN1NwCazZl050\/fp1s2HDBjEtLhBhoGiXwVWeMlKQe5ymjAquGsuWTPNiDH2pg2tcp4wKLsuyJWBK6f+MC7jKIMNIj6n0+vXrRdklofpq+r5WwfXWW2+Zqakp89xzzzX93EV7MWBScMV3GaaKFy5cKFK3KH1rKfllrYELwDp27Jg5cOBAY+BKCSYFVzy4qjLzl4pf1ji4fvnlF7Njx44i4RQL9V544YVs4MoJJgVXWnDZS2Hs5S8YyYh9jW+x2RoaB9enn35q5ubmzM6dO83rr79utm7dmgxcTYJJwRVvqHUk0FKg8hsHF3XJzZs3o8FVBSa8BfFdSLZ4qMkoWyjXHFdno5owPFLg4o5M3E6Tm0P1HaPWZhvyxo72ZZB1fY1ZZ8H1zTffFH2xcePG4qdkpW0bhjNqbbYhbyy4XPEy\/K+rBEgnwfXuu++a9957z9x\/\/\/0GrKI0nagNwxm1NtuQNxW47Hq6PGXsJLgwan399dcFwE6cOGEefPBB0QyuDcMZtTbbkDcHuFyjWVdGsk6CCwoDwF588UUFV81rJQYgMfeK3nSZ5K+qtksso4IrkaW0Yawxbcbcm0hlUdsUcGRoe41Za+DyKUdHLp+Gmt9Dwy+RrERTAG\/LL1NwyeyhsnRThlJ25kPXZLUhb06fi9ONTU8ZFVycXmGUacNYY9qMuZehDlaRtmSgeGnuNWYKLpYZ+Au1YSgxbcbc69cGr0RXZMChFNgvHys0Ul4KrkTabMNQYtqMuTeRyrITGlw5c+lCwcXtAU+5XB1U12xMm7X37t9vzBtvLDb9yCOLP\/ft++P3EdaZS\/QYPdapQsE1woYiNQryNfDImApdvXrV3H333ebOO+9c3Pfj9GljHn20WiOnTiUFmFT+RF01VE0uORRciXosVwfFjlzktNu5mbR2yj6fa\/LcOTP57LN+bSws+MswS7ShMx25NEODZZ4u46wCE+0zX3U+F0asia++8reL6SGmjQkuBVcCJYZUoUFkv9ZgnN9++61Zs2ZNURh\/47IPbKhaxTtk2JgOYlrIuRKNXgoujrIzlFFwuZVKfhN+YvckbJdwxx13FH4TLu6S+CHDXraM34sKLpau1OdiqclfKNdb2CYhXOca4\/uQFdfB4AJ7CGIjwZVLZ1LRcsmh4JL2REX5lB1k+02T77xjJt9+e3Hat2XL4rTvzTcL1i6mzaF7bfq9RieQ4dLHH\/fPTK7cCptB58fIn6jb+tPp0DSyOjkUXIl6KcZQbDC9886kefvtSfOIOW1OmXpavDD03qHh0v3enfIypobz\/\/pXAfLK1CEBnR+js0TdpuBaaoslXYzeuXOT5tlnFw9w8AKrZ1nzN2+mBZcAGMVIWj6c\/Nw5M\/HEE3677\/lsCi6\/qrKUWEqEBoGJ\/CcanaC4LVsWz7OyiTqMWACY75qfmTGXXnkljc9FjUEQys4goTz0O4EMcTIJna\/g8vVwpu9HGVz0VrfJCPzPHp3q1MYFF+qY+9\/\/0oIrpj+FdL6CK0bZEfeOEriIFkdKEe17bo9OlJ7HDiMZPi3eKXAxfLa+SSwsRBEyEaY1dGsukCuhEdhLZb8Jp3Xg85\/\/3Gn+8pfFmFPotcAEFzF3Saj4UGHt+7jg6tH5uYxa+ii55FBwMXuiDCaX3\/TAA9cKpi\/22mf2m\/2ml5VeU1nnwCWg8ylWVsl2Mqj8WD3T\/QougSZTKKtMQqB5O+Z02iwuxXjD7DP0u0BEb1HO6AVa\/NKmTX6fy2GoBRnCudcraakAY\/S69tln5toDD5hly5YVWSbY+LUfShAyllLxXOVT2Iur3sZHru+++85s377d\/Pjjj4U8d911lzl69KiZnp4ekK9pn8tOK8LvdBXpRKdP12aMP2pOJQeYl44\/daqIN7ne\/IQlXx1FMPiZZ9IeM8sEB3QMH\/WHH34o0rdWrlxpkJkvofJTAAt1LAlw0eEL69atM7t37y50gx11L168aA4ePGhuu+22vr5yg6sWTHaOns9YehIvM+mWYpASAI59velhn5rv0eKuWdOf\/2ytcfQFoXuNIFYmDUB7jZpJ55NRU\/tSKt8rB7PAkgAXRq3Dhw8Xn1WrVhWPjvO6ZmZmio89eqUGl02Lu\/adr0x4ZdLL+wsYxC\/FKDOL5RATE+tFdgcnVlasME60hIRpy\/1iQ0bN1HVRQaLk4SUzcmGL6pMnTw6MUjSa4RC8hx56KOnIRUsvJIc4DBmIoMNjRy97oa9vZPIZMhtciQ3VJ5f9\/RC4GP5a\/34F16CqcfDd2bNnneAqH4IXMnIRiDCXp6UYmMuXFwpKDMAIOpwLrrrRiTsy+Z6BQ4jkMFSfXEnAlTAzv9GRi45VRT4f+UW2QjD6HDlyxMzOzvandlyFpgaXi9Gj+fvly5cHWSiukOVyTHCBMQSx4bt8oxM30Oxrhw2uxIbqk6sWXEIqP5Wv2JjPReA6f\/58cYRPGURtgssmISYOH86yFGPIOJgdboMLbsyZM4s1DaTnwSfr5e3lpvK5sbJidyff+qxMMafQzHyi8r2nmTDlbhxcL730kvnwww+L6ZVNlceAy3Wvy+fC+VyYFuKD3wFyoseRGLr2r3+tfkHWUNSSt+pAWcboRXS8c4Mkz1wvB5VfuFKcTI+KHZ1gl2fe8C976W+9FqBcp1H75sU9eYcy8icng2NljYNr165dhVHjUPDPP\/\/cHDhwoDgYPAZcXLaQQIWfoOoffvjhousALE4cJGYphtNGGOB4aGaLeeWVS8MBXZ+x9Brk+msSGw6Jc5G4vntT+GuVRs2k8slfotNM8HfILlatgIvYO\/hKe\/fuNU899ZR5+umni0PpQnyu6DgXk7mLWYpRabyeDq\/sIKbMqaj8svxVsbIiwaq076cdJ2OzjRFUfkqjJpcBL1\/JspdWCA2MXDY1bmdWuHwx7hs1KkODaaiQJTRbnPsc5XKx4EJ9KUavSn+vF4JjDqT8OFkx\/wwLoKcEV78\/BDZCcmeRA\/25gB3orYsIjTK4UIS+w+8hI5fEcJ1UPMP3oTY6Ay6BzFxw+QBEOnD581w2kuWvUUNdApdA342DSwKAnGVjwBWTLR76TJVvP2Znh1D5VW1yR6e6Z2WDi8M2VjSUZcRg6ttmSbPI4Rq5Qo0r9X1OcDFp8U6BiylzGVy+NCjyFf7xj\/n+Mhfposy6PktK5aOhpjLzmfouNtn57LP+Po+hG\/3U6bDxrHguCCszNBhvJtZSDGYMhCtv7duPIbNNx9eed2DJnTtWxhq9aqh8ECY+1jFLZj5D3wAWAIZ+w9IX\/NywYUPSJObRA5dvzuOLczHuD4nd1IKLQeUDKLXEG7MO7suAU84HjCL4TMNlr0JbTO\/9vXuSZ+YL+piWvgytK+MoyFNm9MCFBwqlxX1Kj3DQvfN2QexmqM+YcnMJEYndSJe9hOxilSUzX6Bvb99JFGaVHU1weR42mhYPiN3k6qDiUZn0cqpYmY+NZGK9ETo\/0O4HbsvVdwquqt4R0su5OkgCLpSNHb3KM72u0\/kKrgANhCw5oWZiafGini6Bi+Gg07NzwVVmFmdm5ov0LVxIiMUei088MRHQc3\/cwiJEUDyCzo8SkHy++fngnYvr2h+vkYtrpAGdnXXkYsodEiuDcdijE3YAxvXVV3HAQh3J6fwUSHLUkavvxgtczBgI+01amjMVmfv79pmJxx9PawZMudmxshGh89Mqsbo2BZdA07ExJxfFPNC8z6NPeDA3KGJ81m\/Y4NWAN1bWQTo\/S5zLq6nBAgougcJiYk7RwCI5hT6b\/XgXLlwo\/vz+++\/7\/\/7TlStm5TPPVGqBgPX3v18zf\/vbtcJvwqY77P0AezVzfTZBdxSBZNcuVvN79mTxdSSyoayCS6Axr7IEMZChZpm0uCR24wIT9v5YsWJFsZ8fPsXFlBvrm4bOz8LRPoys3VR0Pjd9K0fakcBUiqJee5FWSC+qclZ8YD3Jb8vCFqaQkgsutFUxerHBFCkvjIYWEmL1NmudUwY6v+oxchm1VG255BgvQkOqdVd5JnNX3NoDF4Hpxo0bhf+Ei0Ym\/L5+\/foUklXWAeOZsDZc9TXGnRqWRydMSUHpD0xHaxrLZdS+5yt\/n0sOBZe0J5jgur55s\/n++PE+mNDMvffeW7SWG0zOR2LKHUrn0\/RqYMn95GQ\/69wlUy6jlnZpLjkUXNKeYNLiANf1f\/6zPTCVn4spN5vO94xINsiqdmnKZdTSLs0lh4JL2hMozxkFuHR84qUvtY\/DkNtL5wv1Zft9ZZDlMmqhiEpoSBSWo9NsEmLlv\/9t\/vTqq5UiIXaDtULkezg3r2wwVtYXlBnngu\/05psTSdc2uUAGuZQtlFh2orJts4VeEuK\/\/608mNs2JKgjNOYkzW9kqZ6xXEfiN7HatArZunGezyWtMEH5HC\/jYoKjVPxi71AmhM3o4f+xJIRzWsSMOUliZQlsbKAK7wsiokECL87m+vXXX83q1asXz+eyg94R9UtvXZLgwoafU1NTxWaj5Sv3yJULTFUdGxpzyjJ6Ca2vzm\/iVFU+8pZGczogA\/XTx7tFNadBYZklBy4A69ixY\/2dfHODi8CEduy0otiRSdiPhRFJYk5dABc9IxdkVQdk+E6b4dYv1bmv\/JIBF+19iD3oMR3AuVy5Ri5MOdChLjANpBX5tJ\/6ewZrVzQZsPQltaiu+soggO+0fPny\/uhD99CBgpUHC1YI2zTIlgy4sDX23Nyc2blzZ7EPfflcLtJ3yLTQZvRwRhc6HSPTmjVrBnP0mrDAujaYMacug4umcfCbbt26VTwtnW28du3aJBrO6ffZAi4ZcNFD0b7xMeCqY\/R+++03c\/vtty9OwyYWaWXpGzSJhVRVwhi9sP0X1oalOocq5nlcfhPpE\/Ll9pucycgT8Qs6oRMFVyCj19TbT2y4npiTva9eG06+j4SoAnzuKV2O+scKXPb5XAcPHjSbNm1KkqOX8+0nBhduYCwhyWFMLlnrSAiUl476ueVOWf\/IgYuOHaKOpPO9ONNCgAvHFOG67777zLZt2wyOkU1FQqTsmCBQBdyUegS2p3H4PZaEqHqk1HKX2ynXX5sVU0Og5MgUaS2IXOdzgcygi0YxgAufV2vSjqQ2O4ogwzOGjMAEJro\/F5jq+iBEbm6f0vMNLRJl+GUjN3L5lOIjNOz77ZMm8X8CGX6muFK8\/VLIIa3D93II9ZukckjL++SW1lc3mnH81bEGl608e8qYA2TU8WiT0zGxhpDifpL58uXL\/eoQd8LHDtx2gXW0n7crIGsNXGXfCUe3gmS4rWJlq+\/kSJypDB\/KvkJOqrRHs9Qgg2y5Oz4VqGzfCfEmijmhfsT32srXkzxf7pmDq36boGkFXASUQ4cOFUe4UnbF888\/78yq4Jx5DLCePXu2FqCSjhknkElIiFF4ObimczlnDlXkCuRonNCgbIrdu3f39YCR5+TJk05wAIyHDx8uPqtWrSruASBnZmaKz\/T0tEFOIS67TgmYqsqW\/TJkgOCFkNMvk9LTIc\/pC976pnq+t3aITE3ck\/vlQJn5aKeV87lcWesuAJGyXcCj0Qw5hJj+1aU8peg0gAxyEJWfesqYm1rORULkljtF37nqyAUyshPUjwRyuCqvvfZa0myYSiq+is0DuPbs2WMwVcRIZF+uKZ9dz2OPPWZ27Nhhzp8\/378txN\/iduQoTBlTB285usllsJy2Q8ukGIFBhuGiFy9+pxBPylkOPWOj4Nq8ebPZvn17AUw8DK66kTC0I8r3dQlkEr8p1fNX1VMGWefyLx2CS0bgpsFUFrdRcLmWlkjiXbHG5vLLcgal7cWAkJ1GKfweuhwjVge+qRe+H5UQhO03QWbMnHChnykRgUYm\/D9lX3P6oTZDw0U++HyuI0eOmNnZ2T6hYftcNFrZgjUJLmo3d1AazBMWZ2LNGi7Em+65556BmBOnc9ooM0pTRvKbvvzyywFXI0c2T0hf1IIrNVt49epVIwVfyENJ7kkxZawjIRB3WlhY6C99GYW4E\/TXRZARmMp+E5hhXPTyTsUQS+zIVbYWXKnjXBilQGjs2rVrwOc6ceJEQZJUBaZjH5JzvwRkLr+JpoD2lM9ut4vGytGLxMfh1Cct4\/KbbDB1BUhicOGGugwNF3Poy9CgQDQxhr6MD2lnxJYvgwydhzfixo0bkyxjH1WQQa9lHyfHKNw2CRFrP\/b9rWXFp3yIHHWV\/TKEDMB0AmwpgsdtjwgxOkv5gogBk+9FXn7G8os9ZxgIbSu4aqyMGCcEpYmBSh2UphEhZKlEDEBS3BvygiAwxTJ6nFQ7+xldqXuYlX3yyScDBFwKvVAdCi6BNiV+maDaftGUI0JI+6H3kA\/qekEQCWGDCe2Q3xRKj3NS7eznwQuyTKYR4GwOIFQHQT5Xysa6VJd0SmHLriCr7knQ4mfOnCkKfPTRR\/2CsWAqt+hLtXOFfaqmiQquhMiUTimqms4dL8u9FCOVSqsYPegn9epxktmXaudKWHABFCz10aNHh1L5UuhmLKeF0ikFR9HjtIgzhoTg6JJTJhZcTSQvjCW4UkwpOKNZDvKjDb8sFQnBAQ23TCy4ci19suUfS3DFdgzHAEbZL8tFQnD0xi3jIih8qXZUN4B18eLFZAt2q2RWcPW2K8g1TRiFRZw5wVSOLcEQjx8\/3s\/Q4YKpXC5kak+yYIvzuq0qQmUq36fgygwuUniXFnH6cvRC6XGXUZZHiVSxJSkp1TSwoIuxBFfMlCLFWy33lNGVpgRGDJe9UDB3jl55iwe0z526cfTsC6fYflU5jc+uv7xhLadtTpmxBFfIlIKjTGmZnCADCUFL2G25Uu8tUvfMVcuT6g49lOqwy+XHElzSKUXuDkyxiNPH6OWMOVXpxzVDQNkmmLrcfcapfyzBBcX4phQc5aUuIwlK5yQhUj2Xggsr+fTqnAbsk14oXmYnEJPAqdOKUipCwaXgSmlPyeuyMz9QuZSEoLOnSbCXX365ds9I3+k0kges2ilMfS6JFrVsNg3YG61IG8HIYefOlVeWl+sr0+S+HZZ98uRmC33tt\/392PpcbSu+ifZdI4RrXxTIUhVEj91+PFecqwn9xbah4BJosKuHUrgeoSqeVLcduaueWHDlytAQdFtrRRVcTNWn3qwHm\/HEGm6d6FULAatIhjqArlu3Lvne\/ky1j3QxBRez+1JvM5fzUAo8UgpwSYDIVONYFVNwMbt71A6liAVX3ZkATJWNfTEFF8MEun4oRTkgju3qcJoMMr+3bt06cJYax+fysYoMlWmRcU3clfZ8DnA1cSiFhC0knRBpk2JZiFTPS628jlyMHs0BriYOpQiJc+3duzfJeiuGWpd8EQVXqYurNo784IMPipL2iZijcChFXYaG7VfhJNDy2WmkmtybZy5VlCm4mD2bmi3s4qEUTFVoMaYGFFxMRaWOc3X5UAqmSrSYRwMKLoGJjNuhFALVaFGHBhRcahaqgUwaUHBlUqxWqxpQcKkNqAYyaUDBlUmxWq1qoFFwuTaGQRdU\/V+7RzUwyhpoFFxQlGtTSM1lG2UTak92V6oWslK2bdtmcu1FKHnaxsHVxgl\/EoVo2dHSgL3SmWKH2NDHzqRp64kaBxce1FYI\/kYGty7Ia8sERrtd+2U9NzdXHK87OztrkM7V9tUKuOyEUigAB3kfOnQoenP+tpWp7bejATu436Vs\/lbAZRMYU1NTWQ99bqe7tdUmNdDGIQuc52sFXDaxgeNcdErI6SotU6UBO\/O\/C0QGydkauOzVs10aytWER0sDxA7Chi5cuGDef\/\/9bGccSzXTGriI2OiSAypVnpZvVwPl+Cj9DamaONzO9\/StgUsDx76u0e99GsB08IsvvhgYqWhG9OSTT7ZOx7cGLt1dyGc6+v2oa6BxcNnL6NXXGnXzUfnrNNA4uLQ7VAPjogEF17j0tD5n4xr4PwnUi+A1G3AiAAAAAElFTkSuQmCC","height":117,"width":156}}
%---
%[output:32a8d39b]
%   data: {"dataType":"text","outputData":{"text":"仰角 (弧度): 1.0471\n","truncated":false}}
%---
%[output:2bad8e04]
%   data: {"dataType":"text","outputData":{"text":"方向角 (弧度): 1.0470\n","truncated":false}}
%---
%[output:549f03d3]
%   data: {"dataType":"text","outputData":{"text":"仰角 (度): 59.9970\n","truncated":false}}
%---
%[output:23fd0ce5]
%   data: {"dataType":"text","outputData":{"text":"方向角 (度): 59.9908\n","truncated":false}}
%---
%[output:188fe135]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAHHpJREFUeF7tXXuIXkWW\/4lhtKNgyCfGZNNrlomYZTQGVvJgxREiu65pMH+YtLAmZDUhwVUQaUkgH+wuk0CaNKxg1rXprOskytjJ\/KFM4rizGTA+yCiRmGZklUQ2GvOCfGJcNc6sbpbfvV916qu+j7p1q+7j67rQJN23HqdO1e+eU6fOOXXFpUuXLsE\/ngOeA9Y5cIUHl3We+gY9BwIOeHD5heA54IgDHlyOGOub9Rzw4PJrwHPAEQc8uBwx1jfrOVBbcB0\/fhzr1q3D6dOng1mcNWsWRkZGMHfuXD+rngOV4EAtwXXx4kVs3rwZM2fOxFNPPRUwcvv27Thz5gy2bt2Knp6eSjDXEzG5OVBLcFFqDQ4OBj\/Tp08PZvCLL77Axo0bgx8vvSb3oq7K6GsJrnfffRejo6MdUkpIs\/7+fixatKgq\/PV0TGIO1BJce\/fuxaFDhyLBtWTJEqxYsWIST6kfelU4MCnA9d5771WF356OCnNg4cKFVqnranARVM888ww8uKyuma5sjMDavXu31bHVElzccw0NDWF4eHjcoBG15yKoVq1ahVZrPr799sZMjJs69WxQPmu9TJ2UVNjO2O4FcLz986cAPitpNHK312LqVNIkz9uPAPwxkTbyo9EYC8BlU3rVEly61kIBrpMn\/woXL86owOR3EwlzAFwP4DCABwF8BOADywNk++eVNtnvCav99PScQ2\/vbzy4yFXdc66J4IqarKzzdDWA7zJW4sF2+EXtnoeaQCjdQ3C9XNuheXApU6fjoeElV571fi2ArxMauBvAG+H7GU3g3JY8nZVa14PLgP3dB64pAL5P4ETSe1nSGDAzsopQ0SSg2WoaaWO11hE8uAx4WQ1wJe0RiltABuxLqSLTvgHACwbqsn2qTFr04DLgWjXAZUC4lSqugcu9Jx\/uP+9oGzasEF54Ix5cBiyf3OAyYJhxlQEAQ8a1y67owWUwA5MbXHcCeNuAa7pVZIMH1cLngNh9komFlXTMBvC5LkEG5ULrsQeXAevsgsuFQcBgUFpVbBw56HTUBk2jCbR0rIXTAHyp03ChZTy4DNhtF1wGBHRtFXFut6B9cPwQgBdrO1oPLoOp8+DKyrQsEo\/uT6+3OxBqYdb+qlHeg8tgHsoHV5VUSZdeIh5cUcuzlr6FujjLB66or7hr83YSAGTTty4HXJajQYMH2jTFPwDglzk6S\/MGydF0YtWwXy+5DPibD1wGHXZNFR3Dg1xmXttxV5cB6keqLHCF9Hpw6c6bVM6Dy4BpqVWEqiuZ169pAt\/oWAtTG48o4Fpb8OAymZUgSJLxXD7kxIh9EZXizqsWA\/idrU4stBNHZ7QrmpdcBiy3By7165nFqmZAeO2qFAEudwfKhYKLkb6rV6+OneJdu3bVIsOSPXAJVujsRWqHDEOCZV7YsBaWZ1l1Di7m\/Vu\/fj3Gxsawdu3a8WSbKudFoOL+\/fsxf\/78jlB7w1lyVs0+uJyRWsOGqWLRUsiAyU0AtlV0DHEq4uWPg1NwEVhbtmxBs9kcz0mhwynTejpt2yjjweVS0sqqcVoksvDksDGrog17bToFl80hV6ktDy6XsyGD6zEAO1x25rRtDy4D9npwGTBNu8qt7bwgNg6RtTs1KJguvT24DNgaDy73ZycG5FakSvpinEDolU3gB1fnXO7ZUii4ZONG1NCqbsgQNHvJ5X5hBj3UClwTDRyFgUtYA8mzul\/HU31wCQlad0m6pp1DoyAwW+6mp+c8ent\/7T5voZBaAwMDtTjLSuJz9cFleZU4ay7lADdIrbZTymPojBAANvwQZWPMFPT0nComKaiQXN1wW4gHl1jkLiTjPW1nXYbh5\/WKdwnG9LYLUwtJChNubtq0Cdu2bav1RXLVA5eLRZ6+eMISBoYK3abhTfFRrIqM51Kz2aoVu9ugYRsAoj3TJC3aK7zEgjY9NLKGr+QfdmGSKyoPe37yy2mhepIrjQ\/2LxlwK7HEeJoAvClend0JkssbNNIAUOf3NowBEW5H2tmfqsm7wiVX9xo0JnO4iA1wyQBpWxGvawIXmLdQve6nmmBSqSoMXN6gUY8FUT6VVGG5j+S9XD61mpZBI807g410t0Gj\/GVbXQrkmCvJQBOohUxnnfXesqSR2pay8X0VKrmqO7nZKCvfoOEinVkF1dpbmsDHtsGVba7zlPbgMuBe+eAyILqWVWiK557LRapqFxbUTiY7BZdp0KNpvaLWT35wqaqJSSi67XOzoriXoZ\/AoPF0yk2UGdoruKhTcHEsPsy\/4Bntpu7+ogm878+5Us+5WMAnqOmmlV\/AWFLBZdM7JUkTyNLPZS3EueQqYAoK7yK\/WhhHcpZJLHzYBXQoxs8bJT8CZjwBnOOlDIcL6Nt+Fx5cBjx1By4DYrquimy1dOX+5NLZmBPiL78zXpYTwZVH4pgYM4xJL6hinrOkuwG8EdK5tAn8lhcx8EC5fo+XXAZzVl\/J5fqLbcDMCVXEx+Z6YOkG4LdUCcV9XTbaL64NDy4DXrsBV56vvcEgKltFCg0JJNcLju8vdseIrgPX9u3bsXMnQ8MvP3KmXzWmbNasWRgZGRkP3kx7z1bdgEud5KLBVgWpJsbcpuVvm8BLZTnu5lH1w7nsKnClpRKIiikjGM+cORMkzeGzefNmzJw5czzttvy+p6cnKFMMuNx9UfVbLuKgOgHUgeR6BcDv2yRX4QOgz73CwSUfKi9btixY1Dt27MCcOXOwYsUKfcojSqbFjFEqDQ4OBj\/Tp08PWmCdjRs3Bj98kt7PnUufPhvgcnezRi4GVqZyG9SB5NrXvnxcJa76hqBCwcVDZObQoBp25MgRHDp0KAAXJQova1i5cmUugKXl6GD\/o6OjHandhDTr7+8PZi\/p\/aJFizTBVa8vbGUwFRDCjLv8+LwO\/H0T+JcPABBg9fsgFQYuVWXbu3fvOLiobvH3PXv25LrdJMoDRN5vqX1yKmW6+LsAvFABo1RNoRa2WvPRat1e8trU8WaX9w93Ani7ZJqTumdEMjM\/nQc2NoFBgoum+a8rTHM0aY3GUTQaY8XnLVQXOoExNDSUC1xRAOXfTpw4Eeyhqg8u3Rs2dABVu7UoEcyPwRRg8xPAVkYh83ZJgqzMJ5uBo9H4LzQaoWfJ7t27sXDhQmvET8ihkSa5ogwHNqiRVUVZFY2STFkll7+21cYMyW3QWshDZKqBdwDb7wWeOg7gZdsd5WhP78NGlXDq1LPFSC6OJm7PtW\/fvuAOLxc3S8rgarVaE6SjuudSpaf8PnnPVaV9VravbI6VZrGqMMO3pRbVwH9vAn\/HLuKMGha7d9BUYXsuQXtUuL961mQ6Tko\/PlQBxSNbCNl3NayFpiPs9no0WlByvRgO9D+bwL1Qbjqpz4ejcHC5XB5R+zbus\/jQzF\/sOZd6RlT0obBLTrtsW+Lbr5vAEwA+ls+6XPZtt+2uApdQPeVLzXltrHx+luaBkfaefXT\/IbIL6aBmvI3rQ7qq9VgTYOr4T2nQoHGAFsPv7SLAYWuFgUsn+5MYp4u9l00edj+4TLll42CXqiHzW5wH\/ucBgCcHR2nU4FMlw0Y6jwoDlyxVVGkizqcIqptvvtnKgXL60M1LeHCZ8C7NlUo2CC0HMA8\/vnQCn2yaCwzy3IsSq70Xgw0QcwwuJPRl3hQGrrRc8fIZFK2H6mGuyXS6quPB5WpRCjP3E8DSa7HswD7sf7oPoB\/2h8JLg\/\/W4ykMXGl+f7Ix4tixY7kPlF2yX+9O5LQvtaBwcfuQ1CXFVWxbXCxO2mgSpHTivmoNcM1sYAOwfWgAT300BDDqhM7xF1iG3iVUHdsBlTwPq0wagM6PTmHgSvNYVyVXXlcol8spXXLZ\/LLTysj2bOVLFwlFoxalrmtUlFomH67y\/\/y4nG1PA4HEaGKqdrRQcEwESntvxbD4GfPCKqy6FsBy4NKNV6Bvyq+w\/\/O+EFzEFbdfp9gs26LXxon2\/\/mCoGMDsjeH8ElM+oi5OaMsDFy6ey6mtFbDPlwCxaTtZHBxIYkQCbl1sajV92kuTypQhUTUTWqZZX+iHhckLTrui2giF48oS6sg008TRDRAsBwXfDsldeN6oMU6XwJXTQPIFg6FZHJoZMdi4KYFJ3DiwJ\/hlXvuxxAG8M6JO0MvKGKU2BEZrinsBIta5DtBy0b4cDzMe8gCAuC0OMbdWBn1d0pVdsbO1bTayc7EhYKLw42yGso54nkQzMU7PDw8HhZiAgCXddIll8ves7St56qTpUX7ZSlBueApWQDcMhv4+DvgZ1djd\/MhrMEL+GHHFICpNA5SWnHPRSCzjiyhiFD+jdl5bZvr5Y+OvlZSOLjsT07xLdYHXMXzRv\/ibkoSAoGShg8XLRdxH9CYHUiz\/sMvY3Tfg6FB41U5lwZFHB8VRElSOqvql\/\/D5MFlsP48uNKYlrYwZWMPAUWQCJVLgO4B3HLpc3w8MC+0wJ\/LknlX15iUNo587wsFlzBq7N+\/P5Lq7rtCSGeSs35Rs0x42iLP0lYRZanaUT2klJoH\/O+CwHiIl2jJIADVOLRqu5QVCq467Kd0lpBbyVU3QOhwLEsZccn4cuC\/bw3tIp9QdNEqWK+nMHAJQ0beUP4qsNctuPKO0MXdXXlpUusnSZwN7UOtxcChe4DHALzPvIU0ViQlB62eFCscXAMDAxBxUbanrKj2qg2uKC4kbfTjjg7iuJlXjU07GmDICc\/0fh+GnNA6fkF4xeftu6gVEvZTGLjS3J+KHXa+3vTBVb+kKvk4k7e2iOei5fBF4BcDgacGLqhZd23x1S1YCwMX2S476NZZeumDK+9iM61vc9FwobM94W2RRFOaVGLdNCPPAIAdoUHjn\/sA\/vpDFkuhKc\/s1ysMXDohJ91nLUyasKzqmO3JT1vkavxVlv5164oDWfKCBouvgR83gU94OHwYWL8WGOZ+i7\/bPhjWAXqWMU8sWxi48pFZrdp2JJf+SX+1Ru+Kmra0ndEEzrWT0qxvAsM0ZNCxUHY9SvswuKIxqt14Wjy4DObBDrgMOp4MVYJ7kAmop4EAXAQVfRSFJ0d9mFAouNJUw8mlFuZdJLY29Wl0FNVPm46fNoGD3N8dBvr7gFEC7UCMM3Qa7Xne59+3FgouOTchAyJFsk6Rt2Lbtm21MNPHS66y91F5FlMV6s4G\/nIN8A6l1AtAADSebdFrt35PYeBSD5HVvO1R2XCrys7qqoU6X1v1sLXsvZ+Up5BhITctD8+LWzuA+x8DXmUYiRzaUtVVUaJBQ41EVm8cibqBpKpszA4u6SrSqg6qdLraDrz3D4RW\/3efA27fABzlL533rZVOqiYBhUkuNRJZvrqHV\/N0N7g0ZyNzsbz7obKlljxgBiW+DvQ3w0iU\/9gCXNMEvhGX30W5N1WJ\/hIlF7tWL0rgHkzcy+XVwszIqmGFuOhpnosRUfcAG+eEQb8HeXDMeH96Z+S9hIHqMs\/Jir0ppTDJJVaCnHJath7WxVLIcWRXC2uIg1iSXTjIMhqZUng28K\/TQjz92xvAdXcDF+rpEU\/2FQ6upGV28OBB3HbbbZUN7xe0T25w6X4osqpsdCK8GvjFtaGgGuRei+E3lDZ0h6qfJbYQcFHl4y0mfKIklNiPnTx5stK5M6oJrjRvBR0Loi5gRDkdH8K4NlV66fZOlZCWwmvDIy16QjFI8hT3W1QZOQa6QeV1gXIhdeN55xxc8rVBNFyo93DJt0GqmXizTnlR5b3kMuU0pRmlkUirNg24si9MncEwtMXAjB1nce7rG8Pt1ugO4CePAR9SNWQBAo1SrKgn7cOVTIdTcEWFmcgm+c8++yyQaLauECqK5fUGV9KCSVLl5AhpluOjphqTZ6CvHdxIIwYlE39YfjlwzTTgmy+B66YFd9wFqdUYcTwH+NniJo5jLn5+YA3wj1JWNkq0H9g+26HvoUh3xrQA1CP59zj6STulHsvoPFmlfbTV1im4oqKPBeC++uorvPXWW5DvLNYZdhXK6IEriuFRk8ZyfGTfuTgAiDbbZusJzBCLKylpZ1SkssjXJwAk5\/3jwhX9EQEi2y1RQe8J\/s7gRtblIxJ+Lgaua\/9JvOK\/\/CF5\/JfktsE149azWIAP8PqJv8EHc27HA\/glPjk797I2SDJYnmk0yEZ2KaJgThFs7FekAhBp1kSSQwKd\/CUgCTA2IKKa5WShgm\/c33EO2CnHTGTHPXIEgJi3cJ4LAZcafUzVcOfOnaiLGqiyNR1ccobZKnwOXNMQ9zGQrgMKSBAiigubIL8RuHJKmJOGILsD6H\/o5UByvf\/cHaEtgyk1fiAo+EPw0JRIZIrEqyJLsPrRkEGhm5ckqxFG5at8O+Z35YHrzJkz2Lp1K8TdxK6n32b7E8Glm\/3WFhVZ1Zakfm22pfYjf9XjFi7B8T1w0z3Bluqq17\/DH05cHYKKeHqVIope8ZSmstiSrYcmhgpdwJnPWWmSiyTL16uaD6H4mumSq3iaqtlj0gJWpd2twHXLQzWQmhuthRRG7\/APlHTFHgDb4KcHlwEXO8H1Jwkm4nzWJgPSCqySRVrLZWXAib0edUIaHOYAe28NNb5\/egO45W7HV7a6nR8PLoPl2Amum1KsZgYd1LJK2oUSSYMShpq5wD88GG6tfn4gVBU\/pRuU7u0r1WJcIeAaGxvTGnVdXKDM1EK3X0ktBo8X0s1xka3VsLRO21FhL\/zbAuCRxaEl8NUPeNUJ8Ckv56pfFDI54RRcJlNThzpm4LI1siqB1MaYaFChsaNtV7+\/GaqFn7wA\/GQN8CEll+0x57UK6o3bg0uPTx2lygWXAcEdVYpZWMlUxlko1wLLbgwNGu9sAa5qAn9gFHJSpt0s\/DCxKmZpv7OsB5cB7+oLLu5taN+2dUulYJ68aPOY9hcAf90XaoEfvgJcs1yK5zKYqJKreHAZTEB1wKWzvzEYYOYqadJQR61rXyt51b2hlniB6iDt8dxz1fOpLbjUSGbBftkDn39btmxZx2G1SIZz+vTpoIrq15j2nnWqAy550eXxVC9y8abQ2WgCLUrXoXawZD1D\/EODxv+ht\/cl7N69GwsXLrTG5CsuXbp0yVprSkPCZ\/H8+fMYGRkBve35qFmkVN\/GKEdi2Uufbaj3Mate\/NUFl21uu\/RgUCWZpFYG+6x2UtAgQWiRqayznN2l87t2kkuEqDCokg\/TsQlwUWqJdG1i6HKWqVOnTmFwcDD4mT59elBEloD8Pem96Keakit9sutRgnF\/7aSgxmqhjhrqnhu1ApeQPP39\/Wg0Gti0aVMHuOScHIJ1cuKbY8eOYXR0tENNlNtknaT34vKIyQuuuLu\/bEo5OhXSG51uT6rjr3tAmPcgpK\/4dy56et5Bb+9v6qUWChVQBpeaXUoGlyh35MgRHDp0KBJcS5YsCaokvV+xYkVQphhwVePra77YstakJZM2eIaH8KCLP7z5rsjgyKw0J5evleSSh0KJVDa4Wq070Gr9ud0ZKaW1POZz2wTT253mwsNA4Xsuu2NpNI6i0RjzkstEcrVa89Fq3W53Rpy0VgcpKGikHyGft4HgUoYiDRr2mC+AxRZrZS2MUgv5NzltW9yea2hoqCMJjrrnSnpf\/T1XHIjSzqHsLSr9luKkpWyxWw5gX4bENHHjL9YzgzygSjh16tnukFwcUL2shbpGgCoCQx9CySUJJLpjyFmdmGKNv\/N8i3suHiLXL5ZLAKwrDBpCmq1bty6wIFLK+HMuWyBw1U6UpBGp1H4HBAfK9VQLuw5cQnqJHIn8vTwPDV3J5Grh1rldwTvv\/hQ1i049NMpeNsWY4sseZVn9M9MUw0941sU9Vz2vD6q15Cpr6tmvB5dL7ssGDWYGdelbaNfdSeVKbc+5XE5vWtsCXDTFf\/stHVH1HlqP+Fyu0wPgol7l1FJJbbkPk584tjiCmczwQsJolgHY337\/UwAfSwkKU5lgocCPAPyxox39sXV2X1troQUu5mpi1apVgQTzj+dAEgfoDf\/444\/Xxyu+CtPpgVWFWag+DTZDTcRou9qgUf0p9RR2Mwc8uLp5dv3YSuWAB1ep7PeddzMHPLhiZldcQiFe79q1K\/AmqesjX73bLWMS41DHVpW8mh5cEWhRUwaoaQnqCDB1TOql8nUcE2mOuv6qKmPz4FJWlXzpnyypojz567Igo5IEyVEGdZbITA+hRkjEzWHR8+XBpcnxOoNLTqEgcpJw2FHpFjTZUeliHlyVnp5O4qJUjxqRj6ivuwAX\/63rFVFxc6De713WXHnJpcF56vBqzg6NapUpMpnAFZejpYzJ8OBK4Xrcwixjskz7nEzgqpL6PqnBlWbCrYp6YQoqUU9NEiT+3m17rqjEsHl5l6f+pAZXEuM4Ua+99lpHpuA8jC6zbjdbC2VzfG9vb6Xu7\/bgilj13QQsWUrJl8dX5Swo70dHaB9VAxbH5cGlzK56wYP8Wk1FkHdhFFm\/Wz001As9ZJ5u2bIFIkFskbwWfXlwlcF13+ek4IAH16SYZj\/IMjjgwVUG132fk4IDHlyTYpr9IMvggAdXGVz3fU4KDnhwTYpp9oMsgwMeXBm5\/uSTT+LkyZMdl0SoTYhbNXVMwbbdq6J863i8cODAAWzYwPzu+R5bXh0c95tvvpnqNFzn0BgProxrTQAnKTKZC5BZp4aHh8evnY3rxja41H6i7pfOOOTx4vLVuj09zL9o\/pBHd911l1Z0d1zIjHnvxdT04MrIZ7FYWW3r1q1QF1nW8JS6gMtmjBR5yEs4mFNS3F+dNg22JGZaPzbfe3AZcJNeAc8++2yk36H6TvWMUL08osCleh2o6qUA+P79YcbbtWvXjqtXslq4dOlSrF+\/HmNjY0E55pbo6+vDvn37JkjVNHcolU4xrocffhjPP\/\/8eB+k5ZFHHunoV5XylES8aI43jvLjpHrFROXAcP0RMlgGqVU8uFJZNLFAnHRSVTBRjgknRUCiqjKqi0b1axQL79FHHx135ZG9v9knAbRy5crgvbrnUmmKygeiozqqoRxibOfPnx\/\/yIiPwqxZszr+pn6IWI4P6Y2iJ8q73abkNJhyoyoeXEZsCy\/w27NnT4cEUENUosqowJTBxXfyvWWCNLmduDKibBq4WC5rAp6oj0nUhyPqQxAHHrHf0g3rqVIQpO6S8eDS5ZRSLu2Ly+KbN2\/GzJkzJ1jEZCkgg+vYsWOBqjQyMtKxF5H7YrtRZbKAS+cjIA83aqxJgBsYGBg3VKh1WY9qLu9mYz4PWW1Os65WKRBSZ9l4cOlwKaaMPNmq2qLui9QmxD5JBZeayYj1VHBFlckCLhkY3IPFfQREmzbBFWWCj+JVlDW2asGQaUvHgyuNQwnv5QjfI0eOdOTZ0NnHsGldyUVpRQtbq9XKLbnYr1A1KWVE23Ep1myCS95vRbFW8I38VCW4l1w5Fmsdq3LCb7jhBhw9ehRLlizpiB+K+tKqe4cy9lyyNLzvvvtSz+SSVEBhSGGbUUYHGZi0AuqY4KPa8XuuOqIjJ83iUDnKfBy16VeNHHmthereTsegwTqyKiab8uPYEWctzAKuRqPRYYIXklvdQyYZguT9XM6pc17dq4U5WZz2RdVJgqPuodRzLnX\/oe5R5LOzKHrEB0A2kYuFvXr1aujkwY8758oCLvYZ5fIk6BNTodKpqs9yYtOc0+e0ugeXU\/ZWu\/G0g2OZ+rLPmbyHRrXXkqdO4kCaxI1ilk3fwiyT4X0Ls3DLly2VA+J6JJ29lkpo0RLEe8WXulR8554D1eTA\/wOkEZhoE9zPegAAAABJRU5ErkJggg==","height":117,"width":156}}
%---
%[output:4dfa0dad]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAEgNJREFUeF7tnU1oXsUax59uCkFBTEBIRIhQcXNVcBENSqH0Qot1J2kW2oLaUtHrQklpISrSGmhowIVVCCleaHSRFl01rgrFL4JuQsPdaIUbWmo2jRdEEbrJ5Tm98955J3POmTPnzJyZM\/8XumjOfP6f+Z2Zec587Nja2toi\/KAAFGhcgR2Aq3FNkSAUyBQAXGgIUMCRAoDLkbBIFgoArhbawOeff04vvfRSL+d9+\/YR\/21oaEhbGhH+1KlT9O6771YusUl+p0+fpvfee0+btlq+v\/76i9566y2an5+n7777jp555pm+ePJzXYK29ahc8ZYjAC6PBtjc3KQXX3yRRkdH6cMPP6SBgYEsd9H4ixrqgw8+SF988QUtLS3Ro48+alTqKvkxXLdu3eorV14mP\/30E73\/\/vt03333ZUHkuvD\/BVxcZvVlwHEnJyfp+PHjmRZd\/gEuj9YtasD87Pvvv9\/Wg\/Hf3njjDfr444+Jwxw6dMi4UVbJrwpcIuwLL7yQQaICXwSXeJksLi4W9tYezeIsK8DlTNr+hMUbmyFRh1Ec8ueff86GherQUG70c3NzlXoX7iFM8zOFS\/SGDPn+\/fsz0FXgy+ASL4wqvbAnMzWaDeBqVM78xLhBcQMumlupseWGzI2Y03j22We18xw1btX8TOFSwdD1uGVwsQbouTw1vBSysWlQHOfs2bO9YZeAjXu+MsdG1fxM4BLQyPMsHfCYc91t0ei5PJFdtbHnNVDTdEzDierneQufeOKJHty6oa2unGXews8++8x43ujJPE6yAVxOZN2eaNVhmmjI165d05awrIFWzc+k51Jd+nLBZAh1wOV5Lj3J30o2gMuT7GUODfV5nvdQHZp9+eWXfd\/Mjh07lrnGb9y4kbm88xwauvyKXPFFQ1LVvZ7X64pw7GUsG9Z6MovTbACXU3n7Ezd1jXMsnRdOpKbOxfKqYJofeyjLeq4iD58KPJeHPzLrvnMVfdPzaAovWQEuLzLfzcT0o26Zq9rUsWGaH5etDK6y57Jj48knn8yFS4C4vr5eyXPq0UyNZQW4GpPSPCHVeSAvLypzY8sOCN1HZ10pivKT08sbFpYNaeUXB3syp6amcuHisCK9p59+2mhFiLmyYYWMFq5ffvmFjh49Sr\/++mum6MjICC0sLNCuXbvCUhilSVaBKOHit\/v09DQNDw9ny2\/4x9+DNjY2aGZmprdmL1mrouJBKBAlXNxrzc7OZv8GBwczIX\/77Tc6ceJE9g+9VxBtK\/lCRAnXDz\/8kH3YlHsp0Zux+\/mpp55K3rAQoH0FooTr0qVLtLKyooVrfHycJiYm2lcWJUhegSTg+vHHH5M3NAQoV2BsbKw8UIUQnYaLofroo48IcFVoEYkGZbB4pX6Tvyjh4jkX723ibebCoaGbczFUvNfoHtpDO2m0km53aD0LXzVepUxaCoy69QvPevxJVzO4muy9ooTL1Fso4LqfXqad9HBLTRnZhq7AHfo3\/Yf+CbjYUKbfuQBX6M06jPIBLsUOJis0AFcYjTf0UgAuCwsBLgvREowCuCyMDrgsREswCuCyMDrgshAtwSiAy8LogMtCtASjAC4LowMuC9ESjAK4LIwOuCxESzAK4LIwOuCyEC3BKIDLwuiAy0K0BKMALgujAy4L0RKM4hUuXhh7+PDhXJkvXLgQxYZEwJUgKRZVdg4Xb5PnAyXX1tboyJEjvbMp1LKKdX3Ly8v0+OOP961Mt6iX0yiAy6m8nUncKVwM1gcffEDvvPNObwuHiXK28UzSbiIM4GpCxe6n4RSursoHuLpq2WbrBbgs9ARcFqIlGAVwWRgdcFmIlmAUr3DJzg2d1qE7MkSZAVeCpFhU2RtcwhvIZYz99FrAZdHSEoziDS7Ra\/Fh+rEfrgm4EiTFosre4BI9VxcO1wRcFi0twSje4GJt+XyKkydP0pkzZ6I+dx1wJUiKRZW9wyVfz6OWFw4NCwsiSrAKeINLd2xZsKqUFAw9V6yW81tub3DBoeHXsMitfQW8wQWHRvvGRgn8KuANLjg0\/BoWubWvgDe4ylZnsBRwaLTfIFCC5hTwBldzRW4\/JTg02rdBDCUAXBZWAlwWoiUYxSlctpsebeP5sh\/g8qV03Pk4hYulwTb\/uBsISm+vgHO45KLhgBp7QyFmfAp4hSs+efQlxrCwK5Z0Ww\/AZaEv4LIQLcEogMvC6IDLQrQEowAuC6MDLgvREowCuCyMDrgsREswSufgOnv2LJ0\/f77PlPJJv2UXipc954QBV4KkWFS5U3CVrbzX7SljGDc2NrJDc\/g3PT1Nw8PDvWO35ecDAwNZGMBl0dISjOIdLvmj8oEDB7JGfe7cORodHaWJiYlaJijbM8a90uzsbPZvcHAwy4vjnDhxIvvHv6Lnu3btAly1LJRWZK9w8UdkPkNjYWGBVldXaWVlJYOLexS+rOHgwYO1ACs7o4PzX1pa6jvaTfRmk5OTmeWLnotTq9BzpQWJbW29waUO2S5dutSDi4db\/P+LFy\/Wut1EtwJEnm+pebJocrn4\/wJ4MQTUDTUBl21zSyueN7jUIZva0BmMubm5WnDpAOW\/ra+vZ3OopuG6h\/bQvbQnrRaD2hor8AddpT\/pKi0uLtLY2JhxvLKAO7a2trbkQGU9l85xUJaJyXN5qCgPRXU9U9WeC3CZWCDNMAIsrr1zuDiTvDnX5cuXszu8XNwsKcO1ubm5rXdU51xq7yk\/x5wrTVBsas1Dwju07qfnEgXUbfcfGRnJnBzCG2dTGY7DvR\/\/eAgofrKHkPOGt9BWXcSrqoC3OVfVgtmE183beJ7FP3bz4zuXjaqIY6tAp+ASQ0\/5UnO+Nlb+fla2AqPsOecBb6Ftc0srnje4TE5\/EtK7mHs1aVbA1aSa3U3LG1xyr6L2JuL7FEP1yCOPNPJB2aXJAJdLdbuTtje4ys6Kl79BsfdQ\/ZgbkuSAKyRrhFsWb3CVrfuTnRHXr1+v\/UHZpeSAy6W63UnbG1xlK9bVnqvuUiiXJgJcLtX9f9pX\/vUP+vvfzvnJzEEu3uAynXPxkdbqtg8H9a6VJOCqJV8ykb3CxarqvIbyGfH8IZgb7\/z8fG9bSGjWAFyhWSTM8niHK0wZqpUKcFXTK9XQgMvC8oDLQrQEo3iFSzg1lpeXtVLjCqEEW2CHq+wVrhjmUya2Rs9lohLCeINLODLqbuUPwWSAKwQrhF8G73BNTU2R2BcVvjz6EgKuWC3nt9ze4Cpb\/uS32vVyA1z19Esltje4WFB5gW7MvRfgSgWPevX0BpfJlhN4C+sZE7HDUsAbXGFVu15p0HPV0y+V2IDLwtKAy0K0BKN4hatsaIhhYYItsMNV9gqXfDYhb4gUh3WKcyvOnDkThZsePVeHiWiwat7gUj8iq+e2607DbbCejSYFuBqVs7OJeYdLfERWbxzR3UASquqAK1TLhFUub3CpO5Hlq3v4MFDAFVbDQGnqK+ANLi6qelECz8HEvVwYFtY3JlIISwGvcHHV5SOnZe9hLJ5CrgOGhWE14lBL4x2uIiG+\/vpreuyxx4Ld3i\/KDrhCbc5hlcsLXDzk41tM+KfrocR87ObNm0GfnQG4wmq8oZfGOVzytUHsuFDv4ZJvg1RP4g1VPPRcoVomrHI5hUu3zUQ+HPTGjRtZj9bUFUK+pAVcvpSOOx+ncOl2Hwvgfv\/9d\/r2229JvrM4FikBVyyWarecXuBSdx\/z0PD8+fMUyzBQNRHgarfRxpJ7a3BtbGzQzMwMibuJYxEMrviYLNVuWVuDi6stX6\/argzVckfPVU2vVEMDLgvLAy4L0RKMArgsjA64LERLMIoXuNbW1oykjWUJFOAyMmfygZzC5VJddVW9yEteDcJ\/O3DgQJ\/jpOxC8bLncGi4tGq30o4SLvH97Pbt27SwsEC88oN\/6o5m9Tub7qO2vGKE01DvBlNXlACubgHgsjbRwSWWS\/ECX\/7x0QACLu61xNEBQjR5x\/OtW7dodnY2+zc4OJgFkXtA\/n\/Rc5EPhoUum2R30o4KLtHzTE5O0tDQEJ08ebIPLnl\/mDCRvAmT71peWlrqGybKaXKcoufiIFPA1R0AXNYkKrhkIRgaGa68O5flcKurq7SysqKFa3x8PEu+6PnExEQWRsB1D+2he2mPS\/sg7YgV+IOu0p90lRYXF2lsbKyxmuzY2traaiw1TUKAy6W6SLuuAgIsTgdwWfRc99PLtJMermsHxO+gAjwkvEPr3ei52D7yEQJ5c665ubm+DZnqnKvoOeZcHaTAYZU6M+dijeAtdNhSkHRlBToFF75zVbY\/IjhUoFNwid5LnNfB\/8cKDYetB0kXKhAtXG3aFd+52lQ\/nrwBl4Wt5O9cO2nUOAX2HvGvShzjxFsOiLptN0C03sKW2xIdOnQo+5iMHxQoUoA\/Hr\/55ptxfURu26QAq20LxJF\/kyszRI2dr9CIQ1qUEgo0rwDgal5TpAgFMgUAFxoCFHCkAODKEVac2SgeX7hwIYqravPaie6e69jrJOqq1i2UYygAl6Y1qruaY7sLWgeYWif1DjZHL2\/nyepOiw6lboBLMb98Rr5YAMxBdIuNnbechjLQnWMiL4SW69lQlt6S4R3s6iLuPBt6K9T\/MgJchorHDFfeVbu6HeGGcgQdDHAFbZ7+wumGHhEVn3Rv99h74yL91euw2rIVei4D5WO6B1pXnZTgyjtGwsDMjQcBXCWS5jXMxi3hMMGU4App+J40XGUu3FCGF3W5U88xEel1bc6lO7uyrnZ14icNV5FwbKivvvqq7zDTOkK3GbfL3kLWVbwkH3rooaCuuwJcmlbfJbDkXkq+ay2Ub0F1XzqhgsX1AlyKddUz6OXH6m7pug3DZ\/yurtBQ7xyQNW37RlTA5bOFI6+kFABcSZkblfWpAODyqTbySkoBwJWUuVFZnwoALp9qI6+kFABcSZkblfWpAOCqqPbbb79NN2\/e7DvHXk1CXPxn4gpuenmVbm0df164cuUKvfbaaxVruz14U6s6uN7ffPMNHT9+vLBMMW+NAVwVm5sAp2gXLzdAPnVqfn6+dzNmXjZNw6Xmo7sCt2KVe8Hl2z8HBgZsk8nisUa7d+822t2dt2WmVgE8RAZcFUUWjZWjzczMkNrIqm5PiQWuJvdIsYZ8jS+fKSmu2C0zQ1M9Zlk+TT4HXBZq8qqATz75RLvuUH2mroxQV3no4FJXHajDSwH48vJyVvojR470hlfysHDv3r107NgxWltby8Lx2RLPP\/88Xb58eVuvWrYcSi2nqNcrr7xCn376aS8PLsurr77al6\/ay3NPxBfN8Y2j\/HJSV8XozsBw\/RKyaAalUQBXqUTbA+T1TuoQTITjAyfF3EIdMqqNRl3XKBre66+\/TuI6Wnn1N+fJAB08eDB7rs651DLpzgMxGTqqWzlE3W7fvt17yYiXwsjISN\/f1BcRh+Mfl1dXHt3q9iZ7TguTW0UBXFay3b1j7OLFi309gLpFRRdGBVOGi58dPXo0GzLJ51rI6eSFEdUog0vMd+RFvGUH8OheJroXh+5FkAePmG+ZbusJaROkaZMBXKZKKeHK3rgcfHp6moaHh7d5xOReQIbr+vXr2VBpYWGhby4i58Xp6sJUgcvkJSBXV1fXIuCmpqZ6LwfdXWw8zOXrowYHB3vbRXjoWuZdDWkjpEmzAVwmKuWEkY2tDlvUeZGahJgnqXCpJxlxPBUuXZgqcMlg8Bws7yUg0mwSLp0LXqeVzhsb2mbIsqYDuMoUKngu7\/BdXV2llZWVngfRZB7DSZv2XNxb8XBxc3Ozds\/F+YqhJvcyIu28I9aahEueb+mkFbqxnmoPjp6rRmONMSob\/IEHHqBr167R+Ph4z+mgm9vw39S5QxtzLrk3fO6550q\/yRUNAYUjhdPUOR1kMNkLaOKC16WDOVeMdNQss\/iorHMf6yb9qpOjrrdQnduZODRkyNmdL7vy8+TI8xZWgWtoaKjPBS96bnUOWeQIkudzNU3nPDqGhTUlLnujmhyCo86h1O9c6vxDnaPI38505REvANlFLhr24cOHyeTM+LzvXFXg4jx1S55E+YQp1HKqw2d2hMTwA1wxWMlRGcs+HMvZtv2dCSs0HDUCJNu8AmU9ri7HJtcWVqkR1hZWUQthW1VAXI9kMtdSC+q7B8Gq+FabCjKHAmEq8F9vBX9KKFw5kwAAAABJRU5ErkJggg==","height":117,"width":156}}
%---
%[output:225cf498]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAIABJREFUeF7tXQ2sVdWVXoymeJEOlEdjgT5lrBY7VbCtgxKlteJUEmw0WmQSwTQFgj\/YoRlasFwrVqwQaXRa\/wiYGNFEtE40I5naEStRg5pWC+00Gm2DOCimvo6YCjpT8ybfuXe9t+56+5z9c\/Y59973zk1elHv379rr2+tnr732qP7+\/n6qPhUFKgpEp8CoClzRaVo1WFEgoUAFrooRKgoURIEKXAURtmq2okDXgeuGG26gH\/zgB8aVO++88+j++++nMWPG0He+8x3atGmTsdyyZcvopptuomuuuSa1DFd85pln6MwzzxzSTtY40P4tt9xCtVqtKzns8OHDA\/RLm3+MiT377LMEOmLNenp6gprs6+ujSy+9NKmb1g7PB2XKXJeuBNf+\/fszicTEnDJlCl177bVOi8aLtGjRooHFyqoIpjCNg9sBIF37dhpgiYVeeeUVWrt2LY0bNy7ptSiGjAmu8ePHE\/5MYwXoFi5cSGVvehW4mkwbC1xoDkxz1VVX0bZt22jatGklwiJOV7xxXHzxxfTd7363sHnEBNf06dNp9+7d9JOf\/KSF5ljXNWvWDBCmqI3CRPkKXAWACzv\/ggUL6Pbbbx9QKaWqxQvxwx\/+cEC6MbivuOIKuvPOO+nxxx9Pisky+De3DUbCB6rvHXfc0dIXt8Vt+OzYcpOZO3duIsW1NHcdq54zgPree+8RaxQmcOn56flrJuaxXHTRRfTiiy\/S7NmzWzQP9PHkk08m1bSmoVV7Sae0Od53331Omg36q8BVALi05DKpqcxEYDgwMC\/mgQMHBiQF2jnrrLOI7R4NWsmIXMakloKJ0JaLbaPHbqrrMlY9Zwk0BowGl6aJi60kNwMs5dNPP92iGmL855xzTgIwCS6t1mu6yQ1K05\/XzKYnVOCKDC4Tc4Npvv3tb7eoLJr5TGqpLgOGwEfacmxPMAPg31u3bm0BkqvKa2JmDXD07zLWLKkEdRNz0GVMdqxJC5BMLcdy2mmntdAZv0Gqr1y5kjZu3DgArkOHDiUbGsYgnVWyfy6jbWcTfdNA1pXgMnkLZ8yYMbDjm1QwJgB7FLV3ypUBuZ0sb2GWKpOm9nD\/csEluK688kqjiiaZ74tf\/GLi5TM5ckzA1EyRpc7KNm1jxRxMQNGbhQQXxmJSQbmOVvd47HLdoBpi\/lwWQOB2bQ4orUKnAdC0UQ4rcHWit5CZYO\/evUPULw12gA+7qQSCjWF9wJV2BGGzX1gKmphFbl62seYBFzO5HkOaraM3Rczhj3\/8Y0Lf6667jhYvXpw4ODS45ObItpaLdKvA1TynKdsVzws9derUFr3f5D1MUwuLklx6p5dMDJXypJNOSiSH6QghzT5MG2secLkehZgkF8bP0hBjePTRR+n6669PzhsluPbt2zfE4YT2TGqhVh19PMFdqRZ2ouTixWYbRe602E21oc0My\/aHqzSw2VxZ6liaaoU2s5hG22ImlSlL5WMVXM9ZqoV88K83RBNd0mwu6RjCmdfXv\/71Ac+epAu8ivrwWm+MWTaXXstKLbS4dkJsrjSQY+EefvjhVK+fVBNZJXFh2DRvIabGZ2omh4qLEZ5mk+hNQ0q5LMmV5S3kjcfmLdTSxBTxYlo3TX\/dDksu6fVjNZFtcrbVIOHTvLWVt7DkCA3pTcP\/s\/tb2zNgMNgG7CLnxcxiWJSRDhHYQVB74BGTh6f6nCvNicPMYfPIyTlBbWT7zzZWbWvCFpRnUS7nXLYzOhO40C48pvLAWG8erGEwDWCPHn\/88XTzzTcnG9XEiRMTqQfaAWCsRvuEg3WdWsjEeO2112jp0qX05ptvJl9NnjyZNm\/eTCeccIJtQxlWv8eIciiLIDYVr6xxuPTjq8mY2uxKcGFHREjLpEmTkvAcfLDjvPXWW3TjjTd2bcBs1qKbzqBCYihdGCtGGUhqlgIcAuainsboO0YbIxZckFobNmxI\/iZMmJDQ8s9\/\/jOtWrUq+Ruu0kure5i3zb0eg9FC29BngTb1NLSfIuqNWHA9\/\/zziV4spRRLM8T0nX766UXQu2qzooAXBbpSLXzooYdo165dRnDNmjWL5s+f70WEqnBFgSIoMCLA9cILLxRBu6rNYUaBmTNnRp3RsAYXQPXTn\/6UKnBF5Zlh2RiABfd9zE9Xggs2F+LAcG7CDg2TzQVQIZymr286HTr0qRS6wXV\/kIj+FJOuNGbMgaS9Q4emENFHhraPJaJ9ze+PaJb5GBH9b45x\/D0R\/V7VR\/\/7m9+dQkS\/FWNLo4nPEHjsXAdtNuZOJPvm3yEdUGfXkE4GaRZjXLr5Tza\/GLrO6LenZ08CrpjSqyvB5eotZHC98cbX6PDhY5rEHU9E71q451wiesJQRjKODwPqsiuJ6Cki+pX44TQieq05tpOJaC8R\/UVVPIuInknpeCoR\/VOz\/l1EhE1jLBH9RpRHGbQrPy704PJoT47pSCL6ax5ClFRXz\/EbRPQzImqMv1Z7m3p7f1GBC6vhes5lBhdacAHJRCJ6p8mgmsl9eeLTTaa3tVMUs2aB0ncupvKyfQ3AGO3LNnhdQtoFfY8asmlV4FK0dInQGAoun13atngnEdHLohAW7QNLpblE9HMi0nUXEvVMJep7jOiY84neRhkp1WxjEb8fUSf6aJ34AswOSfyIWRIfvYQIZAHuD2I+kKh\/JTpiOREE6G5IuvuaddHWwuYctAR0GSPmDemJ+XXOpwJXwFoMgusCOnz4bwNaKKMKA863LzD5Awa1LO8GIiUPNgyofZC8kOImyZsmFaEdAEwAq8\/HpLr61PcvW4HLn2aJlxAOjYbNtZiIHvNsBbbRRs86vsVdJB7UGTAdJBDsKfkBAwNQz1k6PrvJ6CFqG2w5ANn1c6qy9VzrZZVjdRD2EuxOdprIOmEbSwWugPUx21zQdfD5XUCLRVcRNldPnagPDLTF0CmYCKA0MRiDqFntuDrR67c1bb4m802pE+3H\/LHZpDkkmhLp6DrR+zD+pQrsQgc4aKRq67KJuLTrWsbdfq3A5UpTUS7doRHQWGYVSA\/YIDabK6TfkN0YnkKocewVhVr335bOUQbjR712f9KACMBgnL72XjYNK3AFrHe4Kz6gM2MVF69krL7QzviGI6LFoQGgwcUvP7yrX9jq6EiklHSGxBybrS1IOkjTrA2K7TvQFf8PiZr\/U4ErgIblSa6AwQ1UMTF\/nvZ86rLqBnsOEiHtDM2nTZSFs4U9jPg3nyv5tlNO+QpcAXQeBNdFdPjw0c0WYBiDkeThKn4Cg53habj7Dgo7LmwcH9UrywGR5jhAP5BK2vmhxntMnehtAMrXo6fnrcHkQhfQ2uaEMbXj4k2Emg4a8LxWE9E9ykZF\/1AvD1SHyC7LpcsUK7mK3I3TpFnGYfBX6kQ74Sq\/1UCqJU3mco2mYHVWR4q4OAmKPrB24YTzm5slpKctGocqcLmQNB+49MFuSI+dVCcNCHAWYPcX3r\/R9YZAbbHVipxLiJMmdDxpER0AILyZJUsuBMZedtllqbO59957u+JCYrGSK3Sx21nPVRUzqaJQQbEB2c670pw4Zbvi3elcuM2Fa\/LItLNnzx5asmTJQG4KPUSO69u+fTvh2RYZme4+nXJKFgcunCVhq4\/hAAg51F2h1D8fxnVxy+v1SZM0ABLGr72R5axvrF4KBReAtW7dOqrX6wNXOFwGHlrPpe0YZYoDV4zR2Zhc23RgYtRhdQ7qDtzWHJIEox0xe9pR4zLWLJWYwW8K04JtCOAFxkG6DK2lDGxAefgfx9NaKLi859glFYaCSy9OkRNBX5BuAEMZNoa0LaS9pSI2WqaMccH+CgFkkbQrt+0KXAH07mzJ5bLr1ptePh1d8c3m95Io8AgiVApSCNIsKyIDoDqZaDSAR0QfImIegMQRgawHFztCrEx322TfoZsWbzpwLkAqW44OhvBA2oZiYpZ0L2YFrmEHLpcJuXgw0y52on3tKZMgcHVuyHFiPHBtA3BQF6GWQjpDJV3vMiFDmSyp7jJ\/127TNoCzqFZ7kXp7N5VzWVI6N0xD73RHBo\/ZLrl8dlwfp4HrgmeVw24OxjXdfdL2GoOIIy4k4LIkpMnuM4HORcrGmDPacDlLy9PX0IP30iQXewMx\/G7PXmsHV55F0nWzHBRl2FxS7fEBAzMzVETUg2qpx8vpA0xAj7XpSCmlNz0dYZ+1btrGtI+vVnufenv\/rXjJxVILj4d1e3JNd3C5XB0PUaNcwOvDOLI9qZbpfnzHehLRnG80ThY+jB2460JbFzoVV6ZW+z\/q7X2geHCx5BoOyTXdwVXcwqW3bN9Rs0cFtdF2+ZMPdKEmwiOYFdMYct5msuvaQct8fZamFmKYyE+xevVqWr9+fVfnXe9scJkcEVDLYBPwVQoXdRJeQnj7FHCSXBpwMkAyQmV1uZ7B6qT2RuqoC9tVmiz3vw8QtH2UJeVZvXWhWattV6u9Q729\/1G85GJwyed5NDmGj0PDZ6E7oSwz\/\/Km29olEBcgRvQ3JB0iKUzSDvYaAOMCwDx0uFy427OcSaGqctrYGKS6z4a0Lk1ymdKW5SFnO+t2tuRqB2WkKgqgcqoxPkTms7IYY3O5GhKjn5A2WlXy0sA1PB0a59Phw9cEHFKGLJyPFzFG+65tADiQYLYDYd2eTQV07b+TyrF92SbJVTk0XJlB56twrcflwPSwE5BlCrYKDmmx68OGSgsMZrc\/PHGQPjJRzVlE484mOoj2PiBKLkSmJbrJGitURXgdMQafS41QwRDl4XMh1JdmunxoerpGO6VJrsqhYVvomGnDQg5Mfc6wAFbszlleRYwBf5y7AmdaUJtwc9f0yfIq+h4B2Gid53d3wJUGLlt0BqY7shwaMe0QF2YBY4PZ01Jfm1z4iKCHtANA4DmE5AAIYcDj\/6EOQpLIMyfEDUJicnmXsXEZ3hTwX7RpSvHm0157y5YGrvZOM27vYQ6NEGliG7ftIDXLfezqWuYxcEYnl9AuvbufTzT61AIOkm30ifE7aIwNxUeFbYNaGGOqndBGGLiKGnkaSKBm4QP3M7x29pwPjfJwxyPZp8snTZU0efQuJzpiItFHyD\/hmx+Qx5KWTaqIjcs0f788HoVKrtBLj6H1XNghRhk3cIVGJsQYoW7DxHyIuYMUgt0UmnRUz1HaRrZLm3KMkHTYBNJCpLitmNHsRdC5tc1CwYWuqmv+oYsYy1UNWwrMabsyv5DovKlEj+uXTEw2Gg6QoSZlPV3UBBqS1HyIi51ZB8mdtBHxetlUbvu6Fg4uOYThmaDmMxYmsy9CegkX+yZP+6gLLyWiLExSw0X6QC0F2LLAw4wKoKMv34gNzehpkRamy56+9MGmgAgV3xQDQ9XzUsHlO81OLe+mFsYYve8rILJPn0gGXVZLEhvIAVDkoHAJm+IxsgopVdYyJZhLgHLaGpo8q0PtzwpcARjIB65OsRuw+wMMRee5MDlcTAfCsLvg3udEMTZApy2ci7TNWnSfTSmbeSpwlQ6ugA6HVAlloCIkg\/QuajuR1TcX+zHvVZkYdHVpw+aZHIzcr8DlQk9VJp\/kCugwCReCAwGqB5wI8nBVMmXWbo8H93Cwa3JsuERnYEcHY8GNzuqfjdHkXDkCBQfTkJY2B4umkxyje5RECLUH35vleYZF01fgCqB+PnDZpE7WDg7GhJplerjONBENGpzTQC1F\/TSbw0WCgLlxbqYPVnUO+Czict5105ma7wG37gdABp05PMtnE7AxBDYZ\/CE2MtvGHHbguvnmm2nLllbmk5l+bQ+K234H6VvBNaNDwnTg\/IA08PVyYUbMfKYDZIAR6mRWu2OJRq8g+hDMLG04vZFIJmdpLC93yvhKG8CgagLMvtH4NvDE+31YgcuWSsB0pwxgfOutt5KkOfisWbOGJk2aNJB2W\/5eq9WSMvkkV7zF82spSyLZ1CxITDga4HAA0wNwAHJIpAXbfXWiLxHRr9MOjk2qqgSi7RwKwEZfvs\/C+lE1q3Tp4JKHyvPmzUuY+rbbbqOpU6fS\/Pnzc83MdmcMUmnDhg3J34QJE5K+UGfVqlXJHz5Zv59wAhY8BFxgCqhRIVIljSQhdoAM3wHYYHybMi+5LgM7KnwdJZCyACoAAjvQ9GH7MU2NPoNo9LmO8YrYEBCF4huJ4jovc9R+qeDCITJyaGzevJleeukl2rVrVwIuSBQ81nDJJZfkApgtRwf637ZtW0tqN5ZmCxYsSFY463fOWtWZkssv7s0VPq3l0piNJYqLYwRtQD10jXU0jTQrl3xM+yqMSlyrNHBple2hhx4aABfULfz7wQcfzPW6iSkCRNpbuk8QQY4L\/2bAswpoUjXdweW689kWEQzDFwxtZU2\/21QoSCAY6eyggO0DsMLZMJHo6MuJ3sc9LNvj4iFj8zn3C5HWIWNyqSOvx5gdG6WBS6tsmtEBjI0bN+YClwmg+G7v3r2JDRUbXH1906mvDw4N1w8fUOZlkrQwH5vt5DrOlHLH1RvmVkv8YWibLmdfaNvFe6nH4Np26Nhd6k2lnp5HqadnT\/HZn2ySy+Q4cJmCrYxUFaUqapJMvpLLH1y20bb7d9hC+NgeossaJ6QkJF\/nevEajo6soGOfdTC31dOzOwEWPlu3bqWZM2f6NJpZdlR\/f3+\/LpFmcz322GPJG15FvCwpwdXX1zdEOmqbS0tP+XtxNhcMbixSLLXL4sa+tE50P1zmLudALCX5siRW1eYmh02EuUB1ws1kvDIi6+P3c4mWTSTapL2FADge9G7nLWQX2zEbK1AJx4w5UI7k4qGYrvtPnjw5cXKwNy4U4pB++EAF5I\/0EKLv9ngLQ2eEeqHpAGwAcB1Tmk3EHjL0AwDhUFXHKYaodK7j6vxypdlcZZDCZLfBzsIHbv7he86lPWRwY8PtnBViJA9s2Qb0DVrVr1C6rrKUDKEBuq59ta9cqRl3y5im9hji2Vh5fmaLwLD9jjm4ewvLmHFIHzbvYUibXAfOBLjmoXLmcbfnGUNn1K3VRlNv75bibS6X7E9MkiJsr5jk7hxwwU4DM\/sGwWpqNA9qz6s3mvqD64skfL4F1RXSU77gqA19PgxW3\/fUifpkf3k9qTFXOl9bpaqFLFW0NOHvAaoTTzwxyoFyPrJk1+4ccIXOErnVAQacYwEgcB7APoL9FMPLJ+00W6AyzyGWjRhKk\/j1SgOXLVe8PIOC91Af5safeniL5YEr7YavCyNm2DKfqSvpxOdCaZmf0hwT5xNNO5XoFbjutfRk+y1LBc1K9oknW7EBuEpRuZ5ZtmPes0D3BKW12vHU23t9eWph2uN30hnx6quv5j5QDoeOvWYYuNwXpXFdwsUtj4DatHwUUL\/ASOr3M+tE8HZfvY5oXJ3oIJh3LNFXVhCtIKJbiWgn4v34RjARJWCUAGqqg0fXid6H29yUHhseRABOXUtBHWh+O9GvniccHRiz6RoKAI75IjVb0xmTJL8JAZ9pjTkVXazzrxLTWdsi1rXkyhsKZYdIeAkjuKbV6eiX\/0LvrxhL9K\/NfOrcxeY6zVnyBO147lyiWYoZFtTpuAf20uvLpxLdrn7rr9N1tJau37KWaCkCbAcDfy\/oP5lW0K301Z891djcd3PdI2lX\/yN0xmvP06ij+hvHZ58AwF4mWlyn\/qNGJde5Fn6wle57YhGNeq6f+qeOSrTCdfMb7Pu5cf207N27aNOoZl72VXX6n\/Xj6TU6gf5h1IVEN9SJrm32N61OF7\/8M3qKzqa+UXcRJWdoAOYjRF+p07in3qWD940nWgTUApTNQN3r6kTQTiEsH1bz\/mW9Ee0FDL2C31APIP4r0Z31xk2T2QA0vhv8jO5fSR++exTRJwyAu7jewO1S\/KYk\/0P1RgzxV5Xt1zO38eY59oeWMTYlOeaAPScZo\/ysJvrPI6l2xQvU+zeLipdc6NrF5kJKa33tIxwGxdT0l1x8VqVVtSxpxqqNPmcCR8IL5xtF0VT91jRDmBaBIeoNl\/20o4heeYfo1YkN6bVdMgskEMbC\/UEagdkBIH3Qi\/ngd\/1Ygmn++A7IB+h8Pq5Svd1tQnKNod7eTeWAC9M1eQ1ljngcBIN5N23aNHAtxIdMZZT1B1fsUTHgQhjtNKJxcwfVwYEwoMZum+DteYBLgxqog+SExMjInDSj3pSiTfttTr2Bwf+SgAUIoQJCxZOfzoloj7FipTk0Ygy2U9poBReuqoB7Yl7Kk0wW6fWTKXWi\/czgLtdTMAYAgG8hyxQBWAkYbgAabEOAHJIRgNSAYdd78+A4efYVNpU8A5MOE+2s4X+jfRyMxzw78wGzdJK4HXxX4ApAbLjk8rleETCw4CouOfzYo4gLlj655yHxYC+aNh+AEsaODJtyYXgei0tZE1FcvK3BxByoWCq42Kmxfft248hH1hNCLosHxsTBrO8NWm6bpUaWOxwSBaBHRAVLKwaCr9rJ7kY9N3UwPOTgWJZHWdhsISkE0I4Eju\/4eRxpgbtZ3tmh61kquLrBnnJh+XDJ5dK6SxmABoDzyXArAQfmgbQocgdPU2ch+QB2PiIwjQEuOoDL12njQjvXMvmBWRq42JGR9yq\/K2mKLBcHXH67YPp8QlVNXAEBwEwSYjCx5VDnBbyHbFf5nN3pGTTHfUGd6FE4SmQuD+RYxHFGJ39YK0hPhFM6uNIOkTuZjHpsccBVxIxdr3jkv6\/UuJ8Fh0baG8UMegAV7vm0Q3EXu8k3d72Jti79xF2T0sBlC3+KO61iWysXXD5X1qVjIkvlY6nJFxihhq1vJVpiF+E7XJbUSUibKhOkDpx3O6HiAdjCMYEgYJzxJhEUNvVTMj6XdfPIFbvSoa1D7V1Itdo11Nv7i3LOuWSALt\/qDR1+O+uFgStLWtiYL+Zscf4EKZKW0kz3JW0n190f5VAvRio5nztmOKTTbn6eT\/nR9qVJLpcrJ5W3MCaIQttqSrEkbk\/bQppBY6iXpnG6qrehcyy6XoMupYGr6OmU2b5ZcmUZ9yZXOBgVIUQhbnbJfK7ShHNhnEs07oxmhIYL1bLOwDAvBLryHKBu8jOwJoljooMEqKwfM4mMyzxjlmnQugJXAE3D1MKAjoxVfNQkUwNgYDgY2D6C\/QSnw31ESfQEfnuAKIl4T4s4z3NtgzcG9An1VF5VwUYBtVXHG4Z6RKVnMxb9XVX4U6lWe7w8m8umGlZqIe\/saeqW68JmMRIYFeFaMozoG0SLTyK6G99lvToCYMPBol83ka77WEys24Gkh5TME2YWKZTMcYqlSi6ZmxAXIjlZJ+etWL9+PXWDoyOf5MqyJ0JtGF8bhT1xWuXD9wA4X+WAFIE08Y1c19znqrq6cK0rjWL16UvbwTmUBi59iKzztpuy4bqQuh1l8oGLR2xT71yYw9ZGO6iDPnVgcNpxgitQfOcRDgj3nuwRHKWDiw+R9YsjphdI3Cdabsk44MKYi8zClEYT3AeTiWS4HL7HgbDpZrPDmRNSXb8OtY0vRYLBOeko+oDqiGgQHRGygujzY9WVFJS3qXA+ACrTOTK4pqWBS99Elk\/3IBnoyARXGZuCBMaFRHNOJtoBmwsHw9rDWSc6k4iehU3lm6hG7+Rzib50WsMhmhwkp20kU4lOX0j0vEufUiKGOjlCae4D5kYfpYELnemHEmCD8btc3a0W5omxk4uNG77w4qWFFHkwxunN6IkhV9C5DUgGOEia9hWuwcOXAWfisyYvIcCBO1yQPAh7SrtXpaVcHqnRDTGG6WtSq51Evb1ryonQwDBkymnpPewWTyHmkE8txA6Ij+l8y8XOkouJ8pAYrGpp17PB1kFYE0y15EVHKW2+SfTvn25ockPyt7uC2qbKubbD5XBeBCmLoF6WfNgQ0I\/MnwGawptoSpTj22es8p+iWu1I6u29qTxwZQ19586ddMopp3Ts9X4eezi4YjoglhPNGU+0Q79DzKOElMGVDfXc0L80E8OcCMnTPO9asyQRljfcVafH6Hx6fpRBJfxSnejXaX2tIDpiLNFHLPEYtGcR\/fPZDe\/5kGeHlhOdOV5JSUkfjBsRIpyFiuMgHey\/WPiwtpN9s6EUtRAqH14xwcckodgee+ONNzo6d0YucCVpzPgMSatUK4hGjzWkCTuS6CuriXbKeDnBgGmXDqES4q26z2HXh1QbBMtx\/QtpCy2hf1z3BB1X30uvj3qEruo\/kn5HJ9NTq7+aBLuPmvVLouvObphl+xuHyj39l1PfgYlEk9Deb4hmXEi0G+0+R3RTvSFcdsKh8SuiKWc3rpq9rcFGRAAphjWQckBysJSkcICgEZM0QnIb2Ggi\/VvSDOwwMDzTy8dhlBZ7CEmJP\/\/Lm4WDSz4bBMeFfodL5nbXmXitG0ebCoRLLj1gMAPOmkx3l7AZgTkDD2iPqQvmRr9NRptXb6QLm72OaEGdaNu6Rjq0vxDd\/MhK+jnNpR2j+JBYSD0kmtkBZufo+RgH2mkLKF30LjabVH\/LdnRk2VxvFxehYbpmIl+Y3LdvXyLRYj0hVBbW4oErz4hDz4iWEPV8qpmfHQ4DqHrNqIff1hv\/O58lDmwb2DNwsECCyWv8kB4cvqSz7Tbf3zK6\/DFnoBsbirxJ7fOuMtqIbd\/lWQtz3UIll+n2MQPuvffeo6effprkm8Xxp1dMi8WCy2WnNs3Lpx7bLQALwAVw1IluafJ8orLBZoOuZ8u2xKCQY4L3dC\/RuCVEBzl7b554xGLW0a9Vk9oImotEp6rBUsClbx9DNdyyZQt1ixqoF6FYcLksORYVtpe2OWTdLLDBtoH6pBwXyZV72BbwxBl+Nw6N1dbYkgQSE+qetHX8z5pcqJm\/DLy2GKu8bX0y1Wo7ilML9SPjPImi3j\/OTyS3FvzAVaRt4jJeMD0kE+dAP5\/o4lOJHtY53iHNoKaFBMZqO4fnjO\/Rd1oiHaiPkIymc720R9Vd5mwr43O729ZWm2yuLHBhSPJ51fAplF\/TD1xsH4BpQ+5uxZru7DwrAAAKx0lEQVSfBrk9Nq6158hqXXIZU+XUT56oxfFBvMcQBudgkuQ+qrTvOoylWu0P7ZFcwx9cvgfCvouXpzyPDdJqL9G0FeoxAdN5HB9Qxzyr4zkw8E32W555+tbF3KB6hkhv2Re0hQ+oVttZgct3CYZKLjAFPqZ4PHjkcFaj70D59pq3fKxdWmbehUqnHxmX45TqIlRAMC\/TqHk0kFzQlOFWaa50nzMrV1rFthO53xKu+dsuR2oSdEsIVCu45jVzqrczgaUrM9nKgbHhFdP53rPqIZoem4d2rkA64hOSuNQ2Tp\/fy7GvTCMq1FvoQwLfsjqqnuvLaBB8N2\/ePLrxxhupVqslRWwPitt+Rxv+Npfv7NLKp0mfUC8aMx6izQEGGa\/Hwcg4k4K0QSiS7sfmrLH9HosundlOV4KLJeI777xDmzdvJkR+SODwjWZ9zmY61JaeS7Sh3wYzeTbbB66imQiSC3YY7nTZDqkRZgQHjbyzZRufix3KgNcOFA1UrSaiHj76zTDbmIr7vevAxeFSCPDFB0BicEFqceoAJpm88bx\/\/37asGFD8jdhwoSkiJSA+HfW79yPGVw2ZkTrYF64nX2vlMA2gAcNjAM1K9TrGHo1xpA0NJUnY6hhseyrLJsqTaq62qZcP11r6CpwseRZsGAB9fT00OrVq1vAJe+H8drLS5h4a3nbtm0taqJsE3Wyfuf8Hu6SC\/YIbLG0KAe5kGkM5brYRe3AkGQ4HMUcstQ8xEhig5H5NlAXf2n2qC+IODK+qLnGbberwCWnDtBIcKW9uSzLvfTSS7Rr1y4juGbNmpU0n\/X7\/PnzkzIMrr6+6dTXNyPuirS0BmaFlAJz+wSkhkgoeDzhgk7L6f5NonGfTsl3CJAAePhvmlcUGw3sNp2F1zcTLo4EYAN2jvqXxgA9Pbupp2dPZ9zn8uHSzgVXEedATBl+W9iVUmzj2MbEAbmmg2XEGMLRYWJmVoVN10PQJ9qLcYERwIUETwO+Kz2KKId54q\/18XMGFnrcunUrzZw5M1rno\/r7+\/ujtWZoqBPA9cYbX6PDh49JmaaLDVYkhULaBkggKaX0sUVyaMcDX9LU\/cP+AU1MCXBCxirrYNywY7NiLfP2kVUfNIJaPJj4ByrhmDEHhofkwtRlCoE0m2vjxo0tFzK1zZX1u7\/NZVoQF49ZkYyAtl1tF4wVqp5JapnUOVuIVOwIjKIOgW3017YiwIVNqdVRNWxsLpCjvd5C24IU\/btNwvj2z4+IIypdMg1ABY+ljsyI3b9pvFnX6rNsUpta7Esbt\/LDClw6c297zrmKkExw9+LPdrfKbdGHlmp6JD\/fvCyJcKTkJjPu+bPUmko0emEzFQG72\/nSI+Yccjxg8hbCvoIEzIp4CXHYhNImvN6wAhdLL87XgX8PrwiNrIUuw2WfFQkibSpOUYDxSgChPqReWkgU1FUcSuP39kibcCgNrdm14IpJBN+2hp5zMWPrNM6+LRdRnscEewfqHaQObhg7HPZ+pk70h2YSmoGhQbLgE+q9w7kXXP4Aj04PYJh\/koQH5eEIwVygjoZcSXHVKHyOPLLXqwJXAD\/Lc65DhzjsJqAhQrzjsUT0ilNleJ\/wOXToHCL6vVOdOIUglabhiJ3oY4uI\/nfrkGYHxwZ6\/B0R\/UmBYEoTkLa5ohz6etIw9E8ifZR4nPwIIvooc4qt44pDjUYr44jooLXvrjznikmmkLYWLVqUHCZXn4oCWRTA+dbVV1\/dXedc7V7SCljtXoHu6D\/m4THPuPBD5O4gbTXKigLxKVCBKz5NqxYrCiQUqMBVMUJFgYIoUIErAmHT3izjvI\/chU6saruNHWFoA02YUjnce++9pT6\/a7pqZBuX7feYNIrdVgWunBTlaJOJEye2xELq3Ps6KsUWpZJzWEOq65va+g222P3p9tISzNrGZfu96HHnab8CVw7qseSZPXs2HTx4sAVcpl1axlTa4is5l0iO4bVIrVWrVhH++Ja2DIQu8vF4meoBF2dxgZbv25nyq8hxnXjiicmY2zHuGHSvwBVIRckYfX19JKP00xhXpjK47bbbBl7r5CEU9SRultrKL4YGksFajTeR5cuXJ3lPcNmVwWUb1xe+8IUh6RzQoWnjsg6kDQUqcEUgOkAjwZWWwZjL3XrrrfTjH\/+4hdEwDH33LcLQkib0+Lhd09WfWH3qdkw30G3j+vKXv9xC13aMOw89KnDloV6zbgUuOxErcNlpVJUwUKACl50tKnDZaTTiSujko\/pqjEntSkvCo20u1JWPXBRpc+kMXGXbLiaapKnBbFPB5mr3uPMwfKUW5qFeilqYxrgj0VvI5DWBq\/IWRmC+4d6EyTCvzrlaVz1NmtvOsWy\/dzJvVZIrwupkeb3wMid\/RnKERhq4bBEYtt8jLF9hTVTgKoy0VcMjnQIVuEY6B1TzL4wCFbgKI23V8EinQAWukc4B1fwLo0AFrsJIWzU80ilQgWukc0A1\/8IoUIHLk7TsUt6+fTuFXDZMC+r1HIZTcURAPPHEE3T55XgWqJFGXD+95NRQs1DaS6E+bbiUTTvacKnbSWUqcHmuBocoffzjH09qynecPZsqtLjp6du84ALT87ndZz\/72ZbQrZiTqcAVk5pd1BZHDFxwwQW0du3alreeO2kascEl28MdsAcffLDlcmjMuVfgiknNLmlLPhgxZ84cWrZsGV1yySUDl\/9Y9ZI58HlqkydPToCIN55Rb+XKlUn+CpYmM2bMoB\/96EdJcS6LFza5LRkwbFItJfMvXrw46WPPnj1Je9OnT0+AsGPHjkQtNPXFN5TTlkKmJcCN4qVLlyZP8cpbzAyKb33rW7RiBR7qa3y0+qyDob\/\/\/e\/TPffcM9CeCVw630iISl42m1VqoQfFdbwgpBiSjoJx+WF03RwDAUknEQGvgcFMw6FRMtxHf5fWBvrUkipNcgGs3C6XcVFvZc4NpCDArWJdjx+ZlxuBplFaLpE333xzAIQaXOj7jjvuGNASuI0rr7yyZWPzWMpSilbgciSziRGZmbJ2UR14agKXZBwMxwRa2Q7GIqWfD7i0Oqc3DBM59AbBElqP29SWBpPp9jNvMExHCS70laYhFKmaOrJFZrEKXI5U1ExiYmjdlN5x8bsJXJpJTI6HWODS3kIXcJnK6DfVMDeTOifphoQzJqBo2sp2Xn311eROF1RqqboWlRLBkR2cilXgciJTw41tsqVQnW0kvfgmu6QbwaXzL0qSsT0HtbgocF122WXGVTLR3XE5SylWgcuBzCa1iKuZ9P+s8t0Griz7RqvFRYHLJLkclq3tRSpwOSxBlupkssW0nSW7iAku6anUgHZ1xdvUwqzkobpPG7jgWfS1udBHlmcyy5nksLSFFqnA5UDeLLCwrQHVBQb5vn37WjxbuvkY4EKb2lvHqpv2BE6aNGngsNdky7lsHLKNLLtS529EWVumYf4dZdmusnkLTfaewzKWXqQCl4XkJkdGGmBwfrR79+6B8yVdDuBjo16ec\/k6NOAK1zd0161bR3v37k265KQ3rLbJczMfh4ZNqknwwC1+7LHHDskzaKKfPOfC2L73ve\/R3XffnZxzwW51OefCfDm5aOmoceywApcjoapixVFguERkaApV4CqOZ6qWFQVM9qnJNhwuhKvANVxWskvmYUo4oxP3dMlUrMOswGUlUVWgokAYBf4fKTEgDYFtD0YAAAAASUVORK5CYII=","height":117,"width":156}}
%---
%[output:7f236b31]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnX2MV9WZx4+7bZ3pxhSBjQ52oS4jG0PFaBNG6EZ8SdoUGk1QIGmBzYKEikNiW3DQoVFSRqGg626RhgDGALoB1KTNSlNTfPmjkmFXiXZds1sorlt5yfIyTbYFdrs7m++B78\/n98y59577+93zez03mSi\/c1+fcz73+zzPebmXDQ8PD5u4RQtECxRugcsiXIXbNJ4wWsBaIMIVG0K0QCALRLgCGTaeNlogwhXbQLRAIAtEuAIZNp42WiDCFdtAtEAgC0S4Ahk2njZaIMIV20C0QCALRLgCGTaeNlogwhXbQLRAIAtEuAIZttLTvnHkrLlt4pWVHh6PayALNCxcBw8eNIODg2b58uUNZK7wt3L75nfMG0eGzOvLbsoN2fnz5w3+Ro0aFf5GW+gKoezW0HAtWLDA7Ny500ydOrWFqjL9UQjXo1+51jz21WtzPTcayYkTJ8zVV19tOjo6ch3bzjuHsluEq8Fa1WXffc3eUYSrdhUT4aqdret6JcB128SLbt3ry27OdS+hGkmum2jCnUPZrSbKdfjwYbNq1Sqzbt06093dbc2P35YsWWKOHTtm\/z1u3DizdevWUjlirnZzC5HMuH3zoQhXjQFtWrjOnTtn+vv7zaFDh0rw8Leuri6zcuVKa8oNGzaY48ePm4GBAdPZ2WnaHS4kNYafvCNXMwvVSHLdRBPuHMpuwZVr7969Zvfu3dbkVC6o1vr16+3f6NGjbdmZM2dMX1+f\/YO6tTNciLfWvHo0wlUjUJsSLqTSN27caFasWGGViXDhdwBHlYINqWbz5s0zPT09LQ\/XYz87am7rHlWWbodb+JtH7zVTz39gFlzVbwafeSBX8wrVSHLdRBPuHMpuwZQLSrR06VIL1pgxY8piLqjZgQMHnHBNmzbNzJkzp6XhYmyFdihdPwDXvfM+C9cPPzfb\/HD7plxNNVQjyXUTTbhzKLsFgwtKhQ0xlU5o5IGrt7fXKtmUKVOasNrctwy4vvbsB7bw3MD00k6Aa9HOO+2\/Ades7\/Tn6khGIzl58qS56qqrYj9XjtZCu02YMKFQuwWBi+7gli1bbExVDVyw0fz5823msFW2vx08a54+eNY+zt\/P7jK3XNNp\/x+\/3793lv3\/gx3Xm0\/dv6lU5vPsFy5cMKdOnTJjx441l19+uc8hcR9jDOyGv0mTJjU+XFCtbdu2OStu7dq1Zvz48TYWI3xpMReUCyn7VtrWvXncun+Dl19fpk6vv\/qambjjPtM5eZp588iQGfvwrqhcNah4KNfQ0FDhI1uCKJe2h1auds8W9j31vFn81sUuiGPr37cAwVX86Y\/32d8B17n3D5TKfNtXqNjB9\/rNul8ou9UFrnbv55JwvbDkF3YMIUZmIJGx8+SAGTP3u+b0nicNy3wbbahG4nv9Zt0vlN3qAhcqoZ1HaLz0xGpzw9vP2ra4ffoG87W7Z9qRGRGu+uDZ1HBVYrJW7kT+j0fvsW4fNvRnIXmBrXfoZbP8ty+bz6950fZ3aeWC6\/jG4aER\/WO0b6hGUkn9NdMxoexWE+WqxNCtDNeu+XeaGRNHWcDS4DqycJv52l0zS+Zj\/1jSiPlQjaSS+mumY0LZLcIVsBUABmx6ZjHgAjSIq9CftWnU7JJyPfT54zbmgnJpuLLmeoVqJAFN1BCnDmW3CFeg6pUj3PXUERdciLfgEkLRsuDClBTXdJRQjSSQiRrmtKHsFuEKVMVJQ5xwuX+7d1wpI4h4C+qFLCE2pOH\/bM1Ldp8k5cJ+rhHzoRpJIBM1zGlD2S3CFaiKAVffUy\/YZIUGAeAgaQG3kIkN3kYaXJylHOEqttIiXMXaM\/jZfvqTfXa0BeACSIy7AN24vslecME9xB83wJU2HQXn\/vE\/fWieuPv6QofxBDdWnS8Q4apzBaRd3rUcmuzL4igMnCMPXL\/80iLzy5sXlxaq6XngGZsIwVwvvToUBv3i9wenXhnhytlWIlw5DVaL3dPiqiS4fv\/+WzYT6OMWIhb75y8tsskLupmIzfD7PY98vywLyUziLdd02P3j6k\/+LSDC5W+rmu2ZFlfJjmLZGZwXLqTpEbPJiZQRrmKrOMJVrD1znS1pFVyA8vIT37Oj2\/XERgmXzPoVAZeO4\/AwVC78P+aISeXC\/a\/52VHz6FevzTXKPpeRmnjnCFcdK4\/Lnem+paSkBW5V9mW54Jr04jEjAdTZQpT94DddtoOZyoVECDYXXDKTqOGqZhXfOpq9ZpeOcNXM1OUXSourkErHHzaZtMC\/ZV9WHriYIUyCi9NR9LhDmUlMgsvV+Zw0iqRO5q7LZSNcgc2Ot\/uMiVeOWEKasQ7GAOoMXVJcpeGSKXW6hUnKVQ+4MCK\/khV+A1dJzU4f4SrI1EnxE90q3eGb5volxVWEi6PbJVw8H+CSysfHc8EFqNHZjH4z11wvJlYwLvHeU3eMeAmkdT4zhe+Ci6Pw865ZX1BV1ew0Ea4CTA11wqZjJ+n6aXVKc\/0A12cnT7eQ6KFKHIWBtLtMqfN81cIlr5eVSWTM6FpolPGYy2XMGijcKp87inB5wuUDkFYnumpw\/dZ\/5xtlGTXZX6XjnCS4dEZQJiaS4GIsJZWLU1JwT1QuqmFeuJJGdlDV8sLVTEmSrJdAhMsTLjYWrUAw8EuPf89m33QZXTVXYiLL9aOb5oqr2FFcDVzMDGq4OIOZ628kZRJl5\/NfTNg1YpxjCFVLAy\/N1aw0uZJ2HNzeN4+cTf2oRYTLA640906qE4YQyThCun4udcKl0bhdrh9+YyzEcYBJygWoMb0f15NuIVSL19DKpeHCcXA5obKfnTytNHoDcHHVKD2Wkav4ujqfuXaHa4Bx2hdX0lxGnzLXx\/3SoEzySGC3pHgZZT4KG+HKCZcO0KU6uQDi6HQNkG9chXGA9zy81t4l4UJ6HuBRuQgXyjGthFCnwUU1lVBquHg9V5o+LR6TZa7ls9NULc2dTPvGWLVlLijTvBVkQi\/G2clf6oxwecIF1w9v4bzqlASX7K\/So9Tl1BFcU8PFdDvWIGQqH5DkhUuqHJWLM5gRP\/LFkQQXXUatXBo8OcokCzwMIsYET459lNVTLUCuzGXSOaW3oo9LK5P3G+HygCtNnaTr51KnNNePcZVLnWRcxYbm6suie1cJXFBA3D+OTYMrKU2f5jJK8JLg0uvWc2Q\/bIbYb\/13vlmqHQnl0jv+oSzWYfyHnfXLr1JI0kIBxtmul22EK+c3kbNcvySApOvnUifGVVlwcahSUkcxIKkELrixN7yz3SoFZym7lIuZRDRqdohLVYOCuuaWwS4uVUsDj2XSJjhPFng8zgVektdBSPBVGDkFh9eD6\/fQNcdN5+TpZbF0WnuIcOWES6qTfqMCIMKlG4R0\/VxlVCeX68cyun5w02S6XY\/iwDoZPm6hTrLkgUsmO7TLKIdp8SUAu+jxirJMK5dvmbalPM6leGngJZUhG\/iNrV+2datj6QhXgrtXydJq6JPC2x3xkwsuNCAoR6VwITHBeVQ6aaHjKp0RZOUvfmuFfeKshIZ+Ufz+\/QMGIzBcysV9qVwuuFwuIxsfyvD\/UtVkkkQrniyDTaQ7WSl4EgRdd75lWg3TXrZRuXIqV5o6SdfPBRdcvy++vd2+wXViwtVfVQlcvEZeuAALV4YCXHJQr0zvS7iYWmfDlAuN8qN7Ei40RKlqaeBpV3P+rv2ltirLNHgSEq2GskzXj2+ZhjLtZRvhKhAuun6AQgIEI2dl\/Vyun46rfJRLQ5KWipdv3Sy4sC8U+08fft58NP\/PSwuNSheVcKEjmVk1XoOqJuGSZVrVdJmESz4TbJJUlgc8aQsNnl4aXCZX0l62Ea4awOXq8EVjQyezLpMAoTGjc5ZpcjRAmW7Hv9Gw0iCpFi68JDA0Ki9cHOrkC5dWtWrAg02xabgICboTXGWwN2NDehb4t1QnDV6Eq8CYi65fkjr5uH4yMZHX9UNjP7xgm3XhdNICj1mkciGuZIYvDS42Pi40KuMxHIf7ZJnMMrKMA5O1quE4lLlUTZZp5SJcWtV4n0ngES4Nnk8o4GoPUblyKpd0\/XQlcJFNxFWoWPjnGB3vo05MoWt1Skpa1AouZvgwCgQb3UKZpnfBxXhMl2mXkZC4lCupjOckeBIugpBWhufQ4PG4rDIXeLhWhEspmG+2EKlYjhOUcKGCoCKyTMdOSAbQvUMZ1EACxLKi4EKj77nwgV2yOk+20BVzcUQJh1hlwcUsIz\/8gHiMjdalarpMgich0eD5ltFtZrX7ApQGnuuFimcDXLosKleGcnGiH8eNabgYA1UKUFJchYQA1CmvclULF0Y1oLHovjOm97VywS5UtZBw1Qq8JMWjOukuA90epIrWHK69e\/ea1atX2+tOmTLFfrN4+\/btZd833rFjh+np6UmIlIr72Ue59IhnmfVzJR\/yqFMSQDKuciUtENcBSlfMFQIuxEn41CvUMDRcTITkUTUNHuyHaTJ4MSVBCbvCbedqVYyl0bp0GRcCgr1lneMaOhSoG1wAa8+ePaWPgPOj4bNmzTIDAwOms7Oz9DXIdevWBQcsD1xMLRMu6d6hDJ2vXIgzqUy7fj7qlCcjiIZBuKhAvtlCfZxULjRQLBRaDVxFqJoED0qCrB9srgFCGTYXXJWUUZ0InlzrhHDhZafd0JopF79RPG3aNDNnzhx7XX5WddmyZaXf8DsgPHDgQAm44rSq\/Ewh4HIBRPB0Sj0EXHhro6I5frFIuADsn9z1gHUZ\/+7UtaVJodotlMkOKlASXBzwCkhklpFKIgGqFi4qkAs8CZCG0gWXfNnSk6gbXGfOnDFLly41K1asKCkSf5s7d24ZXIODg2bjxo0lhWsUuPKok6wgV+zEt21SXKUrzKcvS9qpEeBiPAa1wP1wjhgziRognQiR6uQLHkFwqZovQHnhYihAN93VXoNOOYlwlc8MRmOBEiTBhXhg6oUPSqPUfeDiXCtUbpFwMZMolWvw4\/N2FWCMg4RySYCoXL5w+YLHWdEuKFnmgouKVw1cOpZ2hQIRLvFayesWauWSbziW0S3MUi7cBkZYA66kuAruHQfSSrj48XBXup3p89BwDR85VFpxasfJAW+4pMvIxu6CS0MiVc1Vlgcul6vJWBr1ouM4GWe76pzgMbMblQtTIA4eNAtSUvFykhzefq\/MOF+WtAgNlxxIK+Fi5fEjdjKuqhQunJPxj07Fu5QLcGF\/lCHZwQQKGiK\/ZIlhU0mqhj6wUHDBpcaLR8dxLoCkqgESCZcug20jXJ5BWgi45HR56fq51AmuX+9vXy6NFvBx\/eSjFQ2XdCdltjAvXMxawmXEHCmqKBq3\/Pj5v\/77\/LIylzr5qppUrlrAlfSyRf3XTbnee+89r6bPPrDRo0cn7s9s47FjFx9IpvRlNpLl48aNM1u3bjXd3d12\/0rhkq4fzoN+oBfHvu7s8GWZCy4MAsXcL7zp9QDcJHXKCxfiALxxOd6OAHE8H4B2JUI0XFAnNHRmC5OUi8dR1TRAIeDCOQEXRqszNV6UciV5KxIu2R7qApcXUTl2IliyPwz9ZsePH7cpfGz9\/f2mq6vLrFy50v5blqNPrVZwQZ3QwDVATJuHhIsmlSlx\/uZyNXWshn3p+hUJF+JJAkt1Inineg7ZF5VL8ehOwoXDCwIwUSkxoVK7fjLLWElZW8Ll6gsDcKtWrTIADtv69evtH9UPGcu+vj77B\/WqFC69TDQbH66JMjlTFmVoSBjvl6RO9YYrLcuYBhfK0KUg1UkqF7KJiBtdkOAYbFzIR7qMLJPHuQDyhQv2ReaS34nGuWQ8JgGqpIyj+flCcGlETVLxRbqF+iEkXKdPnza7d+8u64hmR\/a8efNsX5sPXFzsEu4dK0EDRAXygUuCx\/t3wcVz6rgqr1voo1zVwKU7rSVcnMIhFyHl+opJcOF8UDQNHuECrNhcyoUECkfJEBK+AH63qGNEGVUtL1w4p2wP+tNONYdLXxAdxQsXLjRr164d0YEM9ZGxketmXb\/J4VX79+8fMcpDjxIhXL29vRY2xHlyAwj7\/mbAKo9Up\/G7fm3V6dQTF9++Ei5dJpXr3m2vlB3Ha6FxY8we3ny\/+8kzZfeQVoaYbda3V5eO05AgPuLG2AsNnRuv5zpO3ot0C\/\/4K4vNfz7xTYOYSz47XUaUMWPJ6+gyaS8o1+1fucPOboZbKOEa+\/Au6\/qhTNvZVTbz2\/22TnQZ7h8zF5C5RBk27McEClZ1wthJXYY47qWeodI5eRz+i2RU34wuW\/brNW9nfk0TynXy5EkzYcKEQr8lfdnw8PCwbDFs5DIekuWVDH\/SndSucyTBhWvPnz\/fpuXl9n9H3jF\/+FGv\/QkVz7fmZza+ZWSZPEaXSbj+cu1253GXTbzJfPr+Z1LL\/vfV7QZ\/cvvHyd8wX\/7rXvs7\/nAeNnpAgL+0Le041\/Vwvj\/MmG9t8unfvF\/W4Hm9\/\/nRA6V74LXfvmujmXbrdMMyCden7t9k\/mjizea\/V1wcD1hNGeoIqqfPKaFFGTbWK9xaxn+67K+mv2iev\/GE3fedb++3+7naA8puuaYz1dYXLlww+Js0aVJYuFyjNeSd5R3+5Bq3mAcuKNeSJUtGGMdXnXggGoZWJ0JJBZLn5HE+ZVA0qIlUGSqXqwyqJVXKVfNpx7FMHodz4k8qF8t5PZdyUUlYJgHKq050NfVx8j51mYRLqpN+aeoyvBh\/uuh6q07aI5EvW1\/lGhoaMldffXVYuFwwSOPorF7aKyFJBV2A5o25ZHyUFldJuJC0cB3H2MkVc\/mU+XYU607kNNv5nFMer7OFukx3PrOc8ZErHtOxk7SzLpPXy1PGEf8AU3YUa7j0OQHXvhnnS10srrrDORo25tJzt6A4mzdv9oq5mI6fOXNmKd3OCkBZtdnCIuCS6oSR4RGuA2W85wGvUrjSgPUt00kseS9JfVtyn6DZQtcblO6hzCDqjuCkN28aWDjGpWh5+7mS3lRJhmbWr1KAfI7LGpxbqXJhEqBrqoq0P5MPUCDGdiyXGUGd0NAAuSCRndZa8diXFRouQsLr+XgruKeGhCvNZckq40RL135UQz2CI+8IjXaHixlG2jjC9ZZN5+ut5eDKgs+nPKufqxHg4jp5vJcilYtDohB7uZSrWeFiDORSPK1OLveu7ZXLB56sfVoFLjwnACEscMvopmXZAOVwyaqFq1rXr0i3sCi4NHhpAwCy7FzzmCvrhkKXFwGXVJIQMZePctFOcvHKCNfFxWOSXLhKyiJcOYgsGi6flLpP0kI+ggsurTJFwYWxgDPvnmnnZmklxDVkzNXR0VE2EkMql00ovf9JVrBaVctKaMgXHK5dK+XiyH8MJs7aonI5LJT0htMxEAzNmcF5AUqDspZwoV9HfkBBupkSro93DJjPfPwvbQ+XnFUd4VIWyFIu7O4LF\/ZtFOWCS8glxLIqXcZcWXBBgTBUiXBhEK4eSCtdU17bR7n6nnre4PtgPB+O9e0o1sqVJ2mRltDIirkiXCmtywcuHWjruEpWbNFw4Vo4J764IZUyyy1EZ3WejQkNuR4fJ1HKjGFIuLD46hfffrYiuDjplM9cLVxYgo3DrOQ5tUcS4aoSLr7ZGUPUAi4CmwYX7ovrVuiYqwi45Dn57PWCS77g2LErVa1ouC777mt2ALAEzDVwIMLVYHDhdvQMYK14Ei4CpJXLBRdjpDwuIc6D46CGM9\/ssJMKsUyzD1xo6FimQLpwlbqFacolz5kEF14AiHnx32qVKwsu1k+Eq2C4NAiY1\/ODj7tKyz1njR\/UjVau\/0CAEDMBQDSkPHDlcQX1vvqDE5XChdgJ2xtHhux\/oQA+MVe1cN176g7z4tjXLOx54WKd8pkjXNW0pEvH+sRc2i30gQtLsvU99YKNH+ha6AqkqyPXeJBwobFjaYBmgwuQEC7Ok3LBxWkgmD0MpWw0uHqHXi6L\/6RbGJXLA76QcN2++VCZ354HLiwWuubVo2XrAWq3UGb2PB41c5eilCsPXJtGzS65ob5wcR6VjrkqUS5C4lKuCFdmk0nfoZZw6W\/pMu5yKVcrw4UaAQhwebExxkuDi64mpuIXCRfd7whXlSC5DveFS4JAN4ZK4oq5uFIvVxzCtfVwJBdcvEeqiFzJtlbKpUcbyEydzhZu+txsu\/aETGhkKRf2Hf9fn6ztQbhgM8RL7AIAfLMf\/r7tr8M5Z0y80rppsA3+W4RypcHF+I11Et3CnAA2KlwymcAPHDQqXDe886yNK9nYK4ULz8wXDocVSVUDXFhkBioGoF1wYdYw56ThfFnjB6uFS64QnNX04vCnBAux0rGYCSs8pHLxNtCQOidPt43KBZfPmLasStdqmUe5Fl7Vb0dVyIxgPeHS3QhFwAXQMeYSC47qIW8RrpTWlVe5fOG6mC07a7NgrOA8biFvme5QI8OFlDvWIHQpF54DiQG40lBg9le53EKXchF0aYc05coDFxf7wX\/pHdDuSMXTLUSMh7rTdYD+tAhXneDiZauBC\/HFbd2jLKT1Ui48B9a5RyNEDCTHFkK5suCSSQsXXFIptVsYEi7WD+yqO90lXPKlGJXL1w\/y+BADT+VyC13JB46Kl7dQDVz6UTgCg6MpauEWUoXRtQBlwPp8x36xz3zu9FH7RoeqbHnt64nKxc\/w4DxyvXombfLChevBFR185mJSBPZFJhYZyDzKldZMABeTUdrjwPWYwo\/KVYBy6QYmhwfJzF5ouPgozEbWCi5clw0OcJ04caK0\/l6RcFEZ9LAi6RZqKLPgkkPJkOzwWe9CuvQRrhxqJXf1jblCw8X0vu9jhILrzSNnzevLbnbeRgi49PWS4GJjx43VAq40lzEql2crbRS46OJ43rbdDY0MQXattjS4MFWESQtmCx\/96rVmzc+O2tsjsNot9IVLPqOGC+e0Kx073EIcp+MkH+VKsym\/holQIbqFTeAWVgJXraDiddLgQlJD9kkRKIBgXwSXXgJZcDGGShttDhXDRtecHx+\/99TtBkBLlz3CVetWIq6XR7mSbtMn5oLfr2cHyxEazQ4XbMOGTeVyuZdFwKXrgXAlrYsfQrnQ7cCkEu7HJ\/aNncgVgI436RuHh0z3zvtKa2jI08iPasvfWw0uwtTIcMH+eSeS6ibBr1PCFUUGNcKVAE0RysVTY2p4UrbQtcxZq8AF5Zaxky9cSe8x2AXnxIh5H0XwVS52clfw\/nQewqRShKuOcHEkgO6obBW4qNyMq3ScJU3P9TiyPm0EQBHHRbiyXwUjPn6XfUht9qiFcqW9oQEYhlT5NKLaWCT5KkyHh4od5JUjXP61HeFy2EqO+mgGuPgIjQZXVjNkQiO6hVmWKri8nsqFSgdgGMQa4Sqv2DzKldUkQsPlW3ehXkpRuRJaQIiRFlmNrdryUI2kUrcw63lCwYXr5unID2W3CFeEK4uBsvJmUK5cD2SMiXDltZjYPykVn3bKqFxu60S4\/BtiVK6oXP6txRg7jQVb0iDiPCcL6RbmuY+oXHmspfaNylWF8dShES5\/W7aFcnFgqx44Gt1C\/4bCPSNc\/jZrC7jklHxf08SYKznmim6hXytqC7j8TFG+V4QrwlVJu5HHRLhiQiNXGyraLURnfd4PAua6YY+dY0LDw0hF7hKVy21NPSGySJvX61wtB9fhw4fNkiVLzLFjx6xNx40bZ7Zu3Wq6u7vtv4sc\/lRJpUW4KrFacx7TUnCdO3fO9Pf3m66uLrNy5cX1zDds2GCOHz9uBgYGTGdnZ4SrgnYaqpFUcCtNdUgou9Ul5oJqrV+\/3v6NHj3aVsSZM2dMX1+f\/YN61Vu5mqp1XLrZUI2kGW2R555D2a0ucA0ODprdu3eXVAqGoJrNmzfP9PT0RLjytI4IVwXW+uSQloJr79695sCBA064pk2bZubMmVOCq7e318IWt2wL4KU1NDRkZsyYYTo6OrIPiHtYC9BuCFGKtFtdlMsHLjz0ggULLGRxixYIbYGpU6ea5cuXG\/y3qK2h4YpgFVXN8TxZFigSKl6rLnBBhjdu3Gi2bNlSSmjomCvLGLE8WqDRLVAXuHyyhY1uuHh\/0QJZFqgLXD79XFk3HsujBRrdAnWBC0bRIzTw2\/DwsNm5c2dZdhCdy9u2bSvZ8b777it1PONHJEdWr15dKp81a1ZZFrLRK6Co+0M\/4dKlS817771XOuWOHTvaOtOq25huG1k2yyrPqru6wYUb06MyaIx169bZRoHYbNWqVaVhUbpc\/5vGmDt3rk3nt9OmbYmXzp49e8ri2nayh24brvaWZbOs8ix71g0ugrBixYoRSoWbRp8DHu4LX\/hCGShoNB9++KEtl\/\/PB3V1UGcZodnL9egWPE+7J4hc3T0ADi9rvLwxMkiOCNI2u+6661LLffpe6wZXUoMGUNjQeYzxhxyx4YJn06ZNI+BzJUuaHZ6s+096ZtfLKetcrVwu4cJz6iF4VDe80G+66abUch\/PqKHgkm7dnXfeaWMIrWxM4z\/99NPmySefNBzRwUYhDcgR9q3cYPBsrq4NNhR6Aa1uA5\/nk67yr371qxHdQdJmt956a2o5B5ynXbeh4JJSDrcmwuXTZCJcPlbSYUjWC6ml4NIPmxSTReUa2ZSyGorPW9angTbrPow\/pZeTZbOmgSsrJaqzgjK41G6fTFgg5tJuT7vGXAzUpSscY65PEjty7iDaTFL4QJsh5qrWpnV3C\/Ew+\/btK5uFzDdkzBb6aUXMFrrtxJf6zJkzy\/pGsXeWzZo+W5gGFgP12M\/lB1i1fTJ+V2mevdLAki9vOftd9w1Wa9O6KZdrhAYfWvakxxEafg262tEEfldpnr10u5F3zpErWTbLKs+yRt3gyrqxWB4t0OwWiHA1ew3G+29YC0S4GrZq4o01uwUiXM1eg\/H+G9YCEa6GrZp4Y81ugQhXs9dgvP+GtUCES1UNh8q88sorppLJhknDtkK0AHRn\/PznPzff+ta37Old0yxPgj9dAAADL0lEQVTyXJf3furUKWenfp5zpe2bNPSoqPM3ynkiXKomOHzqiiuusCVcXrtRKoz34VoqoVq40Og563vSpEkjRjUUZYMIV1GWbLLzsFf+7rvvNo899ljQN3g1pikaLnk+zGcKOYs5wlVNzTfpsa75ZHrJAL1mBx+VX2nBDFc5VYZqcuONN5rHH3\/c7s59Dx06VFr\/Q45KcbmWsvEvXry4bL2MKVOm2On8+\/fvtysZu66VNbdNTosfM2aM\/QINl1vgMxKKRYsWmQcffLBUy9p91qNvHnnkEfPcc8+VzueCS9u1Epe80ZpddAtFjejR+VAxLEwq11fUFUgQsKgkpnZoMNhouLCOHFKjf0s6B66plSpJubBYD8\/LfXzcWzmuDl+ZwSxwfRzss3DhQqOHp0kbJa1zgk9FERgNF669efPmEWulLFu2rKnXQolwXaLF1RDZmNLeonpwpwsu2XBwORe08jyuiaK+cGl3zjWdJ+sFweSIvm\/XuTRMXKZBziHjC8YFF64FpXd5CCFd01qoXITrkpVdqwW51EFWin7joswFl24krsRDUXDpD1z4wOXax7WSlsudk3bDNA0XKK5VvbjiMqbby5kPtG8rLNcQ4bpUm0mxlIyRZNzigrFZ4UobQc54DrFkKLjgaro2\/bXRWqhNkdeIcAm1YcwjDUyIpP+v4yy5f7Mpl+v5ZAIDDT8pVsJ+oZSryEZer3NFuC6tnuRyTWQiQQb3Os4KBZeMQzTQvqn4LLcwbfFQfc0s5cJafnljLlwjLTOZlkyqFzS+141wOVb+1caTiY2PPvqoLLOVlBzgknCuxpsVc+GcOltH101nAuXaEK7zpsGVFVPqxMbp06dHLDeWtQoyy3EuflA+K1vYKisntz1cSbGTS43Qf\/Tuu++Wrccu94P7xKC+GriQCtezYNeuXWtXGsbGTByhl\/1meRIaWaom3T64xePHj8+ESx6D9Dvu7aGHHjLbt2+3\/VyIW336ufC8Pgtv+qpIPfZre7jqYfR2u2a7jMjQ9RrhareWHvB5XX2FPq5nwFuq66kjXHU1f+td3LWoi\/7sU+s9tfuJIlztUtPxOWtugf8HnV8mdc79ybcAAAAASUVORK5CYII=","height":117,"width":156}}
%---
%[output:964f43ab]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAG6BJREFUeF7tnQ+QFcWdx79v\/wkaFkShhBWCB\/J3WUBlAS2JgpCENRrghMp5UmjBYXAJ3gmHFcCUF7TYCDkJRES4nMqhBRR6l4hVpydGxaBgSiFZ\/hNBYFFAENhdkP3zrn7z6KVfb89Mz7yZ9\/a995sqCpbp6en+Tn\/21\/3r7l9HotFoFHyxAqxA4ApEGK7ANeUMWQFLAYaLGwIrEJICDFdIwnK2rADDxW2AFQhJAYYrJGE5W1aA4eI2wAqEpADDFZKwnC0rwHBxG2AFQlKA4QpJWM6WFWC4uA2wAiEpwHCFJCxnywowXNwGWIGQFGC4QhKWs2UFGC5uA6xASAowXCEJy9myAgwXtwFWICQFGK6QhOVsWQGGi9sAKxCSAgxXSMJytqwAw8VtgBUISQGGKyRhOVtWgOHiNsAKhKQAwxWSsJwtK8BwcRtgBUJSgOEKSVjOlhVguLgNsAIhKRAqXB9\/\/DEmTZoUV\/SSkhKsWLEC7du3x\/79+zF16lRUVVVZaTp37oyVK1eiR48eIVWXs2UFkqdAqHCtX78eW7ZswVNPPYXWrVvH1er8+fOYO3cuOnXqhNmzZ1v3nnnmGRw7dkybPnmS8JtYgWAUCBUugoUuAY9cZLJaFRUV1h+yYnSdOnUKc+bMsf6w9QrmA3MuqVMgNLiEZRo2bBjuu+++ZjWkLuPatWvjrJR4ZuLEiRgyZEjqVOE3swIBKBAaXGSFpk2bhh07djQVUx5v6bqMbkAGUF\/OghVImgKhwSWcFQsXLmyyQnJX8J133mk2HmO4kvbd+UVJUMARLp31KSsr8+1wkOGhuqnODju4tm7ditLS0iTIwa9gBYJTwBYu4UZ\/+eWX48Y\/1J177rnnfLnMZXi6du2KRYsWNbnlqUq6MdfSpUtBcK1evTq4WnNOrEASFNDC5dY9M3GZE5xO8FxzzTVG3kKCa9myZRZcbL2S0CL4FYEpoIVLdAdnzZql9drpwFFLpMuDxlwEyeOPP24lN5nnIqv1wAMPMFyBfXLOKFkKOFoueYJXLpDT5LCcTh2zqeM1kxUaDFeymgK\/J2gFXMdcCxYsiJunIqtFlidZy5QYrqA\/OeeXLAWa4NJ5Bp0KIc9ZhVlYhitMdTnvMBUIbZ4rqEIzXEEpyfkkWwGGK9mK8\/uyRgHuFmbNp+aKJlsBdmgkW\/G0eN8pAKelktJ2oasBxG8bSouqpLCQobrig6gXj7mCUNE0j\/MAYhtX6dq6dQdKS0ukhzszYKZSAghtEtlDGRyTMlxBKemWz2Wwli5dg2XL1jQ9UF5+P2bMuP\/Sz93dMuL7lxQIbflTUAozXEEp6ZYPWazzUMEST10GjLqHsc2tfDkr4DrmCnLhrp+PwXD5Uc3PMzG4evUaY\/vwnj1vsvXyIG1St5x4KFdTUobLj2p+njlgjbEeeCC27lN3rV698NIYjLuGJgrzPJeJSlmR5oBVS3fLRR5Dcmzw5aZAs3kuekCEPnN72Mv9nTt3YsOGDV4eibNevN3El3TGD\/Xtex3Gjx9hMOZiuExFbWa51DWGQa0hJLAqK3di7NjxpmXjdElWoH\/\/K6w3OnsL2R1v+llcu4VBwUZwNTSA4TL9MilIF4mcR14ez3MFJb0rXOqLBGxeu48MV1CfLNx8CLDc3NPYtWsn+va94dLL2P3uR3VXbyHt55o3b15T4E4\/L6FnGC6\/yiX\/uddf34DcXOCuNr\/CFd2noFX3qckvRAa8MQ4uskoE06hRo1BdXY2RI0daPzNcGfClPVSB4fIglkNSLVz33nuvFRSG4Nq7dy8olgYFm9m4cWOzrEwdHmy5gvlgyciF4QpGZQsuMY4SMJGlomvfvn1WyGkBF4WZpuv9999HeXk5KOAnBY8xievOcAXzwZKRC8MVjMpNlkuEU1OtE1mmJUuWWJaL4QpG9Jaei4DrxvZzcV23cnS8obylF7lFlk\/bLRRjLDEGY8vVIr9daIViuIKRluEKRseMyoXhCuZzxnULaZEsOTLUk0m4WxiM2OmSi4Cr6NpfoEfXh3Etdwt9fbo4hwZBJXv\/srFbuGTJr7Fx4x\/w9NO\/wi23DG4mKjW85cuX2d63+wqvHwImbmp+t20BsHE0UNohdu\/sReCet4A\/nYj9rN53+spVNcDtG4HDNZdT3dcNWHOnt7Yh4Gp93dO4oevD6NOF57m8KRhLre0WPvLII1bgzy5duliJsskVf\/r0aTz22Eyr3osXL8HVV9PqhNh14MB+zJnzGG6\/\/XuYOfNfPOm94FPg3z5zhksHh3hi7Qhg7HftX7n1BFD2FnDmYvM0XgETcJ0v+g1uLvpHhsvTl76cWAtXz549QQclZOsk8iefbMPPf\/6vKCv7URNETtC5aS9boycGAvMG6Z8QAHa5CvigDPhO\/mUrdmsH4PejgcIC52dlSyfy82L9KHeCq\/7sNpwu2YG7Ov0Yvbr8k1sV+b5GAc\/Ln0R89+nTp2uPY7VTOd3muUT38Kc\/LbcWG9PPH3zwHioqFqN79x6eGpNskewskAygbGlEd9INkPvfBdYfBGQIZWvmZvnkChFc1ecIrt24rfNEDC36iaf6cmJNt5DmutatW4cJEyagdWv7MFpi0pnS6c471ombbnAJS3X27BncdNMt+OMfN0GA5rXx2HXZZIjs4DIFRGelVEvY+Sqzkgu4Lvb\/DL2vn4o7OvM2ITPl4lM12yx58uRJ65CFoqIi64gf3ZInOQv1oIZMsVzyGOvs2bMYPLgU8+c\/iVatWnnW2c6ZQRkJS0P\/Fo4MneWi+05dSrovrJdaQC9WS3QLyXIdKjmMa9vehBk9H\/NcZ35AcmjQgXZ0zZ492\/pbd8pjIoKlm+WiugrrdeTIYd9WS270cpdNdnCIxi\/gEF3A3m3jPYducOmcJm7dSd03Jct1tLYSu\/ufQJ823fFoz0cT+fRZ+6ztmIvhgjXOIrc8XYWFhb7GW3YtS9cNdPIWulkuXRcwEYfGrtqD2FF8BoMKizCv57SsBSSRinveLOn3ZelmuYTHkMZZJSUDLBd8YWHbZu55v3rYjbFUwBaXAs9WxuauvDpD5LzcrJ5cD7JcBNefi6sxqE0XPNnrIb\/VzOrnmuASB4x7VUONa2j3fDrBJeazevXq3TTOEpPHsnveq1Zyeju41DyFQ4P+X55sNskrEbg+qzmGvxSfQu823fFMbxFtN5EaZ9+zcXDR9pKnnnrK0VMoS0RAmj6TLnBduHABv\/zlL7Bnz+64bqD4\/23btnpanSFDpBtzyWMiMeYS81zk3dO52HXueXW8Ris+dOM6kyZOv0h2136ON3q3wZC2HbC0zziTxziNokBCcNGcV0VFhfWnfXvnEMfpApfT8ie\/E8lO3kLZM2jnsledEjq4nFZouE1Aq1QQXCurPsOZoV0wsE0XvNDvhwyODwUYLkk03coMVVOTNLrvoGv8unGQmk62YiJfu4nlINcWLj9aia8G98Kt7a7GquLRPpoWP6KF6+jRo5g6dSqqqi6H2ZKlmjJliuWyJ8u1evVqax2i06QzPZsuloubRGz50\/NH\/4ovbi5BfqQBu27\/e5bFhwK2cInuHkFDk8m0C3nIkCFYv349Dh482DQfZvpOhstUqdSnI7j+\/fBeHL65PxoRQdUdZakvVBqWgOFKw48WdpEJrqcPHcLuQTehVU4dvhkxKuxXZmT+DFdGftbEKkVwVRw6iO0DS5GT04CLI+9ILMMsfZrhytIP71Rtgmvhga+wY2ApGnLqEf3+UFbJhwLaFRpiW4mdQ4PeI5wapu\/kMZepUqlPF4PrS+zrPxwXIw34tmxg6guVhiWIg4sW73br1s14G4mX+jJcXtRKbVqC69f7vsaBft9DfU493rj1Btx27ZWpLVQavr0JLgKLAtTMnz\/fiqMR9MVwBa1oePkRXL\/ZeQafF9+BxpxGbBhehFuvtd\/fF15J0jtnCy4vKy38Vpfh8qtc8p8TcB3uNRINuY1Yf2cnDOvgfR9b8kvest7YbMzldQEvx4pvWR80iNIQXL\/9yzl80fMuICeKR4vbYuaAwiCyzqo8eMuJ9LlNljaZpNG1IDVkGqWx20Ki7ig23Uns5R1u3sLfbj+Hqr\/7PhrzGjGzpBA\/G9Qmq8AIorIMl6JiGAt3dY1evFYGxzSdKby6d5g0Gmv5059rUdXtB2jIb7DAmnGzYQAOkxdkSRpXuERg0ETP6EqXMZfTyvcgAoISTCM76UOmOW0l0S3glduovPLe6R0m7ZrquWLrBRz77g8ty3XVR39D+S1X4sEZxSaPc5pLCsTBJY+3xDwWeRFXrVoFdVOkiABFAUNpzaHblS5wUT10Xb9EAoLq9mTptuU7hUejctltlqR7pu9w+050n+B64aM6fFU0Bo35Dcg\/chqF\/7PNevTB8n4MmYmIasRddfOjmEyePHkyNm\/eHLdvK5PhIu3k7mFxcX9rA+WxY8cC2eYvbw0R+7lMtv2bjr2o\/Lp3GLaJGFwf1uN4px8hmteAvGNfo80bH8c9TpANHNIRg0o7mmabdemaWS6xs5iUoNXwnTp1wtixY5ttisx0uOTu4bBht2L9+rWediDbtSS5+yZvYjSByzQOht07TFu3tVnyg0Yc73AvonmNFCMM7V78vfbxgaUd8eCMfgyZRh1buOhQBjrwbsWKFSCQ1B3HmQ6X3D2kfwcVO0MNfyasUZBw2b3DC1y\/ezeK49eMRWN+o\/VY4SuvOT7OkDWXRwsXjaNmzpxpRd6liLq6SeZsgEt0D7dv\/yyQ7qAsvxr2TI5RKG\/99xtkht6VSGi1l\/4vilPtxuFifhSIAK3ffQ85Jy8dveKAGUN2WRwtXDTGIgeE2GHMcAUPlwrNz\/rpI+6axJm3a+t+waRu4X\/9bxRnvjMedQVAYwTI31WJyN5KU+MHhkyKuEs7jMndXlZW1iwCFMPlHy4v3T2\/3kIv7zChg+Ba82YU1a3Hob4ggmgEiOyrRHS\/OVziPdkMmWW5yEtIVoos1vbt27Vw2cXUyMS4hXIDJK9hot1Cp7BncmQn03kur6HVvIa0Jrhe+UMjagrGIZoXQWMe0HhgJxo\/9w5XNkNmO+Zas2YNysvLrcAzbLn8Wy7VLa5aDruTTtR0shteB5dTKGw\/h9+9+t\/1qM0di2hBLqI5QOOZE6jb\/p6J4WPHxyUFtHDR2VvkHRw9ejQ7NAKwXKS1l3V\/bmsL7UKreXmHEwFkuV7dcBHn8WNE83OBvAiiUeDCR84eQy\/kZUN30dYVL8KrLVy40DplMhtd8V4aSyalteBadwHnG+4BCvIQzc2xqndxz2Y0VJ8MtKqZDJktXNQdJCfHli1brLHY0qVLs2qFRqAtKM0ys+B6tRbn6+4G8vOAvFzLHV93bA\/qju8NpTaZCJkjXGIuq2\/fvlaAUDlsdbbMc4XSklp4phZca6px\/sIYID8fyItZrrrj+1D39f5QS59JkLku3CVPolipIceDZ7hCbWMpzdyC6+VvUFvzA0QKCoC8PCASQUPN1\/i26pOklC0TIHPdckKH4L300ku455570LlzZ9\/CptOqeN+VzJAHLVf8f55E7blRiBS0QoTgon5hYwNqDyfuMfQiUzpD5gqXiRBkxebMmWP96dHj8kn3cog2AtPvgd0mZeA0wSlgTSL\/x5eo\/eZO5BS0BvLyY3AhirozB1FfczS4l3nIKd22uyQMl+geioPKBVzi2FdaVU+HNrDl8tCKUpzUgmvVYdScGo5I3pXIySuwShRFFPU1VaivPWb9lKorXSBLCC6xubJ\/\/\/6WzuS2F3CpE88MV6qaovf3WmsLX\/gbar6+DTl5VyGSe4UFVgQRRBvr8O2ZXYhEchCN0op5hsxOYd9wyQeS0zwYLZ+S4VI3XjJc3ht5qp4guFav2I+aE4ORk98GkZx8CyGrYxhtQMOFr9BQV41IhCaX6U7qACONyJJV411LrmXLlqG0tNT694wZM5r+nQotfcMlF5aslAqXmCMTx8D6gUvErHASJpF9VvJBc+ou30RXO6inSdptI7Grm7wx0u5kSnXNYJCH361evhvVJwfHLFck91IxCaZ6RBvOo\/7bU5egio3FUn19hScskOgPhZ2geVn6NwGWqitr4VIbYlBRmOhDqsuXxMcVO4+r64DbNwKHa+w\/uwyXuvlRPCXD5XRsq5+1hS8v34Wa4zcjkkfBQGPzXLErimjDt2isr7ZAS\/VFVsuyXi0weE6Lhsvuw4kt+GfPnok7FNzLh3Zav5dIJCWdNbRbC6iWV3fguGxBnbb56zZGJrJZ0rJcxwcBOeSGl9lqQLSxHtHGiymBq19pjWWR+g2pTWmXz6SthQaXOvnsp1uoq4AA68iRw75jWui6WbLlSiSSkmjQng\/5PgRM3BSrsVwW082STnvBzly0D0Cq05i642S5qr8cAOTk0EDLmkRGtNEacyHaEPs7Cd3Bi\/gc08rv1lon9TQedS+i8GRTyApxyVuk3O6bAOSUJjS4wvAWXrhwwYrCtG3bVt9gyY31n4uB3+0F3BqfaSQlddMiCb\/+YEx+p66ZU\/523T01PyfL5RbzUG0gAq5zX\/a95MawPBmxLqEFVax7GNZFzokZ5eW4iIOYNiMGlnoJsMiJJkL7URhAitAlxvnqz+QHWLdunRUXhlYbud1PtH6hwRXGPJcId5bIZLT4DU+Nc2YxUPaWM1xeIik5Rcx1Asyp+2bnzKD8VOtoN9bzEpKN8o3BVYmzVb2l9hUeTGSdKEQbrcYYNKSjUXdPdZhRQWXHGsGjLmyQPdw33nij432TWJxu8IUGl6is2MGc6AoN4TlMxDsoGqr4TX6k1h0uL5GUVLhEo3aCx26LvvhwTl1USiODo3N8eN2FfBmuvypwuTUlb\/dFd8\/JOnnLMR4uelbdJkX\/J86gGzRokON9CsyU6BUIXCaFSGTMJSLgDh5civnzn0SrVt6Ps9GNXZxc8WqdTJwDMiiyVXEKFKM6T8Z+111NHZC6CL4mZda9LWa5goeLnBFerJO7EvEp5G7fvn37tAvOCS66hg8f7nifVhUlerV4uEQY6cLCtgmFN3PqXgkRnZwQJpGU\/MClg8Lto6pwLb\/NPXKUaUDRIC1XGNbJTht1l4bdbg6G65KCTociuDVA9b4pXK\/cCfzDu8CfTsQ7IUzgonfq3Ol2z7p1Ce3q2JLhSoWrXIylhg0b1nTkMMPlQEgQc1luANp1CxOJ1iRDLKyFXffMDVg7S6jLz6nM6tjMTReTbqFYblSAbparPMixk1v55Puq40zc060aUsdc6qoi+X5Gj7nEOMtN6ETGYXZwmUZSspsctvPaqe5zkzGfk8WV83NaoeF5zs1lzEVgxbx7HVK6vEi448eMGWPtvJAv3TaotPQWugFA9706NFIJF5XXZG2h08oL1XOnG\/OYrtzQgaPLL8i1hbJDQ3aV2807mbSBINM4gSXe4zaP5XY\/0fK2eIdGohXk570rQN3CVcvfwE\/GPWatNk\/l4le70otz43T3xSoMtxUYbve9Kxf\/BMOVqIIZ+DzBlZsLjB8\/PgNrl7wqMVzJ0zpt3mQKV9i\/+dNGMJuCMlzp\/gVDKL8pXGGPWUKoWlKzZLiSKnd6vMwELjdvXBBr89JDLftSMlzp\/gVDKL8JXLrDOagoYu1eEPNEIVQtqVkyXEmVOz1eZgKX2wqIINbmpYdabLnS\/TsltfwMVzBys+UKRseMyoXhCuZzMlzB6JhRuZjA5bZ2j8dc0pnIYbcOr8ufwi4P52+vgAlc7C10b0Fsudw1yroUJnAJz6Acs0KNUZF1wikVZriyvQVo6m8KF6\/QcG48DBfD1UwBU7hYOoaL24BHBRguj4LZJE+q5aqs3InevSkWHl8tWYHdu3eiX7++vCo+wY+UNLh27tyJXbt2JVjc9H28srISR44cwYgRI5BP5wzTGcN1ddi0aROuv\/569OsXi3kuXx9++KH1I4VvpnSUhtKKi\/KjfCnPrVu3ol27dnH50Jho8+bNGDVqFOgAeS9Xnz59QGdh8+VfgaTB5b+ImfGkLoilLrCKqK1wFkyYMAEjR47EtGnTMGvWrKbospROLEF69tlnsXjxYsgBWui+3VxUZija8mvBcCXpG3mFS05PEDJcSfpQAb6G4QpQTKesdAtd5YAp8hYNNa0ak0+8hy1Xkj6ez9cwXD6F8\/qY2KLx8MMPW927qqoq61TGjh074vnnn0dxcbGVJQFDIb9WrlzZdASuXfdRPr2TTlSkS16NbrctxGvZOb0\/BRguf7p5fkoXX093Ksebb74ZB5Z4kW6fFHUdDx48aAEl\/1u2bGvXrm069cNzofmBhBTwBZduXZmfUojQwtmy90c9T4oOpxAWirSwA0tn0dQjdNSfZYcIL6L10zoTf8YXXLrxA\/3fpEmTMGXKlGYBGnXFFOntqrBgwYKm0MSJV7Nl56BCJ5dWPtBNDSemak3Wa968eU2Pq4fBtWwVEi+dk\/c18dy95+ALLtniiA9eUlJiHSpGp0sQZPKlNgIny8eLP71\/RH7isgJe208QW2fs8vAMl+q5UscC6iBaHQs4dVd0g3luOKyAFwXsPLB2edhNkdBRr+RYmj59umMPSrxv48aNzV7hGS4VAC9w6VZRmwgnn2Nrkp7TZLYCavfXS23l4YZbN9IuToj8Pp0jie5TW\/cMlzrwNoWrvLwcc+fOxYABA7B9+3ZMnDgxbrWBrpB28ztexOS0ma+ACQQ6FUR37oknnsCLL74InfVRn5PHsXZdUAGtJ7jEwJteKHu5unXr1mQ63bqFdmab4cp8CMKqoV+4dF1C0zKaWE9PcIl5mXPnzlmHNffo0aNZnDpTuEx+S4iKcrfQ9JNnZzqvTgzRbaMlZV26dLHmAeminpW6PlMoKs9J0nhs0aJFeOihh\/D222\/HzSPKzjpjuMji0CqAcePGYeHChQnDxd3C7AQhjFqbWBF1akc8o3bztmzZ0mzS3c4JJ6+QEbsOfMElRFHd6KZjLjFRzN3CMJpXdueprnSR1dA5LeSzveQYIHYudbtup9tcrbHlcoPr7rvvtiza0KFD8dprr6GiogLt27e3uo10Cbj8eAy5W5jd8DjVXrQn2vOmW+mjc4oRFF988QW6du0KeXmYnffQbiVRqJarqKjI6qfS+Eks5SEhpk6dai1M1Y2Z7BaTskODAfKjgLAedr+AnTzOOjhUJ4fX5wPrFpJlIseGcG6QOG4rse3mBRguP00ru58RDV84JXS7rZ02jOrgUtPr0gjV3cZ6CXcL1c\/rBJfTbwE7uFR4s7s5ce2FAqIL9+mnn2p3EYh0XuFSu4ZOp7bo7gVmucgV7wUuO6tFeYh7YrJZuOqzbfEp4+OugClYlJOTs8PJKrn1wuzWxyYEl3vVOQUrEK4CdsDonGXyth61VAKuyZMnY+bMmXF+ArsaiPzIYtq57UVvy3O3MFzZOHdWIHkKuFkuu5I47epgy5W878dvymIF2HJl8cfnqoerwP8DvCmcGsjmRsYAAAAASUVORK5CYII=","height":117,"width":156}}
%---
%[output:067e5866]
%   data: {"dataType":"text","outputData":{"text":"最高点坐标: (Theta: 30.30, Phi: 59.80), 强度: 43.68 dB\n","truncated":false}}
%---
%[output:866c1bb3]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAHQ1JREFUeF7tnQ2UFcWVx\/9v5s0AgoMOODAPBkcdcJiBEQwMokETUbICQRAFz8lqSI5IUGbJRg2eIO4mgoFE9siBGBXcfHA4J+CqezTgriTxWxZMVnHzgaBZ4gckCrJ8zjjz3us9t5\/1qOmp7q7u1\/3m9bzb53Bgpquqq27Vj1t169atmGEYBvhhCbAEApdAjOEKXKZcIEvAlADDxQOBJRCSBBiukATLxbIEGC4eAyyBkCTAcIUkWC6WJcBw8RhgCYQkAYYrJMFysSwBhovHAEsgJAkwXCEJlotlCTBcPAZYAiFJgOEKSbBcLEuA4eIxwBIISQIMV0iC5WJZAgwXjwGWQEgSYLhCEiwXyxJguHgMsARCkgDDFZJguViWAMPFY4AlEJIEGK6QBMvFsgQYLh4DLIGQJMBwhSRYLpYlwHDxGGAJhCQBhiskwXKxLAGGi8cASyAkCTBcIQmWi2UJMFw8BlgCIUmA4QpJsFwsS4Dh4jHAEghJAgxXSILlYlkCDBePAZZASBJguEISLBfLEggVrnfeeQfz58\/HgQMHTElPmzYNK1asQJ8+fcyfre8TiQTWr1+Puro67hmWQOQl4AjXJ598ggULFuCtt97KNtQKiJ0EBDgrV67EhAkTzGQ\/\/OEPcfDgQRMwepYuXYrq6mrcddddXd4LACMvYW5A0UrAFq6dO3fi5ptvxs9\/\/vMsHCSlxx9\/HA899JCrhqF0O3bs6KKp7r77bhBw9Kxatcr8U1lZaf5MMC9ZssT8I7TXrl27QHVpaWkp2k7ihkdTAkq4WltbTa0yceJE3HDDDV1aJmsgLxqGtJmA6\/Dhw9i8eXMn+MR3586dmwWa4LrpppuwceNGNDc3R1PKkav1JwCOSLWmafzZADLTeX70JKCES0wH77zzzk5aSxRJmuSBBx7AI488ktU6Op8jbbZlyxYz369\/\/esumk0FNcOlI9mg0rQCyKyP6dm16y00NzdJhScYMA+idtRc8npILlM15XP7phVYVRkMl5sUw3x\/Gqy1azdh3bpN2Y8tWvQVtLR85bOfLwizEj2qbNc11\/LlyztNDUlr0dTOi1VPBQ3DVWjjiDRWK6xgiVqeBoymh5k1Mj\/OEsjCpbIMOmVtamrSmhYKsKxaUDW1VK251q5di3Xr1vGaK\/SRnIHrwgun2n7p7be3sfby0A952eeaOnVq1twu6kbGDR1rIcPloTdzSvquuca66aa7bUvZuHHlZ2swnhrqiDo0uMQ+lwosqphKo6mskGzQ0OnGINK8axbirrnIYkiGDX7cJBDatJBA2bBhQ\/b7I0aMwJlnnulWH37fTRKYPfuLmD37So01F8Ol20V5MWhQZe677z7MnDkb9fUNunXjdHmWQFlZRns5WwvZHK\/bLXkzxTNcul3SfelisVbE47zPFVQP5G0TmeEKqsvCLYcA27v3VTQ00AyD9r7oYfO7H6nnzf2J4fLTPd2T5\/vfvw8zr\/4rRgx4ESVnjEf50J90T0Ui\/lXXNZdfx12rXBiu6IwUE65x2zCy5hRi\/cYhft5j0al8AdU0tCMnDFcB9bLHqhBc141+Dg1DTwH9L0bJiPUeS+DkJIHQ9rkYrugOMIJrduN2jBzSitiZn0Os8eHoNqYba95ln4vq4tXbXaf+PC3UkVJhpCG4rj3\/BTQNawX6j0Xsoh8VRsUiVosumsvqY6jrQ+jWbobLTUKF857gmlH3KzRWpRE762LEx60pnMpFqCau08KgYGO4ojMqTLhqX0Rd9UnAKEG\/Ka9Gp\/IFVFNXuKx1FbB5nT4yXAXU6y5VIbgmnf8MxlQBqVgSA6\/+7+hUvoBq6motpPNc99xzj6cTx6r2RQWuNWv+BVu3PoP77\/8Bxo0b36UpTz31BH7843W27+369qm\/AHN\/0\/Vt\/3Jg6xSg+ZzMu2PtwIzngNc+zvxsfe80dqx5Ke3mK4FZ53obcQTXl2q3Y1jiKAwYSFx1OkCRt5KKO3UnuEgrEUxXX301Tpw4gcmTJ5s\/FxNcR44cwR13LDZHxerVa3D22eSdkHneffcdLFlyByZNugKLF3\/L08hZ\/gbwvTed4TpwEpi0FXj\/ZNd0bpCowBKluOW1fk1orurEcZTAwIgr93hqKyfOSEAJ17XXXmseUCS49u7dC4qlQTEztm7d2kVuugaPqGguauBvf\/s6vvOdb2PatC9nIXKCzm0wyQP\/3jHAPWPVOQSANX2Bl6cB\/cpOa7FLzwGengJUlKvzypqRYJpcrZ9XBdfYumdQPeQ4eqUNjP4iw+XWx6r3JlxiHSVgIk1Fz759+8wITQIuispEz0svvYRFixaZIdIoMpNOEM8owUVtFNPDhQsXYdas2ebPL7\/8IlatWo0LLvAWtFTWSHZaRAbwhlpg0xcz3SWgcZsefuV54PH9gAyhFdZEX70hQppr6Mjn0Fj1CdIwMPGKP+pl5FSdJJDVXOLwolU7kWZas2aNqbmKCS6hqY4dO4qLLx6HF174DQRoXsfQro+Bac8BR9s755QhsoNLzutleicDLX9Hp+4EV1XDb5BInETf9KcYO\/oxVPYfp5OV00gSUE4LxRpLrMGKUXORjMQa69ixYxg\/vhnLln0XvXv39jyA7IwZVJDQNPRvYchQaS567zSllCslf89tOqlqjNBclUM+RTpWgssbH8LAios9t7vYMzBcDiNAaK8PPnjft9ai4p2mbPReaCSRTkwB6\/t3thzqwmU1nnjReFQfgqtt9O8xPPEx4kYKX2p4EFUVNgvFYifIof2dpoUUr4IMGXJs+GKdFsrrLvp3RUWFr\/WWnexV00Ana6EXzSW+KSBzW69Z60hwpUa\/ibOGJlFudGBW\/UokKuTgoEyUjgQ6GTQIKtn6F+S08NZbb0VLyzcjc8xfWAxpndXUdJFpgq+o6N\/FPK8jZFUauzWWFbDVzcCDf8iY571qILksXa0nNNfhpncxYOinJlw31n8PtWdyeAavfa2cFt5+++1m4M+amhqzvFxN8RSshiyM99+\/KhJwibXWhRfWZ9dZYvNYNs97Fbac3g4ua5nCoEG\/lzebdcrKFa7Smjh6ox1XVc\/ElYnrcmluUeZVwkWRmgYMGJDzJrIw8R86dAiDBg3C0qX3FjxcbW1tuO++f8Lbb+\/pNA0Uv3\/99V2evDNkiFRmcnnKJtZcYp+LTOeq9ZrKPG9dr5HHRy7Twj83\/Q3lNSUoQxLXVE\/H9IR9sNCiJEej0Z7dn0Q8wttuu015A4r8TQpZvX\/\/fnNPbOHChbjjjm8XPFxO7k9+N5KdrIWyZdDOZG9dM6ngclqv+THFv9F0EmfUGIjBwPXVV2FmYorGcOIksgQ6wUV7XXQLyZw5c7K3P6rEJTQSpVNdMWTNQ+VGAS6VZ4a1LTppVDJTgaNaB1nTyVpMlGu3sRykbyHBlRx6BnqhA9cnJuPGxBeYHI8S6HJYkqZwdMnCkCFDzDu6VC5P8jesFzWovh8VuDzKrscmJ2vhi6NKUFITR69YB25MXIG\/T0zqse0Nq2FZuMjoQI+4QlV1KYLfSjBcfiXXPfkIrucb40jWnIF4LImvJi7FvCETu6cyEf6q7ZqL4Ypwr+ZYdYLr2fr+MM4tAwzgcxUJ\/Kjh2hxLLb7sng9L+hERay4\/Uuu+PATXMxcOQNuQs1BW+inGVQzCo41sLfTaI1m4xAXjXguwxjXkNZdXCRZeeoLrieGD0Da0EuWxdjT3r8Jjo68qvIoWeI06wWW9ANyt7gSkTh7WXG6SLKz3BNcv6obgaKIK\/UrbYBgx7L18dmFVMgK1yQku1QV2dm2O2nmuCPRdaFUkuB47\/zwcS1ThrNJWpFCC9674cmjf66kFM1w9tWdzaBfB9aPaehxLDERZSQqlSOPIlbyJ7FWkSrg+\/PBDzJ8\/HwcOnL5ORi74lltuMU32pLk2btxo+iH26UOXotk\/rLm8dk33pSe4Hjq3AR8lBpumePLS+PQq3kT22iO2cIn7igka2kymU8gTJkyAcGkS+2G6H2S4dCXV\/ekIrvVDx+DDxGAglgRiaaSnXNb9FYtYDRguqcN0XJt00qjGgBfXJOGEK8rxetSE8vkND0B5Ca6N1ePxfmIwOkpSMGJpPD++HlcM4Gt3vfDNcFmkFYbjrm7YM910bh1sdeL1CifBtWnwBBwaVIPW0g6kYwa2X1KHSQM0I9y4VbBI3jNclo528nwPIiCoU9gzp6MkKgdeuzGaq+YjuLYMvAyHBg9DsjSJVCyNX15ai8sGnlEkWATTTKWHhjhWYmfQoE8Lo4ZuNaK05lJN\/XIJCKob9kyVTuewpNwHquMtfjTXv509CUeqapEsS5pRd++qH4A7Gip1u5vTWYOCkvNubW2t1jESr9KLElzUNnl6OGrUaPMA5cGDBwM55q8Ke6Zz7N8NErncfxwF\/OveTDg3t3zWvjQ9NPpfjv875zykS9MwSgx8q+FsfKvxLK\/dXtTpO3nFU4CaZcuWmXE0gn6iBpc8PZw48VI8\/vhmTyeQ7eRnF\/ZMBy63OBhC89HhyMWjTsdK9APXv\/f9Ao4OPB\/peBpGzMA3R\/c3\/\/CjLwETLi+eFvpFd04ZNbio9mJ6SP8OKnaGXdizXOES0Iq12QencoPr6d6TcXTAeVm4Fl9UAfrDj74Euqy5vDrw9sRY8bL4aHq4e\/ebgUwH5XKt8S3kGIXysXydIDOqcNm5muJ\/WToFRwdmpoUoMdBcXY5N1wzQH1mcku9EdhsDYcFlheYfGtURd3XizDvF6BDt8xJ5l9Zc2\/B3OFZ5AVLxlLnmmpAow8bpp298cZMbv+cLx13HQK5weZnu+bUWhgHXs8lpOF55PozSNFLxNDlp4E+3D3SVFyc4LQHXw5IiMGiud3RFcc0lrIa5Tgt1w57p7nPp3HyS67Tw2bYZOHlWHYx4CunSFGJGDL\/\/JlsLvfzn0Qkueb0l9rHIPL9hwwZYD0WKCFAUMJR8Dt2eYoZLN+yZrodGPuD6zxMzcbz\/cKAsZa65DABnr9+Kry1qxNdaRrl1N7+37nNZDz+KzeR58+bhlVdeATnzVlZmNhIZLm\/jJ0jfwnzAtf3oLJzodyFS5WmAjBoA+j37GuJ\/PWT+myAbM6EKY5urvAmiiFJ30VziZDHJgLzhq6urMWvWLBOsYoSriMZCtqlk0Hjh49k4emYGLqOU9JaBvttfQfxvGbjEM6a5Cl9raWTIFAPFFi66lIEuvHvkkUdMLcVwFQ9mBNfLf52NE31HoqM8jZQJF9Drf\/6E8j+qb5lkyLqODyVctI5avHixGXmXIuqqNpmLZVpYPEidbinB9doH16G1TwPay4FUPLPmIrBK9\/zBUSQM2WnxKOGiNdYTTzyRPWHMcBUXYgTXf+2fhbY+DUjFY0jGaXWODFj7nOHi6aICLjphTOb2adOmYcWKFZ2O7TNcxQfXzndmor1XA1JlMaTjMRglQMlHHyH1uxc8CaOYNZmpuchKSHEwSGPt3r1bCZddTA2duIXUG1E1xXsaST0kMWmuXXtmoL1XI4x4Bi7SXMbhj5F80xtcxazJbNdcmzZtMq\/+oRgarLl6CDWazTDh+v10dJQ1wigvgVGagSuWMtD26hOapaiTFZMmU8JFd2+RdXDKlCls0MhpKEUzswnXm9cgGW+EUVZqai\/EYoilDbTufDKQRhUDZLameBFebeXKleYtk2yKD2RMRaIQE67ffQnJkpFAeRxGvCRj0Uin0b73FaROHA6sHT0ZMlu4aDpIRo4dO3aYa7G1a9fmfRNZxKxw6kmv56xUrkhONy\/KnhVuhxVFPb18QyfehdW7w3rTpJN7laiTbt0pvQnXzquQjGXgQrzUZAtpAx0H96Dj432BwdWT12SOcIm9rIaGBjNAaL49NIKGy+5aVOpguyMZ8uDXGaBO35Ah1vUjdAJHnDAOBa7XrkASDUB5WQYuegiuj\/ai4\/A7gcPVEyFzddwlS6Lw1BB+hSSI7txEFkfwjx072ulScLceV13oLR\/XkI\/DqwasDlyqS75Vv9P1gBd5xQnjfmWnz325ndFStddNRkJz7XzpMiSNkYiVl38GF6kuw9RaHUf+rFNMTml6wnTR9cgJ3VDys5\/9DDNmzEAikfAtsKBM8QKsDz5431NMC1VQGGqM6qSvnfbRgcvpTJYcLEbn7Jbd6WQdx127\/zR0OpCmhTufb0YyTXD1AuK0i5zRXEglcer9F3WKCSRNlCFzhUtHQqTFlixZYv6pq6vLZpFDtI0YMQL3378K9fUNOkUq07S1tZlRmF5\/fZcnsJw+KIMk4BG\/qygDfvEF4MYXgPdPAjpwOWkulfaxO9JvjW8op3M7q2X3H4mu4E0PjV+NQTI5ErFefRArpTVXDEYqZRo1Wg++qltUoOmidtwlZ7jE9FBcVC7gEte+klc9xZUPQnOJcGcLFy7CrFnB3BelOshIg\/cnbwM\/\/rxas7mNGKuRQqQX007d08l2R\/9lraQCXgW4W53l9wTXjucakEw2oKSsD2DCVQIQXEYKHcf+gmTbR+Y0sTueqECWE1zicOXo0aNNGZPZXsBl3XjOFS5h3PBqHXTqfBkCO4uhToAY6zesEZ7ovWzh04XrnrGAFX55qkjlWuGyK9sLBKbj7n8MR7JjJErKzkCspNQMDEpaC0YayZMfItX+CQyDznl1D2DUnkKHzDdc8oXktA9G7lMyXNaDl7nAJUKcjR\/fjGXLvovevXt7GSvKtDJYToYBr3BZDRCJvkAukZ7cLIFWuHJZawlBmXA9ex6S7fWIxelqKDJm0EOAJWGk2tBx6r3Pfm98BlnOXeK7AILsBJ43869btw7Nzc3mv1taWrL\/9l14Dhl9wyV\/k7SUFS6xRyacgP3CJcJIV1T0Dyy8mS5Y1EYvcOlqJK+RnqyArW4GHvxDZh1oDfipgtvr+CC4Xt12LpLtwxEr6SVlN8xpoZFuR7LtkPl3jKaLhJ2R8vqZQNP\/DfeaINEfCjtB+7L0bwKsu56ChsvpUgS\/AvMCVlhwydM9WWvqxoW3SxfElJDanIGrFsm2WiBGe1wZMzz9IYiMdAeM5EmkU62SVvtMs\/ntGJ\/5SGvRU4hxPQoWLr97WU59ZBdK2imPk+Zy2quS11jyGkxoGt19LtVelcqM7\/U\/Aqc2Z+Aaho5TQzOGDPEYGbiQ7kA63W5qrnyvuRqbT5oaqXHCqW6d8un8XxAaXNbNZ6\/TQjmUtFNDdNdhTh4RonyV5c0rXLpeILoeGnblWV2gqA1uJnqdAZHVXFtr0H5ycOcsZMAwDKSNDhhGh2ncCPtpx\/9iwaLpSu1kvY3HehZRWLIpZIV45CNSbu9zbVtocOVqLQwaLqdBHyRcVg0iyrazRur4FlrrbndXl87mss6AIc31yjOD0X5ikGkljJm3Imemheaay1xfhWclJONEy6JFppHCbs0kwCIjmgjtR2EA6SYasc63\/kx2gC1btphxYcjbyO29jqyc0oQGVxj7XLk2lvPrScAMUPP0QLSfEGHTZItg8FCRdqIQbV6me1aDGbVMNqwRPFbHBtnCPXz4cMf3OrE43aQZGlyiseIEcxAeGm6N4ffBSCADVyU+PR5e+Gox3WvHfixoyUz7cn1kuKgs6zEp+p24g27s2LGO7ykwU65PIHDpVMLrmkunTE4TjgTCgitsY4Q87du3b5\/S4Zzgoufyyy93fE9eRbk+DFeuEuyB+YOCS6ydgtROduK2ntKwO83BcPXAARulJuUCF2kn8mQfOyGzhsrHI9ZSEydOzF45zHDlQ\/L8Dc8S0IGL1kykkeghy14+tJOqIVbDmUij8hqyrrmsXkXye15zeR42nEFHAm5w0XSPrHtjms\/pVvciYY6fOnWqefJCflTHoCJpLdTpMDZo6EipMNLIcJGGoicDU1Vglr1cW+oElijbbR\/L7X2udWSDRq4S7IH5Ca5nn96NW7\/6z44bud3ZdHFvnKoOwgvDzQPD7X2u7WO4cpVgD8xPcF1\/\/WxQYCJ+\/EuA4fIvux6bUxeusP\/nj7qAGa6o92AI9deFK+w1SwhNy2uRDFdexR2Nj+nA5WaNC8I3LxrSsq8lwxX1Hgyh\/jpwqS7noKoI370g9olCaFpei2S48iruaHxMBy43D4ggfPOiIa0C0VxG8KcVoi7\/gqz\/nj1\/Mi9CdLIWMlzuXZc3zfVHm4uq3atYPCkoctGuXbvMBtPdaPRv8TP97vPNc8yN3Hz47LmZ4Rku93GZN7jcq8IpSAIUtYggE4+IaNSdUYxUPePmu8drLoDhKjCmZU1FVcuHlvIjArYWukuN4XKXEaewkQDvczkPDYYrj+hYoxXRrTHr16\/vdHlFHquT86fYQ4PhynkQBVGA6tyRKlrRhg0bsp+Tw4DRL63OqrfcckunoxZ0zJ2sfOKxhhoLoh1chr4EWHPpyyqnlKpNV3nd8tRTT3UKC2YNHUbWOTrcJzSd9b31Z6FV5syZkz2dm1MDOLNnCTBcnkXmL4P1YgoqRWiza665Bo8++ijuvPPObAw+oanob9qQVXk+kKbav3+\/+V7+t6ih6pv+as+5\/EiA4fIjNR95VHH2VLEf5KJFMBXa81q6dCnmzp3bCT4ZHjLf19bWdtJSdi5KPqrPWXxIgOHyITQ\/WbzCJU\/rJk+ejAULFnTRbGIj98EHH8Tq1ashB2ihOtrtRfmpP+fxLgGGy7vMfOXwCpecnjQcw+VL7N2aieHKk\/hV7kJywBT5iIY1rTUmn7ymeuCBB8CaK0+d6PEzDJdHgflNLtY\/3\/jGN8zp3YEDB2AYBqqqqvDwww9j1KhRZtFWq6Bs+LBO+6xrLmH8EHXkNZff3gomH8MVjBxdS9Hd59q2bZtyY5mtha4iLrgEvuBS+ZX5aZmwhhXL2R8nDw2ShR1YKo3G+1x+Rlx+8\/iCS7V+oN\/dfPPNsHoN2DVHpLd7v3z58qLZ\/LRCJ8tE9rJgDw1nONy2NvKLlk+veFnjiA5vamoyLxWj2yUIMvmxAuek+awXlOVbIPy9aEvA6\/gJ4uiMXRmeNZfVcmVdC1gX0VbPASe3HNViPtpdzbXPtwTsLLB29bDbIqGrXsnd7LbbbnOcQYnvbd26tcsnPMNlBcALXCovah3hWx1YdfJwmp4rAauDspeWyssNt2mk3Wlr+XsqtzN6T2PdM1zWhbcuXMKF56KLLsLu3bu7uPKoKmm3v+NFmJy250tABwKVFMR07t5778VPf\/pTqLSPNZ+8BrabggpoPcElFt70QeGdrQuXsAjaqW2Gq+dDEFYL\/cKlmhLq1lFHe3qCS5w\/On78uHlZc11dXRdvbbc1l9Mc1a5hPC3U7fLiTOfViCGmbeRSVlNTgxUrVpiCI+do60a9kKh89o7WY+QZ8\/Wvfx3bt2838\/fp08dMKhvrtOEiKMjz+rrrrsPKlStzhsvq4c2aqzjBCKLVOlrEurUj8lineTt27OgEiwyi9Wyc6kiPL7iEEKxmdJ4WBjE8uIxcJGA90S2XpTJayHd7HTx4MAuTnUndbtrptlerrbnc4Jo+fbqp0S655BI8+eSTWLVqFSorK81pIz1izeXHYsjTwlyGXs\/OK8YTRclSefqojGIExXvvvYdhw4Zh8+bNWbjsrId2nkShaq4hQ4aY81SysohgK9SV8+fPNx1TxSPDYedMytPCng1BWK0T2sPuP2Ani7MKDquRw2v+wKaFpJnIsCGMGyRAN09su30Bhius4ddzyxUDXxglhFFBbrHTgVEVXNb0TqES3NZ6OU8LrV3nBJfT\/wJ2cFnh7blDhVvmRQJiCvfGG284hqfzCpd1auh0a4vqXWCai0zxXuCy01pUhngnNpvFhh6HB\/My5IojrS5YJA0nY4dbAB83RaH6jz8nuIqj+7iVhSwBO2BUxjKnwKsCrnnz5mHx4sWd7AR27Rflkca0M9sL6DxPCwtZ6Fw3loAXCbhpLruynE51sOby0gOcliXgUwKsuXwKjrOxBNwk8P\/CRJ4aDSVALwAAAABJRU5ErkJggg==","height":117,"width":156}}
%---
%[output:4a06934c]
%   data: {"dataType":"text","outputData":{"text":"最高点坐标: (Theta: 30.40, Phi: 59.30), 强度: 21.81 dB\n","truncated":false}}
%---
%[output:2f8a7577]
%   data: {"dataType":"image","outputData":{"dataUri":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAChCAYAAABUBippAAAAAXNSR0IArs4c6QAAHdZJREFUeF7tnQ2QVUV2gM99PzOACjLEMcCC4N8qiILKAHE1m2K1olAaTQGVSsGiBYvBmbCpwsIq0YopSEHElAZQEbP8lZsCSk1VgrWJy5IYN8Bgwk8UBVxFVwZFBFlQZOa9e1On+\/Z7fe\/r+\/fu7ffmvXdu1WOYud19u0\/3987p06f7GpZlWUAXSYAkkLgEDIIrcZlSgSQBJgGCiwYCSUCTBAguTYKlYkkCBBeNAZKAJgkQXJoES8WSBAguGgMkAU0SILg0CZaKJQkQXDQGSAKaJEBwaRIsFUsSILhoDJAENEmA4NIkWCqWJEBw0RggCWiSAMGlSbBULEmA4KIxQBLQJAGCS5NgqViSAMFFY4AkoEkCBJcmwVKxJAGCi8YASUCTBAguTYKlYkkCBBeNAZKAJgkQXJoES8WSBAguGgMkAU0SILg0CZaKJQkQXDQGSAKaJEBwaRIsFUsSILhoDJAENEmA4NIkWCqWJEBw0RggCWiSAMGlSbBULEmA4KIxQBLQJAGCS5NgqViSAMFFY4AkoEkCvnCdOnUK5s2bBwcOHCg8fsqUKbB06VLo27evpipRsSSB+pCAJ1y7d++GWbNmwcaNG2HChAmF1m7duhWef\/55WLt2LVx99dXapdDZ2QlYl46ODu3PogeQBJKUgBKu8+fPw+OPPw6TJk2CadOmlTzv6aefhuPHj1dEgyFcM2fOhE2bNkFbW1uSbaeySAJaJaCES5iDCxcudGgtURPUJCtWrIA1a9ZAS0uLZwWF9pMT3HjjjYV8H374IcydOxe6urpYkiFDhpRoRIJLa\/9T4Rol4Ku5Bg8eDI8++mjJ49E03LlzZ6Dm8ksntKP8DJVGJLg09j4VrVUCgXOuJUuWOExD1EaPPfZYqDkXwoKXClDUWsuXL2cfof1QYy5atIh9xHyO4NLa\/1S4RgkU4FJ5Bv2eK5t3qnRB8zaEdPPmzQ7tJ\/LMmDGjYI4SXBp7n4rWKgFt61wqWGUgVSajCkiCS2v\/+xae\/64TjOYeSBm3Va8SNfxkbXAJZ8WyZcsKWkg2Bbdv314ybyO4etdI+vbT6yA7oJ196IouAW1mYZCpiPfdThEVXCtXroRVq1aRKz5638bOQXDFE6FWh4a7ajI8w4cPL3Hnq+ZcBFe8Do6T+7v3boL04J9AtuWROMU0bF5trnjVWpgMz6BBg8hb2MuHHcEVr4O0LSKrFqJxzoWRFujKxwujQGidK14H6sxNcMWTrtbwJ7fH0B30SxEa8TpPd27zP\/8AjBEPgXHFHN2PqsvyA+dcvSFwl2ILqzP2CK54cu\/1W05onSteB8fJTXDFkR6AtnWueNUq5ia4kpJk9HIIrugyk3OUrHPhzaBo93iPjJab4Iomr8RSn\/lfOPfOw5Ad8RNovormXOXItURzuZ0QQTGE5Tw0Sh6CK4q0kktrnd7H4MoPHAOX3ro2uYIbqKRAs7DasBFc1RmN5um9cPx\/ZsNFl7YRXGV2QSBc7nIFbJUyHwmuMns2Zrbzpzvh2N4HYeCAiTDoln+MWVpjZg\/0FuJ+rsWLF\/vuONYpOoJLp3S9y0a4ju6dBRdf2gbDbt5YnUrU+FMdcKFWQpjuvPNOOHfuHEyePJn9TnDVeC+XUf3TZ96BT\/bOhAEDxsPIcQRXGSJ0uuIFXPfddx+LREe4Dh8+DHiWBp6ZsW3btpJn6HZ4kOYqp1vj50G4Ot+dC4MvGQs33rgufoENWALTXGIeJWBCTYXXkSNH2G5hARfuEMbrrbfegvb2dsC9Whg9ofOINYKrOqPy6CcvwPufvQyX9x8LN4+hOVc5vVAwC0XEuls7oWZ67rnnmOYiuMoRcW3mOfDZOjj027Xw+\/3Hwu2jn6\/NRlS51so5l5hjCTORNFeVe6kKj0e49hx7Ba64ZBT8aNQ\/VKEGtf9Igqv2+1BLC3Yd+yfo7HoFRlwyCu697u+0PKPeC3WYhTi\/QUeGfDY8mYX1PgTU7dt46Ck4dvZdMCwTHhv\/amMKIWarHQ4NhEr2\/pFZGFO6NZx99eGnoet3ByADeXjq1p\/XcEuqV3WlWfjII4+w3cLDhg1jNSNXfPU6qFpPXnF4Jfzm7CFIQx5W3fJCtapR089VwnXttdcCnnFBi8g13bexKv\/UoZfh8LkjTHNtuGVFrLIaNXPk8CexNX\/+\/PnKN6AkLUha50paouHKe\/zQBvjg7G8gBSZsvXVJuEyUyiEBB1y41rVlyxaYPn2678vtxKIzplO9YihJGRNcSUozfFmT9zwNWcgxs3DbeB5UQFc0CZRsljx58iR7ycLQoUPZ6UyqkCf5Ee4XNUR7fHBqgitYRjpS3LZ7FaQMEzLQA39\/3QwY1\/97Oh5T12UW4HK\/kUR1QGc1JEFwVUPqAON3rYU05Bhgq6+\/j+Aqoxs851wEVxnSrJMse858AXPe\/XcGVsrIw0uj74Zb+g+uk9ZVrhmRN0tWrmr8SaS5Ki1xgN1fn4Qf\/98OyBh5MMCEn42ZDOMHXF75itT4EwtwqV6xGqZt7nMNw+SJkobgiiKtZNL+99en4M\/3\/xekDZN92oePgo4rrk+m8AYqxQGX+2V0QXJQvcAuKE\/U+wRXVInFT\/\/26VMwbf8uMMCCrJGHn17xffirEd+PX3CDlRALLtWrV5OWH8GVtESDy\/vbjz+Cv\/noEwZWBkx4bORVsGjkVcEZKYVDAgQXDYgSCfz1h58xuDKpHJtzLb5yJDxx5QiSVEQJKOE6duwYzJ07F7q6upTFzZkzh71EXH5rSd++fSM+Olxy0lzh5JRkqh\/tOgI7Tp8BM9WDZzLDDwcOgB3jRyf5iIYoyxOu5cuXs\/dnITS4mCxeAo7vMj569CiDqxIXwVUJKTufMfXXR+GXX5+BnlQOLMOEHw7sD7+aQHOuqD1BcEWVWAOkb\/3nQ5BL5aHHyEPeMOH2QRfDv02iOVfUrie4okqsAdIP3foR5NN5yKdMyBt5AAPg1L2jGqDlyTaR4EpWnjVf2q4vLsD0X30OZtrkH8Nkbfr8gWtqvm2VboAyQsP9xkdVpYRTQ3eFac4VT8J7O0\/Avt0nYG\/nl6ygBztGw7i2Vs9Cd39+Af7szS\/BTFkcrpQJlmHBsWlXxqtIA+Z2wIXBuyNGjNC+jSSKnAmuKNLiadetfJf\/XPWeZ+bnNv2RErKZ\/3oadn55Hsy0BRbTXBYzCzdPboWJrX2iV6aBczii4nEgP\/HEE+wcjd5yEVzBPSG00749J2Hv7hMAYAVnslM82D4aHuy4oZD++tVfQT6bByuNH9RcHK6f33UZTLy8OXS5lNB+s2QlIi3KFTbBpZact3YypAzRILvhT66HB189C\/lsjpmEDK40n3P95bhL2Ieu8BIomXNFDeCls+LDCztsyn17TsOGFz6GH\/\/FSBg7fiDLJrRTkLlXfIYBhmGAZSFg4SA798e3Qc+wAWBl8mDiBwFLWezTcfPF0HHLRWGbQOmA3oncKwfBT2fvgv3vfA1gpGHs+EGwd9cxMPPflllXDhlefqDlWy+Db+78AZhoEgqzEAFjZqEFj4zvB+3j+5VZh8bMRvu5elG\/C+20fvUHAEYGjFQWDPYzjd+DYFk5MHPfgmV2l1Fr2VxkqDnK6L5vGnQ3WZBvQrhMgIw970JvYcqCtqEZWPenZBZGEXwgXOJg0Gq9o6ue51z7Or9k5t76VQfVmsVIgWFkJcgyAEYKLLMHzNw3ZUKGj3Jqs9T4P4R8ayt0Zy3oaTIhn7GY9gKcbzHT0ASwDNh4+3lfN36UgdcIaR1wyfMtsY6F7vmXX34Z3JsixQlQeGDohAkTtMmq3uBC9\/i+zq8APXtQmA9ZzGTzNN9kyFIZBhyajAYDrRvyPWfBMi9E7oN0\/8sgfcUogN9rhXzGgFwTcMCyFphZ7tDAD4PMMuDSn\/0LjG1rDVwri1yROs1QApe8YVIsJs+ePRvefvttFsjb0tLCREFwhR8RwrO3fvX7tnnHPXCGkWaaCCz8nTse\/CDj6W1zsWAyFkEzEbTur1Ev+c7R0v1aIHXJZZD53vUAmRRY2RSYGQPyWYBck8Hg6slYYOIHwcrw+l70i19D5ouT7P8EWXD\/e8KFWTEafvDgwXD\/\/fczsAiuYIFiCjkqYl8nrjvJlzDJcA4k5j2pSJAhkKi9UqkmgFQTGKjNhEZj0KUBUlmwzO9YuUY6A5BKAaQzkL5oEEA2A5BJA6RT7KeVMThg7APQg4BlLMinAfIZdMfzta4++z6ApoPchBUXQeY9Jjzhwpcy4Avv1qxZw7SUDrjcYVZDhgxhZybKb6qsFbMwTFREKWQp+08ImcEBw6vEXMS\/c60mOyIQIj4nQ8jsuRmDjf+f\/S2NH4QrA0YmA5BGqNIOuBhgWVuDMS3GNViOwYWQWexroPm9g5B53wkXQRYRLpxHLViwgJ28iyfqqhaZ45qF4ug21IxibxjO744fPw5Lly4tnPjbm+FCoDBmr1Q7hdNw3DREcNxazPbsMZjEhaYZT+eEDDVhmgEmoBL\/58Dxv4OALJ0GI4O\/25Bliz8ZYJkUmE0pMBGsrMG0Vy4DzGOYPXgQUoe8Q6rIXHR9feI7kcWfxIEzOMd69dVX2ZtOcLOkDri8yly0aBHgR2ivasO1b89XTDy43hRdO4WFDAHhWsuyTNux4TIV2dwMuwohExrPtEFDSBE8hKwIlJFu4s4PAR4ClkEtJ7SYbR4iYKjVJA2GpqKJcOFcLA1golX55QmA3f8RqlFkLkqLyLjDGN3tU6ZMcWgOlKQOuFQnR6kOIq0mXOtWvQ8bVh\/h7m8wuevbyocaXOUlKkLG86OpKLSYaZuNhu0AwdsptvblXMHCPEKTodYSGs3WYAhfppmbihn8P\/50QiZMRDYXS3PAkGcjb0Hul1sjNa2RIWNzLhzoqKVQY+3fv18Jl9eZGuWeW4gw79y50\/EsAdekSZMKkfmVhqtEOzHnQQYMQE8dHjZmaYbMDZiADDUXgi3mZhbXZAifMsSJQ4ZOD6bBUs323MyGrQBYUxEwdHQwLYZODnseZgNmIVwWQPeOaHA18pxM6dDAOdcrr7wC7e3t2szC3gRXKHOvABmfI6H5Zlk9tpkW6cs8RGKui0RsoDAZi2ajbTIiaMyNz7UYc3aw3yVdJupdMA37cG2WRu2FYDXZGizLNZgbMNRaaa7BsPjcgbcgf4bvDeNXuLjFRoRMCRe+ewu9g3fddZc2h0Y14RKucr\/9Tt4E2A4EI1MByORaOOdlnCdehyJgGCaF4x235ktOEQNXvnAuhi76Jkhl+nAvY7oPAAKGnyyClgXI4sfWYGnUXujwMAA1Fy4k53\/7PuS6PijM9eQvgSiwNYK56OmKF8erLVu2jL1lMmlXPJqiwtUvFqZVc66VK1eyl6Bv2rQJ2traQnzrq5OE0k6RSrcHO85vAOc+eTb\/ifpNHumRXEVJz+CgOxaimQnL68PsOJae58G\/cxc9Nw1TDC5uLrJ5WAYhswFjWkyYiMKBApA79gH0fH7IdqCIgGBbe5Y0Jlir1TNkvhEaQrvgXAwHeZKLyGG9hQIuBKujoyM0YP4LudGHtL8mc5ppbGBHNJfKr5ETMHS8MK3EPIomc8Sw+SJ34rPoDe6ix4VnG6x0H24mImCyBsOFZrbgLJYLACCXh\/MHf1FwtmBbOWK2WVpYKohmMtYjZL5wibWsUaNGsQNCk4QryjoXdhNChs4NP8gQqHUrMXbPHRVR\/tANn1NoMnsgsjlZpSBTmYxNNmCIFAKAgHGvJ9NlTIMhYLazQwYMNVgTzsXsORhCJq68CecPvekUix3CJcrmN4satrja01iaLDBwV2W+oejiLiJjGWEiNOReRLj8IEO4FszcEZ4HLSlFeJO8TiXWo9yDy38bSLTqOTdHooeTueTZ6xT4+pgBPKoe52gIAg+fErGKqL1cGizbxOdgCJdgxTQhd\/Jj6DnzqT3FwrLt5QmxIG47WfiXi+pqDMgCt5yghtmwYQPce++9gOFJveHyg6y62kuWjgRZ4c\/ekImIeGZMRdg97N0fwlxErSO8iOjnyNomIp8nKQFDhwczEZvBaGrmZqGIvLJMyJ3+BHJnj9vRIlw783keJsJ5p625ZKcKS4XR9QKsYMBE2\/Ccj7ETWmtuu0sgXGFgQi3mjqwIo5miai53XWoTsmL0u3pe5gzsjWJSlfaVAExaC7MdHgxipnEMCTDhou\/LtFiqqS+HC0OlOPUApglW9wW4cHI\/r769BFB05qS4tkQHD5Zf2EojHCuiljJcxf\/7fbG4D9MJMzarmSY2XMI8FC8qF2FLQXMqbLSIuveLLQwjnNqAjH+bcw0lzMHSYNxie4trXaXpORrBl9eCtK3NbB2G5iJz69tzsFS6LxgZ\/PQDo08\/5z4zMw9Wrhvy356A\/Hcn7QiR4tpfQXOJIGR7qaCotYQH0137MO3heWoFslhwic2VY8aMYY1Gt72AK8gbiOlV7n2VBgweRDxFbUImgyb2dalaXNw9zJVI+INn5EXlQqCwHNnBoutxEybfqoKew1S6H6Sa+4ORtc8qxPSotcw8QL4HzJ5zNlw9YOW72boa91CKBXZ+FEFxAVxeQuAgxdPKvR+ysuGS16RwHQzDp2S4gmIHUbjuN1km9ZLzWoKMD0AeWiUWhJ1exiJ8Yb9k\/NN5azP0KPLtKbjQfBGkm1uYSVhwijC4cgAmAnUBzJ7fgZm\/wBat8egBpk1dmpk5TwqbQd0OHJWJGL2V6Ma\/po0fhIpromI9NMrSTfSnBucoGy65aNRSbriCIjAwf5jYwuAmeKeIB1nSgzqoJWLQ85hA\/tWO\/zgXpuN+2xe1iWugs2fa0SfpZkg3D2KA8dhF4YwwAcwcAwmPFbDy58FErYXajK1v9bC5Fjd9iwvP8i7rcOZskKxK738BTzKo8IPHTqBXWSzbRC8tmRx1DZcQUTmQMW2i2KCYjNj9SnFCxp0DYj4iQHPP2eTyws9dOEwCAhGBn2JgpZsulQpFcNDTl+dwWQKwbqbBii53pBAjVbijpOgBLYJc1GLxJXkOdkBHezt0w1GY1zE1foEJl9AQcKkgw79hYDKaDuJybn4UA88eWAkLPlxxzhArpswKcy2xQB3d1OIDXFxOCNJNAyGVHcBvFuZltmlng8NMQPx\/vtsZ8mWbf6UAyXWMAn+plEa3fcM00ugJ34aO1gkn6+RTaYMrKHYQmxImtjD5JjsdHyrI2HFnGOmxB71hXnFzOmrmVabwHOLCtG2+SYvD6i0npWUV21I6wNEERCcGOjMcF1tw5utkTCPZH5x7FX5nGYrrV0nKrBs+hnntU5Xayb2U496LKDzZeGSFuOQtUkH34\/awNriq4S2MKgzZXPSCrHrhVN6gFd35wpST5oceXwalYPF9YSyAl8Ua9ilqKmnxt+BCZ1BxwIpQOSFNyuQT5h77KVkWskQEWOhEE0f7uY+JcP+OfoAtW7awc2EwWDzoftTx5E6vDa5KrnPFFUJtQsbsNmleIx92U9LNxS0oGBnPgnf5dhV5r5hLZfFFYwv1lmQWsigLlTaPb+6h12\/chNZQ5p7KYSY71hAe97KO7I2+5pprfO8ncRanNriwo4IiMILux4WmnPwiCl9oMhSycO32ntAqVctUi85yuqLDQvyVr0u5kZLWzxznKfqtwUWXdBjtFLVUGS7M615Hxb+Jd9CNGzfO9z4ezBT3SgSuuJXojfndkMnmSe+GzAWU9Kscv8j\/7PbilaKWZN\/odkbIZt+RI0dK5vQCLvx5xx13+N4XUUNx2k9weUgPTUW80DGDC5NCk9UmZHGGSPl5dWgnr9q4d2l47eZAzUVwld+nWnLKmsy9p6x2NJkW0TgK1a2dVC1QHWxEcGnqa9UL\/OSX9MWZ6xFkvNPQRY6Lt3jhQq6fZ09TN7NiVY4z\/LsqakiYhfjeb5xzuaOK5Ps05\/LoNZUnSSQN8mLiIahhrkaGDEEa19YKY9su83SVh5Fh3DTiS\/Kee+4pnNosylRtg6pJb2FcISWdX9jVqklp0PqbfE59mHrVO2RCQ01oa4NRbd9UFSa5P\/zAEumC1rGC7ofpf780defQUNnfsgCCovXLXd8oJ34xbufpzF9JZ0Q57RDvjVPlFVEYQREYQffLqZecp+7gUglMnm8FRevHtbVrFTKESWgneW0v7gBr5Px1B5cqLEY2Bbdv3659qwsOqFqALK520v3NX+tg1h1cqg6RTUW8r3sfmVyH3gSZ0E5Rwoz8BrjuOQvBVQMSkOEaPnx4VaLxgyDbt\/sElHe8tn8HxNVOXqUHeePKnbvWwHAKXcW601xBW128juaOc3ZHaGkHmItYDu4piwMZwoQX36bxsTbvnsrris8VsXtx565RZNpb09YdXKrDSnEg4FnzuGiIV1KnTsXp1CBNFmWriy7t5Ne+oAiIJGLz4si3N+StO7hQqO6JtnsTXZwIjaQ7LegUYaHJUAs1wUj2eKGd8IgxvLz2PCVdV7k8gitYunUJV3Cze18KFWQ4gPHv+GmCEfCDthlsIbc3uMoJruAxRHAFy6iiKeSID3ywONGoGtrJr+FBsXs055LeiVzREUQP85SA2OoiEsR5J5lOMZO3MFi6pLmCZUQpPCRA61z+Q4PgqiA6vcmRkkSzKUKD4EpiHMUuI8xWF3cwqnwMGFbAfX\/OnDmOrRYYN7l48eJCXd1e0tiNoAIiSYA0VyRxlZ84aKvL66+\/DsePH4elS5cC7ilzx0iidw7X6dauXcteduG+7\/5daJXp06ezl8bTVXkJEFwVkrnfVpe7774bXnrpJVi4cGHhDD6hqfAnLsiqIh9QUx09epTdl\/8vmqR6ZoWaS4\/B43+s4kHkJBCNEihnq4vY9InHbmNUyYwZMxzwyfDgITq4fV3WUl4hShqbSUVLEiC4KjQcosIlm3WTJ0+GefPmlWg2sZD77LPPwjPPPAOTJk0qgUt1TkSFmtzwjyG4KjQEosIlp0dnCMFVoY5K8DEEV4LC9CsqKFpf3qLhTqsKRsZnkeaqUOeV+RiCq0zBRc0m5j8PP\/wwM++6urrY64BaW1vhxRdfhBtuuIEV6fYK4t+8zgVxz7mE80PUjeZcUXsp2fQEV7Ly9Cwt7DrXG2+8UXC3y4WRt7BCHZXgY8qCSxVXVk6d\/I5AK6e83p7HL0IDZeEFlkqj0TpXb+\/tMl3xqvmDOOXWHTXgJQLVqbhy2iVLljTM4qcbOlkOcpQFRWj4AxV0rF6lcSxLc8kaR3S4OL4M3y4xa9YsRzvcwPlpPvcLyiotEHpebUsg6vhJYuuMVxmR4XJ7rtxzAfck2h054BeWo5rM13ZXU+0rLQH5yOowh+R4LZHgq15xjXD+\/Pm+FpR43rZt20qaGhkuNwBR4FJFUYcRvjuANUweSlO\/EnAHKEdpqTzdCDIjvXZby89ThZ3hfRzrkeFyT7zDwiVCeG666SbYv39\/SSiPqpJe6ztRhElp618CYSBQSUGYc08++SSsX78eVNrHnU+eA3uZoALaSHCJiTc+UERnh4VLnAbkpbYJrvqHQFcLy4XL7204QXUNoz0jwSV2np49e5a9rBm3PpQLV5hvCdFAMguDurqx70d1YgizDUPKhg0bxrb54IXB0e74TCFZedc1zsdWrFgBDz30ELz55puFbUKiXHEGZmi4UONg5PUDDzwAy5Ytiw2XO8KbNFdjAxKn9WG0iHtpR+Rxm3nuo85lEN1741RbemRPeGi4ROPdbvRyNRfBFWc4UV5ZAu6zPOR7KqeF\/G4v9wZV1S4CL7MzaK02MbimTp3KNNrEiRPhtddeg+XLl0NLSwszG\/ESc65yPIZkFhJMXhIQ4wlPyVKd8qtyiiEUn376KeB7AzZv3lww67y8h16RRFo119ChQ5mdivOnIUOGMCcHXnPnzmWBqao5k1cwKZmFBFA5EhDaw+sL2M\/jrILD7eSImj8xsxA1k\/sFBkGR2F7rAgRXOUOrsfOIgS+cEqr3WXtFT6DkVHC50\/sdlRA014ttFrq71w8uv28BL7gq9faRxh6mtdd6YcLt3btXuYtAtCgqXG7T0O+tLap7iWku1cu5\/eDy0looCHFPLDYLVz0dD1Z7A193jcOChfXwc3YEHeATpChUX\/yx4NItOCqfJBAkAS9gVM4y4QtQKQIB1+zZs2HBggUOP4FXHUR5qDG93PaR17mCGkz3SQK1JoEgzeXnofSarpDmqrVRQPWtSQlEdmjUZCup0iSBKkjg\/wEspW8pzgrq2gAAAABJRU5ErkJggg==","height":117,"width":156}}
%---
%[output:450a373f]
%   data: {"dataType":"text","outputData":{"text":"仰角(theta)=30.30°，方位角(phi)=60.00°\n","truncated":false}}
%---
%[output:7a025eb0]
%   data: {"dataType":"text","outputData":{"text":"仰角(theta)=28.40°，方位角(phi)=90.70°\n","truncated":false}}
%---
%[output:906a5286]
%   data: {"dataType":"text","outputData":{"text":"精确仰角: 30.319°, 精确方位角: 59.977°\n","truncated":false}}
%---
