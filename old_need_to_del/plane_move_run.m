function [theta, phi] = plane_move_run(params,move)
%PLANE_MOVE_RUN 此处显示有关此函数的摘要
%   此处显示详细说明

%% Radar parameters
c = physconst('LightSpeed'); %speed of light
BW = 50e6; %bandwidth
fc = 3e9; % carrier frequency
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
Ny=params.Ny;
numTX = params.numRX;
numRX = params.numTX;

% N_L=floor(2*fc/BW);
% N_L=(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L); % 扩展的虚拟阵列
N_L=params.N_L;
Vmax = lambda/(T*4); % Max Unamb velocity m/s

d_y=lambda/2/2;
d_tx= lambda/2;
d_rx = numRX*d_tx; % dist. between rxs
% d_tx = 4*d_rx; % dist. between txs


if move==1
    tr_vel=-d_tx/2/T;
else
    tr_vel=0;
end

% tr_vel=0;


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

% scatter3(tar1_loc(1,1),tar1_loc(1,2),tar1_loc(1,3),'r','filled')
% hold on

r2_radial = 600;
tar2_theta = 30;
tar2_phi = -50;
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

% scatter3(tar2_loc(1,1),tar2_loc(1,2),tar2_loc(1,3),'blue','filled')
% xlabel('x');
% ylabel('y');
% zlabel('z');
% hold off


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

        % scatter3(tx_loc{i,j}(1),tx_loc{i,j}(2),tx_loc{i,j}(3),'b','filled')
        % hold on
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
        % scatter3(rx_loc{i,j}(1),rx_loc{i,j}(2),rx_loc{i,j}(3),'r','filled')
    end
end
% xlabel('x');
% ylabel('y');
% zlabel('z');
% hold off


plane_loc=cell(numTX*numRX,Ny);
for i =1:numTX
    for k = 1:numRX
        for j = 1:Ny
            plane_loc{i+(k-1)*numTX,j}=rx_loc{k,j}+tx_loc{i,j};
            % scatter3(plane_loc{i+(k-1)*numTX,j}(1),plane_loc{i+(k-1)*numTX,j}(2),plane_loc{i+(k-1)*numTX,j}(3),'b','filled')
            % hold on
        end
    end
end

plane_loc_plus=cell(numTX*numRX+N_L-1,Ny);
for i =1:numTX*numRX+N_L-1
    for j=1:Ny
        if i<=(N_L-1)/2
            plane_loc_plus{i,j}(1)=plane_loc{i+(N_L-1)/2,j}(1)-d_tx*((N_L-1)/2+1-i);
        end
    end
end


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




%% Complex signal
phase = @(tx,fx) 2*pi*(fx.*tx+slope/2*tx.^2); % transmitted

snr=params.snr;
% mixed = cell(numTX,numRX,Ny);
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
%             mixed{i,j,l} = signal_1 ;
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
            mixed_signal = signal_1;

            % 添加 AWGN 噪声
            mixed{i, j, l} = awgn(mixed_signal, snr, 'measured');

        end % 结束 j 循环
    end % 结束 i 循环
end % 结束 l 循环

% --- 清理临时变量 (可选) ---
clear t_onePulse_row phase_t_matrix phase_1_matrix phase_2_matrix signal_1_matrix signal_2_matrix signal_1 signal_2 mixed_signal current_delays1 current_delays2 current_delays1_col current_delays2_col numSamplesPerChirp numChirpsTotal numSamplesTotal;





%% Post processing - 2-D FFT
% size(cat(3,mixed{:}))



RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX*Ny);

% --- 参数计算 ---
if mod(N_L, 2) == 0
    error('N_L must be an odd integer for the original indexing logic.');
end
N_half = (N_L - 1) / 2; % Half window size excluding center

numChirpsOutPerCPI = numChirps - (N_L - 1); % Chirps per CPI in output
numChannelsInPerNy = numRX * numTX; % Channels per Ny block in input
numChannelsOutPerNy = numChannelsInPerNy + N_L - 1; % Channels per Ny block in output

numColsIn = numChirps * numCPI;
numPagesIn = numChannelsInPerNy * Ny;
numColsOut = numChirpsOutPerCPI * numCPI;
numPagesOut = numChannelsOutPerNy * Ny;

% --- 预分配输出 ---
RDC_plus = zeros(numADC, numColsOut, numPagesOut);

% --- 向量化 k 循环 ---
k_vec = 1:numChirpsOutPerCPI; % Base vector for output chirp indices within a CPI block

for l = 1:Ny
    % 计算当前 Ny 块在输入和输出页维度上的偏移量
    page_offset_in = (l - 1) * numChannelsInPerNy;
    page_offset_out = (l - 1) * numChannelsOutPerNy;

    % 输入 RDC 中当前 Ny 块的第一个和最后一个通道的页索引
    page_idx_in_first = 1 + page_offset_in;
    page_idx_in_last = numChannelsInPerNy + page_offset_in;

    for i = 1:numCPI
        % 计算当前 CPI 块在输入和输出列维度上的偏移量
        col_offset_in = (i - 1) * numChirps;
        col_offset_out = (i - 1) * numChirpsOutPerCPI;

        % --- 1. 复制中心数据块 (向量化 k) ---
        % 输入 RDC 的列索引 (绝对索引)
        cols_in_center = ((N_half + 1):(numChirps - N_half)) + col_offset_in;
        % 输入 RDC 的页索引 (当前 Ny 块的所有通道)
        pages_in_center = (1:numChannelsInPerNy) + page_offset_in;

        % 输出 RDC_plus 的列索引 (绝对索引)
        cols_out = k_vec + col_offset_out;
        % 输出 RDC_plus 的页索引 (中心通道部分)
        pages_out_center = ((N_half + 1):(N_half + numChannelsInPerNy)) + page_offset_out;

        % 执行复制
        RDC_plus(:, cols_out, pages_out_center) = RDC(:, cols_in_center, pages_in_center);

        % --- 2. 填充边缘数据块 (循环 j, 向量化 k) ---
        for j = 1:N_half
            % --- 左侧填充 ---
            % 输入 RDC 的列索引 (绝对索引, k 向量化)
            cols_in_left = (k_vec + N_half + j) + col_offset_in;
            % 输出 RDC_plus 的页索引 (左侧第 j 个填充通道)
            page_out_left = (N_half + 1 - j) + page_offset_out;

            % 执行复制 (源: RDC特定列, 第一个通道; 目标: RDC_plus特定列, 左填充通道)
            RDC_plus(:, cols_out, page_out_left) = RDC(:, cols_in_left, page_idx_in_first);

            % --- 右侧填充 ---
            % 输入 RDC 的列索引 (绝对索引, k 向量化)
            cols_in_right = (k_vec + N_half - j) + col_offset_in;
            % 输出 RDC_plus 的页索引 (右侧第 j 个填充通道)
            page_out_right = (N_half + numChannelsInPerNy + j) + page_offset_out;

            % 执行复制 (源: RDC特定列, 最后一个通道; 目标: RDC_plus特定列, 右填充通道)
            RDC_plus(:, cols_out, page_out_right) = RDC(:, cols_in_right, page_idx_in_last);
        end % 结束 j 循环
    end % 结束 i 循环
end % 结束 l 循环

% --- 清理临时变量 (可选) ---
clear N_half numChirpsOutPerCPI numChannelsInPerNy numChannelsOutPerNy ...
      numColsIn numPagesIn numColsOut numPagesOut k_vec l page_offset_in ...
      page_offset_out page_idx_in_first page_idx_in_last i col_offset_in ...
      col_offset_out cols_in_center pages_in_center cols_out pages_out_center ...
      j cols_in_left page_out_left cols_in_right page_out_right;





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
ang_phi = -180:0.05:180; % angle axis
ang_theta=0:0.05:90;

RDMs = zeros(numADC,numChirps_new,(numRX*numTX+N_L-1)*Ny,numCPI);
for i = 1:numCPI
    RD_frame = RDC_plus(:,(i-1)*numChirps_new+1:i*numChirps_new,:);
    RDMs(:,:,:,i) = fftshift(fft2(RD_frame,N_range,N_Dopp),2);
    % RDMs(:,:,:,i) = RD_frame;
end

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


M = numCPI; % # of snapshots
% figure
% imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1))))));
% colormap(jet(256))
% % set(gca,'YDir','normal')
% clim = get(gca,'clim');
% caxis([clim(1)/2 0])
% xlabel('Velocity (m/s)');
% ylabel('Range (m)');
if params.cf==1
    %% CA-CFAR

    numGuard = 2; % # of guard cells
    numTrain = numGuard*2; % # of training cells
    P_fa = 1e-5; % desired false alarm rate
    SNR_OFFSET = -5; % dB
    RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

    [RDM_mask, cfar_ranges, cfar_dopps, K] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
    cfar_ranges_real=(cfar_ranges-1)*3;
    % figure
    % h=imagesc(V,R,RDM_mask);
    % xlabel('Velocity (m/s)')
    % ylabel('Range (m)')
    % title('CA-CFAR')


    %% Angle Estimation - FFT

    rangeFFT = fft(RDC_plus(:,1:numChirps_new,:),numADC);

    angleFFT = fftshift(fft(rangeFFT,length(ang_phi),3),3);
    range_az = squeeze(sum(angleFFT,2)); % range-azimuth map





    %% Angle Estimation - MUSIC Pseudo Spectrum



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




    spectrum_data = 10 * log10(abs(reshape(music_spectrum(1,:,:), length(ang_theta), length(ang_phi))));

    % 找到最大值及其索引
    [max_value, max_index] = max(spectrum_data(:));
    [row, col] = ind2sub(size(spectrum_data), max_index);

    % 获取对应的角度
    theta_max = ang_theta(row);
    phi_max = ang_phi(col);

    % 输出结果
    fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max, phi_max, max_value);



    theta=theta_max;
    phi=phi_max;

else


    RDC_plus_rot_frame=RDC_plus(:,1:numChirps_new,:);

    M = numCPI; % # of snapshots

    A = reshape(squeeze(RDC_plus_rot_frame(:,1:100,:)),numADC*100,(numRX*numTX+N_L-1)*Ny);

    Rxx = (A'*A);



    [Q,D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
    [D, I] = sort(diag(D),'descend');
    Q = Q(:,I); % Sort the eigenvectors to put signal eigenvectors first
    Qs = Q(:,1); % Get the signal eigenvectors
    Qn = Q(:,2:end); % Get the noise eigenvectors
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



    %% 方位角坐标修正
    % 生成修正后的方位角坐标 (自动处理-180~180循环)
    ang_phi_shifted = mod(ang_phi  +180+180, 360) - 180;

    %% 数据与坐标对齐
    % 获取修正后坐标的排序索引（保证单调性）
    [ang_phi_sorted, sort_idx] = sort(ang_phi_shifted);

    % 对频谱数据按新方位角排序（沿phi轴重排）
    spectrum_shifted = squeeze(music_spectrum_2D);
    spectrum_shifted = spectrum_shifted(:, sort_idx);

    % % reshape(music_spectr2um(1,:,:),length(ang_theta),length(ang_phi));
    % figure;
    % % mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
    % mesh(ang_phi,ang_theta,10*log10(abs(music_spectrum_2D(:,:))));
    %
    % xlabel('方位角');ylabel('仰角');
    % zlabel('空间谱/db');
    % grid;
    %

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

    % if size(peaks_info, 1) >= 2
    %     second_peak = peaks_info(2, :);
    %     fprintf('仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
    %         ang_theta(second_peak(1)), ang_phi_sorted(second_peak(2)));
    %     % fprintf('修正仰角(theta)=%.2f°，方位角(phi)=%.2f°\n', ...
    %     %     ang_theta(second_peak(1)), mod(ang_phi_sorted(second_peak(2))+360/N_num/numRX+180, 360) - 180);
    % else
    %     disp('未找到明显的次峰值。');
    % end
    theta=ang_theta(main_peak(1));
    phi=ang_phi_sorted(main_peak(2));
end




end

