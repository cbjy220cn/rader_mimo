function [theta,phi]=plane_move_many(param)
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
Ny=6;
numTX = 2;
numRX = 3;

N_L=floor(2*fc/BW);
(mod(N_L, 2) == 0)*(N_L-1)+(mod(N_L, 2) == 1)*(N_L) % 扩展的虚拟阵列
N_L=param.N_L;
Vmax = lambda/(T*4); % Max Unamb velocity m/s

d_y=lambda/2/2;
d_tx= lambda/2;
d_rx = numRX*d_tx; % dist. between rxs
% d_tx = 4*d_rx; % dist. between txs

tr_vel=-d_tx/2/T;
% tr_vel=0;


%% Targets

r1_radial = 3000;
tar1_theta = param.theta;
tar1_phi= param.phi;
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

% r2_radial = 600;
% tar2_theta = 30;
% tar2_phi = 60;
% r2_x = cosd(tar2_phi)*sind(tar2_theta)*r2_radial;
% r2_y = sind(tar2_phi)*sind(tar2_theta)*r2_radial;
% r2_z = cosd(tar2_theta)*r2_radial;
% 
% v2_radial = 0.1; % velocity 1
% v2_x = cosd(tar2_phi)*sind(tar2_theta)*v2_radial;
% v2_y = sind(tar2_phi)*sind(tar2_theta)*v2_radial;
% v2_z = cosd(tar2_theta)*v2_radial;
% r2 = [r2_x r2_y r2_z];
% 
% 
% tar2_loc = zeros(length(t),3);
% tar2_loc(:,1) = r2(1) + v2_x*t;
% tar2_loc(:,2) = r2(2) + v2_y*t;
% tar2_loc(:,3) = r2(3) + v2_z*t;

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
    
       %  scatter3(tx_loc{i,j}(1),tx_loc{i,j}(2),tx_loc{i,j}(3),'b','filled')
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


P1 = rx_loc{1,1};
P2 = r1;


% 计算两点之间的距离
d = sqrt((P2(1) - P1(1))^2 + (P2(2) - P1(2))^2 + (P2(3) - P1(3))^2);

% % 计算仰角
% theta = asin((P2(3) - P1(3)) / d); % 仰角，单位为弧度
% 
% % 计算方向角
% phi = atan2(P2(2) - P1(2), P2(1) - P1(1)); % 方向角，单位为弧度
% 
% % 如果需要，可以将弧度转换为度
% theta_deg = rad2deg(theta);
% phi_deg = rad2deg(phi);
% 
% % 输出结果
% fprintf('仰角 (弧度): %.4f\n', theta);
% fprintf('方向角 (弧度): %.4f\n', phi);
% fprintf('仰角 (度): %.4f\n', theta_deg);
% fprintf('方向角 (度): %.4f\n', phi_deg);


%% TX

delays_tar1 = cell(numTX,numRX,Ny);
% delays_tar2 = cell(numTX,numRX,Ny);
for k = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            delays_tar1{i,j,k} = (vecnorm(tar1_loc-rx_loc_t{j,k},2,2) ...
                            +vecnorm(tar1_loc-tx_loc_t{i,k},2,2))/c; 
            % delays_tar2{i,j,k} = (vecnorm(tar2_loc-rx_loc_t{j,k},2,2) ...
            %                 +vecnorm(tar2_loc-tx_loc_t{i,k},2,2))/c;
        end
    end
end


% r1_at_t = cell(numTX,numRX);
% r2_at_t = cell(numTX,numRX);
% tar1_angles = cell(numTX,numRX);
% tar2_angles = cell(numTX,numRX);
% tar1_velocities = cell(numTX,numRX);
% tar2_velocities = cell(numTX,numRX);


%% Complex signal
phase = @(tx,fx) 2*pi*(fx.*tx+slope/2*tx.^2); % transmitted

snr=param.snr;
mixed = cell(numTX,numRX,Ny);
for l = 1:Ny
    for i = 1:numTX
        for j = 1:numRX
            % disp(['Processing Channel: ' num2str(j) '/' num2str(numRX)]);
            for k = 1:numChirps*numCPI
                phase_t = phase(t_onePulse,fc);
                phase_1 = phase(t_onePulse-delays_tar1{i,j,l}(k*numADC),fc); % received

                signal_t((k-1)*numADC+1:k*numADC) = exp(1j*phase_t);
                signal_1((k-1)*numADC+1:k*numADC) = exp(1j*(phase_t - phase_1));
            end

            mixed{i,j,l} =awgn(signal_1,snr,'measured');
        end
    end
end

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




% figure
% imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1))))));
% colormap(jet(256))
% % set(gca,'YDir','normal')
% clim = get(gca,'clim');
% caxis([clim(1)/2 0])
% xlabel('Velocity (m/s)');
% ylabel('Range (m)');


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

% figure
% colormap(jet)
% imagesc(ang_phi,R,20*log10(abs(range_az)./max(abs(range_az(:))))); 
% xlabel('Azimuth Angle')
% ylabel('Range (m)')
% title('FFT Range-Angle Map')
% set(gca,'clim', [-35, 0])

% doas = zeros(K,length(ang_phi)); % direction of arrivals
% figure
% hold on; grid on;
% for i = 1:K
%     doas(i,:) = fftshift(fft(rangeFFT(cfar_ranges(i),cfar_dopps(i),:),length(ang_phi)));
%     plot(ang_phi,10*log10(abs(doas(i,:))))
% end
% xlabel('Azimuth Angle')
% ylabel('dB')




%% Angle Estimation - MUSIC Pseudo Spectrum


M = numCPI; % # of snapshots
a1=zeros((numRX*numTX+N_L-1)*Ny,length(ang_theta)*length(ang_phi));



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

% reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi));
% figure;
% % mesh(10*log10(abs(reshape(music_spectrum(1,:,:),length(ang_theta),length(ang_phi)))));
% mesh(ang_phi,ang_theta,10*log10(abs(squeeze(music_spectrum(1,:,:)))));
% ax = gca;
% chart = ax.Children(1);
% datatip(chart,phi_max,theta_max,max_value);
% xlabel('方位角');ylabel('仰角');
% zlabel('空间谱/db');
% grid;
% 输出结果
fprintf('最高点坐标: (Theta: %.2f, Phi: %.2f)\n', theta_max, phi_max);
% fprintf('修正点坐标: (Theta: %.2f, Phi: %.2f), 强度: %.2f dB\n', theta_max-2, phi_max-2, max_value);

theta=theta_max;
phi=phi_max;




% %% 改进版抛物线插值函数（带边界检查）
% function [refined_angle, is_valid] = safe_parabolic_interpolation(angles, spectrum_values)
%     % 输入参数检查
%     if numel(spectrum_values) < 3
%         refined_angle = angles(round(numel(angles)/2));
%         is_valid = false;
%         return;
%     end
% 
%     % 三点抛物线插值核心算法
%     y1 = spectrum_values(1);
%     y2 = spectrum_values(2);
%     y3 = spectrum_values(3);
%     denominator = y1 - 2*y2 + y3;
% 
%     % 防止分母过小导致数值不稳定
%     if abs(denominator) < 1e-6
%         refined_angle = angles(2);
%         is_valid = false;
%     else
%         delta = 0.5 * (y1 - y3) / denominator;
%         refined_angle = angles(2) + delta * (angles(2)-angles(1));
%         is_valid = true;
%     end
% end
% 
% %% 改进后的峰值处理流程
% % 初始化存储精确角度的矩阵
% peaks_info = [peaks_info, zeros(size(peaks_info,1), 2)];
% 
% % 对每个检测到的峰值进行亚网格插值
% for i = 1:size(peaks_info,1)
%     row = peaks_info(i,1);
%     col = peaks_info(i,2);
% 
%     %% 仰角(theta)方向插值 --------------------------------
%     % 动态调整索引范围确保3个点
%     theta_start = max(1, row-1);
%     theta_end = min(length(ang_theta), row+1);
%     if (theta_end - theta_start) < 2 % 边界补偿
%         if theta_start == 1
%             theta_end = min(theta_start+2, length(ang_theta));
%         else
%             theta_start = max(1, theta_end-2);
%         end
%     end
% 
%     theta_indices = theta_start:theta_end;
%     theta_vals = ang_theta(theta_indices);
%     spectrum_theta = smoothed_spectrum(theta_indices, col);
% 
%     % 执行安全插值
%     [refined_theta, theta_valid] = safe_parabolic_interpolation(theta_vals, spectrum_theta);
% 
%     %% 方位角(phi)方向插值 --------------------------------
%     % 动态调整索引范围确保3个点
%     phi_start = max(1, col-1);
%     phi_end = min(length(ang_phi_sorted), col+1);
%     if (phi_end - phi_start) < 2 % 边界补偿
%         if phi_start == 1
%             phi_end = min(phi_start+2, length(ang_phi_sorted));
%         else
%             phi_start = max(1, phi_end-2);
%         end
%     end
% 
%     phi_indices = phi_start:phi_end;
%     phi_vals = ang_phi_sorted(phi_indices);
%     spectrum_phi = smoothed_spectrum(row, phi_indices);
% 
%     % 执行安全插值
%     [refined_phi, phi_valid] = safe_parabolic_interpolation(phi_vals, spectrum_phi);
% 
%     %% 结果整合 ------------------------------------------
%     if theta_valid && phi_valid
%         peaks_info(i,4:5) = [refined_theta, refined_phi];
%     else % 插值失败时使用原始网格值
%         peaks_info(i,4:5) = [ang_theta(row), ang_phi_sorted(col)];
%     end
% end
% 
% %% 结果输出（示例）
% if size(peaks_info, 1) >= 1
%     fprintf('精确仰角: %.3f°, 精确方位角: %.3f°\n', peaks_info(1,4), peaks_info(1,5));
% end


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
end