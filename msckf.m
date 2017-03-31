clear;
close all;
clc;
addpath('utils');

tic
dataDir = './dataset/KITTI';
fileName = './dataset/KITTI/dataset_camera_alignedindex_featuretracks.mat'; kStart = 1; kEnd = 111;

load(fileName);

%Set up the camera parameters 
camera.c_u      = cu;                   % Principal point [u pixels] 
camera.c_v      = cv;                   % Principal point [v pixels]
camera.f_u      = fu;                   % Focal length [u pixels]
camera.f_v      = fv;                   % Focal length [v pixels]
camera.w        = w;                    % distortion
camera.q_CI     = rotMatToQuat(C_c_v);  % 4x1 IMU-to-Camera rotation quaternion
camera.p_C_I    = rho_v_c_v;            % 3x1 Camera position in IMU frame

%Set up the noise parameters
%% TODO 图像噪声R
y_var = 11^2 * ones(1,4);               % pixel coord var  
noiseParams.u_var_prime = y_var(1)/camera.f_u^2;
noiseParams.v_var_prime = y_var(2)/camera.f_v^2;

%% TODO [n_g' n_wg' n_a' n_wa'] --> [w_var dbg_var a_var dba_var]
%% TODO Q_imu = E[ [n_g' n_wg' n_a' n_wa']' [n_g' n_wg' n_a' n_wa'] ] 系统噪声协方差Q
w_var = 4e-2 * ones(1,3);               % rot vel var  
a_var = 4e-2 * ones(1,3);               % lin accel var  
dbg_var = 1e-6 * ones(1,3);            % gyro bias change var 
dba_var = 1e-6 * ones(1,3);            % accel bias change var
noiseParams.Q_imu = diag([w_var, dbg_var, a_var, dba_var]);

% state : [q bg ba v p]
q_var_init = 1e-6 * ones(1,3);         % init rot var
bg_var_init = 1e-6 * ones(1,3);        % init gyro bias var
ba_var_init = 1e-6 * ones(1,3);        % init accel bias var
v_var_init = 1e-6 * ones(1,3);         % init velocity var
p_var_init = 1e-6 * ones(1,3);         % init pos var
%% TODO 是IMU的error的初始协方差 就是P0+
noiseParams.initialIMUCovar = diag([q_var_init, bg_var_init, ba_var_init,v_var_init, p_var_init]);
   
% MSCKF parameters
msckfParams.minTrackLength = 10;        % Set to inf to dead-reckon only
msckfParams.maxTrackLength = Inf;      % Set to inf to wait for features to go out of view
msckfParams.maxGNCostNorm  = 1e-2;     % Set to inf to allow any triangulation, no matter how bad
msckfParams.minRCOND       = 1e-12;
msckfParams.doNullSpaceTrick = true;
msckfParams.doQRdecomp = true;

%% ========================== Prepare Data======================== %%
% Important: Because we're idealizing our pixel measurements and the
% idealized measurements could legitimately be -1, replace our invalid
% measurement flag with NaN
prunedStates = {};
% IMU state for plotting etc. Structures indexed in a cell array
%% TODO 创建和image数量一样size的状态数组
imuStates = cell(1,numel(tracks_timestamp));

% idealize feature
%% TODO 去除相机内参
undistored_featuretracks = undistortFeatureTracks(featuretracks,cu,cv,fu,fv,w);

%% TODO 就是转换一下格式 每个image至多1000个feature
y_k_j = transformFeatureTracksFormat(undistored_featuretracks); % transform the tango format 
y_k_j(y_k_j == -1) = NaN;

%% TODO 有多少state就有多少measurement
measurements = cell(1,numel(tracks_timestamp));

%% TODO 有效的landmark的数量
numLandmarks = size(y_k_j,3);

% get camera-gyro aligned index
aligned_index = syn_index;
aligned_imu_reading = aligned_gyro_accel;
first_v_I_G = aligned_gyro_accel(aligned_index(2, 1),8:10)

for state_k = kStart:kEnd
    % get IMU readings during the Period between previous and current image
    % coming 
    start_gyro_accel_index =  aligned_index(3,state_k);
    end_gyro_accel_index   =  aligned_index(3,state_k+1);
    
    measurements{state_k}.index        = state_k;
    measurements{state_k}.imu_reading  = aligned_imu_reading(start_gyro_accel_index:end_gyro_accel_index,:);
    measurements{state_k}.y            = squeeze(y_k_j(1:2,state_k,:));   %% TODO 把每帧的特征拿出来
end

%% ==========================Initial State======================== %%
% fill the field of first imustate
 firstImuState.q_IG = [0;0;0;1];
 firstImuState.b_g = zeros(3,1);
 firstImuState.b_a = zeros(3,1);
 firstImuState.v_I_G = first_v_I_G';
 firstImuState.p_I_G = [0;0;0];
 
% initialize the first state
%% TODO 初始化咯
[msckfState, featureTracks, trackedFeatureIds] = initializeMSCKF(firstImuState, measurements{kStart}, camera, kStart, noiseParams);
imuStates = updateStateHistory(imuStates, msckfState, camera, kStart);
msckfState_imuOnly{kStart} = msckfState;

%% ============================MAIN LOOP========================== %%
numFeatureTracksResidualized = 0;
map = [];

for state_k = kStart:(kEnd-1)
    fprintf('state_k = %4d\n', state_k);
    
    %% ==========================STATE PROPAGATION======================== %%
    %% TODO B. Propagation
    %Propagate state and covariance
    msckfState = propagateMsckfStateAndCovar(msckfState, measurements{state_k}, noiseParams);
    msckfState_imuOnly{state_k+1} = propagateMsckfStateAndCovar(msckfState_imuOnly{state_k}, measurements{state_k}, noiseParams);

    %% ==========================STATE AUGMENTATION======================== %%
    %% TODO C. State Augmentation
    msckfState = augmentState(msckfState, camera, state_k+1);
    
    %% ==========================FEATURE TRACKING======================== %%
    % Add observations to the feature tracks, or initialize a new one
    % If an observation is -1, add the track to featureTracksToResidualize
    featureTracksToResidualize = {};
    
    for featureId = 1:numLandmarks
        %IMPORTANT: state_k + 1 not state_k
        meas_k = measurements{state_k+1}.y(:, featureId);
        outOfView = isnan(meas_k(1,1));
        
        if ismember(featureId, trackedFeatureIds)
            if ~outOfView 
                %Append observation and append id to cam states
                featureTracks{trackedFeatureIds == featureId}.observations(:, end+1) = meas_k;
                
                %Add observation to current camera
                msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
            end
            
            track = featureTracks{trackedFeatureIds == featureId};
            %% TODO 满足MSCKF2007两个Update的触发条件 或者到了最后一帧
            if outOfView ...
                    || size(track.observations, 2) >= msckfParams.maxTrackLength ...
                    || state_k+1 == kEnd
                %Feature is not in view, remove from the tracked features
                [msckfState, camStates, camStateIndices] = removeTrackedFeature(msckfState, featureId);
                
                %Add the track, with all of its camStates, to the 
                %residualized list
                if length(camStates) >= msckfParams.minTrackLength
                    track.camStates = camStates;
                    track.camStateIndices = camStateIndices;
                    featureTracksToResidualize{end+1} = track;
                end
               
                %Remove the track
                featureTracks = featureTracks(trackedFeatureIds ~= featureId);
                trackedFeatureIds(trackedFeatureIds == featureId) = []; 
            end
            
        elseif ~outOfView && state_k+1 < kEnd % && ~ismember(featureId, trackedFeatureIds)
            %% TODO 新的特征
            %Track new feature
            track.featureId = featureId;
            track.observations = meas_k;
            featureTracks{end+1} = track;
            trackedFeatureIds(end+1) = featureId;

            %Add observation to current camera
            msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
        end
    end
    
    %% ==========================FEATURE RESIDUAL CORRECTIONS======================== %%
    %% TODO D. Measurement Model
    if ~isempty(featureTracksToResidualize)
        H_o = [];
        r_o = [];
        R_o = [];

        %% TODO 对所有的需要计算的tracks
        for f_i = 1:length(featureTracksToResidualize)
            track = featureTracksToResidualize{f_i};
            
            %Estimate feature 3D location through Gauss Newton inverse depth
            %optimization
            %% TODO 三角化获得当前的特征的全局坐标
            %% TODO 丢掉误差过大的点
            [p_f_G, Jcost, RCOND] = calcGNPosEst(track.camStates, track.observations, noiseParams);
        
            nObs = size(track.observations,2);
            JcostNorm = Jcost / nObs^2;
            fprintf('Jcost = %f | JcostNorm = %f | RCOND = %f\n',...
                Jcost, JcostNorm,RCOND);
            
            if JcostNorm > msckfParams.maxGNCostNorm ...
                    || RCOND < msckfParams.minRCOND
                break;
            else
                map(:,end+1) = p_f_G;
                numFeatureTracksResidualized = numFeatureTracksResidualized + 1;
                fprintf('Using new feature track with %d observations. Total track count = %d.\n',...
                    nObs, numFeatureTracksResidualized);
            end
            
            %Calculate residual and Hoj 
            %% TODO 计算残差
            [r_j] = calcResidual(p_f_G, track.camStates, track.observations);
            
            %% TODO 测量的noise协方差矩阵就是简单的堆叠起来
            R_j = diag(repmat([noiseParams.u_var_prime, noiseParams.v_var_prime], [1, numel(r_j)/2]));

            %% TODO 残差求导
            [H_o_j, A_j, H_x_j] = calcHoj(p_f_G, msckfState, track.camStateIndices);  % equ (48)

            % Stacked residuals and friends
            if msckfParams.doNullSpaceTrick
                H_o = [H_o; H_o_j];
                if ~isempty(A_j)
                    %% TODO MSCKF2007 (23)
                    r_o_j = A_j' * r_j;

                    %% TODO MSCKF2007 (25)
                    r_o = [r_o ; r_o_j];

                    R_o_j = A_j' * R_j * A_j;

                    %% TODO MSCKF2007 (25)
                    R_o(end+1 : end+size(R_o_j,1), end+1 : end+size(R_o_j,2)) = R_o_j;
                end
            else
                %% TODO MSCKF2007 (25)
                H_o = [H_o; H_x_j];
                r_o = [r_o; r_j];
                R_o(end+1 : end+size(R_j,1), end+1 : end+size(R_j,2)) = R_j;
            end
        end
        
        %% TODO E. EKF Updates
        if ~isempty(r_o)
            % Put residuals into their final update-worthy form
            
            if msckfParams.doQRdecomp
                %% TODO 对Hx矩阵做QR分解
                [T_H, Q_1] = calcTH(H_o);

                %% TODO MSCKF2007 (28)
                r_n = Q_1' * r_o;
                R_n = Q_1' * R_o * Q_1;
            else
                T_H = H_o;
                r_n = r_o;
                R_n = R_o;
            end           
            
            % Build MSCKF covariance matrix
            P = [msckfState.imuCovar, msckfState.imuCamCovar;
                   msckfState.imuCamCovar', msckfState.camCovar];

            % Calculate Kalman gain
            %% TODO MSCKF2007 (29)
            K = (P*T_H') / ( T_H*P*T_H' + R_n ); % == (P*T_H') * inv( T_H*P*T_H' + R_n )

            % State correction
            %% TODO MSCKF2007 (30)
            deltaX = K * r_n;
            msckfState = updateState(msckfState, deltaX);

            % Covariance correction
            %% TODO MSCKF2007 (31)
            tempMat = (eye(15 + 6*size(msckfState.camStates,2)) - K*T_H);
            P_corrected = tempMat * P * tempMat' + K * R_n * K';

            msckfState.imuCovar = P_corrected(1:15,1:15);
            msckfState.camCovar = P_corrected(16:end,16:end);
            msckfState.imuCamCovar = P_corrected(1:15, 16:end);
        end
    end

    %% ==========================STATE HISTORY======================== %% 
    imuStates = updateStateHistory(imuStates, msckfState, camera, state_k+1);
  
    %% ==========================STATE PRUNING======================== %%
    %Remove any camera states with no tracked features
    [msckfState, deletedCamStates] = pruneStates(msckfState);
    
    if ~isempty(deletedCamStates)
        prunedStates(end+1:end+length(deletedCamStates)) = deletedCamStates;
    end    
    plot_traj;
end

%% TODO Save
states = zeros(length(msckfState_imuOnly), 7);
for state_i = 1 : length(msckfState_imuOnly)
    states(state_i, 1:3) = msckfState_imuOnly{state_i}.imuState.p_I_G;
    states(state_i, 4:7) = msckfState_imuOnly{state_i}.imuState.q_IG;
end
save('states_matlab.txt', '-ascii', 'states')