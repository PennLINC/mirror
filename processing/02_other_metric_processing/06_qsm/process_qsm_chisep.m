%% χ-separation Tool

% This tool is MATLAB-based software forseparating para- and dia-magnetic susceptibility sources (χ-separation).
% Separating paramagnetic (e.g., iron) and diamagnetic (e.g., myelin) susceptibility sources
% co-existing in a voxel provides the distributions of two sources that QSM does not provides.

% χ-separation tool v1.2

% Contact E-mail: snu.list.software@gmail.com

% Reference
% H.-G. Shin, J. Lee, Y. H. Yun, S. H. Yoo, J. Jang, S.-H. Oh, Y. Nam, S. Jung, S. Kim, F. Masaki, W.
% Kim, H. J. Choi, J. Lee. χ-separation: Magnetic susceptibility source separation toward iron and
% myelin mapping in the brain. Neuroimage, 2021 Oct; 240:118371.

% χ-separation tool is powered by MEDI toolbox (for BET and Complex data fitting), STI Suite (for V-SHARP), SEGUE toolbox (for SEGUE), and mritools (for ROMEO).
% Code by Byeongpil Moon (mbpil7044@snu.ac.kr)

%% Necessary preparation
% Arguments:
%   example_nifti   - path to raw MEGRE echo-1 magnitude NIfTI (used for niftiinfo)
%   in_dir          - SEPIA output folder containing part-mag.nii.gz, part-phase.nii.gz, header.mat
%   out_dir         - chi-sep output folder (final NIfTIs and temp files written here)
%   is_scaling_flag - 0: use R2pnet to predict R2' from R2*; 1: use scaling factor
%   have_r2prime    - 1: use precomputed R2' map; 0: compute R2' from R2* internally
%   r2starpath      - path to precomputed R2* NIfTI (used only when have_r2prime==1)
%   r2primepath     - path to precomputed R2' NIfTI (used only when have_r2prime==1)
%   brain_mask_path - path to a brain mask NIfTI to use instead of running BET
%                     (optional; empty/omitted falls back to BET)
%   versionStr      - echo-set label ('E12345' or 'E2345') identifying the SEPIA
%                     phase output to read (sub-*_desc-<versionStr>_part-phase_rad)
function process_qsm_chisep(example_nifti,in_dir,out_dir,is_scaling_flag,have_r2prime,r2starpath,r2primepath,brain_mask_path,versionStr)
% Shut down any existing parallel pool, but only if the Parallel Computing
% Toolbox is available (gcp/parpool are not, and the pool is never created
% below, so this is just cleanup). Avoids "Undefined function 'gcp'".
if exist('gcp', 'file')
    delete(gcp('nocreate'));
end

% % Detect how many CPUs were assigned by a scheduler (e.g., SLURM, PBS)
% numCores = str2double(getenv('SLURM_CPUS_PER_TASK'));
% if isnan(numCores)
%     numCores = str2double(getenv('PBS_NUM_PPN'));
% end
% if isnan(numCores)
%     numCores = str2double(getenv('NSLOTS'));  % for SGE/LSF environments
% end

% % Default to 16 if no environment variable found or if more were assigned
% if isnan(numCores) || numCores > 16
%     numCores = 16;
% end

% % Create a local cluster object and restrict it
% cluster = parcluster('local');
% cluster.NumWorkers = min(cluster.NumWorkers, numCores);

% % Start the pool with the chosen number of workers
% parpool(cluster, cluster.NumWorkers);

% fprintf('✅ Parallel pool started with %d workers.\n', cluster.NumWorkers);
% Set x-separation toolbox root with MIRROR_QSM_SOFTWARE_ROOT.
% Toolboxes are resolved tolerantly: each may sit directly under software_root or
% one level deeper (e.g. Chisep_Toolbox_v1.2/Chisep_Toolbox_v1.2,
% STISuite_V3.0/STISuite_V3.0). The chi-sep home must be the directory that
% actually contains the trained networks (models/), since it is passed to
% chi_sepnet_general_sinc to load them. The standalone NIfTI_20140122 toolbox is
% no longer required separately: niftiread/niftiwrite are MATLAB built-ins, and
% load_untouch_nii/save_nii ship inside mritools.
software_root = get_chisep_software_root();
home_directory = resolve_toolbox_dir(software_root, 'Chisep_Toolbox_v1.2', 'models');
toolbox_dirs = { ...
    home_directory, ...
    resolve_toolbox_dir(software_root, 'STISuite_V3.0', 'Core_Functions_P'), ...
    resolve_toolbox_dir(software_root, 'MEDI', 'functions'), ...
    resolve_toolbox_dir(software_root, 'SEGUE_28012021', ''), ...
    resolve_toolbox_dir(software_root, 'mritools', 'matlab')};
for k = 1:numel(toolbox_dirs)
    addpath(genpath(toolbox_dirs{k}));
end
if exist(out_dir,'dir') ~= 7
    mkdir(out_dir);
end
info = niftiinfo(example_nifti);
% Extract subject
subMatch = regexp(in_dir, 'sub-([^/]+)', 'tokens', 'once');
subjectID = subMatch{1};   % '1234'

% Extract session
sesMatch = regexp(in_dir, 'ses-([^/]+)', 'tokens', 'once');
sessionID = sesMatch{1};   % '5678'

%% Run options - User define
RunOptions = struct();

% true: input brain mask | false: calculate brain mask
RunOptions.Mask = false;

% 'MEDI': MEDI brain extraction | 'custom': customize using FSL BET
RunOptions.Mask_method = 'MEDI';

% 'ARLO' | 'NNLS fitting' | 'Use preprocessed R2* or R2'' map'
% When have_r2prime==1 the R2* and R2' maps are loaded from disk; otherwise
% R2* is computed from the MEGRE echoes via ARLO.
if have_r2prime
    RunOptions.R2sfit = 'Use preprocessed R2* or R2'' map';
else
    RunOptions.R2sfit = 'ARLO';
end

% 'ROMEO + weighted echo averaging' | 'nonlinear complex fitting + SEGUE' | 'Laplacian'
RunOptions.Unwrap = 'ROMEO + weighted echo averaging';

% 'V-SHARP'
RunOptions.BFR = 'V-SHARP';

% 'Chi-sepnet' | 'Chi-separation (MEDI)' | 'Chi-separation (iLSQR)'
RunOptions.Chisep = 'Chi-sepnet';

% 'Deep-learning' | 'Region-growing' | 'No'
% Deep-learning requires ONNX (not available on PMACS matlab/2025a); use 'No' or 'Region-growing'.
RunOptions.VesselSeg = 'No';

% GRE smoothing: 0 ~ 0.4(Default)
RunOptions.Tukey = double(0.4);

% 0: No inverse(Default) | 1: Inverse
RunOptions.PhaseInverse = 0;

% 1: have R2' | 0: don't have R2'
RunOptions.HaveR2Prime = have_r2prime;
% r2prime - R2' map in Hz unit (x, y, z). If you don't have R2' map, use chi-sepnet-R2* which doesn't require R2' map.

% 0: generate R2' from R2* using R2pnet | 1: generate R2' from R2* using scaling
RunOptions.is_scaling = is_scaling_flag;
RunOptions.scaling_factor = 0.19;

% false: No denoising for R2s | true: denosing for R2s
RunOptions.denoising = false;

% true: use resolution generalization | false: don't use
RunOptions.resgen = false;
% Determine whether to use resolution generalization pipeline or to interpolate to 1 mm isotropic resolution
% 7T processing is available only with resolution generalization

% Temp files and final outputs both go to out_dir so they stay out of the
% shared SEPIA input folder.
RunOptions.OutputPath = out_dir;
% Output path must not contain ' '(spaces)

% Interpolation options (for B0 direction, Resampling)
% 'sinc' | 'spline'
RunOptions.interp_method = 'sinc';
RunOptions.sinc_window_size = 15;
% 'hann' | 'hamming' | 'blackman'
RunOptions.sinc_window_type = 'hann';

% Last stage Tukey
RunOptions.tukey_strength = 0.5;
RunOptions.tukey_pad = 0.1; %Recommend not to fix this

Data = struct();
Data.RunOptions = RunOptions;

if strcmp(RunOptions.Chisep, 'Chi-sepnet')
    assert_onnx_dependencies('Chi-sepnet');
end
if strcmp(RunOptions.VesselSeg, 'Deep-learning')
    assert_onnx_dependencies('Deep-learning vessel segmentation');
end

%% Data input
pathNifti_mag = sprintf('%s/sub-%s_ses-%s_part-mag_desc-concat_MEGRE.nii.gz', in_dir, subjectID, sessionID);
pathNifti_phs = sprintf('%s/sub-%s_ses-%s_desc-%s_part-phase_rad.nii.gz', in_dir, subjectID, sessionID, versionStr);
pathheader  = sprintf('%s/sub-%s_ses-%s_header.mat', in_dir, subjectID, sessionID);
% magnitude and phase
magnitudedata = niftiread(pathNifti_mag);
phasedata = niftiread(pathNifti_phs);
raw_echo_count = size(magnitudedata, 4);
if size(phasedata, 4) ~= raw_echo_count
    error('Magnitude and phase concat files have different echo counts.');
end
% SEPIA writes echo-selected concat inputs, so use every stored volume.
Data.MGRE_Mag = rot90(double(magnitudedata));
Data.MGRE_Phs = rot90(double(phasedata));
load(pathheader);
TE = double(TE(:)');
if numel(TE) ~= raw_echo_count
    error('Header TE count (%d) does not match concat echo count (%d).', numel(TE), raw_echo_count);
end
Data.VoxelSize = double(voxelSize(:)');
Data.Necho = length(TE);
Data.CF = double(CF);
Data.B0_strength = double(B0);
Data.TE = double(TE);
Data.MatrixSize = double(matrixSize);
Data.nifti_template = magnitudedata;
Data.B0dir = double(B0_dir(:)');

Data.output_root = [RunOptions.OutputPath,filesep,'chisep_output_',char(datetime('now','Format',"MM-dd-yy_HH.mm.ss"))];
mkdir(Data.output_root);

clearvars -except Params Data RunOptions home_directory example_nifti in_dir subjectID sessionID r2primepath out_dir have_r2prime is_scaling_flag r2starpath brain_mask_path versionStr

%% Fill in necessary parameters if empty
% Data.TE = [];                     % [ms]  [row vector]
% Data.B0dir = [];                  % []    [row vector]
% Data.CF = [];                     % [Hz]
% Data.B0_strength = [];            % [T]   B0_strength = CF / 42.58e6;


%% Params_check

% Force even dimension
input_field = {'MGRE_Mag','MGRE_Phs'};
for i = 1:length(input_field)
    [Data.(cell2mat(input_field(i))),x_odd,y_odd,z_odd] = even_pad(Data.(cell2mat(input_field(i))));
end
RunOptions.EvenSizePadding = [x_odd,y_odd,z_odd];
Data.MatrixSize = size(Data.MGRE_Mag);

% TE shape correction
if size(Data.TE,2) > 1
    Data.TE = Data.TE';
end


%% Tukey windowing
imgc = Data.MGRE_Mag .* exp(1i*Data.MGRE_Phs * (-1)^(RunOptions.PhaseInverse));
imgc = tukey_windowing(imgc,RunOptions.Tukey);
Data.MGRE_Mag_Tukey = abs(imgc);
Data.MGRE_Phs_Tukey = angle(imgc);

clearvars imgc


%% Brain mask (Range [0,1])
disp("=================< Brain masking >=================")
if exist('brain_mask_path','var') && ~isempty(brain_mask_path)
    % Use the SEPIA-generated BET brain mask instead of running chi-sep's BET,
    % which is unreliable on the first volume of some echo sets. Match the
    % magnitude orientation (rot90) and the working grid size (the magnitude is
    % even-padded earlier).
    bm = double(rot90(double(niftiread(brain_mask_path))) > 0);
    ref = size(Data.MGRE_Mag);
    ref = ref(1:3);
    Data.Mask = zeros(ref);
    n = min(ref, size(bm));
    Data.Mask(1:n(1), 1:n(2), 1:n(3)) = bm(1:n(1), 1:n(2), 1:n(3));
elseif RunOptions.Mask
    Data.Mask = load('mask.mat');
elseif strcmp(RunOptions.Mask_method,'MEDI')                                % Use MEDI BET
    Data.Mask = BET(Data.MGRE_Mag_Tukey(:,:,:,1), Data.MatrixSize(1:3), Data.VoxelSize);
    %         Data.Mask = double(imerode(Data.Mask, strel('sphere',2)));
    Data.Mask = double(Data.Mask);
else                                                                    % customizing using FSL
    mat2nii_ungz(Data.MGRE_Mag_Tukey,[Data.output_root,'\mag_tmp'])
    cmd = ['/home/user/fsl/bin/bet ',[Data.output_root,'\mag_tmp '],[Data.output_root,'\BET'], ' -m -R -f 0.55 -g 0.15 -S'];%-f 0.7 -g -0.08
    [status, result] = system(fsl_PathCorr(cmd));
    mask_brain = fliplr(rot90(niftiread([Data.output_root,'\BET_mask.nii.gz'])));
    Data.Mask = imerode(imdilate(mask_brain,strel('sphere',2)),strel('sphere',4));
end

clearvars mask_brain


%% R2* fitting (Range [0,100])
disp("==================< R2* fitting >==================")
if strcmp(RunOptions.R2sfit, 'Use preprocessed R2* or R2'' map')
    Data.R2s = rot90(double(niftiread(r2starpath)), 1);

    if RunOptions.HaveR2Prime == 1
        Data.R2p = rot90(double(niftiread(r2primepath)), 1);
    end
else
    % R2s fitting
    if strcmp(RunOptions.R2sfit, 'ARLO')
        Data.dTE = Data.TE(2) - Data.TE(1);
        if (~isempty(find(abs(Data.TE - Data.TE(1) - Data.dTE*(0:length(Data.TE)-1)') > 0.001, 1)) && (length(Data.TE)>2))
            disp("WARNING:ARLO doesn't work!!")
            RunOptions.R2sfit = 'NNLS fitting';
        end
    end

    if strcmp(RunOptions.R2sfit,'ARLO')
        if RunOptions.denoising
            denoised = denoise_SVD(Data.MGRE_Mag_Tukey, Data.Mask);
            Data.R2s = r2star_arlo(denoised,Data.TE*1000,Data.Mask);
        else
            Data.R2s = r2star_arlo(Data.MGRE_Mag_Tukey,Data.TE*1000,Data.Mask);
        end
    elseif strcmp(RunOptions.R2sfit,'NNLS fitting')
        if RunOptions.denoising
            denoised = denoise_SVD(Data.MGRE_Mag_Tukey, Data.Mask);
            Data.R2s = r2star_nnls(denoised,Data.TE,Data.Mask);
        else
            Data.R2s = r2star_nnls(Data.MGRE_Mag_Tukey,Data.TE, Data.Mask);
        end
    end
end
Data.R2s(Data.R2s < 0) = 0;

if RunOptions.HaveR2Prime  % Use Chi-sepnet-R2'
    Data.map = Data.R2p;
else                       % Use Chi-sepnet-R2*
    Data.map = Data.R2s;
end
Data.map(Data.map < 0) = 0;


if (isfield(Data,'MGRE_Phs_BiasCor'))
    phase = Data.MGRE_Phs_BiasCor;
elseif(isfield(Data,'MGRE_Phs_Tukey'))
    phase = Data.MGRE_Phs_Tukey;
else
    phase = Data.MGRE_Phs;
end


%% Phase Unwrapping (Range [-10,10] [rad])
% ROMEO: [unwrapped_phase[w*TE, angle]-> Echo combine -> UnwrappedPhase[w*dTE, angle]]
disp("================< Phase unwrapping >===============")
if(strcmp(RunOptions.Unwrap,'ROMEO + weighted echo averaging'))
    parameters.TE = Data.TE*1000; % on comparison with older code spandey
    parameters.mag = Data.MGRE_Mag_Tukey;
    parameters.mask = double(Data.Mask);
    parameters.calculate_B0 = false;
    parameters.phase_offset_correction = 'on';
    parameters.voxel_size = Data.VoxelSize;
    parameters.additional_flags = '-q';
    parameters.output_dir = fullfile(Data.output_root, 'romeo_tmp');
    mkdir(parameters.output_dir );

    [unwrapped_phase, B0] = ROMEO(double(phase), parameters);
    unwrapped_phase(isnan(unwrapped_phase))= 0;

    % Weighted echo averaging
    TE_s = Data.TE % /1000; % commented by shraddha pandey
    t2s_roi = 0.04;
    W = (TE_s).*exp(-(TE_s)/t2s_roi);
    weightedSum = 0;
    TE_eff = 0;
    for echo = 1:size(unwrapped_phase,4)
        weightedSum = weightedSum + W(echo)*unwrapped_phase(:,:,:,echo)./sum(W);
        TE_eff = TE_eff + W(echo)*TE_s(echo)./sum(W);
    end

    Data.UnwrappedPhase = weightedSum / TE_eff * (TE_s(2)-TE_s(1)) .* Data.Mask;

elseif(strcmp(RunOptions.Unwrap,'nonlinear complex fitting + SEGUE'))
    % Complex fitting from MEDI
    [field, error, residual_, phase0]=Fit_ppm_complex_TE(Data.MGRE_Mag_Tukey.*exp(-1i*phase), Data.TE);

    Inputs.Mask = double(Data.Mask); % 3D binary tissue mask, same size as one phase image
    Inputs.Phase = double(field); % For opposite phase
    Data.UnwrappedPhase = SEGUE(Inputs) .* Data.Mask; % Tissue phase in rad

elseif(strcmp(RunOptions.Unwrap,'Laplacian'))
    % Weighted echo combine + Laplacian
    [phase, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey,Data.MGRE_Phs_Tukey);
    pad_size=[12 12 12];
    [Data.UnwrappedPhase_, ~] = MRPhaseUnwrap(phase,'voxelsize',Data.VoxelSize,'padsize',pad_size);
    Data.UnwrappedPhase = Data.UnwrappedPhase / Data.dTE;

    % Laplacian + Echo sum
    pad_size=[12 12 12];
    [Data.UnwrappedPhase_, ~] = MRPhaseUnwrap(Data.MGRE_Phs_Tukey,'voxelsize',Data.VoxelSize,'padsize',pad_size);
    Data.UnwrappedPhase = sum(Data.UnwrappedPhase_,4) / sum(Data.TE);

    clearvars field_map pad_size
end


%% Background field removal (Range [-5,5])
% [local_field_hz [hz]]
disp("============< Background field removal >============")
if(strcmp(RunOptions.BFR,'V-SHARP'))
    [Data.local_field, Data.mask_brain_new]=V_SHARP(double(Data.UnwrappedPhase), double(Data.Mask),'voxelsize', double(Data.VoxelSize),'smvsize', double(12));
    Data.delta_TE = (Data.TE(2)-Data.TE(1))%/1000; % commented by spandey
    Data.local_field_hz = double(Data.local_field) / (2*pi*Data.delta_TE); % rad to hz
end

%% Compute CSF mask
% Needed for Chi-separation-MEDI, Chi-separation iLSQR,
% and Region-growing algorithm-based vessel segmentation.
Data.mask_CSF = compute_mask_CSF(Data, RunOptions);

%% QSM (required for MEDI / iLSQR chi-separation)
if any(strcmp(RunOptions.Chisep, {'Chi-separation (MEDI)', 'Chi-separation (iLSQR)'}))
    pad_size = [12, 12, 12];
    Data.QSM = QSM_iLSQR(Data.local_field, Data.mask_brain_new, ...
        'TE', Data.delta_TE * 1e3, ...
        'B0', Data.B0_strength, ...
        'H', Data.B0dir, ...
        'padsize', pad_size, ...
        'voxelsize', Data.VoxelSize);
end

%% Chi separation
disp("============< χ-separation processing >============")
switch RunOptions.Chisep
    case 'Chi-sepnet'
        Dr = 114; % This parameter is different from the original paper (Dr = 137) because the network is trained on COSMOS-reconstructed maps
        if RunOptions.resgen
            % Use the resolution generalization pipeline. Resolution of input data is retained in the resulting chi-separation maps
            [Data.x_para, Data.x_dia, Data.x_tot, Data.qsm_map, Data.r2p_map] = chi_sepnet_general_new_wResolGen(home_directory, Data.local_field_hz, Data.map, Data.mask_brain_new, Dr, ...
                Data.B0dir, Data.CF, Data.VoxelSize, RunOptions.HaveR2Prime, Data.B0_strength, RunOptions.is_scaling, RunOptions.scaling_factor, RunOptions.interp_method, RunOptions.sinc_window_size, RunOptions.sinc_window_type);
        else
            % Interpolate the input maps to 1 mm isotropic resolution.
            [Data.x_para, Data.x_dia, Data.x_tot, Data.qsm_map, Data.r2p_map] = chi_sepnet_general_sinc(home_directory, Data.local_field_hz, Data.map, Data.mask_brain_new, Dr, ...
                Data.B0dir, Data.CF, Data.VoxelSize, RunOptions.HaveR2Prime, Data.B0_strength, RunOptions.is_scaling, RunOptions.scaling_factor, RunOptions.interp_method, RunOptions.sinc_window_size, RunOptions.sinc_window_type);
        end
        disp("done")
    case 'Chi-separation (MEDI)'
        Data.mag = sqrt(sum(Data.MGRE_Mag_Tukey.^2,4)) .* Data.mask_brain_new;
        Data.local_field_hz = Data.local_field_hz .* Data.mask_brain_new;
        Data.r2prime = Data.map .* Data.mask_brain_new;
        [~, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey, Data.MGRE_Phs_Tukey);
        params.b0_dir = Data.B0dir;
        params.CF = Data.CF;
        params.voxel_size = Data.VoxelSize;
        params.TE = Data.TE;
        params.lambda = 1;
        params.lambda_CSF = 1;
        params.Dr = 137;
        option_data.qsm = Data.QSM;
        option_data.mask_CSF = Data.mask_CSF;
        option_data.N_std = N_std;
        option_data.wG = [];
        option_data.wG_r2p = [];
        option_data.mask_FastRelax = zeros(size(Data.r2prime));
        option_data.mask_SlowRelax = zeros(size(Data.r2prime));
        [Data.x_para, Data.x_dia, Data.x_tot] = chi_sep_MEDI(Data.mag, Data.local_field_hz, Data.r2prime, N_std, Data.mask_brain_new, params, option_data);

    case 'Chi-separation (iLSQR)'
        Data.mag = sqrt(sum(Data.MGRE_Mag_Tukey.^2,4)) .* Data.mask_brain_new;
        Data.local_field_hz = Data.local_field_hz .* Data.mask_brain_new;
        Data.r2prime = Data.map .* Data.mask_brain_new;
        [~, N_std] = Preprocessing4Phase(Data.MGRE_Mag_Tukey, Data.MGRE_Phs_Tukey);
        params.b0_dir = Data.B0dir;
        params.CF = Data.CF;
        params.voxel_size = Data.VoxelSize;
        params.Dr = 137;
        option_data.qsm = Data.QSM;
        option_data.N_std = N_std;
        [Data.x_para, Data.x_dia, Data.x_tot] = chi_sep_iLSQR(Data.mag, Data.local_field_hz, Data.r2prime, Data.mask_brain_new, params, option_data);

end

Data = sync_chisep_total_maps(Data);

if strcmp(RunOptions.interp_method, 'sinc')
    tukey_strength = RunOptions.tukey_strength;
    tukey_pad = RunOptions.tukey_pad;
    Data.x_para = real(tukey_windowing(Data.x_para,tukey_strength,round(size(Data.x_para).*tukey_pad))) .* Data.mask_brain_new;
    Data.x_dia = real(tukey_windowing(Data.x_dia,tukey_strength,round(size(Data.x_dia).*tukey_pad))) .* Data.mask_brain_new;
    Data.x_tot = real(tukey_windowing(Data.x_tot,tukey_strength,round(size(Data.x_tot).*tukey_pad))) .* Data.mask_brain_new;
    Data.qsm_map = real(tukey_windowing(Data.qsm_map,tukey_strength,round(size(Data.qsm_map).*tukey_pad))) .* Data.mask_brain_new;
    if isfield(Data, 'r2p_map')
        Data.r2p_map = real(tukey_windowing(Data.r2p_map,tukey_strength,round(size(Data.r2p_map).*tukey_pad))) .* Data.mask_brain_new;
        Data.r2p_map(Data.r2p_map < 0) = 0;
    end

    Data.x_para(Data.x_para < 0) = 0;
    Data.x_dia(Data.x_dia < 0) = 0;
end


%% Vessel Segmentation
disp("==============< Vessel segmentation >==============")
if strcmp(RunOptions.VesselSeg, 'Deep-learning') && ~onnx_import_available()
    warning('process_qsm_chisep:OnnxUnavailable', ...
        'Deep-learning vessel segmentation requires ONNX support; skipping vessel segmentation.');
    RunOptions.VesselSeg = 'No';
end
switch RunOptions.VesselSeg
    case 'Deep-learning'
        [Data.vesselMask_para, Data.vesselMask_dia] = vesselSegmentation_Chiseparation_DL(home_directory, Data.x_para, Data.x_dia, Data.mask_brain_new, Data.VoxelSize);

    case 'Region-growing'
        % Params for vessel enhancement filter (MFAT, Default)
        params.tau = 0.02; params.tau2 = 0.35; params.D = 0.3;
        params.spacing = Data.VoxelSize;
        params.scales = 4; params.sigmas = [0.25,0.5,0.75,1];
        params.whiteondark = true;

        % params for Seed Generation
        params.alpha = 2; % Threshold for large vessel seeds
        params.beta = 1; % Threshold for small vessel seeds
        params.mipSlice = round(16 / params.spacing(3) / 2) * 2;
        params.overlap = params.mipSlice / 2;

        % params for Region Growing
        params.limit = [0.5, -0.5]; %% gamma1 and gamma2
        params.Aniso_Thresh = 0.0012;
        params.similarity = 0.5; % see (Eq. 3)

        seedInput.img1 = Data.R2s; seedInput.img2 = Data.x_para .* Data.x_dia;
        baseInput.img1 = Data.x_para; baseInput.img2 = Data.x_dia;

        [paraMask_init, diaMask_init, homogeneityMeasure_p, homogeneityMeasure_d] = ...
            vesselSegmentation_Chiseparation(seedInput, baseInput, Data.mask_brain_new, min(Data.mask_brain_new, 1 - Data.mask_CSF), params);
        Data.vesselMask_para = filterVesselsByAnisotropy(paraMask_init, homogeneityMeasure_p, params.Aniso_Thresh);
        Data.vesselMask_dia  = filterVesselsByAnisotropy(diaMask_init, homogeneityMeasure_d, params.Aniso_Thresh);

        clear paraMask_init diaMask_init homogeneityMeasure_p homogeneityMeasure_d seedInput baseInput

    case 'No'
        % No vessel segmentation
end


%% Save
disp("==================< Saving data >==================")
if ~(sum(RunOptions.EvenSizePadding) == 0)
    input_field = {'x_para', 'x_dia', 'x_tot','qsm_map','Mask','R2s','R2p','UnwrappedPhase','local_field_hz','mask_brain_new','vesselMask_para','vesselMask_dia'};
    for i = 1:length(input_field)
        if isfield(Data,cell2mat(input_field(i)))
            [Data.(cell2mat(input_field(i)))] = even_unpad(Data.(cell2mat(input_field(i))),RunOptions.EvenSizePadding);
        end
    end
end

% Derive a label for output filenames reflecting which R2 map was used.
if have_r2prime
    map_label = 'r2p';
elseif ~is_scaling_flag
    map_label = 'r2primenet';
else
    map_label = 'r2s';
end

info = niftiinfo(example_nifti)
info.Datatype = 'double';
Data.x_para = rot90(Data.x_para,-1);
para_file = sprintf('%s/sub-%s_ses-%s_paramagnetic_%s.nii', out_dir, subjectID, sessionID, map_label);
niftiwrite( Data.x_para, para_file, info);
Data.x_dia = rot90(Data.x_dia,-1);
dia_file = sprintf('%s/sub-%s_ses-%s_diamagnetic_%s.nii', out_dir, subjectID, sessionID, map_label);
niftiwrite( Data.x_dia, dia_file, info);
% total_* NIfTI is the chi-separation net susceptibility map (x_tot), matching the working scripts.
Data.x_tot = rot90(Data.x_tot,-1);
total_file = sprintf('%s/sub-%s_ses-%s_total_%s.nii', out_dir, subjectID, sessionID, map_label);
niftiwrite(Data.x_tot, total_file, info);
Data.r2p_map= rot90(Data.r2p_map,-1);
r2p_file = sprintf('%s/sub-%s_ses-%s_r2prime_%s.nii', out_dir, subjectID, sessionID, map_label);
niftiwrite( Data.r2p_map, r2p_file, info);
Data.R2s = rot90(Data.R2s,-1);
r2s_file = sprintf('%s/sub-%s_ses-%s_r2s_%s.nii', out_dir, subjectID, sessionID, map_label);
niftiwrite(Data.R2s, r2s_file, info);

% Remove temp chisep output (includes romeo_tmp subdirectory).
folder_to_delete = fullfile(Data.output_root);
if exist(folder_to_delete, 'dir')
    rmdir(folder_to_delete, 's');
elseif exist(folder_to_delete, 'file')
    delete(folder_to_delete);
else
    warning('Temp output path does not exist: %s', folder_to_delete);
end
end


function software_root = get_chisep_software_root()
% Resolve the directory holding the chi-sep QSM toolboxes.
software_root = strtrim(getenv('MIRROR_QSM_SOFTWARE_ROOT'));
if ~isempty(software_root)
    return;
end
error('process_qsm_chisep:SoftwareNotFound', ...
    'Set MIRROR_QSM_SOFTWARE_ROOT to the directory containing Chisep_Toolbox_v1.2.');
end

function tb_dir = resolve_toolbox_dir(software_root, name, marker)
% Locate a toolbox under software_root, tolerating one level of nesting.
%   name   - toolbox directory name (e.g. 'STISuite_V3.0')
%   marker - sub-directory that must exist inside the resolved toolbox to
%            confirm it is the real root ('' to accept the directory as-is).
candidates = {fullfile(software_root, name), fullfile(software_root, name, name)};
for i = 1:numel(candidates)
    c = candidates{i};
    if isfolder(c) && (isempty(marker) || isfolder(fullfile(c, marker)))
        tb_dir = c;
        return;
    end
end
if isempty(marker)
    error('process_qsm_chisep:ToolboxNotFound', ...
        'Required toolbox not found: %s under %s', name, software_root);
end
error('process_qsm_chisep:ToolboxNotFound', ...
    'Required toolbox not found: %s (with %s/) under %s', name, marker, software_root);
end

function mask_CSF = compute_mask_CSF(Data, RunOptions)
% Build a CSF mask for MEDI/iLSQR and region-growing vessel segmentation.
mask_CSF = zeros(size(Data.mask_brain_new));
needs_csf = any(strcmp(RunOptions.Chisep, {'Chi-separation (MEDI)', 'Chi-separation (iLSQR)'})) ...
    || strcmp(RunOptions.VesselSeg, 'Region-growing');
if ~needs_csf
    return;
end

R2s = Data.R2s;
brain_mask = Data.Mask;
if ~isequal(size(R2s), size(brain_mask))
    error('process_qsm_chisep:R2sMaskSizeMismatch', ...
        'R2* map size %s does not match brain mask size %s.', ...
        mat2str(size(R2s)), mat2str(size(brain_mask)));
end

min_voxels = 1000;
n_brain = nnz(brain_mask > 0);
if n_brain < min_voxels
    warning('process_qsm_chisep:SmallBrainMask', ...
        'Brain mask has only %d voxels; using empty CSF mask.', n_brain);
    return;
end

% extract_CSF expects the BET mask; V-SHARP mask_brain_new can be too sparse.
try
    mask_CSF = extract_CSF(R2s, brain_mask, Data.VoxelSize);
catch ME
    warning('process_qsm_chisep:ExtractCSFFailed', ...
        'extract_CSF failed (%s). Using R2* percentile CSF mask.', ME.message);
    mask_CSF = fallback_csf_mask_from_r2star(R2s, brain_mask);
end

mask_CSF = double(mask_CSF > 0) .* double(Data.mask_brain_new > 0);
end

function mask_CSF = fallback_csf_mask_from_r2star(R2s, brain_mask)
% Approximate CSF as low-R2* voxels when toolbox extract_CSF fails.
r2_vals = R2s(brain_mask > 0);
r2_vals = r2_vals(isfinite(r2_vals) & r2_vals > 0);
mask_CSF = zeros(size(R2s));
if isempty(r2_vals)
    return;
end
threshold = prctile(r2_vals, 15);
mask_CSF = (R2s <= threshold) & (brain_mask > 0);
end

function Data = sync_chisep_total_maps(Data)
% Chi-sepnet exposes total susceptibility as qsm_map; MEDI/iLSQR use x_tot.
if ~isfield(Data, 'qsm_map')
    Data.qsm_map = Data.x_tot;
elseif ~isfield(Data, 'x_tot')
    Data.x_tot = Data.qsm_map;
end
end

function available = onnx_import_available()
available = exist('importONNXNetwork', 'file') == 2 ...
    || exist('importNetworkFromONNX', 'file') == 2;
end

function assert_onnx_dependencies(feature_name)
% Fail fast before long preprocessing if an ONNX-based step is requested.
if onnx_import_available()
    return;
end

has_dl_toolbox = license('test', 'Deep_Learning_Toolbox');
msg = sprintf('%s requires ONNX model import (importONNXNetwork or importNetworkFromONNX).', feature_name);

if ~has_dl_toolbox
    license_msg = 'Deep Learning Toolbox is not licensed on this MATLAB (license(''test'',''Deep_Learning_Toolbox'') is false).';
else
    license_msg = ['Deep Learning Toolbox is licensed, but the ONNX converter add-on is missing. ', ...
        'Install "Deep Learning Toolbox Converter for ONNX Model Format" ', ...
        '(File Exchange 67296) or ask your HPC admins to add it to the shared MATLAB install.'];
end

error('process_qsm_chisep:MissingOnnxImport', ...
    '%s\n%s\nMATLAB %s\nFor chi-separation without ONNX, set RunOptions.Chisep to ''Chi-separation (MEDI)'' and RunOptions.VesselSeg to ''No'' or ''Region-growing''.', ...
    msg, license_msg, version);
end
