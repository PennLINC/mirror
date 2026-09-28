% Add general Path
% SEPIA install location. Override with the SEPIA_HOME environment variable;
% The caller or MIRROR profile must provide this path.
sepia_home = getenv('SEPIA_HOME');
if isempty(sepia_home)
    error('process_qsm_sepia:MissingSEPIAHome', 'Set SEPIA_HOME in the MIRROR profile or environment.');
end
addpath(genpath(sepia_home));
sepia_addpath;

% Add every toolbox in the qsm-software folder to the path. SEPIA's selected
% algorithms pull in dependencies it does not bundle (e.g. MEDI for the BET
% brain extraction, MRI_susceptibility_calculation_12072021 for the dirTik
% solver), so add the whole tree rather than enumerating them. Resolve the
% folder from MIRROR_QSM_SOFTWARE_ROOT, matching the chi-separation script.
software_root = getenv('MIRROR_QSM_SOFTWARE_ROOT');
if isempty(software_root)
    error('process_qsm_sepia:MissingSoftwareRoot', 'Set MIRROR_QSM_SOFTWARE_ROOT.');
end
addpath(genpath(software_root));

% Define paths
output_dir = '{{ output_dir }}';
% Concatenated phase image
input(1).name = '{{ phase_file }}';

% Concatenated magnitude image
input(2).name = '{{ mag_file }}';
input(3).name = [];  % Leave empty if not needed
input(4).name = '{{ header_file }}';

% Algorithm parameters (same as before)
algorParam = struct();
% Let SEPIA run its BET implementation so the generated mask can be reused by
% chi-separation. process_qsm.py picks up the resulting *_mask_brain.nii.gz and
% passes it to process_qsm_chisep.m, avoiding chi-sep's failing BET call.
algorParam.general = struct( ...
    'isBET', 1, ...
    'fractional_threshold', 0.15, ...
    'gradient_threshold', 0, ...
    'isInvert', 0, ...
    'isRefineBrainMask', 1);
% Define the 'unwrap' sub-structure
algorParam.unwrap = struct( ...
    'echoCombMethod', 'Optimum weights', ...
    'unwrapMethod', 'ROMEO', ...
    'isEddyCorrect', 0, ...
    'isSaveUnwrappedEcho', 1);
% Define the 'bfr' sub-structure
% VSHARP masks out voxels where none of the supplied SMV kernels fit inside the
% brain mask. Include 2 and 1 in the radius ladder so the local field, and
% therefore the Chimap, retains more cortical edge voxels.
algorParam.bfr = struct( ...
    'refine_method', 'None', ...
    'refine_order', 4, ...
    'erode_radius', 0, ...
    'erode_before_radius', 0, ...
    'method', 'VSHARP', ...
    'radius', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
% Define the 'qsm' sub-structure
algorParam.qsm = struct( ...
    'reference_tissue', 'CSF', ...
    'method', 'MRI Suscep. Calc.', ...
    'solver', 'Direct Tikhonov', ...
    'lambda', 0.05);

if ~exist(output_dir, 'dir')
    mkdir(output_dir); % This creates all required parent directories
end

% Run SEPIA process with no supplied mask so SEPIA computes and writes its BET
% brain mask.
sepiaIO(input, output_dir, [], algorParam);
