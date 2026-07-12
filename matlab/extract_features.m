%% extract_features.m
% Loads fault_dataset_raw.mat, extracts features from each of the 900
% samples, and exports everything to features.csv for Python.

clear; clc;
load('fault_dataset_raw.mat');  % loads 'results'

FS = 5000;
F0 = 50;

n_samples = numel(results);
feature_rows = cell(n_samples, 1);

fprintf('Extracting features from %d samples...\n', n_samples);

for i = 1:n_samples
    r = results(i);
    V = r.V_relay;   % 1501 x 3
    I = r.I_relay;   % 1501 x 3

    [pre_idx, fault_idx] = get_fault_windows(r.fault_start, FS);
    pre_idx = pre_idx(pre_idx >= 1 & pre_idx <= size(V,1));
    fault_idx = fault_idx(fault_idx >= 1 & fault_idx <= size(V,1));

    Vf = V(fault_idx, :);
    If = I(fault_idx, :);

    feat = struct();
    feat.fault_type = r.fault_type;
    feat.phases = r.phases;
    feat.location_km = r.location_km;
    feat.r_fault = r.r_fault;
    feat.fault_start = r.fault_start;

    % --- RMS per phase (fault window) ---
    rmsV = sqrt(mean(Vf.^2, 1));
    rmsI = sqrt(mean(If.^2, 1));
    feat.Vrms_a = rmsV(1); feat.Vrms_b = rmsV(2); feat.Vrms_c = rmsV(3);
    feat.Irms_a = rmsI(1); feat.Irms_b = rmsI(2); feat.Irms_c = rmsI(3);
    feat.Vrms_unbalance = (max(rmsV)-min(rmsV)) / (mean(rmsV)+1e-6);
    feat.Irms_unbalance = (max(rmsI)-min(rmsI)) / (mean(rmsI)+1e-6);

    % --- Symmetrical components (fault window) ---
    [V0,V1,V2] = seq_components(Vf, FS, F0);
    [I0,I1,I2] = seq_components(If, FS, F0);
    feat.Vseq0 = abs(V0); feat.Vseq1 = abs(V1); feat.Vseq2 = abs(V2);
    feat.Iseq0 = abs(I0); feat.Iseq1 = abs(I1); feat.Iseq2 = abs(I2);
    feat.Vseq0_ratio = abs(V0)/(abs(V1)+1e-6);
    feat.Vseq2_ratio = abs(V2)/(abs(V1)+1e-6);
    feat.Iseq0_ratio = abs(I0)/(abs(I1)+1e-6);
    feat.Iseq2_ratio = abs(I2)/(abs(I1)+1e-6);

    % --- Harmonics (2nd, 3rd, 5th), summed across phases ---
    for h = [2 3 5]
        feat.(['Vh' num2str(h) '_energy']) = harmonic_energy(Vf, FS, F0, h);
        feat.(['Ih' num2str(h) '_energy']) = harmonic_energy(If, FS, F0, h);
    end

    % --- Wavelet energy (transient signature, phase A, fault window) ---
    [cA, cD1, cD2, cD3] = wavelet_energy(Vf(:,1));
    feat.Vdwt_L1 = cD1; feat.Vdwt_L2 = cD2; feat.Vdwt_L3 = cD3;
    [cA_i, cD1_i, cD2_i, cD3_i] = wavelet_energy(If(:,1));
    feat.Idwt_L1 = cD1_i; feat.Idwt_L2 = cD2_i; feat.Idwt_L3 = cD3_i;

    % --- Apparent impedance (V/I phasor ratio per phase, fault window) ---
    n = size(Vf,1);
    window = hanning(n);
    k = F0*n/FS;
    for ph = 1:3
        Vp = sum(Vf(:,ph) .* window .* exp(-1i*2*pi*k*(0:n-1)'/n));
        Ip = sum(If(:,ph) .* window .* exp(-1i*2*pi*k*(0:n-1)'/n));
        Z = Vp / (Ip + 1e-6);
        feat.(['Zmag_' char('a'+ph-1)]) = abs(Z);
        feat.(['Zang_' char('a'+ph-1)]) = angle(Z);
    end

    feature_rows{i} = feat;

    if mod(i,100) == 0
        fprintf('Processed %d / %d\n', i, n_samples);
    end
end

feature_table = struct2table([feature_rows{:}]);
writetable(feature_table, 'features.csv');
fprintf('Done. Saved features.csv with %d rows, %d columns.\n', ...
    height(feature_table), width(feature_table));


%% ===== Local functions (must stay at bottom of file) =====

function [pre_idx, fault_idx] = get_fault_windows(fault_start, FS)
    pre_end_t = fault_start - 0.002;
    pre_start_t = pre_end_t - 0.02;
    pre_idx = round(pre_start_t*FS)+1 : round(pre_end_t*FS)+1;

    fault_win_start_t = fault_start + 0.002;
    fault_win_end_t = fault_start + 0.022;
    fault_idx = round(fault_win_start_t*FS)+1 : round(fault_win_end_t*FS)+1;
end

function [V0,V1,V2] = seq_components(sig3, FS, F0)
    n = size(sig3,1);
    window = hanning(n);
    k = F0*n/FS;
    a = exp(1i*2*pi/3);
    A = [1 1 1; 1 a a^2; 1 a^2 a] / 3;
    phasors = zeros(3,1);
    for ph = 1:3
        phasors(ph) = sum(sig3(:,ph) .* window .* exp(-1i*2*pi*k*(0:n-1)'/n));
    end
    seq = A * phasors;
    V0 = seq(1); V1 = seq(2); V2 = seq(3);
end

function e = harmonic_energy(sig3, FS, F0, h)
    e = 0;
    n = size(sig3,1);
    freqs = (0:n-1)*(FS/n);
    for ph = 1:3
        spec = abs(fft(sig3(:,ph)));
        [~, idx] = min(abs(freqs - h*F0));
        e = e + spec(idx);
    end
end

function [eA, eD1, eD2, eD3] = wavelet_energy(sig)
    [c, l] = wavedec(sig, 3, 'db4');
    d1 = detcoef(c, l, 1);
    d2 = detcoef(c, l, 2);
    d3 = detcoef(c, l, 3);
    eD1 = sum(d1.^2);
    eD2 = sum(d2.^2);
    eD3 = sum(d3.^2);
    eA = sum(c.^2);  % overall energy, not currently used but available
end
