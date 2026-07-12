%% generate_holdout_test.m
% Generates a small, genuinely NEW batch of simulations (different random
% seed than the original 900) to test the already-trained Python model
% on data it has never seen in any form.

clear; clc;

model_name = 'ml_project';
open_system(model_name);

n_per_class = 4;   % small batch: 4 x 6 classes = 24 samples
fault_types = {'LG','LL','LLG','LLL','LLLG'};
line_length_total = 150;
max_attempts_per_sample = 5;

rng(999);   % <-- DIFFERENT seed than the original (was 42) -- ensures no overlap

results = struct('fault_type', {}, 'phases', {}, 'location_km', {}, ...
                  'r_fault', {}, 'fault_start', {}, 'V_relay', {}, 'I_relay', {});

idx = 1;
fault_block = [model_name '/Three-Phase Fault'];
phases_all = {'A','B','C'};

for ft = 1:length(fault_types)
    fault_type = fault_types{ft};
    n_success = 0;

    while n_success < n_per_class
        attempt_ok = false;
        for attempt = 1:max_attempts_per_sample
            switch fault_type
                case 'LG'
                    sel = phases_all(randperm(3,1)); gnd = 'on';
                case 'LL'
                    sel = phases_all(randperm(3,2)); gnd = 'off';
                case 'LLG'
                    sel = phases_all(randperm(3,2)); gnd = 'on';
                case 'LLL'
                    sel = phases_all; gnd = 'off';
                case 'LLLG'
                    sel = phases_all; gnd = 'on';
            end
            phase_label = strjoin(sel, '');

            faultA_str = 'off'; if any(strcmp(sel,'A')), faultA_str = 'on'; end
            faultB_str = 'off'; if any(strcmp(sel,'B')), faultB_str = 'on'; end
            faultC_str = 'off'; if any(strcmp(sel,'C')), faultC_str = 'on'; end
            set_param(fault_block, 'FaultA', faultA_str, 'FaultB', faultB_str, ...
                'FaultC', faultC_str, 'GroundFault', gnd);

            t_fault_start = 0.08 + rand()*0.02;
            t_fault_end = t_fault_start + 0.08;
            set_param(fault_block,'SwitchTimes', ...
                ['[' num2str(t_fault_start) ' ' num2str(t_fault_end) ']']);

            L1_km = rand()*(line_length_total-10) + 5;
            L2_km = line_length_total - L1_km;

            band = rand();
            if band < 0.5
                Rf = rand()*5;
            elseif band < 0.8
                Rf = 5 + rand()*25;
            else
                Rf = 30 + rand()*70;
            end
            set_param(fault_block,'FaultResistance',num2str(Rf));

            load_pct = 0.7 + rand()*0.6;
            src_pct = 0.95 + rand()*0.10;
            assignin('base','load_scale', load_pct);
            assignin('base','src_scale', src_pct);
            assignin('base','L1_km',L1_km);
            assignin('base','L2_km',L2_km);
            assignin('base','Rf',Rf);

            clear out V_relay I_relay
            try
                out = sim(model_name);
                V_relay = out.v_relay.Data;
                I_relay = out.I_relay.Data;
            catch err
                fprintf('[%s attempt %d] sim error: %s\n', fault_type, attempt, err.message);
                continue
            end

            is_valid = true;
            if any(isnan(V_relay(:))) || any(isnan(I_relay(:)))
                is_valid = false;
            elseif size(V_relay,1) < 1000
                is_valid = false;
            elseif max(abs(I_relay(:))) < 1
                is_valid = false;
            end
            if ~is_valid
                fprintf('[%s attempt %d] rejected\n', fault_type, attempt);
                continue
            end

            attempt_ok = true;
            break
        end
        if ~attempt_ok, continue; end

        results(idx).fault_type = fault_type;
        results(idx).phases = phase_label;
        results(idx).location_km = L1_km;
        results(idx).r_fault = Rf;
        results(idx).fault_start = t_fault_start;
        results(idx).V_relay = V_relay;
        results(idx).I_relay = I_relay;
        idx = idx + 1;
        n_success = n_success + 1;
    end
end

% --- NONE baseline ---
set_param(fault_block,'FaultA','on','FaultB','off','FaultC','off','GroundFault','on');
set_param(fault_block,'FaultResistance','1e9');
n_success = 0;
while n_success < n_per_class
    t_fault_start = 0.08 + rand()*0.02;
    t_fault_end = t_fault_start + 0.08;
    set_param(fault_block,'SwitchTimes', ...
        ['[' num2str(t_fault_start) ' ' num2str(t_fault_end) ']']);
    L1_km = rand()*(line_length_total-10) + 5;
    L2_km = line_length_total - L1_km;
    load_pct = 0.7 + rand()*0.6;
    src_pct = 0.95 + rand()*0.10;
    assignin('base','load_scale', load_pct);
    assignin('base','src_scale', src_pct);
    assignin('base','L1_km',L1_km);
    assignin('base','L2_km',L2_km);
    assignin('base','Rf',1e9);

    clear out V_relay I_relay
    try
        out = sim(model_name);
        V_relay = out.v_relay.Data;
        I_relay = out.I_relay.Data;
    catch
        continue
    end

    results(idx).fault_type = 'NONE';
    results(idx).phases = 'NONE';
    results(idx).location_km = 0;
    results(idx).r_fault = 0;
    results(idx).fault_start = t_fault_start;
    results(idx).V_relay = V_relay;
    results(idx).I_relay = I_relay;
    idx = idx + 1;
    n_success = n_success + 1;
end

fprintf('Generated %d holdout samples.\n', numel(results));

%% ===== Feature extraction (same logic as extract_features.m) =====

FS = 5000; F0 = 50;
n_samples = numel(results);
feature_rows = cell(n_samples, 1);

for i = 1:n_samples
    r = results(i);
    V = r.V_relay; I = r.I_relay;
    [pre_idx, fault_idx] = get_fault_windows(r.fault_start, FS);
    fault_idx = fault_idx(fault_idx >= 1 & fault_idx <= size(V,1));
    Vf = V(fault_idx, :); If = I(fault_idx, :);

    feat = struct();
    feat.fault_type = r.fault_type;
    feat.phases = r.phases;
    feat.location_km = r.location_km;
    feat.r_fault = r.r_fault;
    feat.fault_start = r.fault_start;

    rmsV = sqrt(mean(Vf.^2, 1)); rmsI = sqrt(mean(If.^2, 1));
    feat.Vrms_a = rmsV(1); feat.Vrms_b = rmsV(2); feat.Vrms_c = rmsV(3);
    feat.Irms_a = rmsI(1); feat.Irms_b = rmsI(2); feat.Irms_c = rmsI(3);
    feat.Vrms_unbalance = (max(rmsV)-min(rmsV))/(mean(rmsV)+1e-6);
    feat.Irms_unbalance = (max(rmsI)-min(rmsI))/(mean(rmsI)+1e-6);

    [V0,V1,V2] = seq_components(Vf, FS, F0);
    [I0,I1,I2] = seq_components(If, FS, F0);
    feat.Vseq0 = abs(V0); feat.Vseq1 = abs(V1); feat.Vseq2 = abs(V2);
    feat.Iseq0 = abs(I0); feat.Iseq1 = abs(I1); feat.Iseq2 = abs(I2);
    feat.Vseq0_ratio = abs(V0)/(abs(V1)+1e-6);
    feat.Vseq2_ratio = abs(V2)/(abs(V1)+1e-6);
    feat.Iseq0_ratio = abs(I0)/(abs(I1)+1e-6);
    feat.Iseq2_ratio = abs(I2)/(abs(I1)+1e-6);

    for h = [2 3 5]
        feat.(['Vh' num2str(h) '_energy']) = harmonic_energy(Vf, FS, F0, h);
        feat.(['Ih' num2str(h) '_energy']) = harmonic_energy(If, FS, F0, h);
    end

    [~, cD1, cD2, cD3] = wavelet_energy(Vf(:,1));
    feat.Vdwt_L1 = cD1; feat.Vdwt_L2 = cD2; feat.Vdwt_L3 = cD3;
    [~, cD1i, cD2i, cD3i] = wavelet_energy(If(:,1));
    feat.Idwt_L1 = cD1i; feat.Idwt_L2 = cD2i; feat.Idwt_L3 = cD3i;

    n = size(Vf,1); window = hanning(n); k = F0*n/FS;
    for ph = 1:3
        Vp = sum(Vf(:,ph).*window.*exp(-1i*2*pi*k*(0:n-1)'/n));
        Ip = sum(If(:,ph).*window.*exp(-1i*2*pi*k*(0:n-1)'/n));
        Z = Vp/(Ip+1e-6);
        feat.(['Zmag_' char('a'+ph-1)]) = abs(Z);
        feat.(['Zang_' char('a'+ph-1)]) = angle(Z);
    end

    feature_rows{i} = feat;
end

holdout_table = struct2table([feature_rows{:}]);
writetable(holdout_table, 'features_holdout.csv');
fprintf('Saved features_holdout.csv with %d rows.\n', height(holdout_table));

%% ===== Local functions =====
function [pre_idx, fault_idx] = get_fault_windows(fault_start, FS)
    pre_end_t = fault_start - 0.002; pre_start_t = pre_end_t - 0.02;
    pre_idx = round(pre_start_t*FS)+1 : round(pre_end_t*FS)+1;
    fault_win_start_t = fault_start + 0.002; fault_win_end_t = fault_start + 0.022;
    fault_idx = round(fault_win_start_t*FS)+1 : round(fault_win_end_t*FS)+1;
end

function [V0,V1,V2] = seq_components(sig3, FS, F0)
    n = size(sig3,1); window = hanning(n); k = F0*n/FS;
    a = exp(1i*2*pi/3);
    A = [1 1 1; 1 a a^2; 1 a^2 a] / 3;
    phasors = zeros(3,1);
    for ph = 1:3
        phasors(ph) = sum(sig3(:,ph).*window.*exp(-1i*2*pi*k*(0:n-1)'/n));
    end
    seq = A * phasors;
    V0 = seq(1); V1 = seq(2); V2 = seq(3);
end

function e = harmonic_energy(sig3, FS, F0, h)
    e = 0; n = size(sig3,1); freqs = (0:n-1)*(FS/n);
    for ph = 1:3
        spec = abs(fft(sig3(:,ph)));
        [~, idx] = min(abs(freqs - h*F0));
        e = e + spec(idx);
    end
end

function [eA, eD1, eD2, eD3] = wavelet_energy(sig)
    [c, l] = wavedec(sig, 3, 'db4');
    d1 = detcoef(c, l, 1); d2 = detcoef(c, l, 2); d3 = detcoef(c, l, 3);
    eD1 = sum(d1.^2); eD2 = sum(d2.^2); eD3 = sum(d3.^2);
    eA = sum(c.^2);
end
