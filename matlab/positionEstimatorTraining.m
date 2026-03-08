function modelParameters = positionEstimatorTraining(training_data)
% Ridge regression BMI decoder with dual-LDA direction classification.
%
% STEP 1 : Feature selection  — keep neurons whose firing rate varies across
%           directions (normalised variance > TUNING_THR).
% STEP 2a: Planning LDA       — classify direction from ms 151-320
%           (pre-movement planning window; baseline removed).
% STEP 2b: Execution LDA      — classify from ms 301-470 (early movement).
%           At test time, Bayesian-combine both to resolve ambiguities.
% STEP 3 : Per-direction Ridge — fit  W = (X'X + λI)\X'Y  on standardised
%           firing-rate features. λ=500 chosen to prevent overfitting
%           (consecutive 20ms steps share most of their data → effective
%           n_independent ≈ 80 trials, not ~1200 rows).

SMOOTH_STD = 50;    % Gaussian σ (ms)
WINDOW     = 300;   % Sliding feature window (ms)
LAMBDA     = 500;   % Ridge regularisation (on standardised features)
TUNING_THR = 0.075;   % Neuron-selection threshold (normalised variance)

[n_trials, n_directions] = size(training_data);

%% STEP 1: Feature selection ------------------------------------------------
fprintf('Step 1: Feature selection...\n');
direction_means = zeros(98, n_directions);
for d = 1:n_directions
    rates_d = zeros(n_trials, 98);
    for tr = 1:n_trials
        spikes        = double(training_data(tr, d).spikes);   % (98, T)
        sm            = gaussSmooth1D(spikes, SMOOTH_STD);
        rates_d(tr,:) = mean(sm, 2)';
    end
    direction_means(:,d) = mean(rates_d, 1)';
end

tuning_var       = var(direction_means, 0, 2);
tuning_var_norm  = tuning_var / max(tuning_var);
selected_neurons = find(tuning_var_norm > TUNING_THR);
n_sel            = length(selected_neurons);
fprintf('  Kept %d / 98 neurons (threshold=%.2f)\n', n_sel, TUNING_THR);

%% STEP 2a: Planning LDA (ms 151-320) --------------------------------------
% One row per (trial, direction): mean smoothed rate over [151, 320] ms.
% 0-150ms is near-baseline. Tuning peaks at ~300ms; >320ms is confounded
% with hand position.
fprintf('Step 2a: Training planning LDA (ms 151-320)...\n');
X_cls = zeros(n_trials * n_directions, n_sel);
y_cls = zeros(n_trials * n_directions, 1);
row = 1;
for d = 1:n_directions
    for tr = 1:n_trials
        spikes = double(training_data(tr, d).spikes);
        sm     = gaussSmooth1D(spikes(selected_neurons, 151:320), SMOOTH_STD);
        X_cls(row,:) = mean(sm, 2)';
        y_cls(row)   = d;
        row = row + 1;
    end
end
lda_plan      = fitLDA(X_cls, y_cls);
pred_plan     = predictLDA(lda_plan, X_cls);
fprintf('  Planning LDA train accuracy: %.1f%%\n', mean(pred_plan==y_cls)*100);

%% STEP 2b: Execution LDA (ms 301-470) -------------------------------------
% Re-classifies direction mid-trial using early movement activity.
% Bayesian combination P_combined ∝ P_plan × P_exec can rescue cases where
% the planning signal alone is ambiguous (e.g. 70° vs 110°).
fprintf('Step 2b: Training execution LDA (ms 301-470)...\n');
X_exec = [];
y_exec = [];
for d = 1:n_directions
    for tr = 1:n_trials
        spikes = double(training_data(tr, d).spikes);
        if size(spikes, 2) < 470, continue; end
        sm      = gaussSmooth1D(spikes(selected_neurons, 301:470), SMOOTH_STD);
        X_exec  = [X_exec;  mean(sm, 2)'];   %#ok<AGROW>
        y_exec  = [y_exec;  d];               %#ok<AGROW>
    end
end
lda_exec  = fitLDA(X_exec, y_exec);
pred_exec = predictLDA(lda_exec, X_exec);
fprintf('  Execution LDA train accuracy: %.1f%%\n', mean(pred_exec==y_exec)*100);

%% STEP 3: Per-direction Ridge regression ----------------------------------
fprintf('Step 3: Building per-direction Ridge regression models...\n');
W_list     = cell(n_directions, 1);
b_list     = cell(n_directions, 1);
Xmean_list = cell(n_directions, 1);
Xstd_list  = cell(n_directions, 1);

for d = 1:n_directions
    X_rows = [];
    Y_rows = [];
    for tr = 1:n_trials
        spikes = double(training_data(tr, d).spikes);    % (98, T)
        pos    = training_data(tr, d).handPos(1:2, :);   % (2,  T)
        T      = size(spikes, 2);
        sm     = gaussSmooth1D(spikes, SMOOTH_STD);

        for t = 320:20:(T-1)
            t0   = max(1, t - WINDOW + 1);               % 300ms window
            feat = mean(sm(selected_neurons, t0:t), 2)'; % (1, n_sel)
            X_rows = [X_rows; feat];          %#ok<AGROW>
            Y_rows = [Y_rows; pos(:,t)'];     %#ok<AGROW>
        end
    end

    % Standardise features (firing rates are tiny; without this λI dominates)
    X_mean = mean(X_rows, 1);
    X_std  = std(X_rows,  0, 1) + 1e-8;
    X_norm = (X_rows - X_mean) ./ X_std;

    % Ridge: W = (X_norm'X_norm + λI) \ X_norm'Y
    n_feat = size(X_norm, 2);
    W = (X_norm'*X_norm + LAMBDA*eye(n_feat)) \ (X_norm'*Y_rows);
    b = mean(Y_rows,1) - mean(X_norm,1)*W;

    W_list{d}     = W;
    b_list{d}     = b;
    Xmean_list{d} = X_mean;
    Xstd_list{d}  = X_std;
    fprintf('  Direction %d: %d training rows, W size %dx%d\n', ...
            d, size(X_rows,1), size(W,1), size(W,2));
end

fprintf('Training complete!\n');

% Pack all learned parameters into output struct
modelParameters.W_list           = W_list;
modelParameters.b_list           = b_list;
modelParameters.Xmean_list       = Xmean_list;
modelParameters.Xstd_list        = Xstd_list;
modelParameters.selected_neurons = selected_neurons;
modelParameters.lda_plan         = lda_plan;
modelParameters.lda_exec         = lda_exec;
modelParameters.smooth_std       = SMOOTH_STD;
modelParameters.window           = WINDOW;
end


% ── Local helper: Gaussian smoothing per neuron ----------------------------
function smoothed = gaussSmooth1D(spikes, sigma)
% Smooth each row of (n_neurons × T) matrix with a Gaussian kernel (std=sigma).
    half_win          = ceil(4 * sigma);
    x                 = -half_win : half_win;
    kernel            = exp(-0.5 * (x / sigma).^2);
    kernel            = kernel / sum(kernel);
    [n_neurons, T]    = size(spikes);
    smoothed          = zeros(n_neurons, T);
    for i = 1:n_neurons
        smoothed(i,:) = conv(double(spikes(i,:)), kernel, 'same');
    end
end

% ── Local helper: train Linear Discriminant Analysis ----------------------
function lda = fitLDA(X, y)
% Fit a pooled-covariance LDA classifier.
% X : (n_samples, n_features)   y : (n_samples, 1) integer class labels
    classes   = unique(y(:));
    K         = length(classes);
    [n, p]    = size(X);
    mu        = zeros(K, p);
    prior     = zeros(K, 1);
    SW        = zeros(p, p);     % within-class scatter
    for i = 1:K
        idx       = (y == classes(i));
        n_k       = sum(idx);
        mu(i,:)   = mean(X(idx,:), 1);
        prior(i)  = n_k / n;
        Xc        = X(idx,:) - mu(i,:);
        SW        = SW + Xc' * Xc;
    end
    % Pooled covariance + small diagonal regularisation to avoid singularity
    Sigma          = SW / (n - K) + 1e-6 * eye(p);
    lda.classes    = classes;
    lda.mu         = mu;         % (K, p)
    lda.prior      = prior;      % (K, 1)
    lda.SigmaInv   = inv(Sigma); % (p, p)
end

% ── Local helper: predict with LDA ----------------------------------------
function [labels, scores] = predictLDA(lda, X)
% Classify rows of X using a fitted LDA struct.
% scores : (n_samples, K) posterior probabilities (columns = classes order)
    K          = length(lda.classes);
    n          = size(X, 1);
    log_delta  = zeros(n, K);
    for i = 1:K
        mu_k          = lda.mu(i,:);
        v             = lda.SigmaInv * mu_k';            % (p,1)
        log_delta(:,i) = X * v - 0.5*(mu_k*v) + log(lda.prior(i));
    end
    % Softmax via log-sum-exp for numerical stability
    ld_max   = max(log_delta, [], 2);
    exp_ld   = exp(log_delta - ld_max);
    scores   = exp_ld ./ sum(exp_ld, 2);
    [~, idx] = max(log_delta, [], 2);
    labels   = lda.classes(idx);
end
