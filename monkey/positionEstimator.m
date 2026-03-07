function [x, y] = positionEstimator(test_data, modelParameters)
% Ridge + dual-LDA BMI decoder (online / iterative).
%
% Called at t = 320, 340, 360, … ms (20 ms steps).
%
% Direction is classified with two LDA models:
%   lda_plan  – trained on ms 151-320  (planning; always active)
%   lda_exec  – trained on ms 301-470  (execution; Bayesian-fused at t≥470)
%
% Position is estimated by a probability-weighted blend of per-direction
% Ridge regressors rather than hard argmax, so classification uncertainty
% (e.g. 70° vs 110° confusion) degrades performance gracefully.

SMOOTH_STD = 50;
WINDOW     = 300;

%% Unpack model ---------------------------------------------------------------
W_list           = modelParameters.W_list;
b_list           = modelParameters.b_list;
Xmean_list       = modelParameters.Xmean_list;
Xstd_list        = modelParameters.Xstd_list;
selected_neurons = modelParameters.selected_neurons;
lda_plan         = modelParameters.lda_plan;
lda_exec         = modelParameters.lda_exec;
n_directions     = length(W_list);

spikes    = double(test_data.spikes);   % (98, t_current)
t_current = size(spikes, 2);

%% Planning LDA: classify from ms 151-320 ------------------------------------
sm_pre   = gaussSmooth1D(spikes(selected_neurons, 131:320), SMOOTH_STD);
rate_pre = mean(sm_pre, 2)';                        % (1, n_sel)
[~, plan_scores] = predictLDA(lda_plan, rate_pre);  % (1, n_directions)
probs = plan_scores(1, :);                          % 1 × n_directions

%% Execution LDA: Bayesian fusion at t >= 470 --------------------------------
if t_current >= 470
    sm_exec   = gaussSmooth1D(spikes(selected_neurons, 301:470), SMOOTH_STD);
    rate_exec = mean(sm_exec, 2)';
    [~, exec_scores] = predictLDA(lda_exec, rate_exec);
    combined = probs .* exec_scores(1, :);
    probs    = combined / (sum(combined) + 1e-10);
end

%% Sliding-window firing-rate feature ----------------------------------------
t_start  = max(1, t_current - WINDOW + 1);
sm_sel   = gaussSmooth1D(spikes(selected_neurons, t_start:t_current), SMOOTH_STD);
rate_sel = mean(sm_sel, 2)';                        % (1, n_sel)

%% Probability-weighted position estimate ------------------------------------
pos = zeros(1, 2);
for d = 1:n_directions
    p = probs(d);
    if p < 1e-3, continue; end
    rate_norm_d = (rate_sel - Xmean_list{d}) ./ Xstd_list{d};
    pos = pos + p * (rate_norm_d * W_list{d} + b_list{d});
end

x = pos(1);
y = pos(2);
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

% ── Local helper: predict with LDA ----------------------------------------
function [labels, scores] = predictLDA(lda, X)
% Classify rows of X using a fitted LDA struct from fitLDA.
% scores : (n_samples, K) posterior probabilities
    K          = length(lda.classes);
    n          = size(X, 1);
    log_delta  = zeros(n, K);
    for i = 1:K
        mu_k           = lda.mu(i,:);
        v              = lda.SigmaInv * mu_k';
        log_delta(:,i) = X * v - 0.5*(mu_k*v) + log(lda.prior(i));
    end
    ld_max   = max(log_delta, [], 2);
    exp_ld   = exp(log_delta - ld_max);
    scores   = exp_ld ./ sum(exp_ld, 2);
    [~, idx] = max(log_delta, [], 2);
    labels   = lda.classes(idx);
end
