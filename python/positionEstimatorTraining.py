import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat

def positionEstimatorTraining(training_data):
    """
    training_data: numpy array of shape (n_trials, 8)
    Each element has:
        .spikes  → (98, T) binary array
        .handPos → (3, T) float array [x, y, z]
    
    Returns modelParameters dict containing everything
    positionEstimator needs at test time.
    """

    # ── Constants ──────────────────────────────────────────────────────────
    n_trials, n_directions = training_data.shape   # (50, 8)
    SMOOTH_STD   = 50    # Gaussian smoothing σ in ms — controls noise/temporal resolution tradeoff
    WINDOW       = 300   # sliding window size in ms used at TEST time (keep consistent)
    LAG          = 100   # ms: neural activity leads hand movement by ~100ms
    LAMBDA       = 1e4   # Ridge regularisation strength (we'll discuss tuning this)
    TUNING_THR   = 0.1   # neurons whose rate varies less than this across directions are dropped

    # ── Step 1: Smooth spikes → firing rates ───────────────────────────────
    # gaussian_filter1d applies a Gaussian kernel along axis=1 (time axis)
    # sigma=SMOOTH_STD means we're averaging over ~50ms neighbourhood
    # This converts binary {0,1} spikes into a continuous rate estimate
    # We do this for EVERY trial so we can average properly
    
    # smoothed_all[trial, direction] = (98, T) float array
    smoothed_all = np.empty_like(training_data)   # same shape, will hold arrays
    for tr in range(n_trials):
        for d in range(n_directions):
            spikes = training_data[tr, d]['spikes'][0, 0]   # (98, T)
            smoothed_all[tr, d] = gaussian_filter1d(
                spikes.astype(float), sigma=SMOOTH_STD, axis=1
            )

    # ── Step 2: Trial-average the smoothed firing rates ────────────────────
    # For each (direction, time), average across 50 trials
    # Result: mean_rates[d] = (98, T_d) — one curve per direction per neuron
    # This is the clean signal we'll use for training
    # NOTE: different trials/directions may have different lengths T_d
    
    mean_rates  = []   # list of length 8, each entry (98, T_d)
    mean_pos    = []   # list of length 8, each entry (2, T_d) — just x,y not z

    for d in range(n_directions):
        # Find minimum trial length for this direction (to align trials)
        min_T = min(smoothed_all[tr, d].shape[1] for tr in range(n_trials))
        
        # Stack and average: (n_trials, 98, min_T) → mean over axis 0 → (98, min_T)
        stacked_rates = np.stack(
            [smoothed_all[tr, d][:, :min_T] for tr in range(n_trials)], axis=0
        )
        mean_rates.append(stacked_rates.mean(axis=0))   # (98, min_T)

        # Average hand position across trials too
        stacked_pos = np.stack(
            [training_data[tr, d]['handPos'][0, 0][:2, :min_T] 
             for tr in range(n_trials)], axis=0
        )
        mean_pos.append(stacked_pos.mean(axis=0))   # (2, min_T)

    # ── Step 3: Feature selection — drop non-tuned neurons ─────────────────
    # For each neuron, compute its mean firing rate at each direction
    # (average over time, then look at variance across 8 directions)
    # A neuron that fires the same regardless of direction tells us nothing
    
    # direction_means[d] = (98,) — mean rate of each neuron for direction d
    direction_means = np.stack(
        [mean_rates[d].mean(axis=1) for d in range(n_directions)], axis=1
    )  # shape: (98, 8)

    # For each neuron: how much does its rate vary across the 8 directions?
    # High variance → direction-tuned → keep
    # Low variance  → not tuned → drop
    tuning_variance = direction_means.var(axis=1)   # (98,) one value per neuron

    # Keep neurons above threshold (normalise by max so threshold is interpretable)
    tuning_variance_norm = tuning_variance / tuning_variance.max()
    selected_neurons = np.where(tuning_variance_norm > TUNING_THR)[0]
    
    print(f"Feature selection: keeping {len(selected_neurons)}/98 neurons "
          f"(threshold={TUNING_THR})")

    # ── Step 4: Build training matrices X and Y ─────────────────────────────
    # We pair: firing rates at time t  →  hand position at time t + LAG
    # The LAG accounts for the fact that neural commands precede movement
    # 
    # We concatenate across all 8 directions
    # Each row of X = firing rate vector of selected neurons at one time point
    # Each row of Y = [x, y] hand position LAG ms later

    X_list, Y_list = [], []

    for d in range(n_directions):
        rates = mean_rates[d][selected_neurons, :]   # (n_selected, T_d)
        pos   = mean_pos[d]                          # (2, T_d)
        T_d   = rates.shape[1]

        # Valid time range: start after 320ms (test starts here), 
        # end LAG ms before the end (so t+LAG is still in bounds)
        t_start = 320
        t_end   = T_d - LAG

        if t_end <= t_start:
            continue   # trial too short, skip

        # rates at time t, paired with position at t+LAG
        X_list.append(rates[:, t_start:t_end].T)         # (T_valid, n_selected)
        Y_list.append(pos[:, t_start+LAG:t_end+LAG].T)   # (T_valid, 2)

    X = np.vstack(X_list)   # (N_total, n_selected)
    Y = np.vstack(Y_list)   # (N_total, 2)

    print(f"Training matrix: X={X.shape}, Y={Y.shape}")

    # ── Step 5: Fit Ridge Regression ───────────────────────────────────────
    # Standard linear regression: W = (X'X)⁻¹ X'Y
    # Ridge regression adds λI to make it robust:  W = (X'X + λI)⁻¹ X'Y
    # 
    # Why λI helps with our train/test mismatch:
    #   - Averaged training data is √50 times less noisy than test data
    #   - Without regularisation, W can grow large to fit clean training signal
    #   - Large W → over-reacts to noise at test time → bad predictions
    #   - λI penalises large weights → W stays small → robust to test noise
    
    n_features = X.shape[1]
    W = np.linalg.solve(
        X.T @ X + LAMBDA * np.eye(n_features),   # (n_selected, n_selected)
        X.T @ Y                                   # (n_selected, 2)
    )   # W shape: (n_selected, 2)

    # Store bias (mean subtraction) — important for well-centred predictions
    b = Y.mean(axis=0) - (X.mean(axis=0) @ W)

    # ── Pack everything positionEstimator will need ─────────────────────────
    modelParameters = {
        'W':                W,                  # (n_selected, 2) weight matrix
        'b':                b,                  # (2,) bias term
        'selected_neurons': selected_neurons,   # which neuron indices to use
        'smooth_std':       SMOOTH_STD,         # must match at test time
        'window':           WINDOW,             # sliding window for test-time smoothing
    }

    return modelParameters