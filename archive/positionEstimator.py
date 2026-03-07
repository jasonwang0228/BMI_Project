

def positionEstimator(test_data, modelParameters):
    """
    Called every 20ms. test_data contains:
        test_data['spikes']         → (98, t_current) — all spikes up to now
        test_data['startHandPos']   → (2,) starting position
        test_data['decodedHandPos'] → (2, N) positions decoded so far
    
    Returns: x, y — predicted hand position RIGHT NOW
    """

    # ── Unpack model ───────────────────────────────────────────────────────
    W                = modelParameters['W']
    b                = modelParameters['b']
    selected_neurons = modelParameters['selected_neurons']
    smooth_std       = modelParameters['smooth_std']
    window           = modelParameters['window']

    # ── Get current spikes ─────────────────────────────────────────────────
    spikes = test_data['spikes']   # (98, t_current)
    t_current = spikes.shape[1]

    # ── Compute firing rate via sliding window ─────────────────────────────
    # Use the last `window` ms of spikes to estimate current firing rate
    # This is the single-trial equivalent of our training signal
    # We always use a fixed-length window so the feature distribution
    # stays consistent regardless of how far into the trial we are
    
    t_start = max(0, t_current - window)
    recent_spikes = spikes[:, t_start:t_current]   # (98, window)

    # Smooth then take mean — gives firing rate in spikes/ms for each neuron
    smoothed = gaussian_filter1d(recent_spikes.astype(float), 
                                 sigma=smooth_std, axis=1)
    firing_rate = smoothed.mean(axis=1)             # (98,)

    # ── Select only tuned neurons ──────────────────────────────────────────
    rate_selected = firing_rate[selected_neurons]   # (n_selected,)

    # ── Predict position ───────────────────────────────────────────────────
    # Simple matrix multiply: [x, y] = rate_selected @ W + b
    pos = rate_selected @ W + b   # (2,)

    return pos[0], pos[1]   # x, y