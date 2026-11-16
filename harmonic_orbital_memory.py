import time
import numpy as np


def modrelu(z, b):
    r = np.abs(z)
    scale = np.maximum(0.0, r + b) / np.maximum(r, 1e-12)
    return scale * z


def make_holographic_patterns(n_nodes, n_patterns, rng):
    """Create structured coherent complex wave patterns (standing modes on a ring)."""
    k = np.arange(n_nodes)[None, :]
    freqs = rng.choice(np.arange(1, n_nodes // 2), size=(n_patterns, 2), replace=True)
    phases = rng.uniform(-np.pi, np.pi, size=(n_patterns, 2))
    amps = rng.uniform(0.6, 1.2, size=(n_patterns, 2))
    patterns = (
        amps[:, 0:1] * np.exp(1j * (2 * np.pi * freqs[:, 0:1] * k / n_nodes + phases[:, 0:1]))
        + amps[:, 1:2] * np.exp(1j * (2 * np.pi * freqs[:, 1:2] * k / n_nodes + phases[:, 1:2]))
    ) / np.sqrt(2.0)
    return patterns


def corrupt_batch(clean, mask_frac=0.5, noise_std=0.35, rng=None):
    B, N = clean.shape
    keep = (rng.random((B, N)) >= mask_frac).astype(np.float64)
    noise = noise_std * (rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))) / np.sqrt(2.0)
    return keep * (clean + noise), keep


def train_complex_graph(mode, X_tr, mask_tr, Y_tr, N, T=8, steps=400, lr=0.04, alpha=0.65, seed=195):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, (N, N))
    b = np.full(N, -0.05)
    # Per-neuron 2-harmonic Bohr-wave surface: r_j(th) = r0_j + A1_j*cos(3*th + p1_j) + A2_j*cos(6*th + p2_j)
    n1, n2 = 3, 6
    r0 = np.full((1, N), 0.55 / np.sqrt(N))
    A1 = np.full((1, N), 0.30 / np.sqrt(N))
    p1 = np.zeros((1, N))
    A2 = np.full((1, N), 0.15 / np.sqrt(N))
    p2 = np.full((1, N), 0.5)
    # Gated input skip weight (real scalar per neuron so U(1) gauge equivariance stays exact!)
    gamma = np.full((1, N), 0.35)
    B = X_tr.shape[0]

    m_th, v_th = np.zeros_like(th), np.zeros_like(th)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    m_g, v_g = np.zeros_like(gamma), np.zeros_like(gamma)
    m_s = [np.zeros((1, N)) for _ in range(5)]
    v_s = [np.zeros((1, N)) for _ in range(5)]
    b1_a, b2_a, eps = 0.9, 0.999, 1e-8

    for step_idx in range(1, steps + 1):
        e = np.exp(1j * th)
        if mode == "harmonic_orbital":
            c1, s1 = np.cos(n1 * th + p1), np.sin(n1 * th + p1)
            c2, s2 = np.cos(n2 * th + p2), np.sin(n2 * th + p2)
            r = r0 + A1 * c1 + A2 * c2
        else:
            r = np.ones((N, N)) / np.sqrt(N)
        W = r * e

        zs = [X_tr.copy()]
        pres, abs_l, act_m, scales = [], [], [], []
        for _t in range(T):
            pre = zs[-1] @ W + gamma * X_tr
            ab = np.abs(pre)
            m = (ab + b) > 0
            sc = np.where(m, (ab + b) / np.maximum(ab, 1e-12), 0.0)
            nxt = (1.0 - alpha) * zs[-1] + alpha * (sc * pre)
            zs.append(nxt)
            pres.append(pre); abs_l.append(ab); act_m.append(m); scales.append(sc)

        dz = (zs[-1] - Y_tr) / B
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        dgamma = np.zeros_like(gamma)
        for t in reversed(range(T)):
            pre, ab, m, sc = pres[t], abs_l[t], act_m[t], scales[t]
            dz_act = alpha * dz
            rd = np.real(dz_act.conj() * pre)
            dpre = np.where(m, sc * dz_act - (b / np.maximum(ab**3, 1e-12)) * rd * pre, 0.0)
            db += np.sum(np.where(m, rd / np.maximum(ab, 1e-12), 0.0), axis=0)
            dgamma += np.sum(np.real(dpre.conj() * X_tr), axis=0, keepdims=True)
            dW += zs[t].conj().T @ dpre
            dz = (1.0 - alpha) * dz + dpre @ W.conj().T

        if mode == "harmonic_orbital":
            dr = -n1 * A1 * s1 - n2 * A2 * s2
            g_th = np.real(dW.conj() * (dr + 1j * r) * e)
            proj = np.real(dW.conj() * e)
            g_s = [
                np.mean(proj, axis=0, keepdims=True),
                np.mean(proj * c1, axis=0, keepdims=True),
                np.mean(proj * (-A1 * s1), axis=0, keepdims=True),
                np.mean(proj * c2, axis=0, keepdims=True),
                np.mean(proj * (-A2 * s2), axis=0, keepdims=True),
            ]
        else:
            g_th = np.real(dW.conj() * (1j * W))

        bc1, bc2 = 1 - b1_a ** step_idx, 1 - b2_a ** step_idx
        m_th = b1_a * m_th + (1 - b1_a) * g_th
        v_th = b2_a * v_th + (1 - b2_a) * (g_th ** 2)
        th -= lr * (m_th / bc1) / (np.sqrt(v_th / bc2) + eps)

        m_b = b1_a * m_b + (1 - b1_a) * db
        v_b = b2_a * v_b + (1 - b2_a) * (db ** 2)
        b -= lr * 0.2 * (m_b / bc1) / (np.sqrt(v_b / bc2) + eps)

        m_g = b1_a * m_g + (1 - b1_a) * dgamma
        v_g = b2_a * v_g + (1 - b2_a) * (dgamma ** 2)
        gamma -= lr * 0.2 * (m_g / bc1) / (np.sqrt(v_g / bc2) + eps)

        if mode == "harmonic_orbital":
            arr = [r0, A1, p1, A2, p2]
            for k_i in range(5):
                m_s[k_i] = b1_a * m_s[k_i] + (1 - b1_a) * g_s[k_i]
                v_s[k_i] = b2_a * v_s[k_i] + (1 - b2_a) * (g_s[k_i] ** 2)
                arr[k_i] -= lr * 0.3 * (m_s[k_i] / bc1) / (np.sqrt(v_s[k_i] / bc2) + eps)
            r0, A1, p1, A2, p2 = arr

    if mode == "harmonic_orbital":
        r = r0 + A1 * np.cos(n1 * th + p1) + A2 * np.cos(n2 * th + p2)
        W = r * np.exp(1j * th)
        meta = dict(r0=float(np.mean(r0)), A1=float(np.mean(A1)), A2=float(np.mean(A2)))
    else:
        W = (np.ones((N, N)) / np.sqrt(N)) * np.exp(1j * th)
        meta = {}
    return W, b, gamma, meta


def eval_complex_graph(W, b, gamma, X_te, Y_te, T=8, alpha=0.65, keep_frac=1.0):
    if keep_frac < 1.0:
        mags = np.abs(W)
        thresh = np.quantile(mags, 1.0 - keep_frac)
        W = np.where(mags >= thresh, W, 0.0)
    z = X_te.copy()
    for _ in range(T):
        z = (1.0 - alpha) * z + alpha * modrelu(z @ W + gamma * X_te, b)
    mse_all = np.mean(np.abs(z - Y_te) ** 2)
    cos_phase = np.mean(np.real(z.conj() * Y_te) / (np.maximum(np.abs(z), 1e-12) * np.maximum(np.abs(Y_te), 1e-12)))
    return mse_all, cos_phase


def train_eval_real_baseline(X_tr, Y_tr, X_te, Y_te, X_rot, Y_rot, N, T=8, steps=400, lr=0.04, alpha=0.7, seed=195):
    """Real-valued 2N-dim graph (2N x 2N = 4N^2 params, ~4x params of harmonic_orbital) with Adam."""
    rng = np.random.default_rng(seed)
    Xr = np.concatenate([X_tr.real, X_tr.imag], axis=1)
    Yr = np.concatenate([Y_tr.real, Y_tr.imag], axis=1)
    W = rng.standard_normal((2 * N, 2 * N)) / np.sqrt(4 * N)
    b = np.zeros(2 * N)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    B = Xr.shape[0]

    for step_idx in range(1, steps + 1):
        zs = [Xr.copy()]
        pres = []
        for _t in range(T):
            pre = zs[-1] @ W + Xr
            act = np.tanh(pre + b)
            zs.append((1.0 - alpha) * zs[-1] + alpha * act)
            pres.append(act)
        dz = (zs[-1] - Yr) / B
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        for t in reversed(range(T)):
            act = pres[t]
            dpre = alpha * dz * (1.0 - act ** 2)
            db += np.sum(dpre, axis=0)
            dW += zs[t].T @ dpre
            dz = (1.0 - alpha) * dz + dpre @ W.T
        bc1, bc2 = 1 - 0.9 ** step_idx, 1 - 0.999 ** step_idx
        mW = 0.9 * mW + 0.1 * dW; vW = 0.999 * vW + 0.001 * (dW ** 2)
        mb = 0.9 * mb + 0.1 * db; vb = 0.999 * vb + 0.001 * (db ** 2)
        W -= lr * (mW / bc1) / (np.sqrt(vW / bc2) + 1e-8)
        b -= lr * (mb / bc1) / (np.sqrt(vb / bc2) + 1e-8)

    def _eval(Xin, Yin):
        z = np.concatenate([Xin.real, Xin.imag], axis=1)
        X0 = z.copy()
        for _ in range(T):
            z = (1.0 - alpha) * z + alpha * np.tanh(z @ W + X0 + b)
        pred = z[:, :N] + 1j * z[:, N:]
        mse = np.mean(np.abs(pred - Yin) ** 2)
        cos = np.mean(np.real(pred.conj() * Yin) / (np.maximum(np.abs(pred), 1e-12) * np.maximum(np.abs(Yin), 1e-12)))
        return mse, cos

    return _eval(X_te, Y_te), _eval(X_rot, Y_rot), W.size + b.size


if __name__ == "__main__":
    rng = np.random.default_rng(195)
    N = 32
    patterns = make_holographic_patterns(N, 8, rng)
    idx_tr = rng.integers(0, len(patterns), size=192)
    idx_te = rng.integers(0, len(patterns), size=64)
    Y_tr, Y_te = patterns[idx_tr], patterns[idx_te]
    X_tr, m_tr = corrupt_batch(Y_tr, mask_frac=0.50, noise_std=0.35, rng=rng)
    X_te, m_te = corrupt_batch(Y_te, mask_frac=0.50, noise_std=0.35, rng=rng)

    # Zero-shot U(1) gauge test: unseen random global phase rotation per test sample
    rot = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(len(Y_te), 1)))
    X_rot, Y_rot = X_te * rot, Y_te * rot

    base_mse = np.mean(np.abs(X_te - Y_te) ** 2)
    base_cos = np.mean(np.real(X_te.conj() * Y_te) / (np.maximum(np.abs(X_te), 1e-12) * np.maximum(np.abs(Y_te), 1e-12)))
    print("TASK: 32-node Holographic Wave Recall (50% nodes wiped + 0.35 noise)")
    print(f"corrupted_input   | MSE: {base_mse:.4f} | phase: {base_cos:.4f} | rot_MSE: {base_mse:.4f} | params: 0")

    t0 = time.time()
    (mse_r, cos_r), (rm_r, rc_r), p_r = train_eval_real_baseline(X_tr, Y_tr, X_te, Y_te, X_rot, Y_rot, N)
    print(f"real_tanh_graph   | MSE: {mse_r:.4f} | phase: {cos_r:.4f} | rot_MSE: {rm_r:.4f} | params: {p_r} | {time.time()-t0:.2f}s")

    for mode in ["u1", "harmonic_orbital"]:
        t0 = time.time()
        W, b, gamma, meta = train_complex_graph(mode, X_tr, m_tr, Y_tr, N)
        dt = time.time() - t0
        p_c = N * N + 2 * N + (5 * N if mode == "harmonic_orbital" else 0)
        mse_100, cos_100 = eval_complex_graph(W, b, gamma, X_te, Y_te, keep_frac=1.0)
        rm_100, rc_100 = eval_complex_graph(W, b, gamma, X_rot, Y_rot, keep_frac=1.0)
        print(f"{mode:17s} | MSE: {mse_100:.4f} | phase: {cos_100:.4f} | rot_MSE: {rm_100:.4f} | params: {p_c} | {dt:.2f}s")
        if mode == "harmonic_orbital":
            for kf in [0.50, 0.20]:
                mse_s, cos_s = eval_complex_graph(W, b, gamma, X_te, Y_te, keep_frac=kf)
                rm_s, _ = eval_complex_graph(W, b, gamma, X_rot, Y_rot, keep_frac=kf)
                print(f"  -> pruned {int((1-kf)*100)}%   | MSE: {mse_s:.4f} | phase: {cos_s:.4f} | rot_MSE: {rm_s:.4f} | edges: {int(N*N*kf)}")
