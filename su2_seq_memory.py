import time
import numpy as np


def make_order_dataset(n_samples=600, seq_len=16, seed=195):
    """
    Order-sensitive task (NO positional encodings):
      Each sequence has length L=16 over 4 tokens {0:A, 1:B, 2:C, 3:D}.
      Both A and B appear once; rest are random C/D distractors.
      Label = 1 if A appears before B, else 0.
    Commutative models without positional encodings are mathematically blind (50% ceiling).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, seq_len, 4), dtype=np.float64)
    Y = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        toks = rng.integers(2, 4, size=seq_len)  # C or D distractors
        p1, p2 = rng.choice(seq_len, size=2, replace=False)
        toks[p1] = 0  # A
        toks[p2] = 1  # B
        Y[i] = 1.0 if p1 < p2 else 0.0
        X[i, np.arange(seq_len), toks] = 1.0
    return X, Y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def qmul(a, b):
    # Quaternion multiply: (..., 4) * (..., 4) -> (..., 4)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def train_u1_commutative(X_tr, Y_tr, X_te, Y_te, H=8, steps=300, lr=0.05, seed=195):
    """Commutative U(1) phase-chain: z_t = exp(i * th(x_t)) * z_{t-1}."""
    rng = np.random.default_rng(seed)
    B, L, V = X_tr.shape
    W_th = rng.normal(0, 1.0, (V, H))
    W_out = rng.normal(0, 0.5, (2 * H,))
    b_out = 0.0
    params = [W_th, W_out]
    mp = [np.zeros_like(p) for p in params]
    vp = [np.zeros_like(p) for p in params]
    mb, vb = 0.0, 0.0

    def fwd(X):
        # Total phase is just sum over t (commutative!)
        th = np.einsum("blv,vh->blh", X, W_th)
        tot_th = np.sum(th, axis=1)  # (B, H)
        feat = np.concatenate([np.cos(tot_th), np.sin(tot_th)], axis=1)
        return feat @ W_out + b_out, tot_th, feat

    for step in range(1, steps + 1):
        logits, tot_th, feat = fwd(X_tr)
        dl = (sigmoid(logits) - Y_tr) / B
        gb = np.sum(dl)
        gW_out = feat.T @ dl
        dfeat = np.outer(dl, W_out)
        dtot = -dfeat[:, :H] * np.sin(tot_th) + dfeat[:, H:] * np.cos(tot_th)
        sum_x = np.sum(X_tr, axis=1)  # (B, V)
        gW_th = sum_x.T @ dtot
        bc1, bc2 = 1 - 0.9 ** step, 1 - 0.999 ** step
        for i, g in enumerate([gW_th, gW_out]):
            mp[i] = 0.9 * mp[i] + 0.1 * g; vp[i] = 0.999 * vp[i] + 0.001 * (g * g)
            params[i] -= lr * (mp[i] / bc1) / (np.sqrt(vp[i] / bc2) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * (gb * gb)
        b_out -= lr * (mb / bc1) / (np.sqrt(vb / bc2) + 1e-8)

    te_log, _, _ = fwd(X_te)
    return np.mean((sigmoid(te_log) >= 0.5) == Y_te), W_th.size + W_out.size + 1


def train_su2_quaternion_scan(X_tr, Y_tr, X_te, Y_te, H=4, steps=400, lr=0.05, seed=195):
    """
    Non-commutative SU(2) quaternion sequence memory (ZERO positional encodings!):
      Token v maps to Cayley unit quaternion u_t = (1, -hx, -hy, -hz) / sqrt(1 + ||h||^2)
      State update: q_t = u_t * q_{t-1}   (associative, |q_t| = 1 always, non-commutative!)
    """
    rng = np.random.default_rng(seed)
    B, L, V = X_tr.shape
    # Lookup table of 3D spinor vectors per token & head: (V, H, 3)
    W_h = rng.normal(0, 0.1, (V, H, 3))
    W_out = rng.normal(0, 0.5, (4 * H,))
    b_out = 0.0
    params = [W_h, W_out]
    mp = [np.zeros_like(p) for p in params]
    vp = [np.zeros_like(p) for p in params]
    mb, vb = 0.0, 0.0

    def fwd_seq(X):
        # h_seq: (B, L, H, 3)
        h_seq = np.einsum("blv,vhk->blhk", X, W_h)
        s = 1.0 + np.sum(h_seq ** 2, axis=-1, keepdims=True)
        inv_sqrt_s = 1.0 / np.sqrt(s)
        # Cayley unit quaternion u = (1, -hx, -hy, -hz) / sqrt(1 + ||h||^2)
        u_seq = np.concatenate([np.ones_like(s), -h_seq], axis=-1) * inv_sqrt_s  # (B, L, H, 4)
        qs = [np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (X.shape[0], H, 4)).copy()]
        for t in range(L):
            qs.append(qmul(u_seq[:, t], qs[-1]))
        q_final = qs[-1].reshape(X.shape[0], 4 * H)
        return q_final @ W_out + b_out, (X, h_seq, s, inv_sqrt_s, u_seq, qs, q_final)

    t0 = time.time()
    for step in range(1, steps + 1):
        logits, (X, h_seq, s, inv_sqrt_s, u_seq, qs, q_final) = fwd_seq(X_tr)
        dl = (sigmoid(logits) - Y_tr) / B
        gb = np.sum(dl)
        gW_out = q_final.T @ dl
        dq = np.outer(dl, W_out).reshape(B, H, 4)

        du_seq = np.zeros_like(u_seq)
        for t in reversed(range(L)):
            # q_{t+1} = u_t * q_t => dL/du_t = dL/dq_{t+1} * conj(q_t), dL/dq_t = conj(u_t) * dL/dq_{t+1}
            qt_conj = qs[t].copy(); qt_conj[..., 1:] *= -1
            ut_conj = u_seq[:, t].copy(); ut_conj[..., 1:] *= -1
            du_seq[:, t] = qmul(dq, qt_conj)
            dq = qmul(ut_conj, dq)

        # Backprop through u = (1, -h) / sqrt(s), s = 1 + ||h||^2
        # du_0 = -du_w / sqrt(s) ... wait:
        # dL/dh = (-du_xyz - (du . u) * h) / sqrt(s)
        du_dot_u = np.sum(du_seq * u_seq, axis=-1, keepdims=True)
        dh_seq = (-du_seq[..., 1:] - du_dot_u * h_seq * inv_sqrt_s) * inv_sqrt_s  # wait: check exact coeff!
        # Let's check: u = v * s^(-1/2), v=(1,-h). dv/dh = -I_3 on xyz.
        # d(s^-0.5)/dh = -0.5 * s^-1.5 * 2h = -h * s^-1.5 = -(h/s) * inv_sqrt_s.
        # So dL/dh = -du_xyz * inv_sqrt_s - (du . v) * (h / s) * inv_sqrt_s
        # Since (du . v) * inv_sqrt_s = (du . u), we get: (-du_xyz - (du . u) * h) * inv_sqrt_s? Wait:
        # (du . v) * (-h * s^-1.5) = -(du . u) * h * s^-1 = -(du . u) * h / s!
        dh_seq = -du_seq[..., 1:] * inv_sqrt_s - du_dot_u * h_seq / s
        gW_h = np.einsum("blv,blhk->vhk", X, dh_seq)

        bc1, bc2 = 1 - 0.9 ** step, 1 - 0.999 ** step
        for i, g in enumerate([gW_h, gW_out]):
            mp[i] = 0.9 * mp[i] + 0.1 * g; vp[i] = 0.999 * vp[i] + 0.001 * (g * g)
            params[i] -= lr * (mp[i] / bc1) / (np.sqrt(vp[i] / bc2) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * (gb * gb)
        b_out -= lr * (mb / bc1) / (np.sqrt(vb / bc2) + 1e-8)

    dt = time.time() - t0
    te_log, (_, _, _, _, _, qs_te, _) = fwd_seq(X_te)
    acc = np.mean((sigmoid(te_log) >= 0.5) == Y_te)
    norm_err = np.max(np.abs(np.linalg.norm(qs_te[-1], axis=-1) - 1.0))
    return acc, W_h.size + W_out.size + 1, dt, norm_err


if __name__ == "__main__":
    X_tr, Y_tr = make_order_dataset(600, seq_len=16, seed=195)
    X_te, Y_te = make_order_dataset(200, seq_len=16, seed=999)

    acc_u1, p_u1 = train_u1_commutative(X_tr, Y_tr, X_te, Y_te)
    acc_su2, p_su2, dt_su2, nerr = train_su2_quaternion_scan(X_tr, Y_tr, X_te, Y_te)

    print("TASK: Order-Sensitive Token Check (pos(A) < pos(B), L=16, ZERO positional encodings)")
    print(f"u1_commutative   | test_acc: {acc_u1*100:5.1f}% | params: {p_u1}")
    print(f"su2_quat_memory  | test_acc: {acc_su2*100:5.1f}% | params: {p_su2} | time: {dt_su2:.2f}s | max(|q|-1): {nerr:.2e}")
