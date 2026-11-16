import time
import numpy as np


def make_spirals(n_per_class=200, turns=2.5, noise=0.12, seed=195):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.15, 1.0, n_per_class)
    ang = t * turns * 2 * np.pi
    x0 = np.stack([t * np.cos(ang), t * np.sin(ang)], axis=1) + rng.normal(0, noise, (n_per_class, 2))
    x1 = np.stack([t * np.cos(ang + np.pi), t * np.sin(ang + np.pi)], axis=1) + rng.normal(0, noise, (n_per_class, 2))
    X = np.vstack([x0, x1])
    Y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    perm = rng.permutation(len(Y))
    return X[perm], Y[perm]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_su2_net(X_tr, Y_tr, X_te, Y_te, H_neurons=16, steps=600, lr=0.04, seed=195):
    """
    1-layer SU(2) Pauli Spinor Network (ZERO scalar activations like ReLU/tanh):
      For each neuron j in 1..H:
        h_j(x) = h0_j + x @ V_j   in R^3
        |psi_j> = exp(-i h_j . sigma) |0>  (exact closed-form SU(2))
        expectation vector e_j = <psi_j | sigma | psi_j> in S^2 (R^3, unit norm!)
      Readout: logit = sum_j (e_j . o_j) + c
    """
    rng = np.random.default_rng(seed)
    B, D = X_tr.shape
    h0 = rng.normal(0, 0.8, (H_neurons, 3))
    V = rng.normal(0, 1.2, (D, H_neurons, 3))
    o = rng.normal(0, 0.5, (H_neurons, 3))
    c = 0.0

    # Adam state
    params = [h0, V, o]
    m_p = [np.zeros_like(p) for p in params]
    v_p = [np.zeros_like(p) for p in params]
    mc, vc = 0.0, 0.0

    def forward_and_cache(X):
        # H: (B, H_neurons, 3)
        H = h0[None, :, :] + np.einsum("bd,dhk->bhk", X, V)
        r = np.sqrt(np.sum(H ** 2, axis=-1, keepdims=True) + 1e-12)  # (B, H, 1)
        u = H / r  # unit axis (B, H, 3)
        ca = np.cos(r)  # (B, H, 1)
        sa = np.sin(r)  # (B, H, 1)
        # Exact SU(2) rotation of |0> -> Bloch vector e = <psi|sigma|psi> in R^3!
        # Rotating z-hat = (0,0,1) around axis u by angle 2*r:
        # e = (2*sa*ca*(u_y) + 2*sa^2*(u_x*u_z),
        #     -2*sa*ca*(u_x) + 2*sa^2*(u_y*u_z),
        #      ca^2 + sa^2*(u_z^2 - u_x^2 - u_y^2))
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        c_ = ca[..., 0]
        s_ = sa[..., 0]
        s2 = 2.0 * s_ * c_
        ss = 2.0 * s_ * s_
        ex = s2 * uy + ss * ux * uz
        ey = -s2 * ux + ss * uy * uz
        ez = c_ * c_ + s_ * s_ * (uz * uz - ux * ux - uy * uy)
        E = np.stack([ex, ey, ez], axis=-1)  # (B, H, 3)
        logits = np.sum(E * o[None, :, :], axis=(1, 2)) + c
        return logits, (X, H, r, u, c_, s_, E)

    t0 = time.time()
    for step in range(1, steps + 1):
        logits, (X, H, r, u, c_, s_, E) = forward_and_cache(X_tr)
        probs = sigmoid(logits)
        d_logit = (probs - Y_tr) / B  # (B,)

        # Gradients wrt o, c, E
        gc = np.sum(d_logit)
        go = np.einsum("b,bhk->hk", d_logit, E)
        dE = d_logit[:, None, None] * o[None, :, :]  # (B, H, 3)

        # Backprop through Bloch rotation E(r, u)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        dx, dy, dz = dE[..., 0], dE[..., 1], dE[..., 2]
        # dL/dr (via c_ = cos(r), s_ = sin(r)):
        # d(s2)/dr = 2*(c_^2 - s_^2), d(ss)/dr = 4*s_*c_, d(c_^2)/dr = -2*s_*c_, d(s_^2)/dr = 2*s_*c_
        ds2 = 2.0 * (c_ * c_ - s_ * s_)
        dss = 4.0 * s_ * c_
        dsc = 2.0 * s_ * c_
        dr = (
            dx * (ds2 * uy + dss * ux * uz)
            + dy * (-ds2 * ux + dss * uy * uz)
            + dz * (-dsc + dsc * (uz * uz - ux * ux - uy * uy))
        )
        # dL/du:
        s2 = 2.0 * s_ * c_
        ss = 2.0 * s_ * s_
        dux = dx * (ss * uz) + dy * (-s2) + dz * (-2.0 * s_ * s_ * ux)
        duy = dx * (s2) + dy * (ss * uz) + dz * (-2.0 * s_ * s_ * uy)
        duz = dx * (ss * ux) + dy * (ss * uy) + dz * (2.0 * s_ * s_ * uz)
        du = np.stack([dux, duy, duz], axis=-1)

        # u = H / r => dL/dH = (du - (du . u) u) / r + dr * u
        du_dot_u = np.sum(du * u, axis=-1, keepdims=True)
        dH = (du - du_dot_u * u) / r + dr[..., None] * u  # (B, H, 3)

        gh0 = np.sum(dH, axis=0)
        gV = np.einsum("bd,bhk->dhk", X, dH)

        grads = [gh0, gV, go]
        bc1, bc2 = 1.0 - 0.9 ** step, 1.0 - 0.999 ** step
        for i in range(3):
            m_p[i] = 0.9 * m_p[i] + 0.1 * grads[i]
            v_p[i] = 0.999 * v_p[i] + 0.001 * (grads[i] ** 2)
            params[i] -= lr * (m_p[i] / bc1) / (np.sqrt(v_p[i] / bc2) + 1e-8)
        mc = 0.9 * mc + 0.1 * gc
        vc = 0.999 * vc + 0.001 * (gc ** 2)
        c -= lr * (mc / bc1) / (np.sqrt(vc / bc2) + 1e-8)

    dt = time.time() - t0
    te_logits, _ = forward_and_cache(X_te)
    te_acc = np.mean((sigmoid(te_logits) >= 0.5) == Y_te)
    n_params = h0.size + V.size + o.size + 1
    return te_acc, n_params, dt


def train_real_mlp(X_tr, Y_tr, X_te, Y_te, H_real=48, steps=600, lr=0.04, seed=195):
    """Standard 1-hidden-layer tanh MLP with matched parameter budget (~4*H_real params)."""
    rng = np.random.default_rng(seed)
    B, D = X_tr.shape
    W1 = rng.normal(0, 1.0 / np.sqrt(D), (D, H_real))
    b1 = np.zeros(H_real)
    W2 = rng.normal(0, 1.0 / np.sqrt(H_real), (H_real,))
    b2 = 0.0
    params = [W1, b1, W2]
    m_p = [np.zeros_like(p) for p in params]
    v_p = [np.zeros_like(p) for p in params]
    mb2, vb2 = 0.0, 0.0

    t0 = time.time()
    for step in range(1, steps + 1):
        h = np.tanh(X_tr @ W1 + b1)
        logits = h @ W2 + b2
        d_logit = (sigmoid(logits) - Y_tr) / B
        gW2 = h.T @ d_logit
        gb2 = np.sum(d_logit)
        dh = d_logit[:, None] * W2[None, :] * (1.0 - h ** 2)
        gW1 = X_tr.T @ dh
        gb1 = np.sum(dh, axis=0)
        bc1, bc2 = 1.0 - 0.9 ** step, 1.0 - 0.999 ** step
        for i, g in enumerate([gW1, gb1, gW2]):
            m_p[i] = 0.9 * m_p[i] + 0.1 * g
            v_p[i] = 0.999 * v_p[i] + 0.001 * (g ** 2)
            params[i] -= lr * (m_p[i] / bc1) / (np.sqrt(v_p[i] / bc2) + 1e-8)
        mb2 = 0.9 * mb2 + 0.1 * gb2
        vb2 = 0.999 * vb2 + 0.001 * (gb2 ** 2)
        b2 -= lr * (mb2 / bc1) / (np.sqrt(vb2 / bc2) + 1e-8)

    dt = time.time() - t0
    te_logits = np.tanh(X_te @ W1 + b1) @ W2 + b2
    te_acc = np.mean((sigmoid(te_logits) >= 0.5) == Y_te)
    n_params = W1.size + b1.size + W2.size + 1
    return te_acc, n_params, dt


if __name__ == "__main__":
    X_tr, Y_tr = make_spirals(200, turns=2.5, noise=0.10, seed=195)
    X_te, Y_te = make_spirals(100, turns=2.5, noise=0.10, seed=999)

    acc_mlp, p_mlp, dt_mlp = train_real_mlp(X_tr, Y_tr, X_te, Y_te, H_real=48)
    acc_su2, p_su2, dt_su2 = train_su2_net(X_tr, Y_tr, X_te, Y_te, H_neurons=16)
    print(f"TASK: 2.5-turn Intertwined 2-Spirals (400 train / 200 test)")
    print(f"real_tanh_mlp  | test_acc: {acc_mlp*100:5.1f}% | params: {p_mlp:3d} | time: {dt_mlp:.3f}s")
    print(f"su2_spinor_net | test_acc: {acc_su2*100:5.1f}% | params: {p_su2:3d} | time: {dt_su2:.3f}s")
