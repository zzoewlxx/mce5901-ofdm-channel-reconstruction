"""A small fully connected network in NumPy (ReLU, Adam, optional per-sample loss weights).

Enough for the channel-reconstruction MLP on CPU without installing PyTorch.

    net = MLP([d_in, 128, 128, d_out], seed=0)
    out, cache = net.forward(X)
    net.step(cache, dloss_dout, lr=1e-3)
"""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = 1.0 if i == len(sizes) - 2 else 2.0          # He init for ReLU layers
            self.W.append(rng.standard_normal((a, b)) * np.sqrt(gain / a))
            self.b.append(np.zeros(b))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, X):
        acts, pre = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre.append(z)
            h = z if i == len(self.W) - 1 else np.maximum(z, 0)
            acts.append(h)
        return h, (acts, pre)

    def predict(self, X, batch=65536):
        return np.concatenate([self.forward(X[i:i + batch])[0] for i in range(0, len(X), batch)])

    def step(self, cache, dout, lr, b1=0.9, b2=0.999, eps=1e-8):
        acts, pre = cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (pre[i - 1] > 0)
        self.t += 1
        for k, (p, g) in enumerate(zip(self.params, gW + gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)

    def state(self):
        return [p.copy() for p in self.params]

    def load(self, state):
        for p, s in zip(self.params, state):
            p[...] = s
