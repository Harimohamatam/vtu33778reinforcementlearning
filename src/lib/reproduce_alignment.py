import json, time, numpy as np, pandas as pd

RANDOM_STATE = 42

# Domain: short campus IT/help-desk responses. The corpus is synthetic and generated from phrase slots.
OPENING = ["I can help", "I understand", "Thanks for asking", "I will check", "Let me explain", "We can try"]
ACTION = ["check your account", "reset the password", "review the request", "update your details", "track the payment", "contact support"]
DETAIL = ["right away", "step by step", "with your order", "for your safety", "in a moment", "using the portal"]
CLOSING = ["please wait", "you are welcome", "if needed", "for confirmation", "to resolve this", "before proceeding"]

POLITE = {"please", "thanks", "welcome", "understand", "help"}
ACTION_WORDS = {"check", "reset", "review", "update", "track", "contact", "explain", "try"}
UNHELPFUL = {"ignore", "whatever", "obviously", "never", "impossible"}
FUNCTION_WORDS = {"i", "can", "will", "your", "the", "a", "to", "for", "with", "in", "on", "if", "we", "let", "me", "this", "you", "are", "is", "it", "and", "before", "using"}

def make_corpus(n=2500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        parts = [rng.choice(OPENING), rng.choice(ACTION), rng.choice(DETAIL)]
        if rng.random() < 0.55:
            parts.append(rng.choice(CLOSING))
        out.append(" ".join(parts))
    return out

def gold_reward(text):
    """Held-out rubric: helpful action, respectful tone, concision, and no repeated content words."""
    words = text.lower().split()
    if not words:
        return -2.0
    content = [w for w in words if w not in FUNCTION_WORDS]
    reward = 0.55 * len(POLITE.intersection(words))
    reward += 0.65 * len(ACTION_WORDS.intersection(words))
    reward -= 0.85 * len(UNHELPFUL.intersection(words))
    reward -= 0.12 * max(0, len(words) - 9)
    reward -= 0.9 * max(0, len(content) - len(set(content)))
    if words[0] not in {"i", "thanks", "let", "we"}:
        reward -= 0.3
    return float(reward)

corpus = make_corpus()
VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted({w for s in corpus for w in s.split()})
STOI = {w: i for i, w in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2
V = len(VOCAB)
MAXLEN = 14

def encode(s):
    ids = [BOS] + [STOI[w] for w in s.split()][:MAXLEN - 2] + [EOS]
    return ids + [PAD] * (MAXLEN - len(ids))

def decode(row):
    out = []
    for i in row[1:]:
        if i == EOS or i == PAD:
            break
        out.append(VOCAB[i])
    return " ".join(out)

IDS = np.array([encode(s) for s in corpus])
GRAMMAR = {" ".join([o, a, d] + ([c] if c else []))
           for o in OPENING for a in ACTION for d in DETAIL for c in CLOSING + [None]}
print(f"Synthetic campus help-desk corpus: {len(corpus)} responses, vocabulary {V}")
print("Example:", corpus[0])
print(f"Gold reward is evaluation-only; corpus mean gold reward = {np.mean([gold_reward(s) for s in corpus]):.3f}")

#@title A neural language model written with NumPy (run this cell; open it to read the code)
class NeuralLM:
    """A neural language model: each next token is predicted from the previous K tokens and the position.

    Every training objective of Days 4 and 5 is a weighted sum of token log-probabilities, so one backward pass,
    `grad_weighted`, serves supervised fine-tuning, REINFORCE and DPO alike."""
    def __init__(self, K=3, d=12, H=64, seed=0):
        rng = np.random.default_rng(seed); self.K, self.d = K, d
        self.P = {"E": rng.normal(0, 0.3, (V, d)), "pos": rng.normal(0, 0.3, (MAXLEN, d)),
                  "W1": rng.normal(0, 1 / np.sqrt((K + 1) * d), ((K + 1) * d, H)), "b1": np.zeros(H),
                  "W2": rng.normal(0, 1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}; self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def copy(self):
        c = NeuralLM.__new__(NeuralLM); c.K, c.d = self.K, self.d
        c.P = {k: v.copy() for k, v in self.P.items()}
        c.m = {k: np.zeros_like(v) for k, v in self.P.items()}; c.v = {k: np.zeros_like(v) for k, v in self.P.items()}; c.t = 0
        return c

    def _context(self, ids):
        N, T = ids.shape
        ext = np.concatenate([np.full((N, self.K - 1), BOS), ids[:, :-1]], axis=1)
        return np.stack([ext[:, k:k + T - 1] for k in range(self.K)], axis=2)       # (N, T-1, K)

    def forward(self, ids):
        """Log-probabilities of every next token, shape (N, MAXLEN-1, V)."""
        N, T = ids.shape; ctx = self._context(ids); pos = np.broadcast_to(self.P["pos"][:T - 1], (N, T - 1, self.d))
        X = np.concatenate([self.P["E"][ctx].reshape(N, T - 1, -1), pos], axis=2)
        Hh = np.tanh(X @ self.P["W1"] + self.P["b1"]); logits = Hh @ self.P["W2"] + self.P["b2"]
        logits[..., PAD] = -1e9                                                         # padding is never generated
        lsm = logits - logits.max(-1, keepdims=True); lsm = lsm - np.log(np.exp(lsm).sum(-1, keepdims=True))
        self.cache = (ids, ctx, X, Hh, lsm); return lsm

    def token_logprobs(self, ids):
        lsm = self.forward(ids); y = ids[:, 1:]
        return np.take_along_axis(lsm, y[..., None], 2)[..., 0], (y != PAD).astype(float)

    def seq_logprob(self, ids):
        lp, mask = self.token_logprobs(ids); return (lp * mask).sum(1)

    def grad_weighted(self, ids, w):
        """Gradient of  -sum_n w_n * log pi(sequence n)  with respect to every parameter."""
        lsm = self.forward(ids); ids_, ctx, X, Hh, _ = self.cache; N, T1 = ctx.shape[:2]; y = ids[:, 1:]
        Wt = (y != PAD) * np.asarray(w, float)[:, None]
        G = np.exp(lsm) * Wt[..., None]; np.put_along_axis(G, y[..., None], np.take_along_axis(G, y[..., None], 2) - Wt[..., None], 2)
        g = {"W2": Hh.reshape(-1, Hh.shape[-1]).T @ G.reshape(-1, V), "b2": G.sum((0, 1))}
        dpre = (G @ self.P["W2"].T) * (1 - Hh ** 2)
        g["W1"] = X.reshape(-1, X.shape[-1]).T @ dpre.reshape(-1, dpre.shape[-1]); g["b1"] = dpre.sum((0, 1))
        dX = dpre @ self.P["W1"].T
        g["E"] = np.zeros_like(self.P["E"]); np.add.at(g["E"], ctx, dX[..., :self.K * self.d].reshape(N, T1, self.K, self.d))
        g["pos"] = np.zeros_like(self.P["pos"]); g["pos"][:T1] = dX[..., self.K * self.d:].sum(0)
        return g

    def adam_step(self, g, lr=3e-3, clip=1.0):
        norm = np.sqrt(sum((v ** 2).sum() for v in g.values())); scale = min(1.0, clip / (norm + 1e-12)); self.t += 1
        for k in self.P:
            gk = g[k] * scale; self.m[k] = 0.9 * self.m[k] + 0.1 * gk; self.v[k] = 0.999 * self.v[k] + 0.001 * gk ** 2
            self.P[k] -= lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)

    def sample(self, n, seed=0, temperature=1.0):
        """Generate n sequences token by token."""
        rng = np.random.default_rng(seed); ids = np.full((n, MAXLEN), PAD); ids[:, 0] = BOS; done = np.zeros(n, bool)
        for t in range(1, MAXLEN):
            lsm = self.forward(ids[:, :t + 1])[:, t - 1] / temperature
            p = np.exp(lsm - lsm.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            nxt = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
            if t == MAXLEN - 1:
                nxt = np.full(n, EOS)                                                # every sequence ends by the cap
            nxt[done] = PAD; ids[:, t] = nxt; done |= nxt == EOS
            if done.all():
                break
        return ids

import time
def train_sft(epochs=6, lr=3e-3, seed=RANDOM_STATE):
    """Supervised fine-tuning: maximise the log-probability of the corpus, token by token."""
    lm = NeuralLM(seed=seed); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(len(IDS)), len(IDS) // 128):
            batch = IDS[b]; w = np.ones(len(b)) / ((batch[:, 1:] != PAD).sum())
            lm.adam_step(lm.grad_weighted(batch, w), lr=lr)
    return lm


sft = train_sft()

def annotator_score(text):
    """Simulated annotator: mostly values helpfulness, but also overvalues length and repeated words."""
    words = text.lower().split()
    repeated = len(words) - len(set(words))
    return gold_reward(text) + 0.18 * len(words) + 0.35 * repeated

def make_pairs(model, n_pairs=1000, noise=0.10, seed=7):
    rng = np.random.default_rng(seed)
    outputs = model.sample(2 * n_pairs, seed=seed)
    texts = [decode(row) for row in outputs]
    winners, losers, flipped = [], [], 0
    for i in range(n_pairs):
        a, b = texts[2*i], texts[2*i+1]
        sa, sb = annotator_score(a), annotator_score(b)
        if abs(sa - sb) < 1e-9:
            continue
        win, lose = (outputs[2*i], outputs[2*i+1]) if sa > sb else (outputs[2*i+1], outputs[2*i])
        if rng.random() < noise:
            win, lose = lose, win
            flipped += 1
        winners.append(win); losers.append(lose)
    return np.array(winners), np.array(losers), flipped


Wp, Lp, flipped = make_pairs(sft, noise=0.10, seed=7)

def adam(P, g, state, lr):
    state["t"] = state.get("t", 0) + 1; t = state["t"]
    for k in P:
        m = state.setdefault("m" + k, np.zeros_like(P[k])); v = state.setdefault("v" + k, np.zeros_like(P[k]))
        m[...] = 0.9 * m + 0.1 * g[k]; v[...] = 0.999 * v + 0.001 * g[k] ** 2
        P[k] -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

class RewardModel:
    """The mean of the token embeddings and the length feed a small network that outputs one score.
    Like every learned reward, it is an approximation: It sees which words occur, not their order or repetition."""
    def __init__(self, d=16, H=32, seed=0):
        rng = np.random.default_rng(seed)
        self.P = {"E": rng.normal(0, 0.3, (V, d)), "U1": rng.normal(0, 1 / np.sqrt(d + 1), (d + 1, H)), "c1": np.zeros(H),
                  "u2": rng.normal(0, 1 / np.sqrt(H), H), "c2": np.zeros(1)}
        self.state = {}
    def score(self, ids):
        tok = ids > EOS; n = tok.sum(1, keepdims=True).clip(1)
        F = np.concatenate([(self.P["E"][ids] * tok[..., None]).sum(1) / n, n / 10.0], 1)
        Hh = np.tanh(F @ self.P["U1"] + self.P["c1"]); self.cache = (ids, F, tok, n, Hh)
        return Hh @ self.P["u2"] + self.P["c2"][0]
    def grad(self, gs):
        ids, F, tok, n, Hh = self.cache
        g = {"u2": Hh.T @ gs, "c2": np.array([gs.sum()])}; dpre = np.outer(gs, self.P["u2"]) * (1 - Hh ** 2)
        g["U1"] = F.T @ dpre; g["c1"] = dpre.sum(0); dF = dpre @ self.P["U1"].T
        g["E"] = np.zeros_like(self.P["E"]); np.add.at(g["E"], ids, dF[:, None, :-1] * tok[..., None] / n[..., None])
        return g

def train_rm(W, L, epochs=18, lr=3e-3, seed=0, holdout=0.15):
    """Bradley-Terry: maximise log sigmoid(r(winner) - r(loser)) over the training pairs."""
    rm = RewardModel(seed=seed); rng = np.random.default_rng(seed); ntr = int((1 - holdout) * len(W))
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(ntr), max(1, ntr // 128)):
            sw = rm.score(W[b]); sl = rm.score(L[b]); q = 1 / (1 + np.exp(sw - sl))     # q = 1 - sigmoid(sw - sl)
            rm.score(W[b]); gw = rm.grad(-q / len(b)); rm.score(L[b]); gl = rm.grad(q / len(b))
            adam(rm.P, {k: gw[k] + gl[k] for k in gw}, rm.state, lr)
    acc = float((rm.score(W[ntr:]) > rm.score(L[ntr:])).mean())
    return rm, acc


rm, acc = train_rm(Wp, Lp)

def rlhf(sft, rm, beta=0.2, iters=300, batch=64, lr=1e-3, seed=0, track=25, eval_n=192):
    """REINFORCE with the KL penalty folded into the reward:  r(y) = RM(y) - beta * [log pi(y) - log pi_ref(y)]."""
    pol, ref = sft.copy(), sft; hist = []
    for it in range(iters):
        seqs = pol.sample(batch, seed=1000 * seed + it)
        kl = pol.seq_logprob(seqs) - ref.seq_logprob(seqs)                # a one-sample estimate of the KL
        reward = rm.score(seqs) - beta * kl
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        pol.adam_step(pol.grad_weighted(seqs, adv / batch), lr=lr)
        if it % track == 0 or it == iters - 1:
            ev = pol.sample(eval_n, seed=50000 + it); tt = [decode(r) for r in ev]
            hist.append(dict(iter=it, gold=np.mean([gold_reward(t) for t in tt]), reward_model=float(rm.score(ev).mean()),
                             kl=float((pol.seq_logprob(ev) - ref.seq_logprob(ev)).mean()),
                             length=np.mean([len(t.split()) for t in tt]), in_grammar=np.mean([t in GRAMMAR for t in tt])))
    return pol, pd.DataFrame(hist)

def dpo(sft, W, L, beta=0.1, epochs=4, lr=1e-3, seed=0, batch=96, eval_n=160):
    """Direct preference optimisation: maximise log sigmoid(beta * [(log pi - log pi_ref)(y_w) - (log pi - log pi_ref)(y_l)])."""
    pol, ref = sft.copy(), sft; rng = np.random.default_rng(seed)
    lw_ref, ll_ref = ref.seq_logprob(W), ref.seq_logprob(L); hist = []
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(len(W)), max(1, len(W) // batch)):
            z = beta * ((pol.seq_logprob(W[b]) - lw_ref[b]) - (pol.seq_logprob(L[b]) - ll_ref[b]))
            q = 1 / (1 + np.exp(z))                                       # weight: large for pairs ranked wrongly
            gw = pol.grad_weighted(W[b], beta * q / len(b)); gl = pol.grad_weighted(L[b], -beta * q / len(b))
            pol.adam_step({k: gw[k] + gl[k] for k in gw}, lr=lr)
        ev = pol.sample(eval_n, seed=7000 + ep); tt = [decode(r) for r in ev]
        hist.append(dict(epoch=ep + 1, gold=np.mean([gold_reward(t) for t in tt]),
                         kl=float((pol.seq_logprob(ev) - ref.seq_logprob(ev)).mean()),
                         pair_accuracy=float((z > 0).mean()), in_grammar=np.mean([t in GRAMMAR for t in tt])))
    return pol, pd.DataFrame(hist)


rlhf, rh = rlhf(sft, rm, beta=0.2, iters=120, seed=0, track=20, eval_n=160)

dpo, dh = dpo(sft, Wp, Lp, beta=0.1, epochs=4, seed=0, batch=96, eval_n=160)


state = {"vocab": VOCAB, "maxLen": MAXLEN, "bos": BOS, "eos": EOS, "pad": PAD, "context": 3, "temperature": 1.0, "models": {name: {k: v.tolist() for k, v in model.P.items()} for name, model in (("sft", sft), ("rlhf", rlhf), ("dpo", dpo))}, "rewardModel": {k: v.tolist() for k, v in rm.P.items()}, "training": {"sftEpochs": 6, "rlhfBeta": 0.2, "rlhfIterations": 120, "dpoBeta": 0.1, "dpoEpochs": 4, "preferencePairCount": len(Wp), "rewardModelHeldOutAccuracy": round(acc, 3)}}
with open("src/lib/alignment-checkpoints.json", "w") as f: json.dump(state, f, separators=(",", ":"))
print("reproduced notebook checkpoints", len(Wp), "preference pairs", round(acc, 3), "held-out pairwise accuracy")
