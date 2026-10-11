#!/usr/bin/env python3
"""
recon_bench v2 -- synthetic benchmark for bank-to-ledger transaction matching.

Every number, table and figure in the paper is produced by ONE run of this script
(`./run_all.sh`), which writes everything to results/. See README.md.

Changes from v1 (in response to review):
  * amounts are integer cents; fee detection is an exact integer test (v1 used float equality)
  * decoys are generated with an explicit vendor choice and recorded by category
    (same-vendor / different-vendor / unmatched-amount)
  * vendor-name similarity and reference agreement are separate feature groups
  * stronger deterministic baseline (tiered rule cascade, threshold tuned on training data)
  * stress tests: unseen vendors, unfamiliar name forms, repeated amounts, out-of-range noise
  * counterparty-resolution experiment, calibration, unmatched-item errors, monetary exposure,
    record-level automation, review load, 10 held-out seeds with 95% confidence intervals
  * HistGradientBoosting early stopping disabled explicitly; stable sorts for tie-breaking
"""
import argparse, hashlib, json, math, os, platform, random, sys, time
from collections import Counter
from dataclasses import dataclass, asdict, replace
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

# --------------------------------------------------------------------------- vendors and name forms
VENDORS_A = ["Acme Industrial Supply", "Northwind Logistics", "Contoso Energy", "Globex Metals",
             "Initech Software", "Umbrella Chemicals", "Stark Components", "Wayne Freight",
             "Hooli Cloud Services", "Vandelay Imports", "Soylent Foods", "Cyberdyne Robotics",
             "Tyrell Materials", "Wonka Packaging", "Pied Piper Data", "Massive Dynamic",
             "Gringotts Payments", "Oscorp Tooling", "Aperture Labs", "Monarch Utilities"]
# held-out vendor master, never seen in training
VENDORS_B = ["Bluth Construction", "Prestige Worldwide", "Dunder Paper", "Sterling Cooper Media",
             "Krusty Foods", "Duff Beverages", "Nakatomi Trading", "Weyland Mining",
             "Spacely Sprockets", "Cogswell Cogs", "Rekall Systems", "Virtucon Holdings",
             "Ollivander Supplies", "Gump Seafood Company", "Paper Street Soap", "Pollos Distributors",
             "Wernham Hogg Paper", "Mooby Retail Group", "Planet Express Courier", "InGen Biotech"]
VENDOR_SETS = {"A": VENDORS_A, "B": VENDORS_B}
FEES_C = (1500, 2500, 3000, 4500)  # standard wire / processing fees, in cents


def name_standard(name, rng):
    """Name forms used in training and the main benchmark."""
    words = name.upper().split()
    m = rng.random()
    if m < 0.30:
        return " ".join(w[:4] for w in words)                       # truncated words
    if m < 0.55:
        return words[0] + " " + "".join(w[0] for w in words[1:])    # initials
    if m < 0.75:
        s = list(name.upper())
        if len(s) > 4:
            s[rng.randrange(1, len(s) - 1)] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return "".join(s)                                           # one-character typo
    return name.upper()                                             # full name


def name_unfamiliar(name, rng):
    """Name forms never used in training (stress test only)."""
    words = name.upper().split()
    m = rng.random()
    if m < 0.25:
        return " ".join(w[0] + "".join(ch for ch in w[1:] if ch not in "AEIOU") for w in words)  # vowels dropped
    if m < 0.50:
        return rng.choice(["ACH PMT ", "WIRE OUT ", "SQ *", "PAYPAL *"]) + "".join(words)[:12]   # processor prefix
    if m < 0.75:
        core = words[:-1] if len(words) > 1 else words
        return " ".join(core) + rng.choice([" LLC", " INC", " CORP", " LTD"])                  # suffix, word dropped
    return " ".join(reversed(words))                                                           # word order reversed


NAME_STYLES = {"standard": name_standard, "unfamiliar": name_unfamiliar}


# --------------------------------------------------------------------------- generator
@dataclass(frozen=True)
class Config:
    n: int = 2000                   # ledger entries per run
    d: float = 0.5                  # difficulty: scales every perturbation rate
    seed: int = 0
    vendors: str = "A"              # "A" (training vocabulary) or "B" (unseen)
    names: str = "standard"         # "standard" or "unfamiliar"
    recurring: float = 0.0          # share of ledger entries in same-vendor, same-amount series of three
    recurring_gap: int = 30         # days between invoices in a series (30 = monthly, outside the 10-day window)
    same_vendor_decoy: float = 0.15  # share of extra bank records that reuse the source vendor


@dataclass
class Data:
    bank: list
    gl: list
    truth: set
    causes: dict     # true pair -> list of perturbation tags
    kinds: dict      # item id -> category, for items with no true partner
    vendors: list    # vendor master for this run


def make_dataset(cfg: Config) -> Data:
    rng = random.Random(cfg.seed)
    d = cfg.d
    vendors = VENDOR_SETS[cfg.vendors]
    style = NAME_STYLES[cfg.names]
    gl, bank, truth, causes, kinds = [], [], set(), {}, {}

    def add(vendor, amt_c, day):
        i = len(gl)
        ref = "INV" + str(rng.randint(100000, 999999))
        g = dict(id=f"G{i}", day=day, amount_c=amt_c, vendor=vendor, ref=ref)
        gl.append(g)
        if rng.random() < 0.04 + 0.06 * d:                 # ledger entry with no bank movement
            kinds[g["id"]] = "gl_unmatched"
            return
        tags, bamt = [], amt_c
        r = rng.random()
        if r < 0.10 * d:                                   # fee deducted by the bank
            bamt = amt_c - rng.choice(FEES_C); tags.append("fee")
        elif r < 0.18 * d:                                 # FX / rounding drift up to +-1.5%
            bamt = int(round(amt_c * (1 + rng.uniform(-0.015, 0.015)))); tags.append("fx")
        lag = int(abs(rng.gauss(0, 1 + 3 * d)))
        if lag > 3:
            tags.append("lag>3")
        bref = ref if rng.random() > 0.25 + 0.45 * d else ""
        if not bref:
            tags.append("no_ref")
        if bref and rng.random() < 0.2 * d:
            bref = bref[:-2]; tags.append("trunc_ref")
        b = dict(id=f"B{i}", day=day + lag, amount_c=bamt, name=style(vendor, rng), ref=bref)
        bank.append(b)
        truth.add((b["id"], g["id"]))
        causes[(b["id"], g["id"])] = tags or ["clean"]

    p_series = cfg.recurring / (3 - 2 * cfg.recurring) if cfg.recurring else 0.0
    while len(gl) < cfg.n:
        vendor = rng.choice(vendors)
        amt_c = int(round(math.exp(rng.uniform(math.log(5_000), math.log(25_000_000)))))  # $50 .. $250,000
        if p_series and rng.random() < p_series:          # three monthly invoices, same vendor and amount
            start = rng.randint(0, 89 - 2 * cfg.recurring_gap)
            for k in range(3):
                if len(gl) < cfg.n:
                    add(vendor, amt_c, start + cfg.recurring_gap * k)
        else:
            add(vendor, amt_c, rng.randint(0, 89))

    n_extra = int(cfg.n * (0.03 + 0.05 * d))
    for j in range(n_extra):                               # bank records with no ledger partner
        src = rng.choice(gl)
        same_amt = rng.random() < min(1.0, 0.5 + 0.4 * d)
        same_vendor = rng.random() < cfg.same_vendor_decoy
        vendor = src["vendor"] if same_vendor else rng.choice([v for v in vendors if v != src["vendor"]])
        amt = src["amount_c"] if same_amt else int(round(src["amount_c"] * rng.uniform(0.9, 1.1)))
        bid = f"BX{j}"
        bank.append(dict(id=bid, day=src["day"] + rng.randint(0, 4), amount_c=amt,
                         name=style(vendor, rng), ref=""))
        kinds[bid] = ("decoy_same_vendor" if same_vendor else "decoy_diff_vendor") if same_amt else "unmatched_bank"
    rng.shuffle(bank)
    return Data(bank, gl, truth, causes, kinds, list(vendors))


# --------------------------------------------------------------------------- features
@lru_cache(maxsize=None)
def trigrams(s):
    s = " " + "".join(ch for ch in s.upper() if ch.isalnum() or ch == " ") + " "
    return frozenset(s[i:i + 3] for i in range(len(s) - 2))


@lru_cache(maxsize=None)
def word_tokens(s):
    return frozenset(s.upper().replace("*", " ").split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def resolve_counterparty(name, master):
    """Entity resolution against the vendor master: best trigram match and its margin over the runner-up."""
    t = trigrams(name)
    sims = sorted(((jaccard(t, trigrams(v)), v) for v in master), reverse=True)
    return sims[0][1], sims[0][0] - sims[1][0]


AMT_DATE = ["abs_diff", "rel_diff", "is_fee", "lag", "abs_lag", "log_amt"]
NAME = ["name_tri", "name_tok"]
REF = ["ref_exact", "ref_prefix", "has_ref"]
ER = ["er_same", "er_margin"]
FEATURE_SETS = {"full": AMT_DATE + NAME + REF, "no_name": AMT_DATE + REF,
                "no_ref": AMT_DATE + NAME, "no_text": AMT_DATE, "er": AMT_DATE + NAME + REF + ER}
AMT_TOL, AMT_BAND_C, DAY_TOL = 0.05, 5000, 10


def candidates(data: Data) -> pd.DataFrame:
    """Blocking: ledger entries with |amount gap| <= 5% + $50 and |date gap| <= 10 days."""
    gl = sorted(data.gl, key=lambda g: (g["amount_c"], g["id"]))
    amts = np.array([g["amount_c"] for g in gl])
    cols = {k: [] for k in ["b", "g", "b_amount_c", "diff_c", "abs_diff", "rel_diff", "is_fee", "lag", "abs_lag",
                            "log_amt", "name_tri", "name_tok", "ref_exact", "ref_prefix", "has_ref",
                            "er_same", "er_margin"]}
    for b in data.bank:
        a = b["amount_c"]
        lo = np.searchsorted(amts, (a - AMT_BAND_C) / (1 + AMT_TOL), "left")
        hi = np.searchsorted(amts, (a + AMT_BAND_C) / (1 - AMT_TOL), "right")
        bt, bw = trigrams(b["name"]), word_tokens(b["name"])
        er_vendor, er_margin = resolve_counterparty(b["name"], data.vendors)
        has_ref = int(bool(b["ref"]))
        for g in gl[lo:hi]:
            lag = b["day"] - g["day"]
            if abs(lag) > DAY_TOL:
                continue
            diff = a - g["amount_c"]
            if abs(diff) > AMT_TOL * g["amount_c"] + AMT_BAND_C:
                continue
            cols["b"].append(b["id"]); cols["g"].append(g["id"]); cols["b_amount_c"].append(a)
            cols["diff_c"].append(diff); cols["abs_diff"].append(abs(diff) / 100)
            cols["rel_diff"].append(abs(diff) / g["amount_c"])
            cols["is_fee"].append(int(-diff in FEES_C))                    # exact integer test
            cols["lag"].append(lag); cols["abs_lag"].append(abs(lag))
            cols["log_amt"].append(math.log10(g["amount_c"] / 100))
            cols["name_tri"].append(jaccard(bt, trigrams(g["vendor"])))      # vendor name only, no reference
            cols["name_tok"].append(jaccard(bw, word_tokens(g["vendor"])))
            cols["ref_exact"].append(int(has_ref and b["ref"] == g["ref"]))
            cols["ref_prefix"].append(int(has_ref and g["ref"].startswith(b["ref"])))
            cols["has_ref"].append(has_ref)
            cols["er_same"].append(int(er_vendor == g["vendor"]))
            cols["er_margin"].append(er_margin)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- matchers
def greedy(df, by, ascending, used_b=None, used_g=None):
    used_b = set() if used_b is None else used_b
    used_g = set() if used_g is None else used_g
    out = set()
    d = df.sort_values(by, ascending=ascending, kind="mergesort")
    for b, g in zip(d.b.values, d.g.values):
        if b in used_b or g in used_g:
            continue
        used_b.add(b); used_g.add(g); out.add((b, g))
    return out


def rule_exact(c):
    """R1: same amount to the cent, identical reference, posting lag <= 3 days."""
    return greedy(c[(c.diff_c == 0) & (c.ref_exact == 1) & (c.abs_lag <= 3)], ["abs_lag"], [True])


def rule_relaxed(c):
    """R2: amount within 2%, lag <= 5 days, no text check."""
    return greedy(c[(c.rel_diff <= 0.02) & (c.abs_lag <= 5)], ["abs_diff", "abs_lag"], [True, True])


def cascade_tiers(c, theta):
    return [
        ("T1 exact", (c.diff_c == 0) & (c.ref_exact == 1) & (c.abs_lag <= 3)),
        ("T2 reference", (c.ref_prefix == 1) & ((c.diff_c == 0) | (c.is_fee == 1) | (c.rel_diff <= 0.02)) & (c.abs_lag <= 10)),
        ("T3 amount+name", (c.diff_c == 0) & (c.name_tri >= theta) & (c.abs_lag <= 7)),
        ("T4 fee/drift+name", ((c.is_fee == 1) | (c.rel_diff <= 0.015)) & (c.name_tri >= theta) & (c.abs_lag <= 7)),
    ]


def rule_cascade(c, theta):
    """R3: tiered rule cascade typical of production engines; each tier runs on what earlier tiers left."""
    ub, ug, out, per_tier = set(), set(), set(), {}
    for name, mask in cascade_tiers(c, theta):
        got = greedy(c[mask], ["name_tri", "abs_lag"], [False, True], ub, ug)
        per_tier[name] = got
        out |= got
    return out, per_tier


def assign(c, scores, tau0=0.5):
    """Maximum-weight one-to-one matching over candidate edges with score >= tau0.

    Implementation: a dense matrix over the bank and ledger items that have at least one such edge,
    holding the score on candidate edges and 0 elsewhere, solved with SciPy's rectangular linear
    assignment (Crouse 2016) maximizing total score. Assignments that land on a 0 (non-edge) are
    dropped, so any item may stay unmatched. Returns the matched pairs and their scores.
    """
    keep = scores >= tau0
    if not keep.any():
        return set(), {}
    cb, cg, cs = c.b.values[keep], c.g.values[keep], scores[keep]
    bs, gs = np.unique(cb), np.unique(cg)
    bi = {b: i for i, b in enumerate(bs)}; gi = {g: i for i, g in enumerate(gs)}
    M = np.zeros((len(bs), len(gs)))
    for b, g, s in zip(cb, cg, cs):
        M[bi[b], gi[g]] = s
    rr, cc = linear_sum_assignment(-M)
    out, conf = set(), {}
    for i, j in zip(rr, cc):
        if M[i, j] >= tau0:
            out.add((bs[i], gs[j])); conf[(bs[i], gs[j])] = float(M[i, j])
    return out, conf


# --------------------------------------------------------------------------- metrics
def prf(pred, truth):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(truth) if truth else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def fp_categories(pred, data):
    cats = Counter()
    for b, g in pred - data.truth:
        if b in data.kinds:
            cats[data.kinds[b]] += 1            # bank record with no true partner
        elif g in data.kinds:
            cats["gl_unmatched"] += 1           # true bank record matched to a ledger entry with no partner
        else:
            cats["swap"] += 1                   # both items have true partners elsewhere
    return cats


def unmatched_item_errors(pred, data):
    """For items with no true partner: how many there are, and how many the method matched anyway."""
    pb, pg = {b for b, _ in pred}, {g for _, g in pred}
    out = {}
    for item, kind in data.kinds.items():
        e = out.setdefault(kind, [0, 0])
        e[0] += 1
        e[1] += int(item in (pg if kind == "gl_unmatched" else pb))
    return out


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1)), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def ece(scores, labels, bins=10):
    scores, labels = np.asarray(scores), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            total += m.sum() / len(scores) * abs(scores[m].mean() - labels[m].mean())
    return float(total)


# --------------------------------------------------------------------------- training
def label(c, truth):
    return np.fromiter(((b, g) in truth for b, g in zip(c.b.values, c.g.values)), bool, len(c))


def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=False, random_state=0)
    return m.fit(X, y)


def tune_cascade_theta(train_sets, grid=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)):
    best = None
    for th in grid:
        f = np.mean([prf(rule_cascade(c, th)[0], data.truth)[2] for data, c in train_sets])
        if best is None or f > best[1] + 1e-12:
            best = (th, f)
    return best


# --------------------------------------------------------------------------- evaluation
def evaluate(cfg, models, theta, keep_detail=False):
    data = make_dataset(cfg)
    c = candidates(data)
    truth = data.truth
    y = label(c, truth)
    amount = {b["id"]: b["amount_c"] for b in data.bank}
    scores = {k: m.predict_proba(c[FEATURE_SETS[k]].values)[:, 1] for k, m in models.items()}  # for the M variants

    pred = {}
    pred["R1"] = rule_exact(c)
    pred["R2"] = rule_relaxed(c)
    pred["R3"], tiers = rule_cascade(c, theta)
    for k in models:
        pred[f"M_{k}"] = assign(c, scores[k])[0]
    pred["MG"] = greedy(c[scores["full"] >= 0.5].assign(s=scores["full"][scores["full"] >= 0.5]), ["s"], [False])

    r1b, r1g = {b for b, _ in pred["R1"]}, {g for _, g in pred["R1"]}
    res_mask = ~(c.b.isin(r1b) | c.g.isin(r1g)).values
    res = c[res_mask]
    # hybrids score ONLY the pairs the rule stage left (residual-only scoring)
    res_scores = models["full"].predict_proba(res[FEATURE_SETS["full"]].values)[:, 1] if len(res) else np.array([])
    hm, hconf = assign(res, res_scores)
    pred["H"] = pred["R1"] | hm
    res_scores_er = models["er"].predict_proba(res[FEATURE_SETS["er"]].values)[:, 1] if len(res) else np.array([])
    hm_er, hconf_er = assign(res, res_scores_er)
    pred["H_ER"] = pred["R1"] | hm_er
    r3b, r3g = {b for b, _ in pred["R3"]}, {g for _, g in pred["R3"]}
    res3_mask = ~(c.b.isin(r3b) | c.g.isin(r3g)).values
    res3 = c[res3_mask]
    h3m, _ = assign(res3, models["full"].predict_proba(res3[FEATURE_SETS["full"]].values)[:, 1] if len(res3) else np.array([]))
    pred["H3"] = pred["R3"] | h3m
    # sanity check: residual-only scores equal the full-candidate scores on the same rows
    assert np.array_equal(res_scores, scores["full"][res_mask])

    out = dict(cfg=asdict(cfg), n_bank=len(data.bank), n_gl=len(data.gl), n_true=len(truth), n_cand=len(c),
               cand_recall=len(set(zip(c.b.values, c.g.values)) & truth) / len(truth),
               metrics={k: prf(v, truth) for k, v in pred.items()},
               fp={k: dict(fp_categories(pred[k], data)) for k in ("M_full", "H", "H_ER", "R3", "H3")},
               unmatched={k: unmatched_item_errors(pred[k], data) for k in ("H", "H_ER")},
               r1_true=len(pred["R1"] & truth), r1_pairs=len(pred["R1"]),
               r3_true=len(pred["R3"] & truth), r3_pairs=len(pred["R3"]),
               tier_counts={k: [len(v), len(v & truth)] for k, v in tiers.items()},
               scored_pairs_H=int(res_mask.sum()), scored_pairs_M=len(c), scored_pairs_H3=int(res3_mask.sum()),
               fp_value_c={k: int(sum(amount[b] for b, _ in pred[k] - truth)) for k in ("H", "H_ER", "R2", "R3", "H3")},
               total_value_c=int(sum(amount.values())))
    # same-vendor decoy errors: what did the true partner of the wrongly matched ledger entry look like?
    partner = {g: b for b, g in truth}
    bank_by_id = {b["id"]: b for b in data.bank}
    matched_bank = {b for b, _ in pred["H"]}
    sv = Counter()
    for b, g in pred["H"] - truth:
        if data.kinds.get(b) == "decoy_same_vendor" and g in partner:
            p = bank_by_id[partner[g]]
            sv["n"] += 1
            sv["partner_has_no_reference"] += int(not p["ref"])
            sv["partner_left_unmatched"] += int(p["id"] not in matched_bank)
    out["same_vendor_decoy_partner"] = dict(sv)
    by_cause = {}
    for pair, tags in data.causes.items():
        for t in tags:
            e = by_cause.setdefault(t, [0, 0, 0, 0])
            e[0] += 1; e[1] += pair in pred["R1"]; e[2] += pair in pred["R3"]; e[3] += pair in pred["H"]
    out["by_cause"] = by_cause
    if keep_detail:
        out["_detail"] = dict(R1=pred["R1"], hconf=hconf, truth=truth, amount=amount, n_bank=len(data.bank),
                              n_gl=len(data.gl), res=res, res_scores=res_scores, res_y=y[res_mask])
    return out


def routing(details, taus, recompute=True):
    """Gate the hybrid's model matches at tau. 'filtered' keeps the tau0 = 0.5 assignment and drops pairs
    below tau (what the paper reports); 'recomputed' re-solves the assignment at each tau (a check)."""
    rows = []
    for t in taus:
        agg = Counter()
        for dt in details:
            sets = {"filtered": dt["R1"] | {p for p, s in dt["hconf"].items() if s >= t}}
            if recompute:
                sets["recomputed"] = dt["R1"] | assign(dt["res"], dt["res_scores"], t)[0]
            for mode, auto in sets.items():
                ok = auto & dt["truth"]
                agg[(mode, "auto")] += len(auto); agg[(mode, "ok")] += len(ok)
                agg[(mode, "fp_value")] += sum(dt["amount"][b] for b, _ in auto - dt["truth"])
                agg[(mode, "auto_value")] += sum(dt["amount"][b] for b, _ in auto)
            agg["true"] += len(dt["truth"]); agg["bank"] += dt["n_bank"]; agg["gl"] += dt["n_gl"]
        for mode in (["filtered", "recomputed"] if recompute else ["filtered"]):
            a, ok = agg[(mode, "auto")], agg[(mode, "ok")]
            rows.append(dict(mode=mode, tau=t, auto_precision=ok / a, automated_true_pair_recall=ok / agg["true"],
                             record_automation=a / agg["bank"], bank_review_queue=agg["bank"] - a,
                             ledger_review_queue=agg["gl"] - a,
                             false_auto_matches=a - ok, fp_value_usd=agg[(mode, "fp_value")] / 100,
                             fp_value_share=agg[(mode, "fp_value")] / agg[(mode, "auto_value")]))
    return pd.DataFrame(rows)


def time_scoring(model, feats, n, d, seeds, repeats=7):
    """Like-for-like scoring time of one model on all candidates vs the exact-rule residual."""
    from statistics import median
    rows = []
    for s in seeds:
        data = make_dataset(Config(n=n, d=d, seed=s))
        c = candidates(data)
        r1 = rule_exact(c)
        rb, rg = {b for b, _ in r1}, {g for _, g in r1}
        Xa = c[feats].values
        Xr = c[~(c.b.isin(rb) | c.g.isin(rg)).values][feats].values
        model.predict_proba(Xa[:256])                     # warm-up
        ta, tr = [], []
        for _ in range(repeats):                          # interleaved, so drift affects both equally
            t = time.perf_counter(); model.predict_proba(Xa); ta.append(time.perf_counter() - t)
            t = time.perf_counter(); model.predict_proba(Xr); tr.append(time.perf_counter() - t)
        rows.append(dict(seed=s, pairs_all=len(Xa), pairs_residual=len(Xr), all_s=median(ta), residual_s=median(tr)))
    return rows


def grouped_permutation_importance(model, X, y, groups, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X.values)[:, 1])
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp.values)[:, 1]))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    return base, out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="small smoke test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    N = 400 if args.quick else 2000
    TRAIN_SEEDS = [1, 2, 3]
    TEST_SEEDS = list(range(101, 103 if args.quick else 111))
    DIFFS = [0.0, 0.25, 0.5, 0.75, 1.0]
    TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]

    # ---- training: seeds 1-3 at every difficulty, training vendor set, standard name forms
    train_sets = []
    for s in TRAIN_SEEDS:
        for d in DIFFS:
            data = make_dataset(Config(n=N, d=d, seed=s))
            train_sets.append((data, candidates(data)))
    Xall = pd.concat([c for _, c in train_sets], ignore_index=True)
    yall = np.concatenate([label(c, data.truth) for data, c in train_sets])
    models = {k: fit(Xall[f].values, yall) for k, f in FEATURE_SETS.items()}
    theta, theta_f1 = tune_cascade_theta(train_sets)
    print(f"trained on {len(Xall)} pairs ({int(yall.sum())} positive); cascade theta={theta} ({time.time()-t0:.0f}s)", flush=True)

    # ---- main grid
    main_runs, details = [], []
    for d in DIFFS:
        for s in TEST_SEEDS:
            r = evaluate(Config(n=N, d=d, seed=s), models, theta, keep_detail=d >= 0.5)
            if d >= 0.5:
                details.append(r.pop("_detail"))
            main_runs.append(r)
    print(f"main grid done ({time.time()-t0:.0f}s)", flush=True)
    # wall-clock scoring time, like for like: the SAME full-feature model scores all candidates vs only the
    # residual left by exact rules, interleaved repeats, median per test set. Machine-dependent, so it goes
    # to timing.json, outside the checksummed results.
    timing = {"method": "full-feature model, all candidates vs exact-rule residual; median of 7 interleaved repeats per test set",
              "threads": os.environ.get("OMP_NUM_THREADS") or "default"}
    for d in [0.5, 1.0]:
        rows = time_scoring(models["full"], FEATURE_SETS["full"], N, d, TEST_SEEDS)
        ta, tr = sum(r["all_s"] for r in rows), sum(r["residual_s"] for r in rows)
        per = [1 - r["residual_s"] / r["all_s"] for r in rows]
        timing[str(d)] = dict(test_sets=len(rows), pairs_all=sum(r["pairs_all"] for r in rows),
                              pairs_residual=sum(r["pairs_residual"] for r in rows),
                              pair_reduction=round(1 - sum(r["pairs_residual"] for r in rows) / sum(r["pairs_all"] for r in rows), 4),
                              all_candidates_s=round(ta, 4), residual_only_s=round(tr, 4), time_reduction=round(1 - tr / ta, 4),
                              per_set_time_reduction_min=round(min(per), 4), per_set_time_reduction_max=round(max(per), 4))
    print(f"timing done ({time.time()-t0:.0f}s)", flush=True)

    METHODS = ["R1", "R2", "R3", "M_no_text", "M_no_name", "M_no_ref", "M_full", "MG", "H", "H3", "M_er", "H_ER"]
    raw = [dict(difficulty=r["cfg"]["d"], seed=r["cfg"]["seed"], method=m, precision=r["metrics"][m][0],
                recall=r["metrics"][m][1], f1=r["metrics"][m][2]) for r in main_runs for m in METHODS]
    raw = pd.DataFrame(raw)
    raw.to_csv(f"{args.out}/main_raw.csv", index=False)
    summ = []
    for (d, m), g in raw.groupby(["difficulty", "method"], sort=False):
        row = dict(difficulty=d, method=m)
        for k in ("precision", "recall", "f1"):
            mu, sd, hw = ci95(g[k]); row.update({f"{k}_mean": mu, f"{k}_sd": sd, f"{k}_ci95": hw})
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{args.out}/main_summary.csv", index=False)

    # paired differences per seed
    piv = raw.pivot_table(index=["difficulty", "seed"], columns="method", values="f1")
    paired = {}
    for a, b in [("H", "M_full"), ("MG", "M_full"), ("H_ER", "H"), ("H3", "H"), ("M_full", "M_no_text")]:
        diff = (piv[a] - piv[b]).groupby(level=0)
        paired[f"{a}-{b}"] = {str(d): dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                                            seeds_differing=int((v.abs() > 1e-12).sum())) for d, v in diff}

    # pooled d >= 0.5 diagnostics
    hi = [r for r in main_runs if r["cfg"]["d"] >= 0.5]
    def pool(key, method):
        tot = Counter()
        for r in hi:
            tot.update(r[key][method])
        return dict(tot)
    fp = {m: pool("fp", m) for m in ("M_full", "H", "H_ER", "R3", "H3")}
    unmatched = {}
    for m in ("H", "H_ER"):
        acc = {}
        for r in hi:
            for k, (n_items, n_hit) in r["unmatched"][m].items():
                e = acc.setdefault(k, [0, 0]); e[0] += n_items; e[1] += n_hit
        unmatched[m] = {k: dict(items=v[0], wrongly_matched=v[1], rate=v[1] / v[0]) for k, v in acc.items()}
    cause = {}
    for r in hi:
        for k, v in r["by_cause"].items():
            e = cause.setdefault(k, [0, 0, 0, 0]); e[:] = [x + y for x, y in zip(e, v)]
    cause = {k: dict(pairs=v[0], R1=v[1] / v[0], R3=v[2] / v[0], H=v[3] / v[0]) for k, v in cause.items()}
    pd.DataFrame(cause).T.to_csv(f"{args.out}/recall_by_cause.csv")

    ops = {}
    for d in DIFFS:
        rs = [r for r in main_runs if r["cfg"]["d"] == d]
        ops[str(d)] = dict(
            r1_share_of_true_pairs=float(np.mean([r["r1_true"] / r["n_true"] for r in rs])),
            r1_share_of_hybrid_true_matches=float(np.mean([r["r1_true"] / max(1, round(r["metrics"]["H"][1] * r["n_true"])) for r in rs])),
            r3_share_of_true_pairs=float(np.mean([r["r3_true"] / r["n_true"] for r in rs])),
            scored_pairs_H=float(np.mean([r["scored_pairs_H"] for r in rs])),
            scored_pairs_M=float(np.mean([r["scored_pairs_M"] for r in rs])),
            scoring_avoided=float(np.mean([1 - r["scored_pairs_H"] / r["scored_pairs_M"] for r in rs])),
            cand_recall_mean=float(np.mean([r["cand_recall"] for r in rs])),
            cand_recall_min=float(np.min([r["cand_recall"] for r in rs])))
    svd = Counter()
    for r in hi:
        svd.update(r["same_vendor_decoy_partner"])
    tiers = {}
    for r in hi:
        for k, (n_p, n_ok) in r["tier_counts"].items():
            e = tiers.setdefault(k, [0, 0]); e[0] += n_p; e[1] += n_ok
    tiers = {k: dict(pairs=v[0], precision=v[1] / v[0] if v[0] else None) for k, v in tiers.items()}
    money = {}
    for m in ("H", "H_ER", "R2", "R3", "H3"):
        fpv = sum(r["fp_value_c"][m] for r in hi); tot = sum(r["total_value_c"] for r in hi)
        money[m] = dict(fp_value_usd=fpv / 100, share_of_bank_value=fpv / tot)

    # ---- calibration of the residual scorer (pooled d >= 0.5)
    s_all = np.concatenate([dt["res_scores"] for dt in details]); y_all = np.concatenate([dt["res_y"] for dt in details])
    as_s, as_y = [], []
    for dt in details:
        for p, s in dt["hconf"].items():
            as_s.append(s); as_y.append(p in dt["truth"])
    calib = dict(candidate_pairs=len(s_all), brier=float(np.mean((s_all - y_all) ** 2)), ece_candidates=ece(s_all, y_all),
                 assigned_pairs=len(as_s), ece_assigned=ece(as_s, as_y))
    rel = []
    edges = np.linspace(0, 1, 11)
    for k in range(10):
        m = (s_all >= edges[k]) & ((s_all < edges[k + 1]) | (k == 9))
        if m.any():
            rel.append(dict(bin=f"{edges[k]:.1f}-{edges[k+1]:.1f}", pairs=int(m.sum()), mean_score=float(s_all[m].mean()), match_rate=float(y_all[m].mean())))
    calib["reliability"] = rel

    # ---- routing
    route = routing(details, TAUS)
    route.to_csv(f"{args.out}/routing.csv", index=False)
    print(f"routing done ({time.time()-t0:.0f}s)", flush=True)

    # ---- stress tests at d = 0.5
    STRESS = {"base": Config(n=N, d=0.5), "unseen_vendors": Config(n=N, d=0.5, vendors="B"),
              "unfamiliar_names": Config(n=N, d=0.5, names="unfamiliar"),
              "unseen_and_unfamiliar": Config(n=N, d=0.5, vendors="B", names="unfamiliar"),
              "monthly_recurring": Config(n=N, d=0.5, recurring=0.3, recurring_gap=30),
              "weekly_recurring": Config(n=N, d=0.5, recurring=0.3, recurring_gap=7),
              "noise_beyond_training": Config(n=N, d=1.25)}
    S_METHODS = ["R1", "R3", "M_full", "H", "H_ER"]
    srows, sfp, spaired = [], {}, {}
    for name, base in STRESS.items():
        runs = [r for r in main_runs if r["cfg"]["d"] == 0.5] if name == "base" else \
               [evaluate(replace(base, seed=s), models, theta) for s in TEST_SEEDS]
        for m in S_METHODS:
            for k, idx in (("precision", 0), ("recall", 1), ("f1", 2)):
                mu, sd, hw = ci95([r["metrics"][m][idx] for r in runs])
                srows.append(dict(test=name, method=m, metric=k, mean=mu, sd=sd, ci95=hw))
        spaired[name] = {f"{a}-{b}": ci95([r["metrics"][a][2] - r["metrics"][b][2] for r in runs])
                         for a, b in [("H", "M_full"), ("R3", "H"), ("H_ER", "H")]}
        tot = {}
        for m in ("H", "H_ER"):
            cnt = Counter()
            for r in runs:
                cnt.update(r["fp"][m])
            tot[m] = dict(cnt)
        sfp[name] = tot
        print(f"stress {name} done ({time.time()-t0:.0f}s)", flush=True)
    stress = pd.DataFrame(srows)
    stress.to_csv(f"{args.out}/stress.csv", index=False)

    # ---- feature importance (held-out seed 999, d = 0.75)
    data = make_dataset(Config(n=N, d=0.75, seed=999)); c = candidates(data); yv = label(c, data.truth)
    Xv = c[FEATURE_SETS["full"]]
    base_ap, single = grouped_permutation_importance(models["full"], Xv, yv, {f: [f] for f in FEATURE_SETS["full"]})
    _, grouped = grouped_permutation_importance(models["full"], Xv, yv, {"amount_date": AMT_DATE, "vendor_name": NAME, "reference": REF})

    numbers = dict(
        protocol=dict(N=N, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS, difficulties=DIFFS,
                      train_pairs=int(len(Xall)), train_positive=int(yall.sum()), cascade_theta=theta,
                      cascade_theta_train_f1=theta_f1),
        example_d05_seed101=next(dict(bank=r["n_bank"], gl=r["n_gl"], true_pairs=r["n_true"], candidates=r["n_cand"])
                                 for r in main_runs if r["cfg"]["d"] == 0.5 and r["cfg"]["seed"] == TEST_SEEDS[0]),
        paired_f1_differences=paired, fp_categories_d_ge_05=fp, unmatched_item_errors_d_ge_05=unmatched,
        recall_by_cause_d_ge_05=cause, operations=ops, cascade_tiers_d_ge_05=tiers, monetary_d_ge_05=money,
        calibration_d_ge_05=calib, stress_fp_categories=sfp, stress_paired_f1=spaired,
        same_vendor_decoy_partner_d_ge_05=dict(svd),
        permutation_importance=dict(heldout="seed 999, d=0.75", base_ap=base_ap, single=single, grouped=grouped))
    with open(f"{args.out}/numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, sort_keys=True)
    env = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
               scikit_learn=sklearn.__version__, platform=platform.platform(),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED"),
               runtime_seconds=round(time.time() - t0, 1))
    with open(f"{args.out}/timing.json", "w") as f:
        json.dump(timing, f, indent=1)
    with open(f"{args.out}/environment.json", "w") as f:
        json.dump(env, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
