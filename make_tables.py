#!/usr/bin/env python3
"""Prints the paper's Tables II-VI (markdown, numbered as in the manuscript) and headline numbers from results/ -- no numbers are typed by hand."""
import json, sys
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
S = pd.read_csv(f"{R}/main_summary.csv").set_index(["difficulty", "method"])
ST = pd.read_csv(f"{R}/stress.csv")
RT = pd.read_csv(f"{R}/routing.csv")
N = json.load(open(f"{R}/numbers.json"))
f3 = lambda x: f"{x:.3f}"
pct = lambda x, k=1: f"{100 * x:.{k}f}%"

NAMES = {"R1": "R1 exact rules", "R2": "R2 relaxed rules", "R3": "R3 rule cascade", "M_full": "M learned + assignment",
         "H": "H exact rules → learned", "H3": "H3 cascade → learned", "MG": "MG learned + greedy",
         "M_no_text": "M without name and reference", "M_no_name": "M without vendor name", "M_no_ref": "M without reference",
         "H_ER": "H + counterparty resolution"}


def cell(d, m):
    r = S.loc[(d, m)]
    return f"{f3(r.precision_mean)} / {f3(r.recall_mean)} / {f3(r.f1_mean)} ± {r.f1_ci95:.4f}"


print("TABLE II  (P / R / F1 ± 95% CI half-width)")
print("| Method | d = 0.5 | d = 1.0 |\n| --- | --- | --- |")
for m in ["R1", "R2", "R3", "M_full", "H", "H3"]:
    print(f"| {NAMES[m]} | {cell(0.5, m)} | {cell(1.0, m)} |")

print("\nTABLE III recall by cause, d >= 0.5")
print("| Cause | True pairs | R1 | R3 | H |\n| --- | --- | --- | --- | --- |")
lab = {"no_ref": "Missing reference", "clean": "Clean (no perturbation)", "lag>3": "Posting lag over 3 days",
       "fee": "Bank fee", "fx": "FX / rounding drift", "trunc_ref": "Truncated reference"}
for k, v in sorted(N["recall_by_cause_d_ge_05"].items(), key=lambda kv: -kv[1]["pairs"]):
    print(f"| {lab[k]} | {v['pairs']:,} | {pct(v['R1'])} | {pct(v['R3'])} | {pct(v['H'])} |")

print("\nTABLE IV stress tests at d = 0.5 (F1; recall in parentheses)")
TESTS = [("base", "Training conditions"), ("unseen_vendors", "Unseen vendors"), ("unfamiliar_names", "Unfamiliar name forms"),
         ("unseen_and_unfamiliar", "Unseen vendors + unfamiliar names"), ("monthly_recurring", "Monthly same-vendor, same-amount series"),
         ("weekly_recurring", "Weekly same-vendor, same-amount series"),
         ("noise_beyond_training", "Noise beyond training (d = 1.25)")]
piv = ST.pivot_table(index=["test", "method"], columns="metric", values="mean")
print("| Test | R3 rule cascade | M learned | H exact rules → learned |\n| --- | --- | --- | --- |")
for t, name in TESTS:
    cells = [f"{f3(piv.loc[(t, m), 'f1'])} ({f3(piv.loc[(t, m), 'recall'])})" for m in ["R3", "M_full", "H"]]
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\nTABLE V ablations (F1 ± CI)")
print("| Variant | d = 0.5 | d = 1.0 |\n| --- | --- | --- |")
for m in ["M_full", "M_no_ref", "M_no_name", "M_no_text", "MG", "H_ER"]:
    a, b = S.loc[(0.5, m)], S.loc[(1.0, m)]
    print(f"| {NAMES[m]} | {f3(a.f1_mean)} ± {a.f1_ci95:.4f} | {f3(b.f1_mean)} ± {b.f1_ci95:.4f} |")

print("\nTABLE VI partnerless items wrongly matched by H, d >= 0.5")
print("| Item with no true partner | Items | Wrongly matched | Rate |\n| --- | --- | --- | --- |")
lab2 = {"decoy_same_vendor": "Same-vendor, same-amount bank record", "decoy_diff_vendor": "Different-vendor, same-amount bank record",
        "unmatched_bank": "Bank record, different amount", "gl_unmatched": "Ledger entry, no bank movement"}
for k in ["decoy_same_vendor", "decoy_diff_vendor", "unmatched_bank", "gl_unmatched"]:
    v = N["unmatched_item_errors_d_ge_05"]["H"][k]
    print(f"| {lab2[k]} | {v['items']:,} | {v['wrongly_matched']} | {pct(v['rate'], 2)} |")

print("\nHEADLINES")
fp = N["fp_categories_d_ge_05"]
print("FP totals d>=0.5:", {k: sum(v.values()) for k, v in fp.items()}, "H:", fp["H"])
print("same-vendor partner:", N["same_vendor_decoy_partner_d_ge_05"])
f = RT[RT["mode"] == "filtered"].set_index("tau")
rc = RT[RT["mode"] == "recomputed"].set_index("tau")
print("routing filtered:\n", f[["auto_precision", "automated_true_pair_recall", "record_automation", "bank_review_queue", "ledger_review_queue",
                                 "false_auto_matches", "fp_value_usd", "fp_value_share"]].round(4).to_string())
print("max |filtered - recomputed| false matches:", int((f.false_auto_matches - rc.false_auto_matches).abs().max()),
      "review records:", int((f.bank_review_queue - rc.bank_review_queue).abs().max()))
print("monetary:", N["monetary_d_ge_05"])
print("ops:", {d: {k: round(v, 4) for k, v in o.items()} for d, o in N["operations"].items()})
print("tiers:", N["cascade_tiers_d_ge_05"])
c = N["calibration_d_ge_05"]
print("calibration:", {k: v for k, v in c.items() if k != "reliability"})
for r in c["reliability"]:
    print("  ", r)
pi = N["permutation_importance"]
print("perm grouped:", pi["grouped"]); print("perm single top:", sorted(pi["single"].items(), key=lambda kv: -kv[1][0])[:4])
print("paired:", json.dumps(N["paired_f1_differences"]))
print("stress paired:", json.dumps(N["stress_paired_f1"]))
print("protocol:", N["protocol"], N["example_d05_seed101"])
print("R1 F1 by d:", [round(S.loc[(d, 'R1')].f1_mean, 4) for d in [0, .25, .5, .75, 1]],
      "H F1 by d:", [round(S.loc[(d, 'H')].f1_mean, 4) for d in [0, .25, .5, .75, 1]])
