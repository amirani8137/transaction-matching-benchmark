#!/usr/bin/env python3
"""Checks every number stated in the paper's prose against results/. Exit code 1 if any claim fails."""
import json, sys
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
S = pd.read_csv(f"{R}/main_summary.csv").set_index(["difficulty", "method"])
ST = pd.read_csv(f"{R}/stress.csv").pivot_table(index=["test", "method"], columns="metric", values="mean")
RT = pd.read_csv(f"{R}/routing.csv")
F = RT[RT["mode"] == "filtered"].set_index("tau")
RC = RT[RT["mode"] == "recomputed"].set_index("tau")
N = json.load(open(f"{R}/numbers.json"))
raw = pd.read_csv(f"{R}/main_raw.csv")
fails = []


def check(label, value, stated, nd):
    ok = round(value, nd) == round(stated, nd)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: computed {value:.6g}, paper says {stated}")
    if not ok:
        fails.append(label)


def truth(label, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        fails.append(label)


f1 = lambda d, m: S.loc[(d, m)].f1_mean
pr = lambda d, m: S.loc[(d, m)].precision_mean
check("R1 F1 d=0", f1(0.0, "R1"), 0.857, 3); check("R1 F1 d=1", f1(1.0, "R1"), 0.239, 3)
check("R3 F1 d=1", f1(1.0, "R3"), 0.977, 3); check("R3 F1 d=0.5", f1(0.5, "R3"), 0.998, 3)
check("M F1 d=1", f1(1.0, "M_full"), 0.983, 3); check("M F1 d=0.5", f1(0.5, "M_full"), 0.997, 3)
check("R3 precision d=1", pr(1.0, "R3"), 0.993, 3); check("H precision d=1", pr(1.0, "H"), 0.996, 3)
check("R2 precision d=1", pr(1.0, "R2"), 0.866, 3); check("H3 F1 d=1", f1(1.0, "H3"), 0.990, 3)
others = ["R1", "R2", "R3", "M_no_text", "M_no_name", "M_no_ref", "M_full", "MG", "H", "M_er", "H_ER"]
truth("H3 best F1 for d >= 0.25", all(f1(d, "H3") >= max(f1(d, m) for m in others) for d in [0.25, 0.5, 0.75, 1.0]))
gap = max(f1(d, "M_full") - f1(d, "R3") for d in [0, .25, .5, .75, 1])
check("max learned-over-cascade F1 gap", gap, 0.006, 3)
truth("H3 precision at cascade level (within 0.0002)", all(abs(pr(d, "H3") - pr(d, "R3")) < 2e-4 for d in [0.5, 1.0]))
truth("H == M on all 50 main test sets", all(v["seeds_differing"] == 0 for v in N["paired_f1_differences"]["H-M_full"].values()))
check("candidate recall min (%)", 100 * min(o["cand_recall_min"] for o in N["operations"].values()), 99.17, 2)
fp = {k: sum(v.values()) for k, v in N["fp_categories_d_ge_05"].items()}
check("H false matches", fp["H"], 139, 0); check("H3 false matches", fp["H3"], 236, 0); check("H_ER false matches", fp["H_ER"], 144, 0)
check("H3 extra false matches (%)", 100 * (fp["H3"] / fp["H"] - 1), 70, 0)
c = N["fp_categories_d_ge_05"]["H"]
for k, v in [("decoy_same_vendor", 124), ("decoy_diff_vendor", 5), ("unmatched_bank", 4), ("gl_unmatched", 3), ("swap", 3)]:
    check(f"H FP {k}", c.get(k, 0), v, 0)
check("H_ER same-vendor errors", N["fp_categories_d_ge_05"]["H_ER"]["decoy_same_vendor"], 124, 0)
sv = N["same_vendor_decoy_partner_d_ge_05"]
check("same-vendor with partnerless ledger entry", c["decoy_same_vendor"] - sv["n"], 40, 0)
check("same-vendor displacing partner", sv["partner_left_unmatched"], 84, 0); check("displaced partners without reference", sv["partner_has_no_reference"], 83, 0)
u = N["unmatched_item_errors_d_ge_05"]["H"]
check("same-vendor wrong rate %", 100 * u["decoy_same_vendor"]["rate"], 26.2, 1); check("diff-vendor wrong rate %", 100 * u["decoy_diff_vendor"]["rate"], 0.17, 2)
check("H_ER mean F1 diff d=1", N["paired_f1_differences"]["H_ER-H"]["1.0"]["mean"], -0.0003, 4)
sp = N["stress_paired_f1"]
check("unfamiliar R3-H", sp["unfamiliar_names"]["R3-H"][0], 0.039, 3); check("unfamiliar R3-H CI", sp["unfamiliar_names"]["R3-H"][2], 0.002, 3)
check("unfamiliar H-M", sp["unfamiliar_names"]["H-M_full"][0], 0.034, 3); check("unfamiliar H-M CI", sp["unfamiliar_names"]["H-M_full"][2], 0.003, 3)
check("unseen+unfamiliar R3-H", sp["unseen_and_unfamiliar"]["R3-H"][0], 0.048, 3); check("its CI", sp["unseen_and_unfamiliar"]["R3-H"][2], 0.002, 3)
check("noise H F1", ST.loc[("noise_beyond_training", "H"), "f1"], 0.967, 3); check("noise R3 F1", ST.loc[("noise_beyond_training", "R3"), "f1"], 0.948, 3)
check("noise H-R3", -sp["noise_beyond_training"]["R3-H"][0], 0.019, 3); check("its CI", sp["noise_beyond_training"]["R3-H"][2], 0.002, 3)
check("noise R1 F1", ST.loc[("noise_beyond_training", "R1"), "f1"], 0.124, 3)
check("M recall unfamiliar", ST.loc[("unfamiliar_names", "M_full"), "recall"], 0.842, 3); check("M recall base", ST.loc[("base", "M_full"), "recall"], 0.995, 3)
check("cascade unfamiliar F1", ST.loc[("unfamiliar_names", "R3"), "f1"], 0.986, 3)
truth("unseen vendors / repeated amounts change F1 by <= 0.001", all(abs(ST.loc[(t, m), "f1"] - ST.loc[("base", m), "f1"]) <= 0.001
      for t in ["unseen_vendors", "monthly_recurring"] for m in ["R1", "R3", "M_full", "H", "H_ER"]))
check("weekly H F1", ST.loc[("weekly_recurring", "H"), "f1"], 0.988, 3); check("weekly H precision", ST.loc[("weekly_recurring", "H"), "precision"], 0.990, 3)
check("weekly R3 F1", ST.loc[("weekly_recurring", "R3"), "f1"], 0.994, 3)
check("weekly R3-H", sp["weekly_recurring"]["R3-H"][0], 0.006, 3); check("its CI", sp["weekly_recurring"]["R3-H"][2], 0.002, 3)
wk, bs = N["stress_fp_categories"]["weekly_recurring"]["H"], N["stress_fp_categories"]["base"]["H"]
check("weekly H false matches", sum(wk.values()), 191, 0); check("base H false matches (10 sets)", sum(bs.values()), 26, 0)
check("weekly H FP to partnerless ledger entries", wk["gl_unmatched"], 103, 0); check("weekly H swaps", wk["swap"], 45, 0)
truth("H == M under weekly series", sp["weekly_recurring"]["H-M_full"][0] == 0)
t = N["cascade_tiers_d_ge_05"]
check("T1+T2 pairs", t["T1 exact"]["pairs"] + t["T2 reference"]["pairs"], 22645, 0)
truth("T1 and T2 precision 1.0", t["T1 exact"]["precision"] == 1.0 and t["T2 reference"]["precision"] == 1.0)
check("T3 precision", t["T3 amount+name"]["precision"], 0.993, 3); check("T4 precision", t["T4 fee/drift+name"]["precision"], 0.987, 3)
g = N["permutation_importance"]
check("perm name", g["grouped"]["vendor_name"][0], 0.414, 3); check("perm amount/date", g["grouped"]["amount_date"][0], 0.353, 3)
check("perm reference", g["grouped"]["reference"][0], 0.035, 3); check("base AP", g["base_ap"], 0.998, 3)
mg = N["paired_f1_differences"]["MG-M_full"]
check("MG vs M test sets differing", sum(v["seeds_differing"] for v in mg.values()), 8, 0)
check("MG vs M max abs diff", max(max(abs(v["min"]), abs(v["max"])) for v in mg.values()), 0.0011, 4)
check("tau .5 automated true-pair recall %", 100 * F.loc[0.5, "automated_true_pair_recall"], 98.4, 1)
check("tau .5 record automation %", 100 * F.loc[0.5, "record_automation"], 91.9, 1)
check("tau .5 precision %", 100 * F.loc[0.5, "auto_precision"], 99.74, 2); check("tau .5 review queue", F.loc[0.5, "bank_review_queue"], 4780, 0)
check("tau .5 FP value $M", F.loc[0.5, "fp_value_usd"] / 1e6, 3.99, 2); check("tau .5 FP share of auto value %", 100 * F.loc[0.5, "fp_value_share"], 0.25, 2)
check("tau .99 avoided", F.loc[0.5, "false_auto_matches"] - F.loc[0.99, "false_auto_matches"], 75, 0)
check("tau .99 recall %", 100 * F.loc[0.99, "automated_true_pair_recall"], 73.0, 1)
extra = F.loc[0.99, "bank_review_queue"] - F.loc[0.5, "bank_review_queue"]
check("tau .99 extra review", extra, 14036, 0); check("records per avoided error", extra / 75, 187, 0)
check("tau .99 precision %", 100 * F.loc[0.99, "auto_precision"], 99.84, 2)
check("filtered vs recomputed max diff", max((F.false_auto_matches - RC.false_auto_matches).abs().max(), (F.bank_review_queue - RC.bank_review_queue).abs().max()), 1, 0)
m = N["monetary_d_ge_05"]
check("H value share %", 100 * m["H"]["share_of_bank_value"], 0.23, 2); check("R3 value share %", 100 * m["R3"]["share_of_bank_value"], 0.54, 2)
check("H3 value share %", 100 * m["H3"]["share_of_bank_value"], 0.54, 2); check("R2 value share %", 100 * m["R2"]["share_of_bank_value"], 6.3, 1)
truth("H3 exposure more than twice H", m["H3"]["fp_value_usd"] > 2 * m["H"]["fp_value_usd"])
cal = N["calibration_d_ge_05"]
check("candidate pairs", cal["candidate_pairs"], 559514, 0); check("Brier", cal["brier"], 0.0021, 4)
check("ECE assigned", cal["ece_assigned"], 0.014, 3); check("assigned pairs", cal["assigned_pairs"], 40312, 0)
mid = [b for b in cal["reliability"] if b["bin"] in ("0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5")]
n_mid = sum(b["pairs"] for b in mid)
check("pairs scored 0.1-0.5", n_mid, 797, 0); check("their match rate %", 100 * sum(b["pairs"] * b["match_rate"] for b in mid) / n_mid, 65, 0)
o = N["operations"]
check("R1 share of true pairs d=.5 %", 100 * o["0.5"]["r1_share_of_true_pairs"], 38.0, 1)
check("scoring avoided d=.5 %", 100 * o["0.5"]["scoring_avoided"], 57.3, 1); check("scored share d=.5 %", 100 * (1 - o["0.5"]["scoring_avoided"]), 42.7, 1)
check("scoring avoided d=1 %", 100 * o["1.0"]["scoring_avoided"], 22.2, 1)
p, e = N["protocol"], N["example_d05_seed101"]
check("train pairs", p["train_pairs"], 463193, 0); check("train positive", p["train_positive"], 27930, 0); check("theta", p["cascade_theta"], 0.10, 2)
for k, v in [("bank", 1988), ("gl", 2000), ("true_pairs", 1878), ("candidates", 31239)]:
    check(f"example {k}", e[k], v, 0)
T = json.load(open(f"{R}/timing.json"))  # wall-clock, machine-dependent: checked against the released run only
check("[timing] reduction d=.5 %", 100 * T["0.5"]["time_reduction"], 56, 0)
check("[timing] per-set min d=.5 %", 100 * T["0.5"]["per_set_time_reduction_min"], 52, 0); check("[timing] per-set max d=.5 %", 100 * T["0.5"]["per_set_time_reduction_max"], 60, 0)
check("[timing] reduction d=1 %", 100 * T["1.0"]["time_reduction"], 19, 0)
check("[timing] per-set min d=1 %", 100 * T["1.0"]["per_set_time_reduction_min"], 16, 0); check("[timing] per-set max d=1 %", 100 * T["1.0"]["per_set_time_reduction_max"], 23, 0)
print(f"\n{len(fails)} failed" if fails else "\nall claims match results/")
sys.exit(1 if fails else 0)
