# recon_bench — supplementary material

Code and results for *Rules, Learning, or Both? A Controlled Study of Bank-to-Ledger
Transaction Matching Under Noise and Distribution Shift* (A. Mirani).

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate     # Python 3.13
pip install -r requirements.txt
./run_all.sh                 # ~5-7 min on a laptop CPU; writes results/
python3 make_tables.py       # prints every table in the paper from results/
python3 verify_claims.py     # checks every number stated in the paper's text
```

`run_all.sh` writes `results/SHA256SUMS`. With the pinned versions, a rerun should match the
released file byte for byte. (Two runs, multi-threaded and single-threaded, were identical.)

## Files

| File | Contents |
| --- | --- |
| `recon_bench.py` | data generator, candidate generation, rule matchers (R1, R2, R3), learned matcher, assignment, hybrids, all experiments |
| `make_tables.py` | prints Tables II–VI and headline numbers from `results/` |
| `verify_claims.py` | asserts every number quoted in the paper's prose |
| `results/main_raw.csv` | per seed, difficulty and method: precision, recall, F1 |
| `results/main_summary.csv` | means, standard deviations and 95% CI half-widths |
| `results/stress.csv` | the five stress tests |
| `results/routing.csv` | threshold sweep: automated true-pair recall, record-level automation, review queue, false matches, monetary exposure (filtered and re-solved) |
| `results/recall_by_cause.csv` | recall by perturbation cause |
| `results/numbers.json` | every other reported figure (error categories, calibration, importance, operations) |
| `results/environment.json` | interpreter, library versions and runtime of the released run |

## Notes

* All amounts are integer cents; the fee feature is an exact integer test.
* Seeds: training 1–3, testing 101–110, feature importance on seed 999.
* No real bank, ledger or company data is used; all vendor names are fictitious.
