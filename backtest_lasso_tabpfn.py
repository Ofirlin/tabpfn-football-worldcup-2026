"""Walk-forward: tabprep-style expanded features -> L1-logistic (LASSO) selects top-K
-> TabPFN trained on only those K features -> ensemble (mean proba) of LASSO + TabPFN.
K is tuned on a validation span (<= --val-end) by ensemble log-loss; final metrics are
reported on the held-out test span (> --val-end). Also prints the full per-K curve."""
import argparse, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
import predict_local as P
import features_plus as FP

CLASSES = np.array(["away_win", "draw", "home_win"])
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = [5, 10, 15, 20, 30, 40, 60]


def tabpfn_proba(Xtr, ytr, Xte):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42, device=DEV,
                           model_path=os.environ.get("TABPFN_MODEL_PATH", "auto"))
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)
    idx = [list(clf.classes_).index(c) for c in CLASSES]
    return pr[:, idx]


def metrics(y, pr):
    return accuracy_score(y, CLASSES[pr.argmax(1)]), log_loss(y, pr, labels=CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01")
    ap.add_argument("--val-end", default="2022-12", help="last month used for tuning K")
    ap.add_argument("--end", default="2026-05")
    ap.add_argument("--min-train", type=int, default=2000)
    args = ap.parse_args()
    print(f"device={DEV} span={args.start}..{args.end} val<= {args.val_end} grid={GRID}", flush=True)

    df = P.load_data()
    feats = P.build_features(df)
    Xall, names = FP.expand(feats)
    names = np.array(names)
    feats = feats.reset_index(drop=True)
    Xall = Xall.reset_index(drop=True)
    played_mask = feats["outcome"].notna().values

    months = pd.period_range(args.start, args.end, freq="M")
    rec = {"date": [], "y": [], "LASSO": [], "TabPFN_base": []}
    for K in GRID:
        rec[f"TabPFN_K{K}"] = []
        rec[f"Ens_K{K}"] = []

    for m in months:
        in_test = played_mask & (feats["date"] >= m.start_time).values & (feats["date"] < (m + 1).start_time).values
        in_train = played_mask & (feats["date"] < m.start_time).values
        ntr = in_train.sum()
        if in_test.sum() == 0 or ntr < args.min_train:
            continue
        tr = np.where(in_train)[0][-P.MAX_TRAIN:]
        te = np.where(in_test)[0]
        ytr = feats["outcome"].values[tr]
        yte = feats["outcome"].values[te]
        Xtr_e, Xte_e = Xall.values[tr], Xall.values[te]

        # LASSO on all expanded features (scaled): predictor + selector
        sc = StandardScaler().fit(Xtr_e)
        lasso = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=300,
                                   tol=1e-3, n_jobs=-1).fit(sc.transform(Xtr_e), ytr)
        idx = [list(lasso.classes_).index(c) for c in CLASSES]
        p_lasso = lasso.predict_proba(sc.transform(Xte_e))[:, idx]
        rank = np.argsort(-np.abs(lasso.coef_).sum(0))  # by aggregated |coef|

        # TabPFN reference on base-26 raw features
        base_idx = [list(names).index(c) for c in P.FEATURES]
        p_base = tabpfn_proba(Xtr_e[:, base_idx], ytr, Xte_e[:, base_idx])

        rec["date"].append(feats["date"].values[te]); rec["y"].append(yte)
        rec["LASSO"].append(p_lasso); rec["TabPFN_base"].append(p_base)

        line = [f"{m} n={len(te):4d}"]
        for K in GRID:
            sel = rank[:K]
            p_tab = tabpfn_proba(Xtr_e[:, sel], ytr, Xte_e[:, sel])
            p_ens = 0.5 * (p_lasso + p_tab)
            rec[f"TabPFN_K{K}"].append(p_tab); rec[f"Ens_K{K}"].append(p_ens)
            line.append(f"E{K}={accuracy_score(yte, CLASSES[p_ens.argmax(1)]):.0%}")
        print(" ".join(line), flush=True)

    # assemble
    date = pd.to_datetime(np.concatenate(rec["date"]))
    y = np.concatenate(rec["y"])
    P_ = {k: np.concatenate(v) for k, v in rec.items() if k not in ("date", "y")}
    val = np.asarray(date <= pd.Timestamp(args.val_end) + pd.offsets.MonthEnd(0))
    tst = ~val

    print(f"\n=== n_total={len(y)}  n_val={val.sum()}  n_test={tst.sum()} ===")
    print("\nFULL-SPAN per-K (LASSO is K-independent):")
    a, l = metrics(y, P_["LASSO"]); print(f"  LASSO(all 389)      acc={a:.1%} ll={l:.3f}")
    a, l = metrics(y, P_["TabPFN_base"]); print(f"  TabPFN(base 26)     acc={a:.1%} ll={l:.3f}")
    for K in GRID:
        at, lt = metrics(y, P_[f"TabPFN_K{K}"]); ae, le = metrics(y, P_[f"Ens_K{K}"])
        print(f"  K={K:3d}  TabPFN acc={at:.1%} ll={lt:.3f}   Ensemble acc={ae:.1%} ll={le:.3f}")

    # tune K on val by ensemble log-loss
    Kstar = min(GRID, key=lambda K: log_loss(y[val], P_[f"Ens_K{K}"][val], labels=CLASSES))
    print(f"\n>>> Selected K*={Kstar} (min val ensemble log-loss)")
    print(f"--- HELD-OUT TEST ({args.val_end}+1 .. {args.end}, n={tst.sum()}) ---")
    for name, pr in [("LASSO(all 389)", P_["LASSO"]), ("TabPFN(base 26)", P_["TabPFN_base"]),
                     (f"TabPFN(top-{Kstar})", P_[f"TabPFN_K{Kstar}"]),
                     (f"Ensemble(top-{Kstar})", P_[f"Ens_K{Kstar}"])]:
        a, l = metrics(y[tst], pr[tst]); print(f"  {name:<20} acc={a:.1%} ll={l:.3f}")


if __name__ == "__main__":
    main()
