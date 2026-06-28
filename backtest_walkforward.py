"""Walk-forward backtest: retrain each month on all prior matches, predict that month,
aggregate accuracy + log-loss over the whole held-out span. Compares to PriorLabs'
~59% / ~0.86 claim over 'held-out data' (vs. a single lucky calendar month)."""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tabpfn import TabPFNClassifier
import predict_local as P  # reuse load_data / build_features / FEATURES / MAX_TRAIN


def make_clf():
    return TabPFNClassifier(ignore_pretraining_limits=True, random_state=42,
                            device=os.environ.get("TABPFN_DEVICE", "auto"),
                            model_path=os.environ.get("TABPFN_MODEL_PATH", "auto"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01", help="first test month YYYY-MM")
    ap.add_argument("--end", default="2026-05", help="last test month YYYY-MM (inclusive)")
    ap.add_argument("--min-train", type=int, default=2000, help="skip months with fewer prior matches")
    args = ap.parse_args()

    df = P.load_data()
    feats = P.build_features(df)
    played = feats[feats["outcome"].notna()].copy()

    months = pd.period_range(args.start, args.end, freq="M")
    y_true, y_proba, dates, classes = [], [], [], None

    for m in months:
        test = played[(played["date"] >= m.start_time) & (played["date"] < (m + 1).start_time)]
        train = played[played["date"] < m.start_time].tail(P.MAX_TRAIN)
        if len(test) == 0 or len(train) < args.min_train:
            continue
        clf = make_clf()
        clf.fit(train[P.FEATURES].values, train["outcome"].values)
        classes = clf.classes_
        proba = clf.predict_proba(test[P.FEATURES].values)
        y_true.extend(test["outcome"].values)
        y_proba.extend(proba)
        dates.extend(test["date"].values)
        acc = accuracy_score(test["outcome"], classes[proba.argmax(1)])
        print(f"  {m}  n={len(test):4d}  train={len(train):5d}  acc={acc:.0%}", flush=True)

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    pred = classes[y_proba.argmax(1)]
    yr = pd.to_datetime(dates).year

    print(f"\n===== WALK-FORWARD BACKTEST {args.start}..{args.end} =====")
    print(f"Total held-out matches: {len(y_true)}")
    print(f"Overall accuracy : {accuracy_score(y_true, pred):.1%}")
    print(f"Overall log-loss : {log_loss(y_true, y_proba, labels=classes):.3f}")

    # sanity baselines
    home = np.full(len(y_true), "home_win")
    print(f"\nBaseline always-home accuracy: {accuracy_score(y_true, home):.1%}")
    base = pd.Series(y_true).value_counts(normalize=True)
    print("Class distribution (held-out):", {k: round(v, 3) for k, v in base.items()})

    print("\nPer-year:")
    for y in sorted(set(yr)):
        mask = yr == y
        ll = log_loss(y_true[mask], y_proba[mask], labels=classes)
        print(f"  {y}: n={mask.sum():4d}  acc={accuracy_score(y_true[mask], pred[mask]):.1%}  log-loss={ll:.3f}")


if __name__ == "__main__":
    main()
