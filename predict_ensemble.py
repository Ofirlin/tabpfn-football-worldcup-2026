"""Predict upcoming fixtures with the LASSO-select top-K + TabPFN ensemble.

Pipeline (the tuned configuration from backtest_lasso_tabpfn.py, K=15):
  - expand base-26 -> 389 tabprep features
  - L1-logistic (LASSO) on all played matches = predictor + selects top-K by |coef|
  - TabPFN-2.5 trained on the top-K features
  - prediction = mean(LASSO proba, TabPFN proba)
Outputs predictions_ensemble_<date>.csv with H/D/A probabilities."""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
import predict_local as P, features_plus as FP

CLASSES = np.array(["away_win", "draw", "home_win"])
K = int(os.environ.get("ENS_K", 15))
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    df = P.load_data()
    feats = P.build_features(df)
    Xall, names = FP.expand(feats); names = np.array(names)
    feats = feats.reset_index(drop=True); Xall = Xall.reset_index(drop=True)

    played = feats["outcome"].notna()
    future = feats[feats["home_score"].isna() & (feats["date"] >= P.TODAY)].sort_values("date")
    print(f"device={DEV}  K={K}  played={played.sum()}  upcoming fixtures={len(future)}")
    if len(future) == 0:
        print("No upcoming fixtures (home_score NaN & date>=today)."); return

    pool = feats[played].tail(P.MAX_TRAIN)
    ytr = pool["outcome"].values
    Xtr = Xall.loc[pool.index].values
    Xfut = Xall.loc[future.index].values

    # LASSO: predictor + selector
    sc = StandardScaler().fit(Xtr)
    lasso = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=500, tol=1e-3,
                               n_jobs=-1).fit(sc.transform(Xtr), ytr)
    li = [list(lasso.classes_).index(c) for c in CLASSES]
    p_lasso = lasso.predict_proba(sc.transform(Xfut))[:, li]
    rank = np.argsort(-np.abs(lasso.coef_).sum(0))[:K]
    print(f"top-{K} selected features: {list(names[rank])}")

    # TabPFN on top-K raw features
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42, device=DEV,
                           model_path=os.environ.get("TABPFN_MODEL_PATH", "auto"))
    clf.fit(Xtr[:, rank], ytr)
    ti = [list(clf.classes_).index(c) for c in CLASSES]
    p_tab = clf.predict_proba(Xfut[:, rank])[:, ti]

    p = 0.5 * (p_lasso + p_tab)  # ensemble
    out = future[["date", "home_team", "away_team"]].copy()
    out["predicted"] = CLASSES[p.argmax(1)]
    out["p_home_win"] = p[:, 2]; out["p_draw"] = p[:, 1]; out["p_away_win"] = p[:, 0]
    fn = f"predictions_ensemble_{pd.Timestamp.now():%Y%m%d}.csv"
    out.to_csv(fn, index=False)
    print(f"\n{len(out)} predictions -> {fn}\n")
    for r in out.itertuples():
        print(f"  {r.date.date()}  {r.home_team:>22} vs {r.away_team:<22} -> {r.predicted:<9} "
              f"H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")


if __name__ == "__main__":
    main()
