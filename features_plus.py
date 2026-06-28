"""tabprep-style arithmetic feature expansion (leakage-free, row-wise).

Takes the 26 leakage-free base features from predict_local.build_features and adds:
  - squares of each base feature
  - all pairwise products  (interaction terms)
  - a curated set of domain ratios (relative strength)
Everything is a pure row-wise function of already-safe features -> no leakage,
no fitting needed. Returns (X DataFrame, feature_name list)."""
import itertools, numpy as np, pandas as pd
import predict_local as P

BASE = P.FEATURES

# curated ratios (relative strength); denom stabilized with +1 on |.| to avoid div0
RATIOS = [
    ("home_elo", "away_elo"), ("home_form5", "away_form5"), ("home_gf5", "away_ga5"),
    ("away_gf5", "home_ga5"), ("home_winrate", "away_winrate"), ("home_gf5", "home_ga5"),
    ("away_gf5", "away_ga5"), ("home_rest", "away_rest"), ("home_played", "away_played"),
    ("home_streak", "away_streak"), ("elo_diff", "importance"), ("form10_diff", "gd10_diff"),
]


# extra ratios when FIFA features are in the base set
FIFA_RATIOS = [
    ("home_fifa_pts", "away_fifa_pts"), ("home_fifa_rank", "away_fifa_rank"),
    ("fifa_pts_diff", "importance"), ("home_fifa_pts", "home_elo"),
]


def expand(feats: pd.DataFrame, base=None, ratios=None):
    base = BASE if base is None else base
    ratios = RATIOS if ratios is None else ratios
    cols = {}
    b = feats[base]
    # base
    for c in base:
        cols[c] = b[c].values
    # squares
    for c in base:
        cols[f"sq__{c}"] = b[c].values ** 2
    # pairwise products
    for c1, c2 in itertools.combinations(base, 2):
        cols[f"mul__{c1}__{c2}"] = b[c1].values * b[c2].values
    # curated ratios (skip any whose columns aren't in base)
    for c1, c2 in ratios:
        if c1 in base and c2 in base:
            cols[f"div__{c1}__{c2}"] = b[c1].values / (np.abs(b[c2].values) + 1.0)
    X = pd.DataFrame(cols, index=feats.index)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, list(X.columns)


if __name__ == "__main__":
    df = P.load_data()
    feats = P.build_features(df)
    X, names = expand(feats)
    print(f"base={len(BASE)}  expanded={len(names)} features")
    print("examples:", names[:5], "...", names[-3:])
