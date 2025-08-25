import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("data/processed/als")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# utilities
# ---------------------------

def load_meta():
    meta_path = DATA_DIR / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run preprocessing first.")
    meta = pd.read_csv(meta_path)
    if "track_id" not in meta.columns:
        raise ValueError("meta.csv must contain 'track_id'")
    return meta

def build_synthetic_interactions(meta: pd.DataFrame,
                                 n_users=2000,
                                 items_per_user=30,
                                 seed=42) -> pd.DataFrame:
    """
    Create a fake interactions table (user_id, track_id, count) so we can
    train and demo MF even when we don't have real user logs.
    Bias sampling by genre and popularity (if present).
    """
    rng = np.random.default_rng(seed)
    df = meta.copy()

    # weights for sampling
    pop = df["popularity"].to_numpy() if "popularity" in df.columns else np.ones(len(df))
    pop = np.nan_to_num(pop, nan=0.0)
    pop = pop - pop.min()
    pop = pop / (pop.max() + 1e-8)  # [0,1]
    w_base = 0.2 + 0.8 * pop        # avoid zeros

    genres = df["track_genre"].fillna("unknown") if "track_genre" in df.columns else pd.Series(["unknown"]*len(df))
    unique_genres = genres.unique()

    rows = []
    for u in range(n_users):
        # pick a "home" genre for the user
        g = rng.choice(unique_genres)
        mask = (genres == g).to_numpy()
        if not mask.any():  # fallback if empty
            mask = np.ones(len(df), dtype=bool)

        # genre-biased weights
        w = w_base.copy()
        w[mask] *= 2.0
        w = w / w.sum()

        # sample items (without replacement as much as possible)
        k = min(items_per_user, len(df))
        idxs = rng.choice(len(df), size=k, replace=False, p=w)
        # small random counts (1–3)
        cnts = rng.integers(1, 4, size=k)

        for it, c in zip(idxs, cnts):
            rows.append((f"U{u}", df.iloc[it]["track_id"], int(c)))

    interactions = pd.DataFrame(rows, columns=["user_id", "track_id", "count"])
    return interactions

def load_or_make_interactions(meta: pd.DataFrame, path_csv: str|None):
    if path_csv and Path(path_csv).exists():
        inter = pd.read_csv(path_csv)
    else:
        inter = build_synthetic_interactions(meta)
        synth_path = DATA_DIR / "interactions.synthetic.csv"
        inter.to_csv(synth_path, index=False)
        print(f"ℹ️  built synthetic interactions → {synth_path}")
    # clean
    inter = inter.dropna(subset=["user_id","track_id","count"])
    inter["count"] = inter["count"].astype(np.float32)
    return inter

def build_mappings(inter: pd.DataFrame, meta: pd.DataFrame):
    # only keep items that exist in meta
    valid = inter["track_id"].isin(meta["track_id"])
    inter = inter[valid].copy()

    users = inter["user_id"].astype(str).unique()
    items = meta["track_id"].astype(str).unique()

    u2i = {u:i for i,u in enumerate(users)}
    i2i = {t:i for i,t in enumerate(items)}

    inter["u_idx"] = inter["user_id"].map(u2i)
    inter["i_idx"] = inter["track_id"].map(i2i)

    return inter, u2i, i2i, users, items

def build_csr(inter_idxed: pd.DataFrame, n_users: int, n_items: int):
    rows = inter_idxed["u_idx"].to_numpy()
    cols = inter_idxed["i_idx"].to_numpy()
    data = inter_idxed["count"].to_numpy().astype(np.float32)
    R = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    return R

# ---------------------------
# ALS (explicit) with Cholesky and sparse @ dense
# ---------------------------

def als_explicit(R: sparse.csr_matrix, factors=64, reg=0.1, iters=10, seed=42):
    rng = np.random.default_rng(seed)
    m, n = R.shape  # users, items
    X = rng.normal(scale=0.01, size=(m, factors)).astype(np.float32)  # user factors
    Y = rng.normal(scale=0.01, size=(n, factors)).astype(np.float32)  # item factors

    I = np.eye(factors, dtype=np.float32) * reg

    for it in range(1, iters+1):
        # --- fix Y, update X ---
        YtY = (Y.T @ Y).astype(np.float32) + I
        cY, lower = cho_factor(YtY, lower=True, overwrite_a=True, check_finite=False)
        # RHS: B = R @ Y (sparse @ dense)  -> shape (m, f)
        B = R.dot(Y).astype(np.float32)
        # Solve (YtY)x = b for all rhs columns in B^T, then transpose back
        X = cho_solve((cY, lower), B.T, overwrite_b=False, check_finite=False).T.astype(np.float32)

        # --- fix X, update Y ---
        XtX = (X.T @ X).astype(np.float32) + I
        cX, lower = cho_factor(XtX, lower=True, overwrite_a=True, check_finite=False)
        # RHS: C = R^T @ X -> (n, f)
        C = R.T.dot(X).astype(np.float32)
        Y = cho_solve((cX, lower), C.T, overwrite_b=False, check_finite=False).T.astype(np.float32)

        # simple loss proxy (Frobenius on observed entries)
        # computing exact MSE over sparse entries is expensive; skip in tight loop
        if it % 2 == 0:
            print(f"[ALS] iter {it}/{iters} done")

    return X, Y

def save_model(X, Y, users, items, u2i, i2i):
    np.save(OUT_DIR / "als_user_factors.npy", X)
    np.save(OUT_DIR / "als_item_factors.npy", Y)
    dump({"users": users, "items": items, "u2i": u2i, "i2i": i2i}, OUT_DIR / "mappings.joblib")
    print(f"✅ saved factors to {OUT_DIR}")

def load_model():
    X = np.load(OUT_DIR / "als_user_factors.npy", mmap_mode="r")
    Y = np.load(OUT_DIR / "als_item_factors.npy", mmap_mode="r")
    mp = load(OUT_DIR / "mappings.joblib")
    return X, Y, mp

# ---------------------------
# recommend
# ---------------------------

def recommend_for_user(user_id: str, topk=10):
    meta = load_meta()
    X, Y, mp = load_model()

    if user_id not in mp["u2i"]:
        print("❌ unknown user_id"); return
    u = mp["u2i"][user_id]
    scores = X[u] @ Y.T  # (n_items,)
    # no history masking here because synthetic users may be sparse; remove negatives
    idx = np.argpartition(scores, -topk)[-topk:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    item_ids = [mp["items"][i] for i in idx]
    out = meta.set_index("track_id").loc[item_ids][["track_name","artists","track_genre"]].reset_index()
    out["score"] = scores[idx]
    print(out)

def recommend_similar_song(track_query: str, topk=10):
    meta = load_meta()
    X, Y, mp = load_model()

    # find by name (first match)
    m = meta["track_name"].fillna("").str.lower().str.contains(track_query.lower())
    if not m.any():
        print("❌ song not found in meta.csv"); return
    track_id = meta[m].iloc[0]["track_id"]
    if track_id not in mp["i2i"]:
        print("❌ song not in factor model items"); return

    i = mp["i2i"][track_id]
    q = Y[i]
    sims = Y @ q  # cosine-ish if normalized; here it's dot
    sims[i] = -np.inf
    idx = np.argpartition(sims, -topk)[-topk:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    item_ids = [mp["items"][j] for j in idx]

    out = meta.set_index("track_id").loc[item_ids][["track_name","artists","track_genre"]].reset_index()
    out["similarity"] = sims[idx]
    print(out)

# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=str, default=None,
                    help="CSV with columns: user_id,track_id,count. If missing, builds synthetic.")
    ap.add_argument("--fit", action="store_true", help="Train ALS and save factors.")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rec_user", type=str, default=None, help="Recommend for a given user_id (e.g., U0).")
    ap.add_argument("--rec_song", type=str, default=None, help="Recommend songs similar to this track name.")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    meta = load_meta()

    if args.fit:
        inter = load_or_make_interactions(meta, args.interactions)
        inter, u2i, i2i, users, items = build_mappings(inter, meta)
        R = build_csr(inter, n_users=len(users), n_items=len(items))
        print(f"ℹ️  training ALS on R shape={R.shape}, nnz={R.nnz}, rank={args.factors}")
        X, Y = als_explicit(R, factors=args.factors, reg=args.reg, iters=args.iters)
        save_model(X, Y, users, items, u2i, i2i)

    if args.rec_user:
        recommend_for_user(args.rec_user, args.topk)

    if args.rec_song:
        recommend_similar_song(args.rec_song, args.topk)

if __name__ == "__main__":
    main()
