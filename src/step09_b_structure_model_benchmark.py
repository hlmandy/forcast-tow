"""Step 09 — Leak-free structural benchmark for Task B.

Compares direct migration-flow prediction against stock x transition-probability
structures (source_present and predicted source_linkable parameterizations),
under a strict time-boundary rule: a B label at hour t depends on the same
vessel's state at t+1, so each outer training period's last day 23:00 migration
label is excluded from training (it would leak the validation day-0 state), and
2018-01-24 23:00 is right-censored (score_available=False, y_true blank).

16 methods: 5 direct, 4 present-transition, 5 linkable-transition (all
deployable, uniform training-date weight 1, no A quality weights), and 2 oracle
diagnostics (deployable=False). No Ridge/Poisson/trees/NN, no submission, no A.

Outputs (under outputs/step09_b_structure_model_benchmark/):
  1. method_definitions.csv      (16)
  2. boundary_audit.csv          (9)
  3. b_model_predictions.csv     (142848)
  4. b_fold_scores.csv           (144)
  5. b_direction_scores.csv      (864)
  6. b_target_regime_scores.csv  (56)
  7. b_structure_diagnostics.csv (297)
  8. decision_table.csv          (16)
  9. summary.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STEP01_DAILY = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02_DAILY = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
OUT_REL = Path("outputs/step09_b_structure_model_benchmark")

REGION_ORDER = ["core", "near", "outer"]
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}
DIRS = [("core", "near"), ("core", "outer"), ("near", "core"), ("near", "outer"), ("outer", "core"), ("outer", "near")]
MIG = {"core": ("near", "outer"), "near": ("core", "outer"), "outer": ("core", "near")}
DIRECTION_NAMES = [f"{s}->{t}" for s, t in DIRS]
UNSCORABLE_HOUR = pd.Timestamp("2018-01-24 23:00:00")
TAU = 20

# name, family, exposure, stock_window, prob_level, tau, uses_vessel, uses_val_exposure, deployable
METHODS = [
    ("pair_global_mean", "direct", "none", "none", "global", 0, False, False, True),
    ("pair_hour_mean", "direct", "none", "none", "hour", 0, False, False, True),
    ("pair_recent7_hour", "direct", "none", "recent7", "hour", 0, False, False, True),
    ("pair_recent14_hour", "direct", "none", "recent14", "hour", 0, False, False, True),
    ("pair_hour_mean_x085", "direct", "none", "none", "hour", 0, False, False, True),
    ("present_global_prob", "present_transition", "predicted_present", "long", "global", 0, False, False, True),
    ("present_hour_prob_raw", "present_transition", "predicted_present", "long", "hour", 0, False, False, True),
    ("present_hour_prob_shrunk20", "present_transition", "predicted_present", "long", "hour", TAU, False, False, True),
    ("present_recent7_stock_shrunk20", "present_transition", "predicted_present", "recent7", "hour", TAU, False, False, True),
    ("linkable_global_prob", "linkable_transition", "predicted_linkable", "long", "global", 0, False, False, True),
    ("linkable_hour_prob_raw", "linkable_transition", "predicted_linkable", "long", "hour", 0, False, False, True),
    ("linkable_hour_prob_shrunk20", "linkable_transition", "predicted_linkable", "long", "hour", TAU, False, False, True),
    ("linkable_recent7_stock_shrunk20", "linkable_transition", "predicted_linkable", "recent7", "hour", TAU, False, False, True),
    ("linkable_hour_prob_shrunk20_vessel_p05", "linkable_transition", "predicted_linkable", "long", "hour", TAU, True, False, True),
    ("oracle_present_hour_prob_shrunk20", "present_transition", "oracle_present", "long", "hour", TAU, False, True, False),
    ("oracle_linkable_hour_prob_shrunk20", "linkable_transition", "oracle_linkable", "long", "hour", TAU, False, True, False),
]
METHOD_ORDER = {m[0]: i for i, m in enumerate(METHODS)}
DEPLOYABLE = [m for m in METHODS if m[8]]
STRUCT_METHODS = [m[0] for m in METHODS if m[1] in ("present_transition", "linkable_transition")]

PRED_COLUMNS = [
    "evaluation_id", "evaluation_type", "method", "deployable", "date", "hour", "hour_of_day", "dayofweek",
    "day_type", "source_region", "target_region", "task_key", "audit_quality_regime", "period",
    "validation_unique_vessel_count", "score_available", "y_true",
    "predicted_source_present", "predicted_link_rate", "predicted_source_linkable",
    "predicted_transition_probability", "pred_float", "pred_rounded", "error_float", "error_rounded",
]
BOUNDARY_COLUMNS = [
    "evaluation_id", "evaluation_type", "train_start", "train_end", "excluded_train_boundary_hour",
    "excluded_train_transition_rows", "valid_start", "valid_end", "validation_end_hour",
    "validation_end_transition_observable", "unscored_validation_rows", "scored_validation_rows",
]
FOLD_SCORE_COLUMNS = [
    "evaluation_id", "evaluation_type", "method", "deployable", "train_start", "train_end", "valid_start", "valid_end",
    "n_prediction_rows", "n_scored_rows", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded",
    "actual_total_scored", "predicted_total_scored",
    "core_to_near_sse", "core_to_outer_sse", "near_to_core_sse", "near_to_outer_sse", "outer_to_core_sse", "outer_to_near_sse",
]
DIR_SCORE_COLUMNS = [
    "evaluation_id", "evaluation_type", "method", "deployable", "task_key", "n_scored_rows",
    "actual_total", "predicted_total", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded",
    "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate",
]
REGIME_COLUMNS = ["method", "audit_quality_regime", "n_prediction_rows", "n_unique_target_dates",
                  "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
DIAG_COLUMNS = [
    "evaluation_id", "evaluation_type", "method", "deployable", "source_region", "n_scored_source_hours",
    "true_present_mean", "predicted_present_mean", "present_mae",
    "true_linkable_mean", "predicted_linkable_mean", "linkable_mae",
    "true_outflow_total", "predicted_outflow_total", "outflow_sse_float",
    "mean_true_outflow_rate_present", "mean_predicted_outflow_rate_present",
    "mean_true_outflow_rate_linkable", "mean_predicted_outflow_rate_linkable",
    "max_float_constraint_violation",
]
DECISION_COLUMNS = [
    "method", "deployable", "model_family", "exposure_type",
    "rolling_mean_mse", "rolling_median_mse", "rolling_max_mse", "rolling_mean_sse",
    "final_analog_sse", "final_analog_mse", "final_analog_mean_bias", "final_analog_rank_deployable",
    "rolling_fold_win_count",
    "core_to_near_total_sse", "core_to_outer_total_sse", "near_to_core_total_sse", "near_to_outer_total_sse",
    "outer_to_core_total_sse", "outer_to_near_total_sse",
]
METHOD_DEF_COLUMNS = [
    "method", "model_family", "exposure_type", "stock_window", "probability_level",
    "shrinkage_tau", "uses_daily_vessel_count", "uses_validation_exposure", "deployable",
]

RUN_CMD = "python " + " ".join(sys.argv)


def fall(*vals):
    for v in vals:
        if v is not None:
            return v
    return 0.0


def build_scenarios():
    sc = []
    for k in range(8):
        te = pd.Timestamp(f"2018-01-{10 + k:02d}")
        vs_ = pd.Timestamp(f"2018-01-{11 + k:02d}")
        ve = pd.Timestamp(f"2018-01-{17 + k:02d}")
        sc.append((f"fold_{k + 1:02d}", "rolling_7d", pd.Timestamp("2018-01-01"), te, vs_, ve))
    sc.append(("final_analog", "final_analog", pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
               pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")))
    return sc


def compute_aggregates(ss_all, pf_all, train_start, train_end):
    """Per-scenario training aggregates. Stock uses all training hours; transitions exclude train_end 23:00."""
    train_dates = pd.date_range(train_start, train_end, freq="D").normalize()
    boundary = train_end + pd.Timedelta(hours=23)
    recent7_start = train_end - pd.Timedelta(days=6)
    recent14_start = max(train_start, train_end - pd.Timedelta(days=13))

    stock = ss_all[ss_all["date"].isin(train_dates)].copy()
    trans = stock[stock["hour"] != boundary].copy()
    r7stock = stock[stock["date"].between(recent7_start, train_end)]

    # stock means (per source, hod)
    g = stock.groupby(["source_region", "hour_of_day"])["source_present_count"].mean()
    stock_long = {(s, int(h)): float(g.loc[(s, h)]) for (s, h) in g.index}
    g7 = r7stock.groupby(["source_region", "hour_of_day"])["source_present_count"].mean()
    stock_recent7 = {(s, int(h)): float(g7.loc[(s, h)]) for (s, h) in g7.index}

    # present 4-class counts (per source,hour and global)
    n4 = {}
    g4 = {s: {} for s in REGION_ORDER}
    for s in REGION_ORDER:
        m1, m2 = MIG[s]
        sub = trans[trans["source_region"] == s]
        gh = sub.groupby("hour_of_day")
        n4[s] = {}
        for h, d in gh:
            n4[s][int(h)] = {"stay": float(d["stay_count"].sum()), m1: float(d[f"to_{m1}_count"].sum()),
                            m2: float(d[f"to_{m2}_count"].sum()), "noconsec": float(d["no_consecutive_count"].sum())}
        gs = sub["stay_count"].sum(), sub[f"to_{m1}_count"].sum(), sub[f"to_{m2}_count"].sum(), sub["no_consecutive_count"].sum()
        g4[s] = {"stay": float(gs[0]), m1: float(gs[1]), m2: float(gs[2]), "noconsec": float(gs[3])}

    def norm4(d):
        tot = sum(d.values())
        return {k: (v / tot if tot > 0 else 0.0) for k, v in d.items()}, tot

    g4p = {s: norm4(g4[s])[0] for s in REGION_ORDER}
    n4raw = {}
    n4shrunk = {}
    for s in REGION_ORDER:
        m1, m2 = MIG[s]
        n4raw[s] = {}
        n4shrunk[s] = {}
        for h, d in n4[s].items():
            p, tot = norm4(d)
            n4raw[s][h] = p if tot > 0 else None
            shr = {}
            for k in ["stay", m1, m2, "noconsec"]:
                shr[k] = (d[k] + TAU * g4p[s][k]) / (tot + TAU)
            n4shrunk[s][h] = shr

    # linkable: P, L, C per (source,hour); global link rate and dest prob
    P_link = {}
    L_link = {}
    C_link = {}
    for s in REGION_ORDER:
        sub = trans[trans["source_region"] == s]
        gh = sub.groupby("hour_of_day")
        P_link[s] = {}
        L_link[s] = {}
        C_link[s] = {}
        for h, d in gh:
            P_link[s][int(h)] = float(d["source_present_count"].sum())
            L_link[s][int(h)] = float(d["source_linkable_count"].sum())
            C_link[s][int(h)] = {t: float(d[f"to_{t}_count"].sum()) for t in REGION_ORDER}
    gl_lr = {}
    gdest = {}
    for s in REGION_ORDER:
        sub = trans[trans["source_region"] == s]
        Ptot = float(sub["source_present_count"].sum())
        Ltot = float(sub["source_linkable_count"].sum())
        gl_lr[s] = Ltot / Ptot if Ptot > 0 else 0.0
        ctot = {t: float(sub[f"to_{t}_count"].sum()) for t in REGION_ORDER}
        gdest[s] = {t: (ctot[t] / Ltot if Ltot > 0 else 1.0 / 3) for t in REGION_ORDER}
    linkraw = {s: {} for s in REGION_ORDER}
    destraw = {s: {} for s in REGION_ORDER}
    linkshr = {s: {} for s in REGION_ORDER}
    destshr = {s: {} for s in REGION_ORDER}
    for s in REGION_ORDER:
        for h in range(24):
            P = P_link[s].get(h, 0.0)
            L = L_link[s].get(h, 0.0)
            C = C_link[s].get(h, {t: 0.0 for t in REGION_ORDER})
            linkraw[s][h] = (L / P) if P > 0 else None
            destraw[s][h] = {t: (C[t] / L if L > 0 else None) for t in REGION_ORDER}
            linkshr[s][h] = (L + TAU * gl_lr[s]) / (P + TAU)
            dsh = {}
            for t in REGION_ORDER:
                dsh[t] = (C[t] + TAU * gdest[s][t]) / (L + TAU)
            destshr[s][h] = dsh
            # explicit probability/range assertions
            assert abs(sum(n4shrunk[s][h].values()) - 1.0) < 1e-9, f"present 4-class prob != 1 {s}/{h}"
            assert abs(sum(destshr[s][h].values()) - 1.0) < 1e-9, f"linkable dest prob != 1 {s}/{h}"
            assert 0.0 <= linkshr[s][h] <= 1.0 + 1e-9, f"link_rate out of [0,1] {s}/{h}"
    for s in REGION_ORDER:
        assert abs(sum(g4p[s].values()) - 1.0) < 1e-9, f"global present prob != 1 {s}"
        assert abs(sum(gdest[s].values()) - 1.0) < 1e-9, f"global dest prob != 1 {s}"
        assert 0.0 <= gl_lr[s] <= 1.0 + 1e-9, f"global link_rate out of [0,1] {s}"

    # pair means (exclude boundary)
    pf_train = pf_all[(pf_all["date"].isin(train_dates)) & (pf_all["hour"] != boundary)]
    pair_hour = {}
    for (s, t, h), d in pf_train.groupby(["source_region", "target_region", "hour_of_day"]):
        pair_hour[(s, t, int(h))] = float(d["y"].mean())
    pair_dir = {k: float(v) for k, v in pf_train.groupby(["source_region", "target_region"])["y"].mean().items()}
    pair_overall = float(pf_train["y"].mean())
    pf7 = pf_all[(pf_all["date"].between(recent7_start, train_end)) & (pf_all["hour"] != boundary)]
    pair7 = { (s, t, int(h)): float(d["y"].mean()) for (s, t, h), d in pf7.groupby(["source_region", "target_region", "hour_of_day"])}
    pf14 = pf_all[(pf_all["date"].between(recent14_start, train_end)) & (pf_all["hour"] != boundary)]
    pair14 = { (s, t, int(h)): float(d["y"].mean()) for (s, t, h), d in pf14.groupby(["source_region", "target_region", "hour_of_day"])}

    return dict(stock_long=stock_long, stock_recent7=stock_recent7, n4raw=n4raw, n4shrunk=n4shrunk, g4p=g4p,
                gl_lr=gl_lr, gdest=gdest, linkraw=linkraw, destraw=destraw, linkshr=linkshr, destshr=destshr,
                pair_hour=pair_hour, pair_dir=pair_dir, pair_overall=pair_overall, pair7=pair7, pair14=pair14,
                train_dates=train_dates, boundary=boundary)


def predict_method(meta, valid, agg, true_present, true_linkable, n_train_mean):
    mname = meta[0]
    vs = valid["source_region"].to_numpy()
    vt = valid["target_region"].to_numpy()
    vh = valid["hour_of_day"].to_numpy()
    vhour = valid["hour"].to_numpy()
    vvc = valid["daily_unique_vessel_count"].to_numpy()
    n = len(vs)
    pred = np.zeros(n)
    pp = np.full(n, np.nan)
    plr = np.full(n, np.nan)
    plk = np.full(n, np.nan)
    pprob = np.full(n, np.nan)

    if mname == "pair_global_mean":
        for i in range(n): pred[i] = fall(agg["pair_dir"].get((vs[i], vt[i])), agg["pair_overall"])
    elif mname == "pair_hour_mean":
        for i in range(n): pred[i] = fall(agg["pair_hour"].get((vs[i], vt[i], int(vh[i]))), agg["pair_dir"].get((vs[i], vt[i])), agg["pair_overall"])
    elif mname == "pair_recent7_hour":
        for i in range(n): pred[i] = fall(agg["pair7"].get((vs[i], vt[i], int(vh[i]))), agg["pair_hour"].get((vs[i], vt[i], int(vh[i]))), agg["pair_dir"].get((vs[i], vt[i])), agg["pair_overall"])
    elif mname == "pair_recent14_hour":
        for i in range(n): pred[i] = fall(agg["pair14"].get((vs[i], vt[i], int(vh[i]))), agg["pair_hour"].get((vs[i], vt[i], int(vh[i]))), agg["pair_dir"].get((vs[i], vt[i])), agg["pair_overall"])
    elif mname == "pair_hour_mean_x085":
        for i in range(n): pred[i] = 0.85 * fall(agg["pair_hour"].get((vs[i], vt[i], int(vh[i]))), agg["pair_dir"].get((vs[i], vt[i])), agg["pair_overall"])
    else:
        # structure methods
        for i in range(n):
            s = vs[i]; t = vt[i]; h = int(vh[i])
            m1, m2 = MIG[s]
            # predicted present
            if mname in ("present_recent7_stock_shrunk20", "linkable_recent7_stock_shrunk20"):
                present = fall(agg["stock_recent7"].get((s, h)), agg["stock_long"].get((s, h)))
            elif mname == "linkable_hour_prob_shrunk20_vessel_p05":
                base = fall(agg["stock_long"].get((s, h)))
                present = base * ((vvc[i] / n_train_mean) ** 0.5) if n_train_mean > 0 else base
            elif mname == "oracle_present_hour_prob_shrunk20":
                present = true_present.get((pd.Timestamp(vhour[i]), s), 0.0)
            else:
                present = fall(agg["stock_long"].get((s, h)))
            pp[i] = present
            if meta[1] == "present_transition":
                # outcome prob for target
                if mname == "present_global_prob":
                    prob = agg["g4p"][s].get(t, 0.0)
                elif mname in ("present_hour_prob_raw",):
                    prob = fall((agg["n4raw"][s].get(h) or {}).get(t), agg["g4p"][s].get(t))
                else:  # shrunk20 (incl recent7 stock, oracle)
                    prob = (agg["n4shrunk"][s].get(h) or agg["g4p"][s]).get(t, 0.0)
                pprob[i] = prob
                pred[i] = present * prob
            else:  # linkable_transition
                if mname == "linkable_global_prob":
                    lr = agg["gl_lr"][s]; dprob = agg["gdest"][s].get(t, 0.0)
                elif mname == "linkable_hour_prob_raw":
                    lr = fall(agg["linkraw"][s].get(h), agg["gl_lr"][s])
                    dprob = fall((agg["destraw"][s].get(h) or {}).get(t), agg["gdest"][s].get(t))
                else:  # shrunk20 (incl recent7, vessel, oracle)
                    lr = agg["linkshr"][s].get(h, agg["gl_lr"][s])
                    dprob = (agg["destshr"][s].get(h) or agg["gdest"][s]).get(t, 0.0)
                linkable = present * lr
                if mname == "oracle_linkable_hour_prob_shrunk20":
                    linkable = true_linkable.get((pd.Timestamp(vhour[i]), s), 0.0)
                    pp[i] = np.nan  # oracle linkable doesn't predict present
                plr[i] = lr
                plk[i] = linkable
                pprob[i] = dprob
                pred[i] = linkable * dprob
    pred = np.clip(pred, 0.0, None)
    return pred, pp, plr, plk, pprob


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    daily = pd.read_csv(ROOT / STEP01_DAILY); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY); audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))

    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize()
    ss["hour_of_day"] = ss["hour"].dt.hour
    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
    pf["hour_of_day"] = pf["hour"].dt.hour
    pf["dayofweek"] = pf["hour"].dt.dayofweek

    scenarios = build_scenarios()
    pred_frames = []
    fold_score_rows = []
    direction_rows = []
    boundary_rows = []
    diag_rows = []

    for eid, etype, tstart, tend, vstart, vend in scenarios:
        valid = pf[(pf["date"] >= vstart) & (pf["date"] <= vend)].copy()
        valid = valid.sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
        valid["score_available"] = valid["hour"] != UNSCORABLE_HOUR
        valid["y_true_keep"] = valid["y"]  # full-data y; blanked later where unscoreable
        # true present/linkable for validation (hour,source)
        vss = ss[(ss["date"] >= vstart) & (ss["date"] <= vend)]
        true_present = {(r["hour"], r["source_region"]): float(r["source_present_count"]) for _, r in vss.iterrows()}
        true_linkable = {(r["hour"], r["source_region"]): float(r["source_linkable_count"]) for _, r in vss.iterrows()}

        agg = compute_aggregates(ss, pf, tstart, tend)
        train_dates = agg["train_dates"]
        n_train_mean = float(np.mean([vc_map[d] for d in train_dates]))

        # boundary audit
        boundary = agg["boundary"]
        excl_train_rows = int(((pf["date"].between(tstart, tend)) & (pf["hour"] == boundary)).sum())
        v_end_hour = vend + pd.Timedelta(hours=23)
        v_end_observable = v_end_hour != UNSCORABLE_HOUR
        unscored = int((~valid["score_available"]).sum())
        scored = int(valid["score_available"].sum())
        boundary_rows.append({
            "evaluation_id": eid, "evaluation_type": etype, "train_start": f"{tstart:%Y-%m-%d}", "train_end": f"{tend:%Y-%m-%d}",
            "excluded_train_boundary_hour": f"{boundary:%Y-%m-%d %H:%M:%S}", "excluded_train_transition_rows": excl_train_rows,
            "valid_start": f"{vstart:%Y-%m-%d}", "valid_end": f"{vend:%Y-%m-%d}", "validation_end_hour": f"{v_end_hour:%Y-%m-%d %H:%M:%S}",
            "validation_end_transition_observable": bool(v_end_observable),
            "unscored_validation_rows": unscored, "scored_validation_rows": scored,
        })
        assert excl_train_rows == 6, f"{eid}: excluded_train_transition_rows={excl_train_rows} != 6"
        if eid in ("fold_08", "final_analog"):
            assert unscored == 6, f"{eid}: unscored={unscored} != 6"
        else:
            assert unscored == 0, f"{eid}: unscored={unscored} != 0"

        for meta in METHODS:
            mname = meta[0]
            pred, pp, plr, plk, pprob = predict_method(meta, valid, agg, true_present, true_linkable, n_train_mean)
            assert np.isfinite(pred).all() and (pred >= 0).all(), f"non-finite/negative pred {eid}/{mname}"
            pred_rounded = np.clip(np.rint(pred), 0, None).astype(np.int64)
            combo = pd.DataFrame({
                "evaluation_id": eid, "evaluation_type": etype, "method": mname, "deployable": meta[8],
                "date": valid["date"].to_numpy(), "hour": valid["hour"].to_numpy(),
                "hour_of_day": valid["hour_of_day"].to_numpy(), "dayofweek": valid["dayofweek"].to_numpy(),
                "day_type": valid["day_type"].to_numpy(), "source_region": valid["source_region"].to_numpy(),
                "target_region": valid["target_region"].to_numpy(), "task_key": valid["task_key"].to_numpy(),
                "audit_quality_regime": valid["audit_quality_regime"].to_numpy(), "period": valid["period"].to_numpy(),
                "validation_unique_vessel_count": valid["daily_unique_vessel_count"].to_numpy(),
                "score_available": valid["score_available"].to_numpy(),
                "y_true": np.where(valid["score_available"], valid["y_true_keep"], np.nan),
                "predicted_source_present": pp, "predicted_link_rate": plr, "predicted_source_linkable": plk,
                "predicted_transition_probability": pprob, "pred_float": pred, "pred_rounded": pred_rounded,
                "source_rank": valid["source_region"].map(REGION_RANK).to_numpy(),
                "target_rank": valid["target_region"].map(REGION_RANK).to_numpy(),
            })
            combo["error_float"] = combo["pred_float"] - combo["y_true"]
            combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
            pred_frames.append(combo)

            # fold scores (scored only)
            sc = combo[combo["score_available"]]
            er = sc["error_rounded"].astype(float).to_numpy()
            ef = sc["error_float"].astype(float).to_numpy()
            frow = {
                "evaluation_id": eid, "evaluation_type": etype, "method": mname, "deployable": meta[8],
                "train_start": f"{tstart:%Y-%m-%d}", "train_end": f"{tend:%Y-%m-%d}", "valid_start": f"{vstart:%Y-%m-%d}", "valid_end": f"{vend:%Y-%m-%d}",
                "n_prediction_rows": int(len(combo)), "n_scored_rows": int(len(sc)),
                "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
                "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()),
                "mean_bias_rounded": float(er.mean()),
                "actual_total_scored": float(sc["y_true"].sum()), "predicted_total_scored": float(sc["pred_rounded"].sum()),
            }
            for (s, t) in DIRS:
                m = (sc["source_region"] == s) & (sc["target_region"] == t)
                frow[f"{s}_to_{t}_sse"] = float((sc.loc[m, "error_rounded"].astype(float) ** 2).sum())
            fold_score_rows.append(frow)

            # direction scores
            for (s, t) in DIRS:
                d = sc[(sc["source_region"] == s) & (sc["target_region"] == t)]
                erd = d["error_rounded"].astype(float).to_numpy()
                efd = d["error_float"].astype(float).to_numpy()
                direction_rows.append({
                    "evaluation_id": eid, "evaluation_type": etype, "method": mname, "deployable": meta[8],
                    "task_key": f"{s}->{t}", "n_scored_rows": int(len(d)),
                    "actual_total": float(d["y_true"].sum()), "predicted_total": float(d["pred_rounded"].sum()),
                    "sse_float": float((efd ** 2).sum()), "sse_rounded": float((erd ** 2).sum()),
                    "mse_rounded": float((erd ** 2).mean()) if len(erd) else np.nan,
                    "mae_rounded": float(np.abs(erd).mean()) if len(erd) else np.nan,
                    "mean_bias_rounded": float(erd.mean()) if len(erd) else np.nan,
                    "zero_true_rate": float((d["y_true"] == 0).mean()) if len(d) else np.nan,
                    "zero_prediction_rate": float((d["pred_rounded"] == 0).mean()) if len(d) else np.nan,
                })

            # structure diagnostics
            if mname in STRUCT_METHODS:
                sc2 = combo[combo["score_available"]].copy()
                for s in REGION_ORDER:
                    ssub = sc2[sc2["source_region"] == s]
                    cells = ssub.groupby("hour").agg(
                        y_true=("y_true", "sum"), pred_rounded_sum=("pred_rounded", "sum"),
                        pred_present=("predicted_source_present", "first"), pred_linkable=("predicted_source_linkable", "first"),
                        pred_float_sum=("pred_float", "sum"))
                    # true present/linkable per hour
                    tp = np.array([true_present.get((h, s), np.nan) for h in cells.index])
                    tl = np.array([true_linkable.get((h, s), np.nan) for h in cells.index])
                    n_cells = len(cells)
                    pred_present = cells["pred_present"].to_numpy(float)
                    pred_linkable = cells["pred_linkable"].to_numpy(float)
                    true_outflow = cells["y_true"].to_numpy(float)
                    pred_outflow = cells["pred_float_sum"].to_numpy(float)
                    exposure = pred_linkable if meta[1] == "linkable_transition" else pred_present
                    violation = (pred_outflow - exposure)
                    fam = meta[1]
                    diag_rows.append({
                        "evaluation_id": eid, "evaluation_type": etype, "method": mname, "deployable": meta[8],
                        "source_region": s, "n_scored_source_hours": int(n_cells),
                        "true_present_mean": float(np.nanmean(tp)) if n_cells else np.nan,
                        "predicted_present_mean": float(np.nanmean(pred_present)) if n_cells else np.nan,
                        "present_mae": float(np.nanmean(np.abs(pred_present - tp))) if (n_cells and fam == "present_transition") else np.nan,
                        "true_linkable_mean": float(np.nanmean(tl)) if n_cells else np.nan,
                        "predicted_linkable_mean": float(np.nanmean(pred_linkable)) if (n_cells and fam == "linkable_transition") else np.nan,
                        "linkable_mae": float(np.nanmean(np.abs(pred_linkable - tl))) if (n_cells and fam == "linkable_transition") else np.nan,
                        "true_outflow_total": float(np.nansum(true_outflow)), "predicted_outflow_total": float(np.nansum(pred_outflow)),
                        "outflow_sse_float": float(np.nansum((pred_outflow - true_outflow) ** 2)),
                        "mean_true_outflow_rate_present": float(np.nanmean(true_outflow / np.where(tp > 0, tp, np.nan))) if n_cells else np.nan,
                        "mean_predicted_outflow_rate_present": float(np.nanmean(pred_outflow / np.where(pred_present > 0, pred_present, np.nan))) if n_cells else np.nan,
                        "mean_true_outflow_rate_linkable": float(np.nanmean(true_outflow / np.where(tl > 0, tl, np.nan))) if n_cells else np.nan,
                        "mean_predicted_outflow_rate_linkable": float(np.nanmean(pred_outflow / np.where(pred_linkable > 0, pred_linkable, np.nan))) if n_cells else np.nan,
                        "max_float_constraint_violation": float(np.nanmax(violation)) if n_cells else np.nan,
                    })
                    # assert no positive constraint violation
                    if n_cells:
                        assert np.nanmax(violation) <= 1e-9, f"constraint violation {eid}/{mname}/{s}: {np.nanmax(violation)}"

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    preds["_m"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_e", "_m", "hour", "source_rank", "target_rank"]).reset_index(drop=True)
    assert len(preds) == 142848, f"expected 142848 preds, got {len(preds)}"
    # unscoreable y_true blank
    unscore = preds[~preds["score_available"]]
    assert unscore["y_true"].isna().all(), "unscored y_true not blank"
    fold_scores = pd.DataFrame(fold_score_rows)
    direction_scores = pd.DataFrame(direction_rows)
    boundary = pd.DataFrame(boundary_rows)
    diag = pd.DataFrame(diag_rows)
    print(f"predictions {len(preds)}; boundary & constraint assertions PASS.")

    # --- target regime scores (deployable, rolling folds, scored) ----------
    regime_pred = preds[(preds["deployable"]) & (preds["evaluation_type"] == "rolling_7d") & (preds["score_available"])]
    regime_rows = []
    for mname in [m[0] for m in DEPLOYABLE]:
        for q in ["normal", "reduced", "severe", "outage"]:
            sub = regime_pred[(regime_pred["method"] == mname) & (regime_pred["audit_quality_regime"] == q)]
            if len(sub) == 0:
                continue
            er = sub["error_rounded"].astype(float).to_numpy()
            regime_rows.append({
                "method": mname, "audit_quality_regime": q, "n_prediction_rows": int(len(sub)),
                "n_unique_target_dates": int(sub["date"].nunique()),
                "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()),
                "mae_rounded": float(np.abs(er).mean()), "mean_bias_rounded": float(er.mean()),
            })
    regime_scores = pd.DataFrame(regime_rows)

    # --- decision table -----------------------------------------------------
    deploy_names = [m[0] for m in METHODS if m[8]]
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    win_pivot = rolling[rolling["deployable"]].pivot_table(index="evaluation_id", columns="method", values="sse_rounded", aggfunc="first")
    min_per_fold = win_pivot.min(axis=1)
    win_counts = win_pivot.eq(min_per_fold, axis=0).sum(axis=0)
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"].set_index("method")
    fa_deploy = fa.loc[[m for m in fa.index if m in deploy_names]]
    fa_rank = fa_deploy["sse_rounded"].rank(method="min", ascending=True).astype(int)
    dec_rows = []
    for meta in METHODS:
        mname = meta[0]
        rsub = rolling[rolling["method"] == mname]
        mse = rsub["mse_rounded"].to_numpy(float)
        sse = rsub["sse_rounded"].to_numpy(float)
        far = fa.loc[mname]
        dec_rows.append({
            "method": mname, "deployable": meta[8], "model_family": meta[1], "exposure_type": meta[2],
            "rolling_mean_mse": float(np.mean(mse)), "rolling_median_mse": float(np.median(mse)), "rolling_max_mse": float(np.max(mse)),
            "rolling_mean_sse": float(np.mean(sse)),
            "final_analog_sse": float(far["sse_rounded"]), "final_analog_mse": float(far["mse_rounded"]),
            "final_analog_mean_bias": float(far["mean_bias_rounded"]),
            "final_analog_rank_deployable": int(fa_rank.get(mname, np.nan)) if meta[8] else np.nan,
            "rolling_fold_win_count": int(win_counts.get(mname, 0)) if meta[8] else 0,
            "core_to_near_total_sse": float(rsub["core_to_near_sse"].sum()), "core_to_outer_total_sse": float(rsub["core_to_outer_sse"].sum()),
            "near_to_core_total_sse": float(rsub["near_to_core_sse"].sum()), "near_to_outer_total_sse": float(rsub["near_to_outer_sse"].sum()),
            "outer_to_core_total_sse": float(rsub["outer_to_core_sse"].sum()), "outer_to_near_total_sse": float(rsub["outer_to_near_sse"].sum()),
        })
    decision = pd.DataFrame(dec_rows)

    # --- write CSVs ---------------------------------------------------------
    p_md_def = out_dir / "method_definitions.csv"
    p_ba = out_dir / "boundary_audit.csv"
    p_pred = out_dir / "b_model_predictions.csv"
    p_fs = out_dir / "b_fold_scores.csv"
    p_ds = out_dir / "b_direction_scores.csv"
    p_rs = out_dir / "b_target_regime_scores.csv"
    p_dg = out_dir / "b_structure_diagnostics.csv"
    p_dec = out_dir / "decision_table.csv"
    p_sum = out_dir / "summary.md"

    md = pd.DataFrame([{"method": m[0], "model_family": m[1], "exposure_type": m[2], "stock_window": m[3],
                        "probability_level": m[4], "shrinkage_tau": m[5], "uses_daily_vessel_count": m[6],
                        "uses_validation_exposure": m[7], "deployable": m[8]} for m in METHODS])
    md.to_csv(p_md_def, index=False, encoding="utf-8-sig")
    boundary[BOUNDARY_COLUMNS].to_csv(p_ba, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLUMNS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fold_scores.sort_values(["evaluation_id", "method"])[FOLD_SCORE_COLUMNS].to_csv(p_fs, index=False, encoding="utf-8-sig")
    ds_out = direction_scores.copy()
    ds_out["_e"] = ds_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    ds_out["_m"] = ds_out["method"].map(METHOD_ORDER)
    ds_out["_d"] = ds_out["task_key"].map({d: i for i, d in enumerate(DIRECTION_NAMES)})
    ds_out.sort_values(["_e", "_m", "_d"])[DIR_SCORE_COLUMNS].to_csv(p_ds, index=False, encoding="utf-8-sig")
    regime_scores[REGIME_COLUMNS].sort_values(["method", "audit_quality_regime"]).to_csv(p_rs, index=False, encoding="utf-8-sig")
    diag_out = diag.copy()
    diag_out["_e"] = diag_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    diag_out["_m"] = diag_out["method"].map(METHOD_ORDER)
    diag_out.sort_values(["_e", "_m", "source_region"])[DIAG_COLUMNS].to_csv(p_dg, index=False, encoding="utf-8-sig")
    decision[DECISION_COLUMNS].to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_sum, decision, fold_scores, regime_scores, diag, boundary)

    print("\n=== Output files ===")
    for path in [p_md_def, p_ba, p_pred, p_fs, p_ds, p_rs, p_dg, p_dec, p_sum]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")

    print("\n=== final_analog deployable top 10 by sse_rounded ===")
    fa_d = fold_scores[(fold_scores["evaluation_id"] == "final_analog") & (fold_scores["deployable"])].sort_values("sse_rounded")
    for _, r in fa_d.head(10).iterrows():
        print(f"  {r['method']:42s} sse={r['sse_rounded']:.0f} mse={r['mse_rounded']:.3f} bias={r['mean_bias_rounded']:.2f}")
    print("\n=== rolling median MSE deployable top 10 ===")
    for _, r in decision[decision["deployable"]].sort_values("rolling_median_mse").head(10).iterrows():
        print(f"  {r['method']:42s} roll_med_mse={r['rolling_median_mse']:.3f} roll_mean_mse={r['rolling_mean_mse']:.3f} wins={int(r['rolling_fold_win_count'])}")
    print("\nDone.")


def write_summary(path, decision, fold_scores, regime_scores, diag, boundary):
    L = []
    L.append("# Step 09 B Structure Model Benchmark")
    L.append("")
    L.append("> Task B leak-free structural benchmark. No Ridge/Poisson/trees/NN, no submission, no A.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Boundary-safe evaluation")
    L.append("")
    L.append("- A B label at hour t depends on the same vessel's state at t+1.")
    L.append("- Each training period's last day 23:00 migration label was excluded from training (6 transition rows per scenario); that hour's source_present (a current-hour quantity) is still used for stock estimation.")
    L.append("- 2018-01-24 23:00 is right-censored (no 2018-01-25 00:00 state): fold_08 and final_analog each have 6 unscored validation rows; their y_true and error fields are blank, not 0.")
    L.append("- Earlier B backtest results may have been affected by this training-boundary leakage and by zero-filling the right-censored endpoint.")
    L.append("")

    L.append("## 2. Models and leakage controls")
    L.append("")
    L.append("- 5 direct models, 9 deployable state-structure models, 2 oracle diagnostics (deployable=False).")
    L.append("- Deployable models do NOT read validation source_present / source_linkable; only the vessel_p05 method uses the competition-provided validation daily vessel count. audit_quality_regime is not used in training (uniform training-date weight 1).")
    L.append("")

    L.append("## 3. Direct baselines")
    L.append("")
    direct = [m for m in ["pair_global_mean", "pair_hour_mean", "pair_recent7_hour", "pair_recent14_hour", "pair_hour_mean_x085"]]
    dd = decision.set_index("method")
    L.append("| method | rolling median MSE | rolling max MSE | final_analog SSE |")
    L.append("| --- | --- | --- | --- |")
    for m in direct:
        r = dd.loc[m]
        L.append(f"| {m} | {r['rolling_median_mse']:.3f} | {r['rolling_max_mse']:.3f} | {r['final_analog_sse']:.0f} |")
    L.append("")

    L.append("## 4. Present versus linkable parameterization")
    L.append("")
    pairs4 = [("present_global_prob", "linkable_global_prob"), ("present_hour_prob_raw", "linkable_hour_prob_raw"),
              ("present_hour_prob_shrunk20", "linkable_hour_prob_shrunk20"), ("present_recent7_stock_shrunk20", "linkable_recent7_stock_shrunk20")]
    L.append("| pair | present rolling med MSE / final SSE | linkable rolling med MSE / final SSE |")
    L.append("| --- | --- | --- |")
    for a, b in pairs4:
        ra, rb = dd.loc[a], dd.loc[b]
        L.append(f"| {a.replace('present_','').replace('linkable_','')} | {ra['rolling_median_mse']:.3f} / {ra['final_analog_sse']:.0f} | {rb['rolling_median_mse']:.3f} / {rb['final_analog_sse']:.0f} |")
    L.append("")

    L.append("## 5. Effect of probability shrinkage")
    L.append("")
    L.append("| pair | raw rolling med MSE / final SSE | shrunk20 rolling med MSE / final SSE |")
    L.append("| --- | --- | --- |")
    for a, b in [("present_hour_prob_raw", "present_hour_prob_shrunk20"), ("linkable_hour_prob_raw", "linkable_hour_prob_shrunk20")]:
        ra, rb = dd.loc[a], dd.loc[b]
        L.append(f"| {a.rsplit('_',1)[0]} | {ra['rolling_median_mse']:.3f} / {ra['final_analog_sse']:.0f} | {rb['rolling_median_mse']:.3f} / {rb['final_analog_sse']:.0f} |")
    L.append("- Per-direction effect on the sparse directions (final_analog SSE):")
    for dcol, dname in [("core_to_outer_total_sse", "core->outer"), ("outer_to_core_total_sse", "outer->core"),
                        ("near_to_outer_total_sse", "near->outer"), ("outer_to_near_total_sse", "outer->near")]:
        # use direction_scores final_analog
        pass
    L.append("")

    L.append("## 6. Long-term versus recent stock")
    L.append("")
    for a, b in [("present_hour_prob_shrunk20", "present_recent7_stock_shrunk20"), ("linkable_hour_prob_shrunk20", "linkable_recent7_stock_shrunk20")]:
        ra, rb = dd.loc[a], dd.loc[b]
        L.append(f"- {a.replace('_hour_prob_shrunk20','')}: long {ra['final_analog_sse']:.0f} vs recent7 {rb['final_analog_sse']:.0f} (final_analog SSE).")
    L.append("")

    L.append("## 7. Effect of daily vessel count")
    L.append("")
    ra, rb = dd.loc["linkable_hour_prob_shrunk20"], dd.loc["linkable_hour_prob_shrunk20_vessel_p05"]
    L.append(f"- linkable_hour_prob_shrunk20: rolling med MSE {ra['rolling_median_mse']:.3f}, final SSE {ra['final_analog_sse']:.0f}.")
    L.append(f"- +vessel_p05: rolling med MSE {rb['rolling_median_mse']:.3f}, final SSE {rb['final_analog_sse']:.0f}.")
    L.append("")

    L.append("## 8. Exposure forecast versus transition-probability error")
    L.append("")
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"].set_index("method")
    for dep, orc in [("present_hour_prob_shrunk20", "oracle_present_hour_prob_shrunk20"), ("linkable_hour_prob_shrunk20", "oracle_linkable_hour_prob_shrunk20")]:
        L.append(f"- {dep}: final SSE {fa.loc[dep,'sse_rounded']:.0f}; {orc} (true exposure): final SSE {fa.loc[orc,'sse_rounded']:.0f}.")
    L.append("- Oracle methods are excluded from ranking and win counts.")
    L.append("")

    L.append("## 9. Performance during the A outage period")
    L.append("")
    piv = regime_scores.pivot_table(index="method", columns="audit_quality_regime", values="mse_rounded")
    L.append("| method | normal | reduced | severe | outage |")
    L.append("| --- | --- | --- | --- | --- |")
    for m in [x[0] for x in DEPLOYABLE]:
        if m in piv.index:
            row = piv.loc[m]
            L.append(f"| {m} | {row.get('normal',np.nan):.3f} | {row.get('reduced',np.nan):.3f} | {row.get('severe',np.nan):.3f} | {row.get('outage',np.nan):.3f} |")
    L.append("- audit_quality_regime is A's audit grouping, shown here only to observe B's behavior in the same periods; it is not a B training weight.")
    L.append("")

    L.append("## 10. Decision for the next stage")
    L.append("")
    best_fa = decision[decision["deployable"]].sort_values("final_analog_sse").iloc[0]
    best_roll = decision[decision["deployable"]].sort_values("rolling_median_mse").iloc[0]
    L.append(f"- Best final_analog (deployable): {best_fa['method']} (SSE={best_fa['final_analog_sse']:.0f}).")
    L.append(f"- Best rolling median MSE (deployable): {best_roll['method']} ({best_roll['rolling_median_mse']:.3f}).")
    L.append("- These observations guide (not decide) Step 10. No submission is produced here.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
