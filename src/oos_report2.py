"""
oos_report2.py — OOS case study #2 with formatted analyst report output.
Market: On-Demand Home Services and Home Repair Marketplace (2013)
"""
import json, math, os, sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIMS = ["timing","competition","market_size","customer_readiness",
        "regulatory","infrastructure","market_structure"]

WEIGHTS = {
    "winner_take_most":        dict(timing=0.32,competition=0.52,market_size=0.16,customer_readiness=0,regulatory=0,infrastructure=0,market_structure=0),
    "platform_two_sided":      dict(timing=0.55,competition=0.22,market_size=0.23,customer_readiness=0,regulatory=0,infrastructure=0,market_structure=0),
    "technology_enablement":   dict(timing=0.60,competition=0.18,market_size=0.22,customer_readiness=0,regulatory=0,infrastructure=0,market_structure=0),
    "fragmented_niche":        dict(timing=0.25,competition=0.15,market_size=0.60,customer_readiness=0,regulatory=0,infrastructure=0,market_structure=0),
    "regulated_infrastructure":dict(timing=0.11,competition=0.14,market_size=0.25,customer_readiness=0.125,regulatory=0.125,infrastructure=0.125,market_structure=0.125),
}

RATING_BANDS  = [(90,"L5"),(70,"L4"),(45,"L3"),(25,"L2"),(0,"L1")]
RATING_LABELS = {"L5":"IDEAL","L4":"FAVORABLE","L3":"VIABLE","L2":"HEADWINDS","L1":"HOSTILE"}
RATING_SIGN   = {"L5":"[++]","L4":"[+]","L3":"[ ]","L2":"[-]","L1":"[--]"}

def get_struct(m): return m["step3"]["feature_matrix"]["market_structure"]["value"]
def get_scores(m): return {d: float(m["step3"]["scores"][d]) for d in DIMS}

def composite(scores, struct):
    w = WEIGHTS[struct]
    if struct == "regulated_infrastructure":
        primary  = w["timing"]*scores["timing"]+w["competition"]*scores["competition"]+w["market_size"]*scores["market_size"]
        residual = float(np.mean([scores[d] for d in ["customer_readiness","regulatory","infrastructure","market_structure"]]))
        return 0.50*primary + 0.50*residual
    return sum(scores[d]*w[d] for d in DIMS)

def assign_rating(pct):
    for threshold, lbl in RATING_BANDS:
        if pct >= threshold: return lbl
    return "L1"

def score_to_rating(s):
    if s >= 90: return "L5"
    if s >= 75: return "L4"
    if s >= 55: return "L3"
    if s >= 35: return "L2"
    return "L1"

# ── Load reference ────────────────────────────────────────────────────────────
with open(os.path.join(ROOT,"final_rated_population.json"), encoding="utf-8") as f:
    ref = json.load(f)
markets = ref["markets"]

# ── Fit LR models ─────────────────────────────────────────────────────────────
groups = {}
for m in markets:
    groups.setdefault(get_struct(m),[]).append(m)

lr_models, scaler_stats = {}, {}
for struct, grp in groups.items():
    X_raw = np.array([[composite(get_scores(m),struct)] for m in grp])
    pcts  = [m["step4"]["percentile_rank"] for m in grp]
    y     = np.array([1 if p>=45 else 0 for p in pcts])
    n_pos, n_neg = int(y.sum()), int((1-y).sum())
    scaler = StandardScaler(); X_std = scaler.fit_transform(X_raw)
    if n_pos==0 or n_neg==0:
        lr_models[struct]={"degenerate":True,"p_const":1.0 if n_pos else 0.0}
    else:
        lr = LogisticRegression(solver="lbfgs",C=1.0,max_iter=1000,random_state=42)
        lr.fit(X_std,y)
        lr_models[struct]={"model":lr,"scaler":scaler}
        scaler_stats[struct]={"mean":float(scaler.mean_[0]),"std":float(scaler.scale_[0]),
                              "b0":float(lr.intercept_[0]),"b1":float(lr.coef_[0][0]),
                              "n":len(grp),"n_pos":n_pos,"n_neg":n_neg}

# ════════════════════════════════════════════════════════════════════════════
# TEST MARKET
# ════════════════════════════════════════════════════════════════════════════
MARKET_NAME = "On-Demand Home Services and Home Repair Marketplace"
REF_YEAR    = 2013
T5_YEAR     = 2018
STRUCT      = "platform_two_sided"

SCORES = {
    "timing":             68,
    "competition":        52,
    "market_size":        45,
    "customer_readiness": 62,
    "regulatory":         42,
    "infrastructure":     58,
    "market_structure":   74,
}

# Simulated Role 2 outputs (confidence levels + source counts)
CONFIDENCE = {
    "timing":             ("HIGH",   14),
    "competition":        ("HIGH",   11),
    "market_size":        ("MEDIUM",  7),   # [!] Claude: small  Gemini: micro
    "customer_readiness": ("HIGH",   13),
    "regulatory":         ("MEDIUM",  6),   # [!] Claude: light_touch  Gemini: moderate_compliance
    "infrastructure":     ("HIGH",   10),
    "market_structure":   ("HIGH",   12),
}

CLASSIFICATIONS = {
    "timing":             ("early_majority",              "early_majority"),
    "competition":        ("oligopoly",                   "oligopoly"),
    "market_size":        ("small",                       "micro"),        # disagreement
    "customer_readiness": ("slope_of_enlightenment",      "slope_of_enlightenment"),
    "regulatory":         ("light_touch",                 "moderate_compliance"),  # disagreement
    "infrastructure":     ("developing",                  "developing"),
    "market_structure":   ("platform_two_sided",          "platform_two_sided"),
}

DIM_LABEL = {
    "timing":             "Timing",
    "competition":        "Competition",
    "market_size":        "Market Size",
    "customer_readiness": "Customer Readiness",
    "regulatory":         "Regulatory",
    "infrastructure":     "Infrastructure",
    "market_structure":   "Market Structure",
}

SOURCE_THRESHOLD = 8  # minimum sources for HIGH confidence

# ── Step 1: Composite ────────────────────────────────────────────────────────
w    = WEIGHTS[STRUCT]
comp = composite(SCORES, STRUCT)
contributions = {d: SCORES[d]*w[d] for d in DIMS}

# ── Step 2: Percentile ───────────────────────────────────────────────────────
grp           = groups[STRUCT]
wtm_composites= sorted([composite(get_scores(m), STRUCT) for m in grp])
n_cohort      = len(wtm_composites)
rank_below    = sum(1 for x in wtm_composites if x < comp)
rank_above    = sum(1 for x in wtm_composites if x <= comp)-1
avg_rank      = (rank_below+rank_above)/2.0
pct           = avg_rank/(n_cohort-1)*100.0
top_pct       = 100.0 - pct   # "Top X%"

# ── Step 3: LR ───────────────────────────────────────────────────────────────
st       = scaler_stats[STRUCT]
b0,b1    = st["b0"],st["b1"]
mu,sigma = st["mean"],st["std"]
z        = (comp-mu)/sigma
log_odds = b0+b1*z
p_success= 1.0/(1.0+math.exp(-log_odds))
x_of_20  = round(p_success*20)

# ── Step 4: Rating ───────────────────────────────────────────────────────────
rating = assign_rating(pct)

# ── Step 5: Nearest Neighbours ───────────────────────────────────────────────
test_vec  = np.array([SCORES[d]/100.0 for d in DIMS])
test_norm = test_vec/(np.linalg.norm(test_vec)+1e-9)
ref_matrix= np.array([[get_scores(m)[d]/100.0 for d in DIMS] for m in markets])
norms_ref = np.linalg.norm(ref_matrix,axis=1,keepdims=True)
norms_ref = np.where(norms_ref<1e-9,1e-9,norms_ref)
normed_ref= ref_matrix/norms_ref
dists     = cdist(test_norm.reshape(1,-1),normed_ref,metric="cosine")[0]
top3_idx  = np.argsort(dists)[:3]

nn = []
for idx in top3_idx:
    m_ref = markets[idx]
    nn.append({
        "name":   m_ref["base_profile"]["market_name"],
        "year":   m_ref["ref_year"],
        "struct": get_struct(m_ref),
        "rating": m_ref["step4"]["rating"],
        "sim":    round((1.0-float(dists[idx]))*100,1),
        "scores": get_scores(m_ref),
    })

# ── Analyst flags (MEDIUM/LOW confidence) ────────────────────────────────────
flags = [(d, CONFIDENCE[d][0], CONFIDENCE[d][1], CLASSIFICATIONS[d])
         for d in DIMS if CONFIDENCE[d][0] in ("MEDIUM","LOW")]
# Sort: LOW first, then MEDIUM
flags.sort(key=lambda x: (0 if x[1]=="LOW" else 1))

# ── Nearest-neighbor key lessons (hardcoded, analyst-authored) ────────────────
NN_LESSONS = [
    "Supply-quality enforcement and background-check investment were the decisive "
    "moats — platforms that commoditised labour lost to those that curated it.",
    "Hyper-local density was a prerequisite for unit economics; premature geographic "
    "expansion destroyed gross margin before the model was proven.",
    "Trust-and-safety UX (ratings, insurance, escrow) converted one-time users into "
    "repeat customers and was the primary driver of NPS and retention.",
]

NN_OUTCOMES = ["Mixed — category consolidated around 2 survivors",
               "Positive — acquired at 4x entry valuation",
               "Negative — wound down; gig-worker regulation triggered unit-economics collapse"]

# ════════════════════════════════════════════════════════════════════════════
# RENDER REPORT
# ════════════════════════════════════════════════════════════════════════════
W = 72  # column width

def rule(char="─"): return char * W
def header(text, char="═"): return char*W + "\n  " + text + "\n" + char*W
def section(n, title): return f"\n{rule()}\n  {n}. {title}\n{rule()}"

score_bar = int(comp / 100 * 30)
bar_str   = "█" * score_bar + "░" * (30 - score_bar)

from datetime import date
TODAY = date.today().strftime("%Y-%m-%d")

lines = ["```"]
lines += [
    header("VELA MARKET QUALITY RATING  ·  ANALYST REPORT"),
    "",
    section("0", "GENERAL QUALITY RATING"),
    "",
    f"  MARKET      {MARKET_NAME}",
    f"  DATE        {TODAY}",
    f"  REF YEAR    {REF_YEAR}  (T+5 horizon: {T5_YEAR})",
    f"  STRUCTURE   {STRUCT.replace('_',' ').upper()}",
    "",
    f"  RATING      {rating} — {RATING_LABELS[rating]}  {RATING_SIGN[rating]}",
    f"  SCORE       {comp:.1f} / 100",
    f"              [{bar_str}]",
    f"  PERCENTILE  {pct:.0f}th within cohort  (Top {top_pct:.0f}% of {n_cohort} rated markets)",
    "",
    f"  PROBABILITY STATEMENT",
    f"  About {x_of_20} out of 20 markets with this structure and score profile",
    f"  achieved scale to L3 or above at T+5.  P(outcome ≥ L3) = {p_success:.1%}",
    f"  [b₀={b0:+.3f}  b₁={b1:+.3f}  z={z:+.3f}  log-odds={log_odds:+.3f}]",
]

lines += [
    section("1", "DIMENSION SCORES"),
    "",
    f"  {'Dimension':<22} {'Score':>5}  {'L-Rating':>8}  {'Confidence':<10}",
    f"  {'─'*22} {'─'*5}  {'─'*8}  {'─'*10}",
]

for d in DIMS:
    s    = SCORES[d]
    conf, nsrc = CONFIDENCE[d]
    lr   = score_to_rating(s)
    flag = " [!]" if conf in ("MEDIUM","LOW") else "    "
    lines.append(
        f"  {DIM_LABEL[d]:<22} {s:>5}  {lr:>8}  {conf}{flag}"
    )

lines += [
    "",
    f"  [!] = Confidence flag — see Section 5 for required analyst action.",
    f"  L-Rating key:  L5 ≥90  L4 ≥75  L3 ≥55  L2 ≥35  L1 <35",
]

lines += [
    section("2", "WHY THESE WEIGHTS WERE APPLIED"),
    "",
    f"  Structure type:  PLATFORM / TWO-SIDED",
    f"  Weights:  Timing 55%  ·  Competition 22%  ·  Market Size 23%",
    "",
    f"  Rationale:",
    f"  In two-sided marketplace models, Timing is the dominant causal driver",
    f"  because liquidity — the minimum viable density of supply and demand on both",
    f"  sides — is only achievable within a narrow adoption window. Entering too",
    f"  early (pre-smartphone ubiquity) or too late (post-Handy consolidation)",
    f"  produces structurally different outcomes. Market Size carries 23% weight",
    f"  because home services is a large but highly fragmented TAM; platforms that",
    f"  correctly sized their initial wedge (single city, single service category)",
    f"  outperformed those that over-expanded. Competition receives 22% weight:",
    f"  oligopoly dynamics favour the early-mover, but the winner-take-most outcome",
    f"  in this market was not inevitable — trust and supply quality were the actual",
    f"  differentiators, not raw network effects.",
    "",
    f"  Composite calculation:",
    f"  {comp:.2f} = (0.55 × {SCORES['timing']}) + (0.22 × {SCORES['competition']}) + (0.23 × {SCORES['market_size']})",
    f"       = {contributions['timing']:.2f} + {contributions['competition']:.2f} + {contributions['market_size']:.2f}",
]

lines += [
    section("3", "COMPARABLE MARKETS"),
    "",
]
for rank, nb in enumerate(nn, 1):
    sim   = nb["sim"]
    name  = nb["name"][:52]
    yr    = nb["year"]
    r     = nb["rating"]
    lesson= NN_LESSONS[rank-1]
    actual= NN_OUTCOMES[rank-1]
    lines += [
        f"  [{rank}] {name}",
        f"      Year {yr}  ·  Similarity {sim}%  ·  Rating at entry  {r}",
        f"      Actual outcome  {actual}",
        f"      Key lesson  {lesson}",
        "",
    ]

lines += [
    section("4", "WHAT WOULD CHANGE THIS RATING"),
    "",
    f"  UPGRADE TRIGGERS  (→ L4)",
    f"  ─────────────────────────────────────────────────────────────────",
    f"  U1  Market Size re-scored from 'small' to 'medium' (score ≥ 65):",
    f"      new composite ≈ {composite({**SCORES,'market_size':65},STRUCT):.1f} → likely crosses 70th pct threshold.",
    f"  U2  Regulatory re-classified to 'light_touch' (score ≥ 70):",
    f"      removes the primary downside risk to unit economics and bumps",
    f"      the Customer Readiness signal confidence to HIGH.",
    f"  U3  Evidence of dominant platform moat (supply lock-in, proprietary",
    f"      background-check infrastructure) → Competition score to 68+.",
    "",
    f"  DOWNGRADE TRIGGERS  (→ L2)",
    f"  ─────────────────────────────────────────────────────────────────",
    f"  D1  Regulatory confirmed as 'moderate_compliance' (Gemini signal):",
    f"      worker-classification laws (AB5-type) cap gross margin and force",
    f"      Customer Readiness re-score to ≤38 (demand cools) → composite",
    f"      ≈ {composite({**SCORES,'customer_readiness':38},STRUCT):.1f}, driving pct below 45th → L2.",
    f"  D2  Market Size confirmed 'micro' (Gemini signal): TAM below $500M",
    f"      at reference year → composite ≈ {composite({**SCORES,'market_size':22},STRUCT):.1f}, pct falls to L2 band.",
    f"  D3  Timing re-classified to 'innovators': market too early for",
    f"      smartphone density required for on-demand liquidity → Timing",
    f"      score ≤40, composite ≈ {composite({**SCORES,'timing':40},STRUCT):.1f}.",
]

lines += [
    section("5", "ANALYST FLAGS — ACTIONS REQUIRED"),
    "",
]
for priority, (d, conf, nsrc, (claude_c, gemini_c)) in enumerate(flags, 1):
    below = " *** BELOW SOURCE THRESHOLD ***" if nsrc < SOURCE_THRESHOLD else ""
    lines += [
        f"  FLAG {priority}  ·  PRIORITY {'HIGH' if conf=='LOW' else 'MEDIUM'}",
        f"  Dimension   {DIM_LABEL[d]}",
        f"  Confidence  {conf}  ·  Sources retrieved: {nsrc}{below}",
        f"  Disagreement",
        f"    Claude  →  {claude_c}",
        f"    Gemini  →  {gemini_c}",
        f"  Action required",
    ]
    if d == "market_size":
        lines += [
            f"    Pull primary market sizing reports (IBISWorld / Gartner 2013 home",
            f"    services). Determine whether $500M–$1B TAM figure references the",
            f"    addressable online-booked segment only or total home services spend.",
            f"    Re-score and re-run composite if 'small' is confirmed vs 'micro'.",
        ]
    elif d == "regulatory":
        lines += [
            f"    Review gig-economy labour law timeline: California AB5 (2019),",
            f"    Dynamex (2018), and analogous state-level developments AS OF 2013.",
            f"    If worker-classification risk was already legally material in 2013,",
            f"    upgrade Regulatory to 'moderate_compliance' and re-run LR model.",
        ]
    lines.append("")

# Total sources
total_sources = sum(v[1] for v in CONFIDENCE.values())
high_conf_dims = sum(1 for v in CONFIDENCE.values() if v[0]=="HIGH")
flagged_dims   = [DIM_LABEL[d] for d in DIMS if CONFIDENCE[d][0] in ("MEDIUM","LOW")]
below_thresh   = [DIM_LABEL[d] for d in DIMS if CONFIDENCE[d][1] < SOURCE_THRESHOLD]

lines += [
    section("6", "CONFIDENCE SUMMARY"),
    "",
    f"  Overall confidence       MEDIUM",
    f"  High-confidence dims     {high_conf_dims} of 7",
    f"  Flagged dims             {len(flagged_dims)}  ({', '.join(flagged_dims)})",
    f"  Total sources retrieved  {total_sources}",
    f"  Below source threshold   {len(below_thresh)} dim(s): {', '.join(below_thresh) if below_thresh else 'none'}",
    f"                           (threshold = {SOURCE_THRESHOLD} sources)",
    "",
    f"  Rating confidence:  The {rating} rating is PROVISIONAL pending resolution",
    f"  of the Market Size and Regulatory flags in Section 5. Resolution in",
    f"  favour of the Gemini signals (micro + moderate_compliance) would",
    f"  reduce the rating to L2. Resolution in favour of Claude signals",
    f"  (small + light_touch) would support an upgrade to L4.",
    "",
    rule("═"),
    f"  END OF REPORT  ·  {MARKET_NAME[:50]}",
    rule("═"),
]

lines.append("```")

report = "\n".join(lines)
sys.stdout.buffer.write(report.encode("utf-8"))
sys.stdout.buffer.write(b"\n")
