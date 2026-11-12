 TraceChain - Specification (v2, built from scratch)
 1. Goal
 Given a Bitcoin wallet address, return:
  1. an illicit-risk score (0 to 1) with a per-wallet explanation, and
  2. the nearest VASP-like hub in the address graph, with hop distance, confidence, and the path to it.

 Serve this through a REST API and a web dashboard that shows a risk gauge, the fund-flow path as a graph,
 the explanation, and the model metrics.

 2. Honest scope (this must appear in the README and in the slides)
  • Public data has no real exchange labels. Elliptic++ labels addresses only as illicit, licit, or unknown.
  • So the "VASP" seeds are proxies: addresses with exchange-like structure (very high degree, high fan-in
    and fan-out, long activity). The system also supports an optional registry file of real VASP addresses. If
    present, the registry replaces the proxies.
  • The output is "nearest likely VASP-like hub", not an identity. Risk scores are for triage, never for automatic
    action.
  • Proof of concept on Bitcoin only (Elliptic++ Actors dataset).
  • The UI must show whether a seed is a proxy or from the registry.


 3. Data (a human places the files in data/raw/)
 Elliptic++ Actors Dataset, three files: wallets_features.csv, wallets_classes.csv, AddrAddr_edgelist.csv. Source:
 github.com/git-disl/EllipticPlusPlus (the README links to the Google Drive download).
 The dataset README states about 822,942 addresses, 49 time steps, 56 features, classes illicit 14,266 / licit
 251,088 / unknown 557,588, and 2,868,964 address-to-address edges. Treat these as expectations to verify. If
 the files differ, the files win.
 Do not assume column names. scripts/00_inspect_data.py must print headers, dtypes, row counts, class
 counts, and write DATA_NOTES.md. All later code uses the names found there. DATA_NOTES.md must settle
 and record:
  • how many rows each address has (features are per address per time step),
  • whether the edge list has a time column,
  • the address id type, and how addresses join across the three files.


 4. Pipeline
 4.1 Preparation (scripts/01_prepare.py)
  • Load and clean the three files. Map classes to illicit=1, licit=0, unknown=excluded from training and scoring
    metrics (unknown addresses stay in the graph).
  • Temporal split on labelled rows: train on time steps 1 to 34, test on 35 to 49.
  • Serving uses an address's latest row. Record that choice in DATA_NOTES.md.
  • Leakage check: report what share of test addresses also appear in the training period. Evaluate on all test
    rows and also on the "cold" subset (test addresses never seen in training).
  • Build the address graph as a scipy sparse matrix with an address-to-index mapping. Save under artifacts/.


 4.2 Graph features
 Compute without using labels: in-degree, out-degree, PageRank (own sparse power iteration, about 50
 iterations), mean degree of neighbours. Do not duplicate features the dataset already has; record any overlap



TraceChain v2 - Specification                                                                                  Page 1
 in DATA_NOTES.md.

 4.3 VASP seeds and attribution (src/tracechain/vasp.py, scripts/02_build_vasp_index.py)
  • Seed score: rank-average of total degree, in-degree, out-degree, number of active time steps, and total
    value moved if such a feature exists.
  • Take the top K=30 addresses as proxy seeds, skipping addresses labelled illicit in the training period only.
    Name them VASP-PROXY-01 and so on.
  • Registry override: if data/registry/vasp_registry.csv exists (columns: address, name), those addresses
    become seeds with kind "registry".
  • Attribution: multi-source shortest path (BFS) on the undirected address graph using scipy.sparse.csgraph.
    Process seeds in chunks to limit memory. For every address store the nearest seed id and the hop count.
  • confidence_score = 1 / (1 + hops). Label: high if hops <= 2, medium if hops <= 4, low otherwise.
    Unreachable means "unattributed" with no hop value.
  • Path reconstruction on demand. /api/trace must answer in under 2 seconds warm.
  • Attribution checks, saved to reports/attribution_stats.json: share of addresses reachable within 2, 4 and 6
    hops; median hops for illicit vs licit labelled addresses; the same numbers for K=10, 30 and 100.

 4.4 Risk model (scripts/03_train.py)
  • Two Random Forests, same settings: baseline (original features) and graph-enhanced (original plus graph
    features and hop distance to nearest seed).
  • Settings: n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1,
    random_state=42. Keep the model small enough to load fast.
  • Report for the illicit class: precision, recall, F1, PR-AUC, confusion matrix, at threshold 0.5. Report on all
    test rows and on the cold subset.
  • Report the result honestly. If graph features do not help, say so.
  • Save the graph-enhanced model and feature list under artifacts/. Save reports/metrics.json and
    reports/figures/ (PR curves for both models, confusion matrix, top 15 feature importances).

 4.5 Explanations (src/tracechain/explain.py)
  • Per wallet, use SHAP TreeExplainer on that single row, for the illicit class. Return the top 5 features with
    value and signed contribution.
  • If SHAP is slower than 2 seconds or fails, fall back to feature importance times how unusual the value is,
    and set explanation_method to "importance_x_deviation". Never present the fallback as SHAP.


 5. API contract (FastAPI, api/main.py)
 All routes start with /api. Load artifacts once at startup. Allow CORS for the web dev origin.
  • GET /api/health returns {"status": "ok", "model_loaded": bool, "graph_nodes": int}
  • GET /api/wallets/sample?label=illicit|licit&n=5 returns test-period addresses with known labels, for demos.
    Response: {"wallets": [{"address": str, "label": str, "time_step": int}]}
  • GET /api/score/{address} returns:

 { "address": str, "time_step": int, "risk_score": float, "risk_label": "illicit" | "licit", "nearest_vasp": {"id": str, "name":
 str, "kind": "proxy" | "registry", "hops": int | null, "confidence": "high" | "medium" | "low", "confidence_score": float
 | null}, "top_features": [{"name": str, "value": float, "contribution": float}], "explanation_method": "shap" |
 "importance_x_deviation" }
  • GET /api/trace/{address} returns {"path": [str], "nodes": [{"id": str, "kind": "wallet" | "path" | "vasp" |
    "neighbor"}], "edges": [{"source": str, "target": str}]}. At most 60 nodes: the path plus up to 12 neighbours of
    the wallet.
  • GET /api/metrics returns the contents of reports/metrics.json plus attribution_stats.json.



TraceChain v2 - Specification                                                                                              Page 2
  • Errors: 404 with {"detail": "..."} for unknown address, 503 if artifacts are not loaded.


 6. Reports (committed)
 reports/metrics.json, reports/attribution_stats.json, reports/figures/*.png, reports/screenshots/*.png,
 reports/RESULTS.md (tables generated from the JSON files by a script, not typed by hand).

 7. Web dashboard (web/)
 One page, no router needed.
  • Search box plus "try a sample" buttons (illicit and licit) using /api/wallets/sample.
  • Result card: SVG risk gauge, risk label, nearest VASP with a proxy/registry badge, hops, confidence.
  • Fund-flow graph with react-force-graph-2d. Colour by node kind. Hover shows the address.
  • Explanation panel: horizontal bars for top features, with the explanation_method shown.
  • Metrics tab: baseline vs graph-enhanced table, PR curve image, attribution stats.
  • Loading and error states everywhere. Works at laptop width.
  • Build first against mock JSON in web/src/mock (switch with VITE_USE_MOCK=true), then against the real
    API.


 8. Repository layout
 tracechain/ AGENTS.md SPEC.md README.md DATA_NOTES.md requirements.txt .gitignore
 .antigravityignore data/raw/ data/registry/ (not committed) artifacts/ (not committed) src/tracechain/ data.py
 features.py vasp.py model.py explain.py service.py scripts/ 00_inspect_data.py 01_prepare.py
 02_build_vasp_index.py 03_train.py 04_make_results.py api/main.py web/ Vite + React tests/ reports/

 9. Definition of done
  • Fresh clone plus data in data/raw/ plus the README steps reproduces everything.
  • pytest passes.
  • /api/score and /api/trace answer for one illicit and one licit test-period address.
  • reports/metrics.json exists and the dashboard shows it.
  • Five screenshots exist in reports/screenshots/.
  • README states the honest scope from section 2.




TraceChain v2 - Specification                                                                                     Page 3
