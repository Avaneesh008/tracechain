 PROMPTS.md - building TraceChain v2 in Antigravity
 Commands are for Windows PowerShell. Run them from the repo root unless stated.
 Team note: this file has no git commands after section A. For every step, commit with the branch commands in
 TEAM.md section 5 (own branch, pull request, never straight to main). Start each prompt with this line, filled in
 for the person running it: "Owner scope: you may edit only <the files that person owns in TEAM.md>. Do not
 run any git commands."

 A. One-time setup (about 20 minutes)
  1. Download the Elliptic++ Actors Dataset (wallets_features.csv, wallets_classes.csv, AddrAddr_edgelist.csv)
     from the Google Drive link in github.com/git-disl/EllipticPlusPlus. Start this first, it is the slowest step.

  2. Create the new repo folder and put the three files in this layout:

   mkdir tracechain
   cd tracechain
   git init -b main
   mkdir data\raw
   mkdir data\registry
   mkdir artifacts
   mkdir reports

 Copy the three CSV files into data\raw\. Copy all four files (AGENTS.md, SPEC.md, PROMPTS.md, TEAM.md)
 into the repo root.
  3. Create .gitignore:

   @"
   data/
   artifacts/
   .venv/
   __pycache__/
   node_modules/
   web/dist/
   *.pkl
   *.parquet
   *.npz
   "@ | Set-Content .gitignore

  4. Create .antigravityignore so agents do not read the huge CSVs:

   @"
   data/
   artifacts/
   node_modules/
   .venv/
   "@ | Set-Content .antigravityignore

  5. Python environment:

   python -m venv .venv
   .venv\Scripts\Activate.ps1

  6. First commit, and connect the new empty GitHub repo (Avaneesh creates it, then paste its URL):

   git add AGENTS.md SPEC.md PROMPTS.md TEAM.md .gitignore .antigravityignore
   git commit -m "Add project spec, agent rules, prompts and team plan"
   git remote add origin <NEW-REPO-URL>
   git push -u origin main

  7. In Antigravity: File, Open Folder, choose the tracechain folder (the repo root, not its parent). Sign in. In the
     settings pick a review-driven policy so the agent asks before risky terminal commands. Check that
     AGENTS.md shows up as an active rule; if your version looks for a rules folder instead, add the same text



TraceChain v2 - Prompts                                                                                          Page 1
     as a rule under .agents\rules\ (some guides write .agent\rules\, so check which one your version shows).

  8. In the chat box choose Planning mode for steps P1 to P5. Read the implementation plan the agent writes
     and correct it before it runs. Use the strongest model you have for P3 and P4.


 B. Order of work
 P0 runs first (Avaneesh) and is merged quickly. After that, in parallel:
  • P1 (Sriraghava) needs P0.
  • P5 (Balu) needs only P0. It works from mock data and only touches web\.
  • P4 skeleton (Avaneesh) needs only P0 and the contract in SPEC section 5. Stubs are fine until real artifacts
    exist.
  • P2 (Avaneesh) needs the graph artifacts from P1. Do P2 before polishing P4, because P3 waits on it.
  • P3 (Sriraghava) needs P1 and P2: the graph-enhanced model uses the hop distance to the nearest seed,
    which comes from P2. The baseline model can be trained earlier.
  • P4 is finished after P3 (it serves the model and metrics).
  • P6 (Balu) needs the API and model merged.



 P0 - Scaffold and data check (Fast mode)
   Read AGENTS.md and SPEC.md fully.
   Create the repository skeleton exactly as in SPEC section 8 (empty files with a one-line
     docstring are fine), requirements.txt with pinned versions for: numpy, pandas, scipy,
     scikit-learn, shap, matplotlib, pyarrow, fastapi, uvicorn, pytest, httpx, joblib. Add a
     README.md stub.
   Then write scripts/00_inspect_data.py as described in SPEC section 3. Run it against
     data/raw/ and write DATA_NOTES.md from the real output.
   README.md, DATA_NOTES.md and requirements.txt are stubs created in P0; after P0 is merged
     their owners in TEAM.md edit them (other people change requirements.txt only in a small
     branch of its own).
   If any of the three files is missing, stop and tell me which one.
   Show me the real headers and counts you found.

   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

 Commit and push with the commands for this step in TEAM.md section 5.

 P1 - Preparation and graph features (Planning mode)
   Implement SPEC sections 4.1 and 4.2 in src/tracechain/data.py, src/tracechain/features.py
     and scripts/01_prepare.py.
   Use only the column names recorded in DATA_NOTES.md. Settle the open points listed in SPEC
     section 3 and write the decisions into DATA_NOTES.md.
   Build the address graph as a scipy sparse matrix, never networkx. Save artifacts under
     artifacts/.
   Write tests with a tiny synthetic graph for the PageRank and degree code.
   Run the script on the real data and show me: row counts per split, class counts per split,
     the leakage check numbers, run time, and peak memory if you can measure it.

   pytest

 Commit and push with the commands for this step in TEAM.md section 5.

 P2 - VASP seeds and attribution (Planning mode)
   Implement SPEC section 4.3 in src/tracechain/vasp.py and scripts/02_build_vasp_index.py.
   Use scipy.sparse.csgraph with seeds processed in chunks. Support the optional registry file
     at data/registry/vasp_registry.csv.
   Write tests on a tiny synthetic graph: nearest seed, hop counts, unreachable nodes, registry
      override.




TraceChain v2 - Prompts                                                                                    Page 2
   Run it on the real graph. Produce reports/attribution_stats.json as described, including
     K=10, 30 and 100.
   Show me the real stats and how long one path reconstruction takes.

   pytest

 Commit and push with the commands for this step in TEAM.md section 5.

 P3 - Model training and evaluation (Planning mode, strongest model)
   Implement SPEC sections 4.4 and 4.5 in src/tracechain/model.py, src/tracechain/explain.py
     and scripts/03_train.py, plus scripts/04_make_results.py that turns the JSON files into
     reports/RESULTS.md tables.
   Train baseline and graph-enhanced Random Forests with the exact settings in the spec. Report
      all test rows and the cold subset.
   Save metrics.json and the figures. Report the result honestly even if graph features do not
     help.
   Test the explanation function on three real addresses and show the time each takes. If SHAP
     is too slow, use the fallback and label it.
   Show me the real metrics table and the model file size.

   pytest

 Commit and push with the commands for this step in TEAM.md section 5.
 (Models and data stay uncommitted. Share artifacts between teammates by drive, or regenerate with the
 scripts.)

 P4 - API (Planning mode)
   Implement SPEC section 5 in api/main.py and src/tracechain/service.py. Load all artifacts
     once at startup. Follow the JSON contract exactly.
   Write contract tests with FastAPI TestClient: 200 shapes, 404 for an unknown address, 503
     when artifacts are missing (use a temporary empty artifacts folder).
   Run uvicorn on port 8000. Call /api/health, /api/wallets/sample for illicit and licit, then
     /api/score and /api/trace for one address of each. Show me the real responses and response
      times.

   pytest

 Commit and push with the commands for this step in TEAM.md section 5.

 P5 - Web dashboard (Planning mode, can run in parallel after P0)
   Build the dashboard described in SPEC section 7 inside web/ using Vite, React and plain CSS.
      Use react-force-graph-2d. No 3D libraries, no Spline. Keep CSS additive: new classes or
     new files, never rewrite existing rules.
   Start with mock JSON in web/src/mock that matches SPEC section 5 exactly, switched by
     VITE_USE_MOCK=true. When the real API at VITE_API_URL is available, switch to it.
   Add loading and error states for every request.
   Run the dev server. Use the browser to open it and check: search works, sample buttons work,
      the graph renders, the metrics tab renders. Fix what is broken.

 Commit and push with the commands for this step in TEAM.md section 5.

 P6 - Screenshots, README, results (Fast mode)
 Start the API and the web app against the real data first.
   Using the browser, take these screenshots at laptop width and save them in
     reports/screenshots/:
   01_search.png (empty page), 02_illicit_result.png (a test-period illicit sample wallet, full
      result), 03_licit_result.png (a licit sample wallet), 04_trace_graph.png (graph close-up
     with the path to the VASP), 05_metrics.png (metrics tab).
   Then finish README.md: what it is, the honest scope from SPEC section 2, how to reproduce
     from a fresh clone (data placement, pip install, scripts 00 to 04, uvicorn, npm run dev),
     and a results section copied from reports/RESULTS.md.
   Do not write any number that is not in reports/*.json.




TraceChain v2 - Prompts                                                                                  Page 3
 Commit and push with the commands for this step in TEAM.md section 5.



 C. If time runs short, cut in this order
  1. Sensitivity runs for K=10 and K=100 (keep K=30).
  2. SHAP (use the labelled fallback).
  3. Registry support (keep the code path, skip the file).
  4. Metrics tab in the web app (keep the table only).

 Never cut: the temporal split, the baseline vs graph-enhanced comparison, the honest scope text, the
 screenshots.

 D. Things to check yourself, not the agent
  • Open reports/metrics.json and read the numbers before they go on a slide.
  • Try one illicit and one licit wallet from the sample buttons in front of your guide.
  • Do not delete the old prototype repo until the new one passes a full demo.




TraceChain v2 - Prompts                                                                                 Page 4
