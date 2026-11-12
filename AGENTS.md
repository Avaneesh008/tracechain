 AGENTS.md - standing rules for every agent in this repo
 Project: TraceChain. Read SPEC.md before doing anything. If SPEC.md and a prompt disagree, say so and
 ask; do not guess.

 Hard rules
  • Write all code from scratch. Do not copy, adapt, fetch or look at the old prototype repo (Avaneesh008/SIH).
  • Never invent column names, file contents, or numbers. Open the real files, print the headers, and write what
    you find in DATA_NOTES.md before coding against them.
  • Every number in reports, README or slides must come from a script run in this repo. Save outputs to
    reports/.
  • Never mark a task done unless you ran the code successfully and showed the real output.
  • Do not commit anything inside data/ or artifacts/. Do commit reports/metrics.json, reports/figures/ and
    reports/screenshots/.
  • If a step needs a human action (downloading data, credentials), stop and say exactly what is needed. Do
    not fake or generate substitute data.


 Python rules
  • Python 3.10 or newer. Type hints on public functions. Small functions. Docstrings in plain language.
  • Pipeline is scripts in scripts/, logic in src/tracechain/. No notebooks for the pipeline.
  • The address graph is large (about 822K nodes, about 2.9M edges). Use scipy.sparse and
    scipy.sparse.csgraph for it. Use networkx only for small display subgraphs (under 200 nodes).
  • Read CSVs with explicit dtypes. Use float32 where possible. Watch memory.
  • Pin versions in requirements.txt.
  • Add tests in tests/ using tiny synthetic graphs for the logic. Run them with pytest before finishing a task.


 Frontend rules
  • Vite, React, plain CSS. Use react-force-graph-2d for the graph.
  • No 3D libraries. No Spline. No external 3D embeds.
  • CSS changes are additive: add new classes or new files. Do not rewrite or delete existing rules.
  • API base URL comes from VITE_API_URL.
  • Every API call needs a loading state and an error state.


 Team and git rules
  • Three people work in this repo, each on their own machine and branch. See TEAM.md for who owns which
    files.
  • Only edit the files the current person owns (listed in the prompt you are given). If a change is needed
    elsewhere, stop and say which file and why.
  • Do not run git commit, git push, git merge or git rebase. The human commits, so the commit is under the
    right name.
  • Do not edit requirements.txt, SPEC.md or AGENTS.md unless the prompt says so. If you need a new
    dependency, tell the human the exact line to add.


 Style
  • Plain language in comments, README and commit messages.
  • No emojis anywhere (code, UI text, docs, commit messages).



TraceChain v2 - Agent rules                                                                                    Page 1
  • Before large work, write a short plan. After each task, list what is done and what is still not done.




TraceChain v2 - Agent rules                                                                                 Page 2
