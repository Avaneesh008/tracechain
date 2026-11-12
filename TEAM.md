 TEAM.md - three people, three owners, three sets of
 commits
 Team: Sriraghava, Avaneesh (repo owner), Balu. This file replaces the single-person git commands in
 PROMPTS.md. Use the prompts from PROMPTS.md, but commit with the commands below.

 1. Who owns what
 Each person drives their own Antigravity agents on their own machine and commits their own work under their
 own name. Do not commit another person's work under your name.

 Person                        Steps                       Files they own                  Branches

 Avaneesh                      P0 scaffold, P4 API, P2     api/,                           feat/scaffold, feat/api,
                               VASP attribution            src/tracechain/service.py,      feat/vasp
                                                           src/tracechain/vasp.py,
                                                           scripts/00_inspect_data.py, s
                                                           cripts/02_build_vasp_index.p
                                                           y, tests for these

 Sriraghava                    P1 data prep and graph      src/tracechain/data.py,         feat/data-prep, feat/model
                               features, P3 model and      features.py, model.py,
                               explanations                explain.py,
                                                           scripts/01_prepare.py,
                                                           03_train.py,
                                                           04_make_results.py,
                                                           DATA_NOTES.md, tests for
                                                           these

 Balu                          P5 web dashboard, P6        web/, reports/screenshots/,     feat/web, feat/docs
                               screenshots and README      README.md

 Stub files: P0 (Avaneesh) creates README.md, DATA_NOTES.md, requirements.txt and the tests/ folder as
 stubs. After P0 is merged, README.md belongs to Balu, DATA_NOTES.md and tests for data, features, model
 and explain belong to Sriraghava, tests for api, service and vasp belong to Avaneesh, and requirements.txt is a
 shared file (see section 4).
 Swap steps between you if someone has a better fit. Just keep the rule: one person per step, and the person
 who ran the step makes the commit.

 2. Order and dependencies
  1. Avaneesh does P0 first and merges it quickly (about 30 minutes). Everyone else waits for it only to pull the
     skeleton.
  2. Then in parallel:
        • Sriraghava: P1, then P3. The graph-enhanced model needs the hop distance from Avaneesh's P2, so
          P3 is finished only after P2 is merged and its artifacts/ folder is shared. The baseline model can be
          trained earlier.
        • Balu: P5 (works from mock data, needs nothing from the others).
        • Avaneesh: P4 API skeleton against the contract in SPEC section 5 (stubs are fine) while waiting for P1,
          then P2 as soon as P1 has produced the graph artifacts. Do P2 before polishing P4, because
          Sriraghava's P3 waits on it. Finish P4 after P3.
  3. Artifacts (graph, model) are not in git. Whoever creates them shares the artifacts/ folder by drive or USB.
     For the final screenshots, Balu either copies artifacts/ and runs the API locally, or points VITE_API_URL at
     the API running on Avaneesh's or Sriraghava's machine.
  4. Everyone who runs a pipeline step needs the three CSVs in data/raw/.




TraceChain v2 - Team plan                                                                                             Page 1
 3. One-time setup (every person)
   git config --global user.name "Your Name"
   git config --global user.email "the-email-used-on-your-github-account"
   git clone <NEW-REPO-URL>
   cd tracechain
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

 Use the email that is linked to your own GitHub account, otherwise GitHub will not count the commits as yours.
 Avaneesh, once, on GitHub:
  1. Settings, Collaborators: add Sriraghava and Balu with write access.
  2. After the P0 pull request is merged: Settings, Branches, add a rule for main that requires a pull request
     before merging.


 4. Rules that keep three people from colliding
  • Nobody commits directly to main after P0. Everything goes through a branch and a pull request.
  • Edit only the files you own (table above). Before every commit run git status and check that only your
    files appear.
  • Review rotation: Avaneesh's pull requests are reviewed by Sriraghava, Sriraghava's by Balu, Balu's by
    Avaneesh. Reviewer reads the diff and the shown real output, then merges.
  • Merge with "Create a merge commit", not "Squash", so each person's individual commits stay under their
    own name.
  • Shared files (requirements.txt, SPEC.md, AGENTS.md): change them in a small branch of their own, and
    tell the others to pull.
  • Agents do not run git commit or git push. The human does, so the commit is under the right name.
  • Check authorship any time with: git log --format="%an | %s" -15


 5. Exact commands per person and step
 Pattern used everywhere: update main, make a branch, let the agent work, check status, commit, update from
 main, push, open a pull request.

 Avaneesh
 P0 (the repo and first commit already exist from PROMPTS.md section A, so push P0 on a branch):
   git checkout main
   git pull
   git checkout -b feat/scaffold

 Run the P0 prompt, then:
   git status
   git add .
   git commit -m "Scaffold project and inspect real data"
   git push -u origin feat/scaffold

 Open a pull request on GitHub. Sriraghava reviews and merges.
 P4:
   git checkout main
   git pull
   git checkout -b feat/api

 Run the P4 prompt, then:
   pytest
   git status



TraceChain v2 - Team plan                                                                                    Page 2
   git add .
   git commit -m "FastAPI service with score, trace, sample and metrics routes"
   git pull origin main
   git push -u origin feat/api

 P2:
   git checkout main
   git pull
   git checkout -b feat/vasp

 Run the P2 prompt, then:
   pytest
   git status
   git add .
   git commit -m "VASP proxy seeds and nearest-seed attribution index"
   git pull origin main
   git push -u origin feat/vasp

 Sriraghava
 P1:
   git checkout main
   git pull
   git checkout -b feat/data-prep

 Run the P1 prompt, then:
   pytest
   git status
   git add .
   git commit -m "Data preparation, temporal split, graph features"
   git pull origin main
   git push -u origin feat/data-prep

 P3 (start after feat/data-prep and feat/vasp are merged and you have Avaneesh's artifacts/ folder. To start
 earlier, tell the agent in the prompt to train only the baseline first, then add the graph-enhanced model when P2
 arrives):
   git checkout main
   git pull
   git checkout -b feat/model

 Run the P3 prompt, then:
   pytest
   git status
   git add .
   git commit -m "Train baseline and graph-enhanced models, metrics, figures, explanations"
   git pull origin main
   git push -u origin feat/model

 Balu
 P5:
   git checkout main
   git pull
   git checkout -b feat/web

 Run the P5 prompt, then:
   git status
   git add .
   git commit -m "Dashboard: gauge, trace graph, explanation, metrics tab"
   git pull origin main
   git push -u origin feat/web

 Smaller commits are better here. Commit after the search page, again after the graph, again after the metrics
 tab, using the same add and commit commands with a matching message.


TraceChain v2 - Team plan                                                                                    Page 3
 P6 (after the API and model are merged):
   git checkout main
   git pull
   git checkout -b feat/docs

 Run the P6 prompt, then:
   git status
   git add .
   git commit -m "Add screenshots, results and README"
   git pull origin main
   git push -u origin feat/docs


 6. If a pull request conflicts
   git checkout main
   git pull
   git checkout feat/your-branch
   git merge main

 Open the conflicted files, keep both sides' intent, then:
   git add .
   git commit -m "Merge main into feat/your-branch"
   git push


 7. Last step tonight
 Avaneesh merges the final pull requests, then everyone runs once:
   git checkout main
   git pull
   git log --format="%an | %s" -30

 Each of the three names should appear on real work. This is also the log you show your guide.




TraceChain v2 - Team plan                                                                        Page 4
