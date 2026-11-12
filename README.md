# TraceChain

Bitcoin wallet illicit-risk scoring and nearest VASP-like hub attribution on the Elliptic++ Actors dataset.

## Scope

- Public data has no real exchange labels. Elliptic++ labels addresses only as illicit, licit, or unknown.
- VASP seeds are structural proxies (high degree, high fan-in/fan-out, long activity), with optional registry override (`data/registry/vasp_registry.csv`).
- Outputs represent the nearest likely VASP-like hub, not verified legal entity identity. Risk scores are for analyst triage only.
- Proof of concept on Bitcoin only (Elliptic++ Actors dataset).

Full reproduction steps and evaluation results will be populated in P6.

