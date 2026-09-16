# Session status — 2026-09-14 (overnight)

Written by the main agent before the user slept, so that a compacted context, a
new session, or a different reader can resume without asking anyone. Everything
here is verifiable from the repository; nothing depends on conversation memory.

## Where things stand

| Item | Value |
| --- | --- |
| Branch | `integration`, pushed to `origin` (32 commits, in sync at the time of writing) |
| HEAD | `249ac51` + the cleanup commits before it; check `git log --oneline -5` |
| Goal | active, round budget 300 (raised from 40 by the user's request) |
| Plan of record | `final_connection.md` — §5 steps, §6.2 subagent budget rules, §7 decision log D-01…D-22, §8 core measurements |
| Entry point for humans | `README.md` (rewritten in `32ee369`) |
| Frozen feature contract | `results/kuleuven_audit.md` §7–8 |

## Done and verified

Steps 1–7 plus the D-22 cleanup: baseline, KU Leuven conversion (320 trials), envelope
precompute, data audit + contract freeze, decoder training (two contracts), transport port,
frontend vendor, plus `antio` test fix, ANT report verification/correction, and the
repository-wide cleanup (`4f4a9e5`, `32ee369`, `c95d075`).

Independently re-run by the main agent at the last check: `scripts/run_tests.py --suite
auditory --suite streaming` → 477 tests green. The frontend suite needs
`ATTUNE_PYTHON=<repo>\.venv\Scripts\python.exe` on Windows.

## The numbers that matter

| Contract | Split | Balanced accuracy |
| --- | --- | --- |
| 64ch | held-out story | 0.6195 |
| 64ch | leave-one-subject-out | 0.6115 |
| 20ch (live rig) | held-out story | 0.5974 |
| 20ch (live rig) | leave-one-subject-out | 0.5891 |

Majority null is **0.6586** (labels are 66.1 % A by time), so raw accuracy below the null is
expected and only balanced accuracy is meaningful — this is a **weak effect**, and the plan
says so rather than dressing it up. Window length curve (64ch balanced): 0.637 (5 s) → 0.698
(10 s) → 0.796 (30 s) → 0.875 (60 s).

## In flight when this was written

Step 5.5 (offset sweep) — script and test written, `results/aad_shift_sweep_20260913-032304.json`
appeared; its own commit is pending. Next after it: steps 8, 9, 9.5, 10/10.5, 11, 12.

## Constraints later steps must carry (plan §3.17)

1. The default quality policy halts on KU Leuven, so the demo run must choose a relaxed policy
   **explicitly** and record it.
2. The chain drops roughly the last second of every trial; window counts come from measurement.
3. `evaluation.assert_held_out` refuses leave-one-subject-out by construction.

## How to resume

Read `final_connection.md` §5 for the step table with budgets and evidence, then dispatch the
next step to a subagent with that row quoted. Never let the main agent write implementation
code (§9). One deliverable gets at most three subagents; after that, report to the user.
