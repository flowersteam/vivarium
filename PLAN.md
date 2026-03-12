# Vivarium Release Plan

_Living document — updated at the end of every working session._
_Last updated: 2026-03-11_

---

## Goal

Release a clean, documented version of Vivarium demonstrating its three core use cases:

1. **Headless simulation in JAX** — for researchers comfortable with JAX
2. **Programmatic Pythonic control** — for CS students who know Python but not JAX
3. **Web interface** — for students with little or no programming background

The release standard is "good enough": clean, usable, and honest — not top-notch.

---

## Phases

### Phase 1 — Codebase Audit
_Goal: understand the true current state before planning details._

**Approach**: bottom-up audit. Explore each package independently, produce a local `CLAUDE.md` per package, then synthesize into the global picture.

#### Step 1 — Per-package audit
For each package under `vivarium/`, audit independently and produce a local `CLAUDE.md` capturing the following. Before starting each audit, read any already-produced local `CLAUDE.md` files for context on previously audited packages. Each local `CLAUDE.md` should include: purpose, key files, public API, internal patterns, known issues, and state (solid / brittle / incomplete). Each audit should also flag: unnecessary complexity, architecture flaws, inconsistencies, missing documentation, missing test coverage for critical logic, and potential refactoring or renaming opportunities.

Packages to audit:
- [x] `vivarium/environment/` (state, components, rendering)
- [x] `vivarium/simulator/` (simulator core, gRPC server/client)
- [x] `vivarium/controllers/` (client-side controllers)
- [x] `vivarium/interface/` (Panel web UI)
- [x] `vivarium/utils/` (runtime, scene configs, server management)
- [x] `conf/` (Hydra configuration structure)
- [x] `scripts/` (entry points)
- [x] `tests/` (coverage, passing/failing, gaps)
- [x] `notebooks/` (tutorials, sessions — what's current vs outdated)

#### Step 2 — Cross-package synthesis

**Approach**: Read all local CLAUDE.md files first for the map, then re-explore the actual code to spot cross-cutting patterns (shared base classes, repeated workarounds, naming inconsistencies, common anti-patterns). Update local CLAUDE.md files if the cross-package view reveals something missed per-package.

- [x] Map inter-package dependencies (textual dependency map in global CLAUDE.md)
- [x] Identify cross-package issues: architectural flaws, unnecessary coupling, inconsistent patterns between packages
- [x] Audit the `from_config` pattern across all classes: consistency, workarounds, mismatch between Hydra config structure and `__init__` signatures, and whether all classes can be instantiated via normal `__init__` without Hydra
- [x] Identify other cross-cutting patterns and concerns not visible from single-package audits
- [x] Identify dead code and files to remove
- [x] For each audience journey (researcher, CS student, web UI), assess readiness: what works, what's missing, what's blocking
- [x] Build a fresh global `CLAUDE.md` from scratch, based solely on the local audit files and synthesis findings. Do NOT read `docs/claude_md_v1.md` (the previous version) until the new one is written. Only then, review the old version to check for anything valuable that was missed.

#### Step 3 — Planning discussion
_Deliberative conversation to agree on Phase 2–4 task lists. Full discussion log in `docs/audit_discussion.md`._

**Format**: One topic at a time, layered from architectural decisions down to specifics. Before presenting topics, carefully read the global `CLAUDE.md` and analyze how issues from local `CLAUDE.md` files relate to each other — group symptoms of the same root cause together. Before proposing solutions, analyze the relevant code files. 

**Layers**:
1. Architectural/structural decisions (high-level, cascade down)
2. Per-package cleanup priorities (within agreed architecture)
3. Documentation and feature scope (what we commit to for the release)

**Output**:
- [ ] Populate Phase 2 tasks (cleanup & refactoring)
- [ ] Populate Phase 3 tasks (documentation & tutorials)
- [ ] Populate Phase 4 tasks (feature fixes)
- [ ] Review and agree on task lists before any changes begin

---

### Phase 2 — Cleanup & Refactoring
_Detailed tasks TBD after audit._

Rough scope:
- Remove outdated/incomplete files
- Refactor module organization for clarity if needed
- Ensure all tests pass after changes
- Update CLAUDE.md to reflect any structural changes

---

### Phase 3 — Documentation & Tutorials
_Detailed tasks TBD after audit._

Rough scope, one track per audience:
- **Researchers**: document headless JAX simulation workflow and document each class or function
- **CS students**: Using the content of `notebooks/sessions/` series, write a tutorial explaining all the functionalities demonstrated in Sessions 1 to 4 and the miniproject
- **Younger students**: document the web interface workflow
- **Developers/students extending the code**: write a "how to add a new component" guide (critical for incoming Master's student)

---

### Phase 4 — Feature Fixes & Additions _(time permitting)_
_Detailed tasks TBD after audit._

Rough scope:
- Fix brittle components: reproduction, consumption
- Extend Braitenberg sensing/motor abilities
- Add logging for headless simulation

---

### Phase 5 — Release Prep
_Detailed tasks TBD after audit._

Rough scope:
- Full test run on a clean environment
- Version bump and changelog
- GitHub release

---

## Key Decisions

| Decision | Choice |
|---|---|
| SceneBuilder | Removed — not in release scope |
| Multi-spawn | Implemented and tested — keep |
| Target audiences | Researchers, CS students, younger students (web) |
| Release standard | Clean and usable, not exhaustive |

## Open Questions

- Which specific files/modules to remove? → determined by audit
- Is module reorganization needed, and at what scope? → determined by audit
- Which Phase 4 components can realistically be fixed within the timeline? → determined by audit
