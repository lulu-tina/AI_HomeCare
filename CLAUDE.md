# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an AI dispatch/scheduling engine for home-care caregivers (長照居家照顧派單系統), built for an AI competition. The core logic reads caregiver, client, task, and historical-service data from an Excel workbook, matches caregivers to today's pending tasks under hard/soft constraints, solves the assignment as a mixed-integer program, and then reports a causal (DiD) comparison of AI vs. human dispatch outcomes.

There is no git repository, package manifest, or test suite yet — this is an early-stage prototype.

## File layout

- `caregiver_engine.py` — all core logic (data loading, `PipelineConfig` dataclass of tunable dispatch-policy parameters, Phase 1/2/3 functions). Both entry points below import from here; do not duplicate logic into either of them.
- `ai_caregiver_pipeline.py` — thin CLI entry point. Runs the three phases once with default `PipelineConfig` values and prints results to stdout.
- `app.py` — Streamlit web dashboard, button-triggered (not live-recompute). Sidebar exposes 5 curated policy sliders (`buffer_mins`, `travel_penalty_weight` — drives both `travel_penalty_weight` and `objective_travel_weight`, `urgent_priority_bonus`, a "continuity bonus" slider mapped to `preferred_caregiver_bonus`, a "skill bonus" slider mapped to both `cert_bonus_dementia` and `cert_bonus_other`); the rest of `PipelineConfig` stays at its dataclass defaults. Block ① lets users upload a replacement `.xlsx` (or fall back to `DEFAULT_EXCEL_PATH`) and edit `Caregiver_Profiles`/`Client_Profiles`/`Today_Pending_Tasks` directly via `st.data_editor` (row add/delete built in; a small "新增欄位" control adds columns) — edited data lives in `st.session_state` and is only reset when the upload/file identity changes, not on every rerun. Block ② runs Phases 1–3 against the edited data when "🚀 執行 AI 最佳化派單" is clicked, then shows KPIs (派單成功率, 平均轉場時間, 平均適配得分), a 4-panel `st.pyplot`/matplotlib+seaborn dashboard (score histogram, per-caregiver load, assigned/unassigned pie, Phase 3 DiD satisfaction bars), and a results table with a `assigned_results.csv` download. There is no Plotly map in this version.

## Running the pipeline

CLI (prints to stdout, no visualization):
```bash
python ai_caregiver_pipeline.py
```

Web dashboard (interactive, adjustable parameters):
```bash
streamlit run app.py
```

Requires: `pandas`, `numpy`, `openpyxl` (for reading `.xlsx`), `ortools` (for the MIP solver), `requests` (for OSRM travel-time queries — see "Travel time model" below), and — for the web dashboard only — `streamlit`, `matplotlib`, and `seaborn`. Both entry points must be run from the repo root since data loads via the relative path `./00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx` (used by `app.py` only as the fallback when no file is uploaded). Live scheduling runs need outbound internet access to `router.project-osrm.org`; without it, every travel-time lookup falls back to the flat-degree estimate (see below) and prints a warning per unique coordinate pair or batch.

There are no lint/test/build commands configured in this repo.

## Data source

All input data lives in `00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx`, with four sheets that map directly onto DataFrames in the script:

- **Caregiver_Profiles** (`df_cg`) — one row per 居服員 (caregiver): home lat/lon, certifications, daily hour caps, current fatigue (monthly cumulative hours), average satisfaction score, exclusion conditions (e.g. refuses walk-ups/pets/smoking environments), and up to two pre-existing time-blocked appointments for today.
- **Client_Profiles** (`df_cl`) — one row per 案家 (client household): service location, required caregiver gender, care level (CMS), special care needs, environment tags, heavy-lift-assistance flag, and historical preferred caregiver ID.
- **Today_Pending_Tasks** (`df_tasks`) — today's unassigned tasks: time window, duration, priority, and requested service type. Merged with `Client_Profiles` on 案家ID at load time to form `tasks`.
- **Historical_Service_Logs** (`df_hist`) — 300 historical service records with a `歷史媒合機制(Treatment)` flag (1 = AI-matched, 0 = human-matched) used purely for the Phase 3 outcome comparison, not for scheduling.

Column names are Traditional Chinese; keep them as-is when reading/writing this workbook (spot checks with the Bash tool over this data mangle the encoding — read Excel contents via Python/pandas or the Read tool instead of piping through the terminal).

Other files in `00_DB/` (PDFs, other `.xlsx` files) are reference/source materials for the long-term-care service statistics this project is built around — not consumed by the script.

## Pipeline architecture (`caregiver_engine.py`)

Three sequential phases, each a standalone function that both `ai_caregiver_pipeline.py` and `app.py` call in order:

1. **Phase 1 — `run_phase1_matching`** (candidate generation): for every (task × caregiver) pair, apply hard filters, then score survivors starting from `config.base_score` (default 60):
   - Hard filters (reject pair entirely): gender requirement, heavy-lift capability requirement, `EXCLUSION_MAP` environment conflicts (caregiver aversion vs. client environment tag), and daily-hour-cap overflow. These are not exposed as adjustable — they are eligibility rules, not dispatch policy.
   - Soft scoring (`score`, floored at 0): +certification match (`cert_bonus_dementia` / `cert_bonus_other`), +`preferred_caregiver_bonus` for historical preferred caregiver, +satisfaction deviation from `satisfaction_baseline` weighted by `satisfaction_weight`, −travel-time penalty (weighted by `travel_penalty_weight`, capped at `travel_penalty_cap`) computed from `calc_travel_minutes` (real OSRM road-network minutes, falling back to the `calc_distance_km` flat-degree approximation on API failure — see "Travel time model" below), −fatigue penalty (`fatigue_weight`) based on **service-intensity-weighted** monthly cumulative hours vs. `fatigue_reference_hours` (see "Fatigue model" below).
   - Output: `df_matches`, one row per surviving (task, caregiver) candidate pair.

2. **Phase 2 — `run_phase2_optimization`** (time-conflict filtering + OR-Tools MIP):
   - Drops candidate pairs that conflict with a caregiver's pre-existing appointments today, accounting for travel time plus `config.buffer_mins` transition buffer (parking/stairs/access/handoff).
   - Builds a binary assignment MIP with OR-Tools' `SCIP` backend: decision var `X[task_id, cg_id]`; constraint 1 — each task assigned to at most one caregiver; constraint 2 — no caregiver double-booked into two new tasks whose windows (plus travel+buffer) overlap.
   - Objective: maximize `match_score − objective_travel_weight×travel_minutes + priority_bonus` (priority_bonus is `urgent_priority_bonus` if the task priority string contains "緊急" (urgent), else `normal_priority_bonus`) — this weighting is where dispatch policy actually lives, not in the hard filters.
   - Output: dict with `df_valid`, `df_result` (final task→caregiver assignment, includes lat/lon for map plotting), `status`, `assigned_count`.

3. **Phase 3 — `run_phase3_did`**: splits `Historical_Service_Logs` into AI-treated vs. human-control groups on `歷史媒合機制(Treatment)` and reports the uplift in mean satisfaction and reduction in early-termination rate attributable to AI matching. This is a retrospective analysis over historical data, independent of Phases 1–2's live scheduling run, and is **not** affected by `PipelineConfig` — it does not take config as an argument.

All dispatch-policy coefficients named above live in the `PipelineConfig` dataclass; `app.py`'s sidebar exposes one slider per field. When adding a new tunable coefficient, add it to `PipelineConfig` first, thread it through the relevant Phase function, then add a sidebar slider — never hardcode a new policy constant directly in `app.py` or `ai_caregiver_pipeline.py`.

### Travel time model (OSRM)

- `calc_travel_minutes(lat1, lon1, lat2, lon2, config)` is the single entry point for transition travel time, used identically in four places: Phase 1 scoring, Phase 2's existing-schedule conflict check, Phase 2's new-task-pair conflict check, and `_diagnose_caregiver_change`'s pair check — keep them consistent if changing the transit-time model.
- Internally it calls `get_osrm_travel_time`, which queries the free OSRM public API (`router.project-osrm.org`, `biking` profile) for real road-network minutes between two points, backed by a module-level dict cache (`_OSRM_TRAVEL_TIME_CACHE`, keyed on coordinates rounded to 6 decimals). On request timeout (`OSRM_TIMEOUT_SECONDS`), network failure, or a non-`Ok` response, it prints an `[OSRM Warning]` log and falls back to the `calc_distance_km` flat-degree estimate (1° lat ≈ 111 km, 1° lon ≈ 101 km) × `travel_min_per_km`.
- Before their pairwise loops, `run_phase1_matching` and `run_phase2_optimization` each call `prefetch_osrm_travel_times(origins, destinations, travel_min_per_km)`, which batches the whole origin×destination coordinate set into one (or a few, chunked at `OSRM_TABLE_MAX_COORDS`) OSRM `/table` matrix requests and pre-populates the shared cache — this is what keeps a 20-caregiver × 25-task run to a handful of HTTP calls instead of hundreds; without it the per-pair fallback path still works, just far slower. A failed batch prints a warning and leaves those pairs to be resolved (and independently degrade) on the next per-pair `get_osrm_travel_time` call.
- `OSRM_HOST` / `OSRM_PROFILE` / timeouts are module-level constants, not `PipelineConfig` fields (same pattern as `calc_distance_km`'s constants) — they're plumbing, not dispatch policy.

### Fatigue model (service-intensity-weighted)

- `get_service_intensity_weight(task)` maps a task to a strain coefficient via `SERVICE_INTENSITY_WEIGHT`: 1.5 for heavy transfer/joint mobility (`需重度移位協助(0/1)==1`), 1.2 for meal care/tube feeding/hygiene (`特殊照護需求` in `{餐食照顧/管灌, 管路安全與特殊日常照護}`), else 0.8 (general housework/companionship, including dementia care).
- Phase 1 no longer uses the flat `cg["當月累計服務時數(疲勞度)"] / fatigue_reference_hours` ratio; it now multiplies the caregiver's cumulative hours by the *current candidate task's* intensity weight first: `fatigue_penalty = (cg累計時數 × intensity_weight) / fatigue_reference_hours × fatigue_weight`. This is evaluated per (task, caregiver) pair (Phase 1 has no per-service-type history to sum over), so assigning a physically demanding task to an already-fatigued caregiver is penalized more than assigning a light one.

### Other invariants when modifying this pipeline


- `EXCLUSION_MAP` keys are caregiver `特殊排斥條件` values; values are the matching `案家環境特徵` string they conflict with. Both sides are exact-string-matched against the Excel data, so any new exclusion category must be added to both the map and used consistently in the source spreadsheet.
- Time strings in the workbook use `"HH:MM-HH:MM"` (or the literal `"無既定行程"` sentinel for "no existing appointment"); `parse_time` and the task-time parsing in Phase 2 assume this exact format.
- `app.py` only re-runs Phases 1–3 when the "🚀 執行 AI 最佳化派單" button is clicked (not on every slider/data-editor change); the last result is kept in `st.session_state["last_result"]` and re-displayed on subsequent reruns until the button is clicked again. Raw sheet reads are `st.cache_data`-cached (keyed on upload identity or file path + mtime), but the edited copies of `Caregiver_Profiles`/`Client_Profiles`/`Today_Pending_Tasks` live separately in `st.session_state` (`edit_cg`/`edit_cl`/`edit_tasks`) so user edits survive reruns and aren't clobbered by the cache.
- `setup_chinese_font()` in `app.py` (sets `plt.rcParams["font.sans-serif"]`) must be called *after* `sns.set_style(...)`, not before — seaborn's `set_style` overwrites `font.sans-serif` back to its own default (Arial-first) list, which silently drops all CJK glyphs from the matplotlib dashboard.
