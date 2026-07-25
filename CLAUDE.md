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

Requires: `pandas`, `numpy`, `openpyxl` (for reading `.xlsx`), `ortools` (for the MIP solver), and — for the web dashboard only — `streamlit`, `matplotlib`, and `seaborn`. Both entry points must be run from the repo root since data loads via the relative path `./00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx` (used by `app.py` only as the fallback when no file is uploaded).

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
   - Soft scoring (`score`, floored at 0): +certification match (`cert_bonus_dementia` / `cert_bonus_other`), +`preferred_caregiver_bonus` for historical preferred caregiver, +satisfaction deviation from `satisfaction_baseline` weighted by `satisfaction_weight`, −travel-time penalty (weighted by `travel_penalty_weight`, capped at `travel_penalty_cap`) computed from `calc_distance_km` (a flat-degree approximation, not haversine — 1° lat ≈ 111 km, 1° lon ≈ 101 km, tuned for Taiwan's latitude; not exposed as adjustable), −fatigue penalty (`fatigue_weight`) based on monthly cumulative hours vs. `fatigue_reference_hours`.
   - Output: `df_matches`, one row per surviving (task, caregiver) candidate pair.

2. **Phase 2 — `run_phase2_optimization`** (time-conflict filtering + OR-Tools MIP):
   - Drops candidate pairs that conflict with a caregiver's pre-existing appointments today, accounting for travel time plus `config.buffer_mins` transition buffer (parking/stairs/access/handoff).
   - Builds a binary assignment MIP with OR-Tools' `SCIP` backend: decision var `X[task_id, cg_id]`; constraint 1 — each task assigned to at most one caregiver; constraint 2 — no caregiver double-booked into two new tasks whose windows (plus travel+buffer) overlap.
   - Objective: maximize `match_score − objective_travel_weight×travel_minutes + priority_bonus` (priority_bonus is `urgent_priority_bonus` if the task priority string contains "緊急" (urgent), else `normal_priority_bonus`) — this weighting is where dispatch policy actually lives, not in the hard filters.
   - Output: dict with `df_valid`, `df_result` (final task→caregiver assignment, includes lat/lon for map plotting), `status`, `assigned_count`.

3. **Phase 3 — `run_phase3_did`**: splits `Historical_Service_Logs` into AI-treated vs. human-control groups on `歷史媒合機制(Treatment)` and reports the uplift in mean satisfaction and reduction in early-termination rate attributable to AI matching. This is a retrospective analysis over historical data, independent of Phases 1–2's live scheduling run, and is **not** affected by `PipelineConfig` — it does not take config as an argument.

All dispatch-policy coefficients named above live in the `PipelineConfig` dataclass; `app.py`'s sidebar exposes one slider per field. When adding a new tunable coefficient, add it to `PipelineConfig` first, thread it through the relevant Phase function, then add a sidebar slider — never hardcode a new policy constant directly in `app.py` or `ai_caregiver_pipeline.py`.

### Key invariants when modifying this pipeline

- `calc_distance_km` and the `travel_mins = dist_km * config.travel_min_per_km + config.buffer_mins` formula are used identically in three places (scoring, existing-schedule conflict check, new-task-pair conflict check) — keep them consistent if changing the transit-time model.
- `EXCLUSION_MAP` keys are caregiver `特殊排斥條件` values; values are the matching `案家環境特徵` string they conflict with. Both sides are exact-string-matched against the Excel data, so any new exclusion category must be added to both the map and used consistently in the source spreadsheet.
- Time strings in the workbook use `"HH:MM-HH:MM"` (or the literal `"無既定行程"` sentinel for "no existing appointment"); `parse_time` and the task-time parsing in Phase 2 assume this exact format.
- `app.py` only re-runs Phases 1–3 when the "🚀 執行 AI 最佳化派單" button is clicked (not on every slider/data-editor change); the last result is kept in `st.session_state["last_result"]` and re-displayed on subsequent reruns until the button is clicked again. Raw sheet reads are `st.cache_data`-cached (keyed on upload identity or file path + mtime), but the edited copies of `Caregiver_Profiles`/`Client_Profiles`/`Today_Pending_Tasks` live separately in `st.session_state` (`edit_cg`/`edit_cl`/`edit_tasks`) so user edits survive reruns and aren't clobbered by the cache.
- `setup_chinese_font()` in `app.py` (sets `plt.rcParams["font.sans-serif"]`) must be called *after* `sns.set_style(...)`, not before — seaborn's `set_style` overwrites `font.sans-serif` back to its own default (Arial-first) list, which silently drops all CJK glyphs from the matplotlib dashboard.
