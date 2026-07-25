"""長照居家照顧派單系統 - 核心運算引擎

將原始三階段流程（適配度評分 -> OR-Tools 最佳化 -> DiD 效益回溯）拆成可
重複呼叫的函式，並把所有具派單政策意義的係數集中到 PipelineConfig，供
CLI (ai_caregiver_pipeline.py) 與網頁儀表板 (app.py) 共用同一份邏輯。
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DEFAULT_EXCEL_PATH = "./00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx"

# 環境排斥條件對照表 (居服員排斥 -> 案家環境)，兩側皆為 Excel 原始字串，需完全相符
EXCLUSION_MAP = {
    "拒爬高樓(無電梯)": "傳統公寓無電梯",
    "拒寵物環境": "有養寵物",
    "拒菸害環境": "有抽菸",
}


@dataclass
class PipelineConfig:
    """所有可調整的派單政策參數，預設值與原始腳本一致。"""

    # 轉場緩衝與交通時間模型
    buffer_mins: float = 15.0
    travel_min_per_km: float = 3.0

    # Phase 1：軟性適配度評分
    base_score: float = 60.0
    cert_bonus_dementia: float = 15.0
    cert_bonus_other: float = 10.0
    preferred_caregiver_bonus: float = 20.0
    satisfaction_baseline: float = 4.0
    satisfaction_weight: float = 10.0
    travel_penalty_weight: float = 0.5
    travel_penalty_cap: float = 15.0
    fatigue_reference_hours: float = 160.0
    fatigue_weight: float = 10.0

    # Phase 2：OR-Tools 目標函數權重
    objective_travel_weight: float = 0.5
    urgent_priority_bonus: float = 50.0
    normal_priority_bonus: float = 20.0


# ==========================================
# 資料載入
# ==========================================
def load_data(excel_path: str = DEFAULT_EXCEL_PATH):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到檔案 {excel_path}，請確認檔案與腳本在同一目錄下。")

    df_cg = pd.read_excel(excel_path, sheet_name="Caregiver_Profiles")
    df_cl = pd.read_excel(excel_path, sheet_name="Client_Profiles")
    df_tasks = pd.read_excel(excel_path, sheet_name="Today_Pending_Tasks")
    df_hist = pd.read_excel(excel_path, sheet_name="Historical_Service_Logs")

    tasks = df_tasks.merge(df_cl, on="案家ID", how="left")
    return df_cg, df_cl, df_tasks, df_hist, tasks


# ==========================================
# 共用工具
# ==========================================
def calc_distance_km(lat1, lon1, lat2, lon2):
    """以台灣緯度估算直線距離 km（1度緯度約111km，1度經度約101km）。"""
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 101.0
    return np.sqrt(dlat**2 + dlon**2)


def parse_time(time_str):
    if pd.isna(time_str) or time_str == "無既定行程":
        return None, None
    parts = str(time_str).split("-")
    return (
        datetime.strptime(parts[0].strip(), "%H:%M"),
        datetime.strptime(parts[1].strip(), "%H:%M"),
    )


# ==========================================
# Phase 1: 適配度過濾與評分機制
# ==========================================
def run_phase1_matching(tasks: pd.DataFrame, df_cg: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    match_results = []

    for _, task in tasks.iterrows():
        t_id = task["任務ID"]
        c_id = task["案家ID"]
        req_gender = task["指定居服員性別"]
        req_lift = task["需重度移位協助(0/1)"]
        client_env = task["案家環境特徵"]
        client_lat = task["服務地點_緯度"]
        client_lon = task["服務地點_經度"]
        task_duration_hrs = task["服務歷時(分鐘)"] / 60.0
        pref_cg = task["歷史首選居服員ID"]
        req_type = task["特殊照護需求"]

        for _, cg in df_cg.iterrows():
            cg_id = cg["居服員ID"]

            # --- Hard Constraints (硬性過濾，不合格者直接剔除) ---
            if req_gender == "限女性" and cg["性別"] != "女":
                continue
            if req_gender == "限男性" and cg["性別"] != "男":
                continue

            if req_lift == 1 and cg["具備重度移位體力(0/1)"] == 0:
                continue

            cg_excl = cg["特殊排斥條件"]
            if cg_excl in EXCLUSION_MAP and client_env == EXCLUSION_MAP[cg_excl]:
                continue

            if (
                cg["今日已佔用工時(小時)"] + task_duration_hrs
                > cg["每日工時上限(小時)"]
            ):
                continue

            # --- Soft Match Scoring ---
            score = config.base_score

            cert = cg["核心專長證照"]
            if req_type == "失智引導與精神陪伴" and cert in [
                "失智症照顧專長",
                "精神疾病照顧專長",
            ]:
                score += config.cert_bonus_dementia
            elif req_type in ["管路安全與特殊日常照護", "餐食照顧/管灌"] and cert == "單一級照服證照":
                score += config.cert_bonus_other

            if cg_id == pref_cg:
                score += config.preferred_caregiver_bonus

            score += (cg["歷史滿意度均值"] - config.satisfaction_baseline) * config.satisfaction_weight

            dist_km = calc_distance_km(
                cg["服務起點_緯度(家)"],
                cg["服務起點_經度(家)"],
                client_lat,
                client_lon,
            )
            travel_time_min = dist_km * config.travel_min_per_km
            score -= min(travel_time_min * config.travel_penalty_weight, config.travel_penalty_cap)

            fatigue_penalty = (cg["當月累計服務時數(疲勞度)"] / config.fatigue_reference_hours) * config.fatigue_weight
            score -= fatigue_penalty

            match_results.append(
                {
                    "任務ID": t_id,
                    "案家ID": c_id,
                    "居服員ID": cg_id,
                    "適配度分數": round(max(score, 0), 2),
                    "預估交通時間(分)": round(travel_time_min, 1),
                    "任務開始時間": task["時間窗_開始"],
                    "任務結束時間": task["時間窗_結束"],
                    "優先級": task["任務優先級"],
                    "地點緯度": client_lat,
                    "地點經度": client_lon,
                }
            )

    return pd.DataFrame(match_results)


# ==========================================
# Phase 2: 時空路徑衝突過濾 + OR-Tools 多目標最佳化
# ==========================================
def run_phase2_optimization(
    df_matches: pd.DataFrame,
    tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    config: PipelineConfig,
):
    # 建立居服員今日既定行程時間阻擋塊 (Time Blocks)
    cg_busy = {}
    for _, cg in df_cg.iterrows():
        cg_id = cg["居服員ID"]
        busy_intervals = []
        for i in [1, 2]:
            t1, t2 = parse_time(cg[f"今日既定行程{i}_時段"])
            if t1:
                busy_intervals.append(
                    (
                        t1,
                        t2,
                        cg[f"今日既定行程{i}_地點緯度"],
                        cg[f"今日既定行程{i}_地點經度"],
                    )
                )
        cg_busy[cg_id] = busy_intervals

    # 建立任務時間表
    task_times = {}
    for _, task in tasks.iterrows():
        t_id = task["任務ID"]
        t1 = datetime.strptime(task["時間窗_開始"].strip(), "%H:%M")
        t2 = datetime.strptime(task["時間窗_結束"].strip(), "%H:%M")
        task_times[t_id] = (t1, t2, task["服務地點_緯度"], task["服務地點_經度"])

    # 過濾掉與既定行程衝突（含轉場緩衝時間）的配對
    valid_rows = []
    for _, row in df_matches.iterrows():
        t_id = row["任務ID"]
        cg_id = row["居服員ID"]
        t_start, t_end, t_lat, t_lon = task_times[t_id]

        conflict = False
        for b_start, b_end, b_lat, b_lon in cg_busy[cg_id]:
            dist = calc_distance_km(t_lat, t_lon, b_lat, b_lon)
            travel_mins = dist * config.travel_min_per_km + config.buffer_mins
            if not (
                t_end + timedelta(minutes=travel_mins) <= b_start
                or b_end + timedelta(minutes=travel_mins) <= t_start
            ):
                conflict = True
                break

        if not conflict:
            valid_rows.append(row.to_dict())

    df_valid = pd.DataFrame(valid_rows)

    result = {
        "df_valid": df_valid,
        "status": None,
        "df_result": pd.DataFrame(),
        "assigned_count": 0,
    }

    if df_valid.empty:
        return result

    # 建立 OR-Tools 混合整數規劃 (MIP) 求解器
    solver = pywraplp.Solver.CreateSolver("SCIP")

    X = {}
    for _, row in df_valid.iterrows():
        X[(row["任務ID"], row["居服員ID"])] = solver.IntVar(
            0, 1, f"x_{row['任務ID']}_{row['居服員ID']}"
        )

    # 限制條件 1：每個任務至多只能派給一位居服員
    for t_id in df_valid["任務ID"].unique():
        solver.Add(
            sum(
                X[(t_id, cg_id)]
                for cg_id in df_valid[df_valid["任務ID"] == t_id]["居服員ID"]
            )
            <= 1
        )

    # 限制條件 2：同一居服員若被派兩個新任務，時間不能重疊且須預留轉場緩衝
    for cg_id in df_valid["居服員ID"].unique():
        cg_tasks = df_valid[df_valid["居服員ID"] == cg_id]["任務ID"].tolist()
        for i in range(len(cg_tasks)):
            for j in range(i + 1, len(cg_tasks)):
                t1_id, t2_id = cg_tasks[i], cg_tasks[j]
                t1_start, t1_end, t1_lat, t1_lon = task_times[t1_id]
                t2_start, t2_end, t2_lat, t2_lon = task_times[t2_id]

                dist = calc_distance_km(t1_lat, t1_lon, t2_lat, t2_lon)
                travel_mins = dist * config.travel_min_per_km + config.buffer_mins

                if not (
                    t1_end + timedelta(minutes=travel_mins) <= t2_start
                    or t2_end + timedelta(minutes=travel_mins) <= t1_start
                ):
                    solver.Add(X[(t1_id, cg_id)] + X[(t2_id, cg_id)] <= 1)

    # 目標函數：Z = 適配度分數 - w*交通時間 + 優先級權重
    objective = solver.Objective()
    for _, row in df_valid.iterrows():
        w_match = row["適配度分數"]
        w_travel = row["預估交通時間(分)"]
        priority_bonus = (
            config.urgent_priority_bonus
            if "緊急" in str(row["優先級"])
            else config.normal_priority_bonus
        )

        coeff = w_match - (config.objective_travel_weight * w_travel) + priority_bonus
        objective.SetCoefficient(X[(row["任務ID"], row["居服員ID"])], coeff)

    objective.SetMaximization()

    status = solver.Solve()
    result["status"] = status

    if status == pywraplp.Solver.OPTIMAL:
        assigned_count = 0
        results = []
        for (t_id, cg_id), var in X.items():
            if var.solution_value() > 0.5:
                assigned_count += 1
                row_data = df_valid[
                    (df_valid["任務ID"] == t_id) & (df_valid["居服員ID"] == cg_id)
                ].iloc[0]
                results.append(
                    {
                        "任務ID": t_id,
                        "案家ID": row_data["案家ID"],
                        "派單居服員": cg_id,
                        "適配分數": row_data["適配度分數"],
                        "預估車程(分)": row_data["預估交通時間(分)"],
                        "服務時段": f"{row_data['任務開始時間']}-{row_data['任務結束時間']}",
                        "任務優先級": row_data["優先級"],
                        "地點緯度": row_data["地點緯度"],
                        "地點經度": row_data["地點經度"],
                    }
                )

        df_result = pd.DataFrame(results)
        if not df_result.empty:
            df_result = df_result.sort_values(by="任務ID")
        result["df_result"] = df_result
        result["assigned_count"] = assigned_count

    return result


# ==========================================
# Phase 3: 效益產出評估 (DiD 雙重差分比較)
# ==========================================
def run_phase3_did(df_hist: pd.DataFrame) -> dict:
    ai_group = df_hist[df_hist["歷史媒合機制(Treatment)"] == 1]
    human_group = df_hist[df_hist["歷史媒合機制(Treatment)"] == 0]

    ai_sat = ai_group["案家滿意度(1-5)"].mean()
    human_sat = human_group["案家滿意度(1-5)"].mean()
    ai_dropout = ai_group["不滿意導致提早結案(0/1)"].mean()
    human_dropout = human_group["不滿意導致提早結案(0/1)"].mean()

    return {
        "ai_group": ai_group,
        "human_group": human_group,
        "ai_sat": ai_sat,
        "human_sat": human_sat,
        "ai_dropout": ai_dropout,
        "human_dropout": human_dropout,
        "uplift_sat": ai_sat - human_sat,
        "uplift_dropout": human_dropout - ai_dropout,
    }
