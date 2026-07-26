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
import requests
from ortools.linear_solver import pywraplp

DEFAULT_EXCEL_PATH = "./00_DB/AI_Caregiver_Allocation_Ultimate_Database.xlsx"

# OSRM 公用路網 API 設定：預設查詢騎乘(機車/腳踏車)路網時間，逾時即降級為概算公式。
OSRM_HOST = "http://router.project-osrm.org"
OSRM_PROFILE = "biking"
OSRM_TIMEOUT_SECONDS = 3.0
OSRM_TABLE_TIMEOUT_SECONDS = 10.0
OSRM_TABLE_MAX_COORDS = 90  # 單次 /table 批次查詢座標數上限，避免超出公用伺服器限制

# 服務項目強度加權係數：依體力耗費強度分級，用於疲勞度模型。
SERVICE_INTENSITY_WEIGHT = {
    "重度移位": 1.5,  # 重度移位／肢體關節活動
    "餐食管灌": 1.2,  # 餐食照顧／管灌洗頭
    "一般照護": 0.8,  # 一般家務／陪伴看護
}

# 環境排斥條件對照表 (居服員排斥 -> 案家環境)，兩側皆為 Excel 原始字串，需完全相符
EXCLUSION_MAP = {
    "拒爬高樓(無電梯)": "傳統公寓無電梯",
    "拒寵物環境": "有養寵物",
    "拒菸害環境": "有抽菸",
}

# 法定 20 小時特照培訓：案家需求類別 -> 合格居服員「核心專長證照」集合。
# 居服員若不具備對應證照，該配對於 Phase 1 直接硬性剔除，不得派單。
SPECIAL_CERT_REQUIREMENTS = {
    "失智引導與精神陪伴": {"失智症照顧專長", "精神疾病照顧專長"},
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
    preferred_caregiver_bonus: float = 60.0
    continuity_performance_bonus: float = 15.0
    continuity_satisfaction_threshold: float = 4.3
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


# 字典快取已查詢過的經緯度對 -> 路網時間(分鐘)，Phase 1 / Phase 2 共用同一份快取，
# 避免同一對座標於不同階段重複發送 API 請求。
_OSRM_TRAVEL_TIME_CACHE: dict = {}


def _round_coord(v: float) -> float:
    return round(float(v), 6)


def _osrm_fallback_minutes(lat1, lon1, lat2, lon2, travel_min_per_km, reason) -> float:
    """降級為 Haversine/歐式距離估算，並印出 Warning Log。"""
    print(
        f"[OSRM Warning] 路網查詢失敗 ({lat1},{lon1}) -> ({lat2},{lon2})，"
        f"降級為 Haversine/歐式距離估算: {reason}"
    )
    return calc_distance_km(lat1, lon1, lat2, lon2) * travel_min_per_km


def get_osrm_travel_time(lat1, lon1, lat2, lon2, travel_min_per_km=3.0):
    """查詢 OSRM 公用路網 API，回傳兩點間真實路網騎乘時間（分鐘）。

    以字典快取已查詢過的經緯度對，避免重複發送 API 請求（大量座標對可先呼叫
    `prefetch_osrm_travel_times` 以單次 /table 批次請求暖身快取）。若 API 逾時、
    網路斷線或回傳失敗，自動降級為 calc_distance_km 的概算距離公式，並印出
    Warning Log，確保 Phase 1 / Phase 2 的排程運算不因外部服務中斷而失敗。
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    key = (_round_coord(lat1), _round_coord(lon1), _round_coord(lat2), _round_coord(lon2))
    if key in _OSRM_TRAVEL_TIME_CACHE:
        return _OSRM_TRAVEL_TIME_CACHE[key]

    url = f"{OSRM_HOST}/route/v1/{OSRM_PROFILE}/{lon1},{lat1};{lon2},{lat2}"
    try:
        resp = requests.get(url, params={"overview": "false"}, timeout=OSRM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            raise ValueError(f"OSRM 回傳異常狀態: {data.get('code')}")
        travel_min = data["routes"][0]["duration"] / 60.0
    except Exception as exc:
        travel_min = _osrm_fallback_minutes(lat1, lon1, lat2, lon2, travel_min_per_km, exc)

    _OSRM_TRAVEL_TIME_CACHE[key] = travel_min
    return travel_min


def _fetch_osrm_table_chunk(origins, destinations, travel_min_per_km):
    """對一批 origin x destination 座標呼叫 OSRM /table 矩陣 API，一次查詢多組配對的
    路網時間，寫入共用快取。origins / destinations 皆為已四捨五入的 (lat, lon) tuple 清單。
    """
    coords = origins + destinations
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    sources = ";".join(str(i) for i in range(len(origins)))
    dest_indices = ";".join(str(len(origins) + i) for i in range(len(destinations)))
    url = f"{OSRM_HOST}/table/v1/{OSRM_PROFILE}/{coord_str}"

    try:
        resp = requests.get(
            url,
            params={"annotations": "duration", "sources": sources, "destinations": dest_indices},
            timeout=OSRM_TABLE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM /table 回傳異常狀態: {data.get('code')}")
        durations = data["durations"]
    except Exception as exc:
        print(
            f"[OSRM Warning] /table 批次路網查詢失敗（{len(origins)}x{len(destinations)} 座標對），"
            f"個別配對將於逐筆查詢時各自降級為概算公式: {exc}"
        )
        return

    for i, (o_lat, o_lon) in enumerate(origins):
        for j, (d_lat, d_lon) in enumerate(destinations):
            key = (o_lat, o_lon, d_lat, d_lon)
            duration_sec = durations[i][j]
            if duration_sec is None:
                _OSRM_TRAVEL_TIME_CACHE[key] = _osrm_fallback_minutes(
                    o_lat, o_lon, d_lat, d_lon, travel_min_per_km, "OSRM /table 無法規劃該路徑"
                )
            else:
                _OSRM_TRAVEL_TIME_CACHE[key] = duration_sec / 60.0


def prefetch_osrm_travel_times(origins, destinations, travel_min_per_km=3.0) -> None:
    """批次暖身 OSRM 路網時間快取，大幅降低 Phase 1 / Phase 2 的 API 呼叫次數。

    origins / destinations 為 (lat, lon) 的可迭代物件（例如所有居服員住家座標 x 所有
    任務地點座標）。以 OSRM `/table` 矩陣 API 一次查詢整批配對，取代逐筆呼叫
    `get_osrm_travel_time` 各自發送一次 HTTP 請求；查詢結果寫入與 `get_osrm_travel_time`
    共用的字典快取，之後逐筆呼叫即直接命中快取。任一批次查詢失敗僅印出 Warning Log
    並跳過，未快取的配對會在後續逐筆查詢時各自降級為概算公式，不影響排程結果正確性。
    """
    unique_origins = sorted({(_round_coord(lat), _round_coord(lon)) for lat, lon in origins})
    unique_destinations = sorted({(_round_coord(lat), _round_coord(lon)) for lat, lon in destinations})

    pending_origins = [
        o
        for o in unique_origins
        if any((o[0], o[1], d[0], d[1]) not in _OSRM_TRAVEL_TIME_CACHE for d in unique_destinations)
    ]
    if not pending_origins or not unique_destinations:
        return

    dest_chunk_size = max(1, OSRM_TABLE_MAX_COORDS - len(pending_origins))
    for i in range(0, len(unique_destinations), dest_chunk_size):
        dest_chunk = unique_destinations[i : i + dest_chunk_size]
        _fetch_osrm_table_chunk(pending_origins, dest_chunk, travel_min_per_km)


def calc_travel_minutes(lat1, lon1, lat2, lon2, config: "PipelineConfig") -> float:
    """兩點間轉場車程（分鐘，不含轉場緩衝）。統一經由 get_osrm_travel_time 查詢，
    Phase 1 評分與 Phase 2 衝突檢查皆呼叫此函式，確保交通時間模型一致。
    """
    return get_osrm_travel_time(lat1, lon1, lat2, lon2, config.travel_min_per_km)


def get_service_intensity_weight(task) -> float:
    """依服務項目之體力耗費強度，回傳該任務對應的疲勞度加權係數。"""
    if task["需重度移位協助(0/1)"] == 1:
        return SERVICE_INTENSITY_WEIGHT["重度移位"]
    if task["特殊照護需求"] in ("餐食照顧/管灌", "管路安全與特殊日常照護"):
        return SERVICE_INTENSITY_WEIGHT["餐食管灌"]
    return SERVICE_INTENSITY_WEIGHT["一般照護"]


def parse_time(time_str):
    if pd.isna(time_str) or time_str == "無既定行程":
        return None, None
    parts = str(time_str).split("-")
    return (
        datetime.strptime(parts[0].strip(), "%H:%M"),
        datetime.strptime(parts[1].strip(), "%H:%M"),
    )


def _check_hard_constraints(task, cg, config: "PipelineConfig"):
    """檢查單一 (任務, 居服員) 配對的硬性條件。全數通過回傳 None，否則回傳未通過原因。

    抽成獨立函式供 Phase 1 過濾與「更新居服員原因」診斷共用同一套判斷邏輯。
    """
    req_gender = task["指定居服員性別"]
    if req_gender == "限女性" and cg["性別"] != "女":
        return "案家指定女性居服員，原居服員性別不符"
    if req_gender == "限男性" and cg["性別"] != "男":
        return "案家指定男性居服員，原居服員性別不符"

    if task["需重度移位協助(0/1)"] == 1 and cg["具備重度移位體力(0/1)"] == 0:
        return "案家需重度移位協助，原居服員不具備相關體力條件"

    cg_excl = cg["特殊排斥條件"]
    if cg_excl in EXCLUSION_MAP and task["案家環境特徵"] == EXCLUSION_MAP[cg_excl]:
        return f"原居服員排斥「{EXCLUSION_MAP[cg_excl]}」環境條件"

    task_duration_hrs = task["服務歷時(分鐘)"] / 60.0
    if cg["今日已佔用工時(小時)"] + task_duration_hrs > cg["每日工時上限(小時)"]:
        return "原居服員今日工時已達每日上限"

    required_certs = SPECIAL_CERT_REQUIREMENTS.get(task["特殊照護需求"])
    if required_certs and cg["核心專長證照"] not in required_certs:
        return "原居服員缺乏該需求類別之法定專長認證"

    return None


# ==========================================
# Phase 1: 適配度過濾與評分機制
# ==========================================
def run_phase1_matching(tasks: pd.DataFrame, df_cg: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    match_results = []

    # 進入逐筆比對迴圈前，先以單次 OSRM /table 批次查詢暖身快取（居服員住家 x 任務地點），
    # 取代 N x M 次個別 HTTP 請求。
    prefetch_osrm_travel_times(
        ((cg["服務起點_緯度(家)"], cg["服務起點_經度(家)"]) for _, cg in df_cg.iterrows()),
        ((task["服務地點_緯度"], task["服務地點_經度"]) for _, task in tasks.iterrows()),
        config.travel_min_per_km,
    )

    for _, task in tasks.iterrows():
        t_id = task["任務ID"]
        c_id = task["案家ID"]
        client_lat = task["服務地點_緯度"]
        client_lon = task["服務地點_經度"]
        pref_cg = task["歷史首選居服員ID"]
        req_type = task["特殊照護需求"]

        for _, cg in df_cg.iterrows():
            cg_id = cg["居服員ID"]

            # --- Hard Constraints (硬性過濾，不合格者直接剔除) ---
            if _check_hard_constraints(task, cg, config) is not None:
                continue

            cert = cg["核心專長證照"]

            # --- Soft Match Scoring ---
            score = config.base_score

            if req_type == "失智引導與精神陪伴" and cert in [
                "失智症照顧專長",
                "精神疾病照顧專長",
            ]:
                score += config.cert_bonus_dementia
            elif req_type in ["管路安全與特殊日常照護", "餐食照顧/管灌"] and cert == "單一級照服證照":
                score += config.cert_bonus_other

            if cg_id == pref_cg:
                # 照護連續性為最高指導原則：基礎加分已大幅提高，避免微幅車程/成本優化
                # 就任意更換案家熟悉的居服員；表現優良（滿意度達門檻）者再疊加動態加成。
                score += config.preferred_caregiver_bonus
                if cg["歷史滿意度均值"] >= config.continuity_satisfaction_threshold:
                    score += config.continuity_performance_bonus

            score += (cg["歷史滿意度均值"] - config.satisfaction_baseline) * config.satisfaction_weight

            travel_time_min = calc_travel_minutes(
                cg["服務起點_緯度(家)"],
                cg["服務起點_經度(家)"],
                client_lat,
                client_lon,
                config,
            )
            score -= min(travel_time_min * config.travel_penalty_weight, config.travel_penalty_cap)

            intensity_weight = get_service_intensity_weight(task)
            weighted_fatigue_hours = cg["當月累計服務時數(疲勞度)"] * intensity_weight
            fatigue_penalty = (weighted_fatigue_hours / config.fatigue_reference_hours) * config.fatigue_weight
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
                    "具備失智症20小時認證(0/1)": int(cert == "失智症照顧專長"),
                    "具備精神疾病20小時認證(0/1)": int(cert == "精神疾病照顧專長"),
                }
            )

    return pd.DataFrame(match_results)


def _diagnose_caregiver_change(
    task_row,
    pref_cg_id,
    assigned_cg_id,
    df_cg: pd.DataFrame,
    df_matches: pd.DataFrame,
    df_valid: pd.DataFrame,
    other_assigned_task_ids,
    task_times: dict,
    config: "PipelineConfig",
) -> str:
    """回傳「更新居服員原因」文字。案家為新客戶（無歷史首選居服員）或本次
    仍指派給原居服員時回傳空字串；僅在確實更換居服員時才需要說明原因。
    """
    if pd.isna(pref_cg_id) or str(pref_cg_id).strip() == "":
        return ""
    if pref_cg_id == assigned_cg_id:
        return ""

    t_id = task_row["任務ID"]

    pref_in_matches = ((df_matches["任務ID"] == t_id) & (df_matches["居服員ID"] == pref_cg_id)).any()
    if not pref_in_matches:
        cg_rows = df_cg[df_cg["居服員ID"] == pref_cg_id]
        if cg_rows.empty:
            return "原居服員資料異動，查無此居服員"
        return _check_hard_constraints(task_row, cg_rows.iloc[0], config) or "原居服員不符合硬性派單條件"

    pref_in_valid = ((df_valid["任務ID"] == t_id) & (df_valid["居服員ID"] == pref_cg_id)).any()
    if not pref_in_valid:
        return "原居服員今日既定行程與本任務時段衝突"

    t_start, t_end, t_lat, t_lon = task_times[t_id]
    for other_t_id in other_assigned_task_ids:
        o_start, o_end, o_lat, o_lon = task_times[other_t_id]
        travel_mins = calc_travel_minutes(t_lat, t_lon, o_lat, o_lon, config) + config.buffer_mins
        if not (
            t_end + timedelta(minutes=travel_mins) <= o_start
            or o_end + timedelta(minutes=travel_mins) <= t_start
        ):
            return "原居服員該時段已媒合其他案家任務"

    return "原居服員符合派單資格，惟系統整體最佳化後綜合適配分數較低，改派其他居服員"


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

    # 進入衝突檢查迴圈前，先以 OSRM /table 批次查詢暖身快取（任務地點 x 既定行程地點、
    # 任務地點 x 任務地點），取代逐筆個別 HTTP 請求。
    task_coords = [(lat, lon) for _, _, lat, lon in task_times.values()]
    busy_coords = [
        (b_lat, b_lon) for intervals in cg_busy.values() for (_, _, b_lat, b_lon) in intervals
    ]
    if busy_coords:
        prefetch_osrm_travel_times(task_coords, busy_coords, config.travel_min_per_km)
    prefetch_osrm_travel_times(task_coords, task_coords, config.travel_min_per_km)

    # 過濾掉與既定行程衝突（含轉場緩衝時間）的配對
    valid_rows = []
    for _, row in df_matches.iterrows():
        t_id = row["任務ID"]
        cg_id = row["居服員ID"]
        t_start, t_end, t_lat, t_lon = task_times[t_id]

        conflict = False
        for b_start, b_end, b_lat, b_lon in cg_busy[cg_id]:
            travel_mins = calc_travel_minutes(t_lat, t_lon, b_lat, b_lon, config) + config.buffer_mins
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

                travel_mins = calc_travel_minutes(t1_lat, t1_lon, t2_lat, t2_lon, config) + config.buffer_mins

                if not (
                    t1_end + timedelta(minutes=travel_mins) <= t2_start
                    or t2_end + timedelta(minutes=travel_mins) <= t1_start
                ):
                    solver.Add(X[(t1_id, cg_id)] + X[(t2_id, cg_id)] <= 1)

    # 限制條件 3：居服員今日新派任務總歷時 + 既有已佔用工時，不得超過每日工時上限。
    # Phase 1 僅逐筆過濾單一任務是否超時，無法阻擋「多筆任務加總後超派」的組合，
    # 此為求解器層級的產能限制式，修復該缺口。
    task_duration_hrs = {
        row["任務ID"]: row["服務歷時(分鐘)"] / 60.0 for _, row in tasks.iterrows()
    }
    cg_capacity = {
        row["居服員ID"]: (row["今日已佔用工時(小時)"], row["每日工時上限(小時)"])
        for _, row in df_cg.iterrows()
    }
    for cg_id in df_valid["居服員ID"].unique():
        used_hours, cap_hours = cg_capacity.get(cg_id, (0.0, float("inf")))
        cg_task_ids = df_valid[df_valid["居服員ID"] == cg_id]["任務ID"].unique()
        solver.Add(
            sum(X[(t_id, cg_id)] * task_duration_hrs[t_id] for t_id in cg_task_ids)
            + used_hours
            <= cap_hours
        )

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

        # 每位居服員本次新指派到的任務清單，供「更新居服員原因」判斷同時段衝突用
        assigned_task_ids_by_cg = {}
        for row in results:
            assigned_task_ids_by_cg.setdefault(row["派單居服員"], []).append(row["任務ID"])

        for row in results:
            t_id = row["任務ID"]
            task_row = tasks[tasks["任務ID"] == t_id].iloc[0]
            pref_cg_id = task_row["歷史首選居服員ID"]
            other_task_ids = [
                tid for tid in assigned_task_ids_by_cg.get(pref_cg_id, []) if tid != t_id
            ]
            row["更新居服員原因"] = _diagnose_caregiver_change(
                task_row,
                pref_cg_id,
                row["派單居服員"],
                df_cg,
                df_matches,
                df_valid,
                other_task_ids,
                task_times,
                config,
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


# ==========================================
# 居督人工覆寫稽核日誌 (Human-in-the-Loop override audit log)
# ==========================================
OVERRIDE_LOG_PATH = os.path.join("output_results", "supervisor_override_log.csv")

OVERRIDE_LOG_COLUMNS = [
    "時間戳記", "任務ID", "案家ID", "AI推薦居服員ID", "居督指定居服員ID", "變更原因",
]


def save_override_log(
    task_id,
    client_id,
    ai_cg_id,
    supervisor_cg_id,
    reason: str,
    log_path: str = OVERRIDE_LOG_PATH,
) -> None:
    """將居督一筆人工覆寫紀錄以附加 (append) 方式寫入 CSV 稽核日誌。

    每次呼叫寫入一列；檔案不存在時先建立目錄與標題列。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = pd.DataFrame(
        [{
            "時間戳記": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "任務ID": task_id,
            "案家ID": client_id,
            "AI推薦居服員ID": ai_cg_id,
            "居督指定居服員ID": supervisor_cg_id,
            "變更原因": reason,
        }],
        columns=OVERRIDE_LOG_COLUMNS,
    )
    write_header = not os.path.exists(log_path)
    entry.to_csv(log_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def load_override_log(log_path: str = OVERRIDE_LOG_PATH) -> pd.DataFrame:
    """讀取現有稽核日誌；檔案不存在時回傳空的標準欄位 DataFrame。"""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=OVERRIDE_LOG_COLUMNS)
    return pd.read_csv(log_path, encoding="utf-8-sig")
