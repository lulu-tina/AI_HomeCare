"""長照居家照顧 AI 派單系統 - 網頁儀表板

執行方式：
    streamlit run app.py

工作流程：
    1. 側邊欄調整五項核心派單政策參數。
    2. 區塊一上傳／檢視／直接編輯居服員、案家、今日待派任務三張表格
       （可新增列與欄位），做為派單運算的輸入資料。
    3. 區塊二按下「執行 AI 最佳化派單」後，依序呼叫 caregiver_engine.py
       的 Phase 1～3，並顯示 KPI、四合一分析儀表板與指派明細表。

核心運算邏輯（適配度評分、OR-Tools 最佳化、DiD 效益回溯）皆定義於
caregiver_engine.py，本檔僅負責資料編輯介面與結果呈現，不重複實作演算法。
"""

import pathlib

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from ortools.linear_solver import pywraplp

from caregiver_engine import (
    DEFAULT_EXCEL_PATH,
    PipelineConfig,
    load_override_log,
    run_phase1_matching,
    run_phase2_optimization,
    run_phase3_did,
    save_override_log,
)

# ==========================================
# 跨平台中文字型處理（避免 st.pyplot 圖表出現亂碼）
# ==========================================
# 直接隨 repo 附帶一份開源中文字型（Noto Sans TC, SIL Open Font License），
# 因為系統套件（packages.txt: fonts-noto-cjk）是否真的裝進 Streamlit Cloud 容器、
# 以及字型註冊名稱是否符合猜測，皆不受我們控制；自帶字型檔可確保任何部署環境
# 都一定找得到，不必依賴系統/雲端環境是否裝好中文字型。
_BUNDLED_FONT_PATH = pathlib.Path(__file__).parent / "assets" / "fonts" / "NotoSansTC-Regular.ttf"


def setup_chinese_font():
    # 依平台猜測系統字型名稱不可靠：猜錯字型 matplotlib 不會報錯，只會靜默退回
    # DejaVu Sans（無中文字形，顯示為方框）。優先使用自帶字型，系統字型僅作備援。
    bundled_name = None
    if _BUNDLED_FONT_PATH.exists():
        fm.fontManager.addfont(str(_BUNDLED_FONT_PATH))
        bundled_name = fm.FontProperties(fname=str(_BUNDLED_FONT_PATH)).get_name()

    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei",  # Windows
        "PingFang TC", "PingFang SC", "Heiti TC", "Arial Unicode MS",  # macOS
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC",  # Linux / Streamlit Cloud
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    found = [name for name in candidates if name in available]

    ordered = ([bundled_name] if bundled_name else []) + found + ["DejaVu Sans"]
    seen = set()
    plt.rcParams["font.sans-serif"] = [n for n in ordered if not (n in seen or seen.add(n))]
    plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 色彩定義（來源：dataviz 色票，固定順序、非隨機挑色）
# ==========================================
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GOOD = "#0ca30c"

REQUIRED_SHEETS = [
    "Caregiver_Profiles",
    "Client_Profiles",
    "Today_Pending_Tasks",
    "Historical_Service_Logs",
]

REQUIRED_COLUMNS = {
    "Caregiver_Profiles": [
        "居服員ID", "性別", "服務起點_經度(家)", "服務起點_緯度(家)", "核心專長證照",
        "每日工時上限(小時)", "當月累計服務時數(疲勞度)", "歷史滿意度均值",
        "具備重度移位體力(0/1)", "今日既定行程1_時段", "今日既定行程1_地點經度",
        "今日既定行程1_地點緯度", "今日既定行程2_時段", "今日既定行程2_地點經度",
        "今日既定行程2_地點緯度", "特殊排斥條件", "今日已佔用工時(小時)",
    ],
    "Client_Profiles": [
        "案家ID", "指定居服員性別", "服務地點_經度", "服務地點_緯度", "特殊照護需求",
        "案家環境特徵", "需重度移位協助(0/1)", "歷史首選居服員ID",
    ],
    "Today_Pending_Tasks": [
        "任務ID", "案家ID", "時間窗_開始", "時間窗_結束", "服務歷時(分鐘)", "任務優先級",
    ],
    "Historical_Service_Logs": [
        "居服員ID", "案家ID", "歷史媒合機制(Treatment)", "不滿意導致提早結案(0/1)", "案家滿意度(1-5)",
    ],
}

st.set_page_config(page_title="長照居家照顧 AI 派單系統", page_icon="🏠", layout="wide")
st.title("🏠 長照居家照顧 AI 派單系統")
st.caption("居督與營運團隊可上傳／編輯今日排班資料、調整派單政策參數，並一鍵執行 AI 最佳化派單")

# ==========================================
# 側邊欄：五項核心派單政策參數
# ==========================================
defaults = PipelineConfig()

with st.sidebar:
    st.header("⚙️ 核心派單政策參數")

    if st.button("🔄 還原預設值", width="stretch"):
        for key in ["cfg_buffer_mins", "cfg_travel_penalty_weight", "cfg_urgent_priority_bonus",
                    "cfg_continuity_bonus", "cfg_skill_bonus"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    buffer_mins = st.slider(
        "轉場與交接緩衝時間（分鐘）", 0.0, 60.0, defaults.buffer_mins, 1.0,
        key="cfg_buffer_mins",
        help="居服員完成一項任務後，考量停車、上下樓、門禁核對與交接紀錄所需的緩衝時間。",
    )
    travel_penalty_weight = st.slider(
        "車程扣分權重（分／分鐘）", 0.0, 2.0, defaults.travel_penalty_weight, 0.1,
        key="cfg_travel_penalty_weight",
        help="每一分鐘車程對適配度分數與派單目標函數的扣分幅度，數值越高代表越優先就近派案。",
    )
    urgent_priority_bonus = st.slider(
        "緊急任務優先加分（分）", 0.0, 100.0, defaults.urgent_priority_bonus, 5.0,
        key="cfg_urgent_priority_bonus",
        help="任務優先級標示為「緊急」時，在派單目標函數中額外獲得的加分，數值越高代表緊急任務越優先被指派。",
    )
    continuity_bonus = st.slider(
        "照護連續性加分（分）", 0.0, 100.0, defaults.preferred_caregiver_bonus, 1.0,
        key="cfg_continuity_bonus",
        help="當案家的歷史首選居服員恰為候選居服員時的加分，鼓勵維持既有照護關係與默契。"
        "表現優良（滿意度達門檻）的首選居服員另有動態加成，避免因車程等微幅優化而被替換。",
    )
    skill_bonus = st.slider(
        "核心專長匹配加分（分）", 0.0, 50.0, defaults.cert_bonus_dementia, 1.0,
        key="cfg_skill_bonus",
        help="居服員持有之核心專長證照符合案家特殊照護需求時的加分。",
    )

config = PipelineConfig(
    buffer_mins=buffer_mins,
    travel_penalty_weight=travel_penalty_weight,
    objective_travel_weight=travel_penalty_weight,
    urgent_priority_bonus=urgent_priority_bonus,
    preferred_caregiver_bonus=continuity_bonus,
    cert_bonus_dementia=skill_bonus,
    cert_bonus_other=skill_bonus,
)

# ==========================================
# 區塊一：Excel 載入與現況檢視／動態編輯
# ==========================================
st.header("① 資料載入與編輯")
uploaded_file = st.file_uploader("上傳派單資料庫（.xlsx，未上傳則使用預設檔案）", type=["xlsx"])


@st.cache_data(show_spinner="讀取 Excel 資料中...")
def _read_raw_sheets(file_or_path):
    xl = pd.ExcelFile(file_or_path)
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in xl.sheet_names]
    if missing_sheets:
        raise ValueError(f"檔案缺少必要工作表：{'、'.join(missing_sheets)}")

    sheets = {name: pd.read_excel(xl, sheet_name=name) for name in REQUIRED_SHEETS}

    for name, required_cols in REQUIRED_COLUMNS.items():
        missing_cols = [c for c in required_cols if c not in sheets[name].columns]
        if missing_cols:
            raise ValueError(f"工作表「{name}」缺少必要欄位：{'、'.join(missing_cols)}")

    return sheets


try:
    if uploaded_file is not None:
        source_key = f"upload::{uploaded_file.name}::{uploaded_file.size}"
        sheets = _read_raw_sheets(uploaded_file)
    else:
        import os

        if not os.path.exists(DEFAULT_EXCEL_PATH):
            st.error(f"找不到預設檔案：{DEFAULT_EXCEL_PATH}，請改用上方上傳功能。")
            st.stop()
        source_key = f"default::{os.path.getmtime(DEFAULT_EXCEL_PATH)}"
        sheets = _read_raw_sheets(DEFAULT_EXCEL_PATH)
except ValueError as e:
    st.error(f"⚠️ 資料格式錯誤：{e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ 無法讀取上傳的檔案，請確認上傳的是有效的 Excel（.xlsx）檔案。錯誤訊息：{e}")
    st.stop()

# 僅在資料來源變更時（首次載入／換檔案／原檔被覆寫）重設編輯區，避免使用者的編輯內容被覆蓋
if st.session_state.get("_data_source_key") != source_key:
    st.session_state["_data_source_key"] = source_key
    st.session_state["edit_cg"] = sheets["Caregiver_Profiles"].copy()
    st.session_state["edit_cl"] = sheets["Client_Profiles"].copy()
    st.session_state["edit_tasks"] = sheets["Today_Pending_Tasks"].copy()
    st.session_state["data_hist"] = sheets["Historical_Service_Logs"].copy()


def render_editable_sheet(session_key: str):
    with st.expander("➕ 新增欄位"):
        c1, c2 = st.columns([3, 1])
        new_col_name = c1.text_input("欄位名稱", key=f"newcol_input_{session_key}", label_visibility="collapsed",
                                      placeholder="輸入新欄位名稱")
        if c2.button("新增欄位", key=f"newcol_btn_{session_key}", width="stretch"):
            df_current = st.session_state[session_key]
            if not new_col_name:
                st.warning("請先輸入欄位名稱。")
            elif new_col_name in df_current.columns:
                st.warning(f"欄位「{new_col_name}」已存在。")
            else:
                df_current[new_col_name] = None
                st.session_state[session_key] = df_current
                st.rerun()

    df = st.session_state[session_key]
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key=f"editor_{session_key}_{len(df.columns)}",
    )
    st.session_state[session_key] = edited


tab_cg, tab_cl, tab_tasks = st.tabs([
    "居服員資料 (Caregiver_Profiles)",
    "案家資料 (Client_Profiles)",
    "今日待派任務 (Today_Pending_Tasks)",
])

with tab_cg:
    st.caption("可直接編輯儲存格、新增／刪除列（勾選列號後按 Delete），或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_cg")

with tab_cl:
    st.caption("可直接編輯儲存格、新增／刪除列，或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_cl")

with tab_tasks:
    st.caption("可直接編輯儲存格、新增／刪除列，或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_tasks")

# ==========================================
# 區塊二：一鍵執行與成果儀表板
# ==========================================
st.header("② 執行最佳化派單與成果儀表板")

if st.button("🚀 執行 AI 最佳化派單", type="primary"):
    df_cg = st.session_state["edit_cg"].copy()
    df_cl = st.session_state["edit_cl"].copy()
    df_tasks = st.session_state["edit_tasks"].copy()
    df_hist = st.session_state["data_hist"].copy()

    try:
        tasks = df_tasks.merge(df_cl, on="案家ID", how="left")
        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        did = run_phase3_did(df_hist)
    except Exception as e:
        st.error(
            "⚠️ 派單運算過程發生錯誤，請確認表格中的 ID 對應是否存在、數值欄位是否填寫正確"
            f"（例如經緯度、工時、時間格式「HH:MM」）。錯誤訊息：{e}"
        )
        st.stop()

    st.session_state["last_result"] = {
        "df_tasks": df_tasks,
        "df_cg": df_cg,
        "tasks": tasks,
        "df_matches": df_matches,
        "phase2": phase2,
        "did": did,
    }
    # 每次重新執行最佳化後，AI 建議已改變，先前的居督覆寫不再對應同一份建議，故一併清空。
    st.session_state["overrides"] = {}

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    df_tasks = res["df_tasks"]
    df_matches = res["df_matches"]
    phase2 = res["phase2"]
    did = res["did"]

    df_valid = phase2["df_valid"]
    df_result = phase2["df_result"]
    assigned_count = phase2["assigned_count"]
    solver_ok = phase2["status"] == pywraplp.Solver.OPTIMAL

    if not solver_ok:
        st.warning("OR-Tools 無法在目前資料與參數下找到最佳解，請檢查資料是否有效或調整參數。")

    total_tasks = len(df_tasks)
    assign_rate = (assigned_count / total_tasks * 100) if total_tasks else 0.0
    avg_transition = (
        (df_result["預估車程(分)"] + config.buffer_mins).mean() if not df_result.empty else 0.0
    )
    avg_score = df_result["適配分數"].mean() if not df_result.empty else 0.0

    st.subheader("📌 KPI 指標")
    k1, k2, k3 = st.columns(3)
    k1.metric("派單成功率", f"{assign_rate:.1f}%", help=f"{assigned_count} / {total_tasks} 筆任務成功指派")
    k2.metric("平均轉場時間", f"{avg_transition:.1f} 分", help="已派單任務的平均車程時間＋轉場緩衝時間")
    k3.metric("平均適配得分", f"{avg_score:.1f} 分", help="已派單配對的平均適配度分數")

    st.subheader("📊 派單分析儀表板")
    if df_matches.empty:
        st.info("目前無候選配對可供分析（可能所有配對皆被硬性條件過濾）。")
    else:
        sns.set_style("whitegrid")
        setup_chinese_font()  # 須在 sns.set_style 之後呼叫，否則字型會被 seaborn 預設值覆蓋
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # (1) 適配度分數分布
        ax = axes[0, 0]
        ax.hist(df_matches["適配度分數"], bins=20, color=BLUE, edgecolor="white")
        mean_score = df_matches["適配度分數"].mean()
        ax.axvline(mean_score, color=MUTED, linestyle="--", label=f"平均 {mean_score:.1f}")
        ax.set_title("適配度分數分布（全部候選配對）")
        ax.set_xlabel("適配度分數")
        ax.set_ylabel("候選配對數")
        ax.legend()

        # (2) 各居服員派單量
        ax = axes[0, 1]
        if not df_result.empty:
            load_counts = df_result["派單居服員"].value_counts().sort_values(ascending=True)
            ax.barh(load_counts.index.astype(str), load_counts.values, color=BLUE)
            ax.set_xlabel("派單任務數")
        else:
            ax.text(0.5, 0.5, "無派單結果", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("各居服員派單量")

        # (3) 派單狀態占比
        ax = axes[1, 0]
        unassigned = total_tasks - assigned_count
        if total_tasks > 0:
            ax.pie(
                [assigned_count, unassigned],
                labels=["已派單", "未派單"],
                colors=[BLUE, MUTED],
                autopct="%1.1f%%",
                startangle=90,
            )
        ax.set_title("任務派單狀態占比")

        # (4) AI vs 人工歷史成效比較（DiD）
        ax = axes[1, 1]
        ax.bar(
            ["AI 派單\n(Treatment)", "人工派單\n(Control)"],
            [did["ai_sat"], did["human_sat"]],
            color=[BLUE, ORANGE],
        )
        for i, v in enumerate([did["ai_sat"], did["human_sat"]]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 5)
        ax.set_title(f"歷史平均滿意度比較（淨提升 +{did['uplift_sat']:.2f} 分）")
        ax.set_ylabel("案家滿意度（1-5分）")

        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("📋 指派明細表")
    if df_result.empty:
        st.info("目前無派單結果。")
    else:
        display_cols = [
            "任務ID", "案家ID", "派單居服員", "適配分數", "預估車程(分)", "服務時段", "任務優先級",
            "更新居服員原因",
        ]
        st.dataframe(df_result[display_cols], width="stretch", hide_index=True)

        csv = df_result[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 匯出指派結果 (assigned_results.csv)",
            csv,
            file_name="assigned_results.csv",
            mime="text/csv",
        )

    # ==========================================
    # 區塊三：居督人工覆寫（Supervisor Override）與稽核日誌
    # ==========================================
    st.header("③ 居督人工覆寫（Supervisor Override）")
    st.caption(
        "當 AI 建議排單不符合實際場域狀況時（例如居服員臨時請假、案家臨時改期），"
        "居督可於此針對個別任務手動重新指定居服員；每一筆變更皆會記錄原因並寫入稽核日誌，供後續演算法迭代分析。"
    )

    st.session_state.setdefault("overrides", {})
    overrides = st.session_state["overrides"]

    OVERRIDE_KEEP_AI = "（維持 AI 建議）"
    OVERRIDE_UNASSIGN = "撤銷指派（不指派）"
    OVERRIDE_REASONS = ["車程太遠", "居服員請假", "案家臨時改期", "長者情緒抗拒", "其他"]
    OVERRIDE_TRAVEL_ALERT_THRESHOLD = 3

    override_log_df = load_override_log()
    travel_reason_count = (
        (override_log_df["變更原因"] == "車程太遠").sum() if not override_log_df.empty else 0
    )
    if travel_reason_count >= OVERRIDE_TRAVEL_ALERT_THRESHOLD:
        st.warning(
            f"📈 稽核日誌累計已有 {travel_reason_count} 筆覆寫原因為「車程太遠」，"
            "建議提高側邊欄的『車程扣分權重』，讓 AI 派單更優先考量就近指派。"
        )

    df_cg_run = res.get("df_cg", pd.DataFrame())
    cg_id_options = (
        sorted(df_cg_run["居服員ID"].astype(str).unique().tolist()) if not df_cg_run.empty else []
    )
    assigned_map = dict(zip(df_result["任務ID"], df_result["派單居服員"])) if not df_result.empty else {}

    for _, task_row in df_tasks.iterrows():
        t_id = task_row["任務ID"]
        cl_id = task_row["案家ID"]
        ai_cg = assigned_map.get(t_id)
        ai_label = str(ai_cg) if ai_cg is not None else "未指派"

        existing_override = overrides.get(t_id)
        current_label = (
            ("未指派" if existing_override["cg_id"] is None else str(existing_override["cg_id"]))
            if existing_override
            else ai_label
        )

        header = f"任務 {t_id}（案家 {cl_id}）— AI 建議：{ai_label}"
        if existing_override:
            header += f"　→　居督已覆寫為：{current_label}（{existing_override['reason']}）"

        with st.expander(header):
            options = [OVERRIDE_KEEP_AI] + cg_id_options + [OVERRIDE_UNASSIGN]
            if existing_override:
                default_val = (
                    OVERRIDE_UNASSIGN if existing_override["cg_id"] is None else str(existing_override["cg_id"])
                )
                default_idx = options.index(default_val) if default_val in options else 0
            else:
                default_idx = 0

            chosen = st.selectbox(
                "居督手動指定居服員",
                options,
                index=default_idx,
                key=f"override_select_{t_id}",
            )

            if chosen == OVERRIDE_KEEP_AI:
                new_cg = ai_cg
            elif chosen == OVERRIDE_UNASSIGN:
                new_cg = None
            else:
                new_cg = chosen

            override_happens = new_cg != ai_cg

            reason = ""
            if override_happens:
                reason = st.selectbox(
                    "換人原因（必填）",
                    OVERRIDE_REASONS,
                    key=f"override_reason_{t_id}",
                )
                if reason == "其他":
                    reason = st.text_input("請說明其他原因（必填）", key=f"override_reason_detail_{t_id}")

            col_confirm, col_clear = st.columns(2)
            if col_confirm.button(
                "✅ 確認覆寫", key=f"override_confirm_{t_id}", disabled=not override_happens, width="stretch"
            ):
                if not reason.strip():
                    st.warning("請填寫換人原因後再確認。")
                else:
                    save_override_log(
                        t_id, cl_id, ai_label, new_cg if new_cg is not None else "未指派", reason
                    )
                    overrides[t_id] = {"cg_id": new_cg, "reason": reason}
                    st.success("已記錄居督覆寫，並寫入稽核日誌。")
                    st.rerun()

            if existing_override:
                if col_clear.button(
                    "↩️ 清除覆寫，還原 AI 建議", key=f"override_clear_{t_id}", width="stretch"
                ):
                    del overrides[t_id]
                    st.rerun()

    st.subheader("📄 最終派單結果（含居督覆寫）")
    final_rows = []
    for _, task_row in df_tasks.iterrows():
        t_id = task_row["任務ID"]
        cl_id = task_row["案家ID"]
        base = df_result[df_result["任務ID"] == t_id] if not df_result.empty else pd.DataFrame()
        ai_row = base.iloc[0] if not base.empty else None
        ai_cg = ai_row["派單居服員"] if ai_row is not None else None

        override = overrides.get(t_id)
        if override:
            final_cg = override["cg_id"]
            change_note = f"居督覆寫：{override['reason']}"
        else:
            final_cg = ai_cg
            change_note = ai_row["更新居服員原因"] if ai_row is not None else ""

        final_rows.append({
            "任務ID": t_id,
            "案家ID": cl_id,
            "AI建議居服員": ai_cg if ai_cg is not None else "未指派",
            "最終派單居服員": final_cg if final_cg is not None else "未指派",
            "適配分數": ai_row["適配分數"] if ai_row is not None else None,
            "預估車程(分)": ai_row["預估車程(分)"] if ai_row is not None else None,
            "服務時段": ai_row["服務時段"] if ai_row is not None else "",
            "任務優先級": ai_row["任務優先級"] if ai_row is not None else task_row.get("任務優先級", ""),
            "備註": change_note,
        })

    df_final = pd.DataFrame(final_rows)
    st.dataframe(df_final, width="stretch", hide_index=True)

    csv_final = df_final.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 匯出最終派單結果（含覆寫） (final_assignment_with_overrides.csv)",
        csv_final,
        file_name="final_assignment_with_overrides.csv",
        mime="text/csv",
    )

    with st.expander("🗂️ 檢視完整稽核日誌 (supervisor_override_log.csv)"):
        if override_log_df.empty:
            st.info("尚無居督覆寫紀錄。")
        else:
            st.dataframe(override_log_df, width="stretch", hide_index=True)
else:
    st.info("請先於上方確認／編輯資料，再按下「🚀 執行 AI 最佳化派單」開始運算。")
