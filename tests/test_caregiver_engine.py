"""caregiver_engine.py 核心邏輯的單元測試（合成資料，不依賴 Excel 檔案）。

執行方式：
    python -m unittest tests.test_caregiver_engine -v
"""

import os
import sys
import unittest

import pandas as pd
from ortools.linear_solver import pywraplp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caregiver_engine import (
    PipelineConfig,
    run_phase1_matching,
    run_phase2_optimization,
)


def make_caregiver(
    cg_id,
    lat=25.05,
    lon=121.55,
    cert="單一級照服證照",
    used_hours=0.0,
    daily_cap=8.0,
    satisfaction=4.0,
    fatigue=0.0,
    lift_ok=1,
    gender="女",
    exclusion="無",
    busy1="無既定行程",
    busy2="無既定行程",
):
    return {
        "居服員ID": cg_id,
        "性別": gender,
        "服務起點_經度(家)": lon,
        "服務起點_緯度(家)": lat,
        "核心專長證照": cert,
        "每日工時上限(小時)": daily_cap,
        "當月累計服務時數(疲勞度)": fatigue,
        "歷史滿意度均值": satisfaction,
        "具備重度移位體力(0/1)": lift_ok,
        "今日既定行程1_時段": busy1,
        "今日既定行程1_地點經度": lon,
        "今日既定行程1_地點緯度": lat,
        "今日既定行程2_時段": busy2,
        "今日既定行程2_地點經度": lon,
        "今日既定行程2_地點緯度": lat,
        "特殊排斥條件": exclusion,
        "今日已佔用工時(小時)": used_hours,
    }


def make_task(
    t_id,
    c_id,
    start,
    end,
    duration_min,
    lat=25.05,
    lon=121.55,
    req_gender="無特殊要求",
    req_lift=0,
    req_type="一般家務與照顧",
    pref_cg="",
    priority="一般",
    client_env="無",
):
    return {
        "任務ID": t_id,
        "案家ID": c_id,
        "指定居服員性別": req_gender,
        "需重度移位協助(0/1)": req_lift,
        "案家環境特徵": client_env,
        "服務地點_緯度": lat,
        "服務地點_經度": lon,
        "服務歷時(分鐘)": duration_min,
        "歷史首選居服員ID": pref_cg,
        "特殊照護需求": req_type,
        "時間窗_開始": start,
        "時間窗_結束": end,
        "任務優先級": priority,
    }


class CapacityConstraintTests(unittest.TestCase):
    """Phase 2 MIP 必須阻擋「多筆任務加總後超派」，即便每筆任務個別未超時。"""

    def test_solver_never_exceeds_daily_hour_cap(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=8.0, used_hours=0.0)])

        # 兩筆任務時間窗不重疊（含轉場緩衝），但總歷時 10 小時 > 每日工時上限 8 小時。
        tasks = pd.DataFrame(
            [
                make_task("TK01", "CL01", "08:00", "13:00", 300),  # 5 小時
                make_task("TK02", "CL02", "14:00", "19:00", 300),  # 5 小時
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        # Phase 1 逐筆檢查未超時，兩筆候選配對都應存活。
        self.assertEqual(len(df_matches), 2)

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        self.assertEqual(phase2["status"], pywraplp.Solver.OPTIMAL)

        df_result = phase2["df_result"]
        assigned_minutes = 0
        if not df_result.empty:
            assigned_task_ids = df_result["任務ID"].tolist()
            assigned_minutes = tasks[tasks["任務ID"].isin(assigned_task_ids)]["服務歷時(分鐘)"].sum()

        # 已佔用工時(0) + 新派任務總歷時，不得超過每日工時上限(8小時=480分鐘)。
        self.assertLessEqual(assigned_minutes / 60.0, 8.0)
        # 兩筆任務加總必超派，因此不可能兩筆都指派給唯一的居服員。
        self.assertLessEqual(phase2["assigned_count"], 1)

    def test_capacity_constraint_accounts_for_already_used_hours(self):
        """Phase 2 求解器本身須獨立核算「已佔用工時」，不能只依賴 Phase 1 的逐筆過濾。

        直接手動建構 df_matches（略過 Phase 1），確保限制式 3 本身確實把
        今日已佔用工時 + 新派任務歷時 一併納入運算，而非只在 Phase 1 起作用。
        """
        config = PipelineConfig()
        # 今日已佔用 6 小時，上限 8 小時，僅剩 2 小時可派，新任務歷時 3 小時應被排除。
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=8.0, used_hours=6.0)])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "11:00", 180)])

        df_matches = pd.DataFrame(
            [
                {
                    "任務ID": "TK01",
                    "案家ID": "CL01",
                    "居服員ID": "CG01",
                    "適配度分數": 60.0,
                    "預估交通時間(分)": 0.0,
                    "任務開始時間": "08:00",
                    "任務結束時間": "11:00",
                    "優先級": "一般",
                    "地點緯度": 25.05,
                    "地點經度": 121.55,
                }
            ]
        )

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        self.assertEqual(phase2["assigned_count"], 0)


class SpecialCertHardFilterTests(unittest.TestCase):
    """法定 20 小時特照培訓認證：缺乏對應認證者，Phase 1 必須直接剔除該配對。"""

    def test_uncertified_caregiver_excluded_from_dementia_task(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_no_cert", cert="單一級照服證照"),
                make_caregiver("CG_dementia", cert="失智症照顧專長"),
            ]
        )
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:00", "10:00", 120, req_type="失智引導與精神陪伴")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        matched_cg_ids = set(df_matches["居服員ID"])

        self.assertNotIn("CG_no_cert", matched_cg_ids)
        self.assertIn("CG_dementia", matched_cg_ids)

    def test_certified_pair_carries_cert_flag_columns(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG_dementia", cert="失智症照顧專長")])
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:00", "10:00", 120, req_type="失智引導與精神陪伴")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        row = df_matches.iloc[0]
        self.assertEqual(row["具備失智症20小時認證(0/1)"], 1)
        self.assertEqual(row["具備精神疾病20小時認證(0/1)"], 0)


class ContinuityPriorityTests(unittest.TestCase):
    """照護連續性應為最高指導原則：不得為了微幅車程優化而更換熟悉且評價良好的居服員。"""

    def test_preferred_high_performing_caregiver_outscores_closer_alternative(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                # 案家熟悉、表現優良的首選居服員，但距離較遠。
                make_caregiver("CG_preferred", lat=25.20, lon=121.70, satisfaction=4.6),
                # 明顯較近的替代居服員，滿意度普通。
                make_caregiver("CG_closer", lat=25.051, lon=121.551, satisfaction=4.0),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK01", "CL01", "08:00", "10:00", 120,
                    lat=25.05, lon=121.55, pref_cg="CG_preferred",
                )
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        score_preferred = df_matches.loc[df_matches["居服員ID"] == "CG_preferred", "適配度分數"].iloc[0]
        score_closer = df_matches.loc[df_matches["居服員ID"] == "CG_closer", "適配度分數"].iloc[0]

        self.assertGreater(score_preferred, score_closer)

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]
        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_preferred")

    def test_non_performing_preferred_caregiver_gets_no_dynamic_bonus(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [make_caregiver("CG_preferred", satisfaction=config.continuity_satisfaction_threshold - 0.5)]
        )
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="CG_preferred")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        score = df_matches.iloc[0]["適配度分數"]
        expected_without_dynamic_bonus = (
            config.base_score
            + config.preferred_caregiver_bonus
            + (df_cg.iloc[0]["歷史滿意度均值"] - config.satisfaction_baseline) * config.satisfaction_weight
        )
        self.assertAlmostEqual(score, round(expected_without_dynamic_bonus, 2), places=2)


class CaregiverChangeReasonTests(unittest.TestCase):
    """指派明細表「更新居服員原因」欄：新客戶或維持原居服員時應為空值，
    確實更換居服員時必須附上可解釋的具體原因。
    """

    def test_new_client_without_preferred_caregiver_is_blank(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01")])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["更新居服員原因"], "")

    def test_unchanged_preferred_caregiver_is_blank(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01")])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="CG01")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG01")
        self.assertEqual(df_result.iloc[0]["更新居服員原因"], "")

    def test_hard_constraint_failure_gives_specific_reason(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_male_pref", gender="男"),
                make_caregiver("CG_female_alt", gender="女"),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK01", "CL01", "08:00", "10:00", 120,
                    req_gender="限女性", pref_cg="CG_male_pref",
                )
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_female_alt")
        self.assertIn("性別不符", df_result.iloc[0]["更新居服員原因"])

    def test_existing_schedule_conflict_gives_specific_reason(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_busy_pref", busy1="08:00-10:00"),
                make_caregiver("CG_free_alt"),
            ]
        )
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:30", "10:30", 120, pref_cg="CG_busy_pref")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_free_alt")
        self.assertEqual(df_result.iloc[0]["更新居服員原因"], "原居服員今日既定行程與本任務時段衝突")

    def test_double_booked_with_other_new_task_gives_specific_reason(self):
        config = PipelineConfig()
        # CG_pref 是 TK_A、TK_B 兩案家共同的歷史首選居服員，但兩任務時段重疊，
        # 無法同時服務兩者；TK_A 額外要求重度移位協助，只有 CG_pref 具備，
        # 因此系統必須把 CG_pref 留給 TK_A，TK_B 改派唯一的替代居服員 CG_alt。
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_pref", lift_ok=1),
                make_caregiver("CG_alt", lift_ok=0),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK_A", "CL_A", "08:00", "10:00", 120,
                    req_lift=1, pref_cg="CG_pref",
                ),
                make_task(
                    "TK_B", "CL_B", "09:00", "11:00", 120,
                    req_lift=0, pref_cg="CG_pref",
                ),
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"].set_index("任務ID")

        self.assertEqual(df_result.loc["TK_A", "派單居服員"], "CG_pref")
        self.assertEqual(df_result.loc["TK_A", "更新居服員原因"], "")
        self.assertEqual(df_result.loc["TK_B", "派單居服員"], "CG_alt")
        self.assertEqual(df_result.loc["TK_B", "更新居服員原因"], "原居服員該時段已媒合其他案家任務")


if __name__ == "__main__":
    unittest.main()
