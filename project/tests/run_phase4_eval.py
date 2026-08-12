#!/usr/bin/env python
"""
Phase 4 正式评测脚本 — 200 条测试集分维度统计

三维度评测：
  Object Recall  — 纯物体/属性检索（CLIP 视觉理解）
  Event Accuracy — 纯事件检索（TQUM 路由 + 事件引擎）
  Combo Recall   — 组合查询（物体+事件多模态融合）

评测模式：TQUM Phase 3（CLIP-primary + event boost 置信度分级路由）
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from services.embedding_service import EmbeddingService
from services.query_rewrite_service import QueryRewriteService
from services.hybrid_search_service import HybridSearchService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth 视频集合（基于事件元数据分析）
# ---------------------------------------------------------------------------
ALL_VIDEOS = [
    "MVI_40851", "MVI_40852", "MVI_40853", "MVI_40854", "MVI_40855",
    "MVI_40863", "MVI_40864", "MVI_40891", "MVI_40892", "MVI_40901",
    "MVI_40902", "MVI_40903", "MVI_40904", "MVI_40905",
]

# Object ground truth
BUS_VIDEOS = ["MVI_40851", "MVI_40852", "MVI_40853", "MVI_40855", "MVI_40863",
              "MVI_40864", "MVI_40892", "MVI_40901", "MVI_40903", "MVI_40904", "MVI_40905"]
TRUCK_VIDEOS = ["MVI_40855", "MVI_40864", "MVI_40901", "MVI_40903", "MVI_40905"]
CAR_VIDEOS = ALL_VIDEOS[:]  # all 14

# Event ground truth (high-confidence subset, conf > 0.8)
WW_HIGH = ["MVI_40901", "MVI_40855", "MVI_40905", "MVI_40903", "MVI_40904", "MVI_40902", "MVI_40853"]
CL_HIGH = ["MVI_40901", "MVI_40863", "MVI_40855", "MVI_40864", "MVI_40905",
           "MVI_40904", "MVI_40891", "MVI_40892", "MVI_40853", "MVI_40852", "MVI_40902"]
RL_VIDEOS = ["MVI_40891"]

# Combo ground truth (label + event intersection)
BUS_WW = ["MVI_40901", "MVI_40855", "MVI_40905", "MVI_40903", "MVI_40904",
          "MVI_40853", "MVI_40892", "MVI_40852", "MVI_40851"]
TRUCK_CL = ["MVI_40901", "MVI_40855", "MVI_40864", "MVI_40905", "MVI_40903"]
CAR_RL = ["MVI_40891"]
TRUCK_WW = ["MVI_40901"]
BUS_CL = ["MVI_40863", "MVI_40864", "MVI_40853", "MVI_40852"]
ALL_EVENT = ALL_VIDEOS[:]

# ---------------------------------------------------------------------------
# 200 条正式测试集
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Dict] = [

    # ==================================================================
    # OBJECT (70 queries) — 纯物体/属性检索
    # ==================================================================

    # --- Object-Bus (25) ---
    {"query": "公交车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "巴士", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大客车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "公共汽车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大型客车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "一辆公交车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大巴车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "客车行驶", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "公交车在道路上", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "巴士经过路口", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "找一辆公交车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "有巴士的画面", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "公交车的视频", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大型公交车辆", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "道路上的巴士", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "公共汽车行驶中", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "找一下公交车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "有没有大巴车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "客运车辆", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "公交大巴", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大巴士", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "城市公交车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "看到一辆大巴", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "长途客车", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},
    {"query": "大型巴士", "expected": BUS_VIDEOS, "category": "object", "subcategory": "bus"},

    # --- Object-Truck (20) ---
    {"query": "货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "大货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "重型卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "一辆货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "卡车在道路上", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "货车经过", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "大卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "运货车辆", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "找一辆货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "有卡车的画面", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "货车的视频", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "大型货运车辆", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "道路上的卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "找一下货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "大卡车行驶", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "载货汽车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "货运卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "有没有大货车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},
    {"query": "看到一辆卡车", "expected": TRUCK_VIDEOS, "category": "object", "subcategory": "truck"},

    # --- Object-Car/General (25) ---
    {"query": "汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "轿车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "小汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "机动车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "车辆", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "一辆汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "道路上的汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "小轿车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "机动车辆", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "行驶中的车辆", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "路上的车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "找一辆汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "有车辆的画面", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "小型客车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "机动车行驶", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "一辆轿车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "小汽车在道路上", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "汽车经过", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "看看有没有车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "路上的机动车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "行驶的轿车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "路面上的小汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "机动汽车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "道路上的小轿车", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},
    {"query": "汽车行驶中", "expected": CAR_VIDEOS, "category": "object", "subcategory": "car"},

    # ==================================================================
    # EVENT (70 queries) — 纯事件检索
    # ==================================================================

    # --- Event-WrongWay (25) ---
    {"query": "车辆逆行", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "逆向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "机动车逆向驾驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆方向与道路规定相反", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "机动车违规反向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆驶入对向车道", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "反方向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆朝相反方向移动", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆错误方向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "汽车逆着车流行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆进入反向道路", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "逆行车辆", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "反向驾驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆反向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "违法逆行", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "机动车逆向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "违章逆行车辆", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆违反方向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "逆向通行", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆不走正确方向", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "有车逆向行驶", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "逆行违章", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "车辆驶向相反方向", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "机动车反方向通行", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},
    {"query": "逆向行驶的车辆", "expected": WW_HIGH, "category": "event", "subcategory": "wrong_way"},

    # --- Event-CrossLine (25) ---
    {"query": "车辆压线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "压线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆跨越道路标线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆驶过车道边界", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "机动车压道路实线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆跨线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "大型车辆越过车道线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆没有保持车道", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "货车偏离正常车道", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "机动车压道路标线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆压着标线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "跨越车道线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆越过道路标线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "压线违章", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆压线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "不按规定车道行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆偏离车道", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "机动车跨越标线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "压线车辆", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆越线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "跨线违章", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆不保持车道", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "压道路实线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "车辆越过车道分隔线", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},
    {"query": "机动车越线行驶", "expected": CL_HIGH, "category": "event", "subcategory": "cross_line"},

    # --- Event-RedLight (20) ---
    {"query": "车辆闯红灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "闯红灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "汽车违反交通信号灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "红灯亮时车辆通过路口", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆无视红灯继续行驶", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "机动车闯信号灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "小车冲过红灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆抢红灯通过路口", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆未遵守红绿灯规则", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆不按交通信号通行", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "红灯时通过路口", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "闯信号灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆在红灯时通过", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "抢红灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "违章闯红灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "红灯期间通过路口", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "机动车违反信号灯", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "车辆闯红灯通过", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "无视红灯通过", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},
    {"query": "红灯时强行通过", "expected": RL_VIDEOS, "category": "event", "subcategory": "red_light"},

    # ==================================================================
    # COMBO (60 queries) — 组合查询（物体+事件）
    # ==================================================================

    # --- Combo-Bus+WrongWay (15) ---
    {"query": "公交车逆行", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "巴士逆向行驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "大客车反向行驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公共汽车逆向驾驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公交车驶入对向车道", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "巴士反方向行驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "大客车走错方向", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公共汽车错误方向行驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公交车违规逆行", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "客车逆向通过道路", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "大巴车反方向开", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公交车方向错误", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "巴士违章逆行", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "大客车逆向通行", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},
    {"query": "公共汽车反向驾驶", "expected": BUS_WW, "category": "combo", "subcategory": "bus_ww"},

    # --- Combo-Truck+CrossLine (15) ---
    {"query": "货车压线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "卡车跨越道路标线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "大货车压道路实线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "重型卡车越过车道线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "货车没有保持车道", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "卡车偏离正常车道", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "大货车压道路标线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "货车跨越车道线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "卡车压线行驶", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "大货车越线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "重型货车压标线", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "卡车不保持车道", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "货车偏离车道", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "大卡车跨线行驶", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},
    {"query": "货车压着标线开", "expected": TRUCK_CL, "category": "combo", "subcategory": "truck_cl"},

    # --- Combo-Car+RedLight (10) ---
    {"query": "汽车闯红灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "轿车闯红灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "小汽车违反信号灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "机动车红灯时通过", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "车辆在路口闯红灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "汽车抢红灯通过", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "轿车无视红灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "小汽车闯信号灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "汽车红灯期间通过", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},
    {"query": "轿车不遵守红灯", "expected": CAR_RL, "category": "combo", "subcategory": "car_rl"},

    # --- Combo-Truck+WrongWay (5) ---
    {"query": "货车逆行", "expected": TRUCK_WW, "category": "combo", "subcategory": "truck_ww"},
    {"query": "卡车逆向行驶", "expected": TRUCK_WW, "category": "combo", "subcategory": "truck_ww"},
    {"query": "大货车反向行驶", "expected": TRUCK_WW, "category": "combo", "subcategory": "truck_ww"},
    {"query": "卡车走错方向", "expected": TRUCK_WW, "category": "combo", "subcategory": "truck_ww"},
    {"query": "货车驶入对向车道", "expected": TRUCK_WW, "category": "combo", "subcategory": "truck_ww"},

    # --- Combo-Bus+CrossLine (10) ---
    {"query": "公交车压线", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "巴士跨越标线", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "大客车越过车道线", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "公共汽车压线行驶", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "公交车没有保持车道", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "客车偏离车道", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "巴士压道路标线", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "大客车越线行驶", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "公共汽车跨线", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},
    {"query": "公交车偏离车道", "expected": BUS_CL, "category": "combo", "subcategory": "bus_cl"},

    # --- Combo-Vague (5) ---
    {"query": "有车辆违反交通规则", "expected": ALL_EVENT, "category": "combo", "subcategory": "vague"},
    {"query": "道路上存在异常驾驶行为", "expected": ALL_EVENT, "category": "combo", "subcategory": "vague"},
    {"query": "寻找违规车辆", "expected": ALL_EVENT, "category": "combo", "subcategory": "vague"},
    {"query": "驾驶行为存在安全隐患", "expected": ALL_EVENT, "category": "combo", "subcategory": "vague"},
    {"query": "交通违法事件", "expected": ALL_EVENT, "category": "combo", "subcategory": "vague"},
]

# ---------------------------------------------------------------------------
# 事件元数据加载
# ---------------------------------------------------------------------------

def load_event_metadata() -> Dict[str, List[Dict]]:
    event_dir = PROJECT_ROOT / "metadata" / "events"
    video_events: Dict[str, List[Dict]] = {}
    if not event_dir.exists():
        return video_events
    for json_file in sorted(event_dir.glob("*.json")):
        if json_file.name == ".gitkeep":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            video_name = data.get("video_name", "")
            stem = Path(video_name).stem if video_name else json_file.stem
            events = data.get("events", [])
            video_events[stem] = events
        except Exception:
            pass
    logger.info("Loaded event metadata for %s videos", len(video_events))
    return video_events


# ---------------------------------------------------------------------------
# 帧加载与 CN-CLIP 编码
# ---------------------------------------------------------------------------

def load_frame_metadata() -> List[Dict]:
    metadata_path = PROJECT_ROOT / "index" / "frame_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("Loaded %s frame metadata entries", len(metadata))
    return metadata


def resolve_existing_frames(metadata: List[Dict]) -> tuple[List[Path], List[Dict]]:
    frame_paths: List[Path] = []
    valid_metadata: List[Dict] = []
    for item in metadata:
        raw_path = item.get("frame_path", "")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            alt_path = PROJECT_ROOT / "frames" / Path(raw_path).name
            if alt_path.exists():
                path = alt_path
            else:
                continue
        frame_paths.append(path)
        valid_metadata.append(item)
    logger.info("Resolved %s existing frame files", len(frame_paths))
    return frame_paths, valid_metadata


def encode_frames(embedding_service: EmbeddingService, frame_paths: List[Path], batch_size: int = 16) -> np.ndarray:
    all_vectors: List[np.ndarray] = []
    total = len(frame_paths)
    start = time.time()
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = frame_paths[batch_start:batch_end]
        records = embedding_service.batch_encode(frame_paths=batch_paths, batch_size=len(batch_paths))
        batch_vectors = np.asarray([r["embedding"] for r in records], dtype=np.float32)
        all_vectors.append(batch_vectors)
        if batch_end % 64 == 0 or batch_end == total:
            logger.info("Encoded %s/%s frames (%.1fs)", batch_end, total, time.time() - start)
    matrix = np.vstack(all_vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms
    return matrix


def build_faiss_index(embeddings: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# 检索核心
# ---------------------------------------------------------------------------

def normalize_video_stem(video_name: str) -> str:
    return Path(str(video_name)).stem.casefold()


def clip_search(embedding_service, index, frame_metadata, queries, top_k=5):
    merged: Dict[str, Dict] = {}
    candidate_k = max(top_k * 3, top_k)
    for q in queries:
        q_vector = embedding_service.encode_text(q)
        q_vector = q_vector.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q_vector, candidate_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(frame_metadata):
                continue
            meta = frame_metadata[idx]
            key = str(meta.get("frame_id") or meta.get("frame_path", ""))
            existing = merged.get(key)
            if existing is None or score > existing["score"]:
                merged[key] = {
                    "rank": 0, "score": float(score),
                    "video_name": meta.get("video_name", ""),
                    "timestamp": float(meta.get("timestamp_seconds", meta.get("timestamp", 0.0))),
                    "frame_path": meta.get("frame_path", ""),
                }
    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k * 2]
    return results


def dedup_by_video(results, top_k=5):
    best: Dict[str, Dict] = {}
    for r in results:
        stem = normalize_video_stem(r["video_name"])
        existing = best.get(stem)
        if existing is None or r["score"] > existing["score"]:
            best[stem] = r
    deduped = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return deduped


def event_search(event_metadata, event_types, top_k=5):
    type_set = set(event_types)
    best: Dict[str, Dict] = {}
    for vstem, events in event_metadata.items():
        for ev in events:
            et = ev.get("event_type", "")
            if type_set and et not in type_set:
                continue
            conf = float(ev.get("confidence", 0.0))
            existing = best.get(vstem)
            if existing is None or conf > existing["score"]:
                best[vstem] = {
                    "score": conf, "video_name": vstem + ".avi",
                    "event_type": et, "event_confidence": conf,
                }
    results = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k * 4]


def evaluate_query(results, expected_videos):
    expected_set = {normalize_video_stem(v) for v in expected_videos}
    hit1 = hit5 = False
    first_rank = 0
    for r in results:
        rv = normalize_video_stem(r["video_name"])
        rank = r["rank"]
        if rv in expected_set:
            if rank == 1:
                hit1 = True
            if rank <= 5:
                hit5 = True
            if first_rank == 0:
                first_rank = rank
    return {"hit_at_1": hit1, "hit_at_5": hit5, "first_relevant_rank": first_rank}


# ---------------------------------------------------------------------------
# TQUM Phase 3 检索
# ---------------------------------------------------------------------------

def run_tqum_phase4(embedding_service, index, frame_metadata, event_metadata, top_k=5):
    tqum = QueryRewriteService(use_chinese_clip=True)

    class RouteHelper:
        route_by_confidence = HybridSearchService.route_by_confidence
        _intent_value = HybridSearchService._intent_value

    route_helper = RouteHelper()

    query_reports = []
    VISUAL_TOKENS = {"红色", "白色", "黑色", "蓝色", "黄色", "浅色",
                     "汽车", "轿车", "货车", "卡车", "公交车", "巴士",
                     "摩托车", "机动", "大车", "小车", "红灯", "压线", "逆行"}

    for i, item in enumerate(TEST_QUERIES, start=1):
        query = item["query"]
        expected = item["expected"]
        category = item["category"]
        subcategory = item.get("subcategory", "")

        intent = tqum.parse_query_intent(query)
        event_types = intent.event_types
        event_conf = intent.event_confidence
        rewrites = intent.rewritten_queries
        attributes = intent.attributes if hasattr(intent, "attributes") else {}

        route_info = route_helper.route_by_confidence(intent)
        route = route_info["route"]

        has_visual_attr = bool(attributes.get("color") or attributes.get("vehicle_type"))
        results = []

        if route in ("event_primary", "clip_primary", "vague_event"):
            event_boost = route_info.get("event_boost", 0.02)
            search_types = event_types if route != "vague_event" else []
            event_candidates = event_search(event_metadata, search_types, top_k=top_k * 4) if (search_types or route == "vague_event") else []

            event_confs = {}
            for ec in event_candidates:
                stem = normalize_video_stem(ec["video_name"])
                event_confs[stem] = max(event_confs.get(stem, 0.0), ec["event_confidence"])

            if has_visual_attr:
                clip_queries = [r for r in rewrites if any(tok in r for tok in VISUAL_TOKENS)]
                if not clip_queries:
                    clip_queries = rewrites[:4]
            else:
                clip_queries = [query] + [r for r in rewrites if r != query][:3]

            clip_results = clip_search(embedding_service, index, frame_metadata, clip_queries, top_k * 4)
            clip_results = dedup_by_video(clip_results, top_k * 4)

            reranked = []
            for cr in clip_results:
                stem = normalize_video_stem(cr["video_name"])
                ec = event_confs.get(stem, 0.0)
                combined = cr["score"] + event_boost * ec
                # Event-confidence score floor: ensure high-confidence event matches
                # surface even when CLIP score is low (e.g., red_light visual ambiguity)
                if ec > 0:
                    combined = max(combined, ec * 0.60)
                reranked.append({"score": combined, "video_name": cr["video_name"],
                                 "clip_score": cr["score"], "event_confidence": ec})

            clip_stems = {normalize_video_stem(cr["video_name"]) for cr in clip_results}
            for ec in event_candidates:
                stem = normalize_video_stem(ec["video_name"])
                if stem not in clip_stems:
                    # Event-only fallback: base score + event boost
                    combined = 0.35 + event_boost * ec["event_confidence"]
                    combined = max(combined, ec["event_confidence"] * 0.60)
                    reranked.append({"score": combined, "video_name": ec["video_name"],
                                     "clip_score": 0.0, "event_confidence": ec["event_confidence"]})

            reranked.sort(key=lambda x: x["score"], reverse=True)
            results = reranked[:top_k]
        else:
            results = clip_search(embedding_service, index, frame_metadata, rewrites, top_k * 3)
            results = dedup_by_video(results, top_k)

        for rank_idx, r in enumerate(results, start=1):
            r["rank"] = rank_idx

        eval_result = evaluate_query(results, expected)

        report = {
            "query": query, "category": category, "subcategory": subcategory,
            "expected_count": len(expected),
            "hit_at_1": eval_result["hit_at_1"], "hit_at_5": eval_result["hit_at_5"],
            "first_relevant_rank": eval_result["first_relevant_rank"],
            "route": route, "event_conf": event_conf,
            "top5_results": [{"rank": r["rank"], "video": r["video_name"],
                              "score": round(r["score"], 4)} for r in results],
        }
        query_reports.append(report)

        status = "OK" if eval_result["hit_at_1"] else ("TOP5" if eval_result["hit_at_5"] else "MISS")
        logger.info("[%3d/200] [%s] [%s] %s -> %s | route=%s",
                    i, status, category, query,
                    results[0]["video_name"] if results else "N/A", route)

    return query_reports


# ---------------------------------------------------------------------------
# 分维度统计
# ---------------------------------------------------------------------------

def compute_stats(reports):
    total = len(reports)
    hit1 = sum(1 for r in reports if r["hit_at_1"])
    hit5 = sum(1 for r in reports if r["hit_at_5"])
    rr = sum((1.0 / r["first_relevant_rank"]) if r["first_relevant_rank"] > 0 else 0.0 for r in reports)
    return {
        "count": total,
        "recall_at_1": round(hit1 / total, 4) if total else 0,
        "recall_at_5": round(hit5 / total, 4) if total else 0,
        "mrr": round(rr / total, 4) if total else 0,
        "hit1": hit1, "hit5": hit5,
    }


def build_category_report(query_reports):
    categories = ["object", "event", "combo"]
    result = {}
    for cat in categories:
        cat_reports = [r for r in query_reports if r["category"] == cat]
        stats = compute_stats(cat_reports)
        # Subcategory breakdown
        subcats = sorted(set(r.get("subcategory", "") for r in cat_reports))
        sub_stats = {}
        for sc in subcats:
            sc_reports = [r for r in cat_reports if r.get("subcategory", "") == sc]
            sub_stats[sc] = compute_stats(sc_reports)
        result[cat] = {"overall": stats, "subcategories": sub_stats}
    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    settings = get_settings()
    settings.clip_backend = "cnclip"
    settings.cnclip_model_name = "ViT-B-16"

    logger.info("Phase 4 evaluation starting (200 queries, model=%s)", settings.cnclip_model_name)

    all_metadata = load_frame_metadata()
    frame_paths, valid_metadata = resolve_existing_frames(all_metadata)
    event_metadata = load_event_metadata()

    embedding_service = EmbeddingService(settings)
    embeddings = encode_frames(embedding_service, frame_paths, batch_size=16)
    index = build_faiss_index(embeddings)

    logger.info("=" * 60)
    logger.info("Running TQUM Phase 4 evaluation (200 queries)")
    logger.info("=" * 60)

    query_reports = run_tqum_phase4(embedding_service, index, valid_metadata, event_metadata, top_k=5)

    # Overall stats
    overall = compute_stats(query_reports)
    category_report = build_category_report(query_reports)

    # Print results
    print("\n" + "=" * 70)
    print("Phase 4 正式评测结果（200 条测试集）")
    print("=" * 70)

    print(f"\n{'维度':<20} {'数量':>6} {'R@1':>8} {'R@5':>8} {'MRR':>8}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'Overall':<20} {overall['count']:>6} {overall['recall_at_1']:>7.1%} {overall['recall_at_5']:>7.1%} {overall['mrr']:>8.4f}")

    for cat in ["object", "event", "combo"]:
        cat_data = category_report[cat]
        s = cat_data["overall"]
        label_map = {"object": "Object Recall", "event": "Event Accuracy", "combo": "Combo Recall"}
        print(f"{label_map[cat]:<20} {s['count']:>6} {s['recall_at_1']:>7.1%} {s['recall_at_5']:>7.1%} {s['mrr']:>8.4f}")
        # Subcategories
        for sc, sc_stats in cat_data["subcategories"].items():
            print(f"  {sc:<18} {sc_stats['count']:>6} {sc_stats['recall_at_1']:>7.1%} {sc_stats['recall_at_5']:>7.1%} {sc_stats['mrr']:>8.4f}")

    # 达标判定
    print("\n" + "=" * 70)
    print("达标判定")
    print("=" * 70)

    obj_r1 = category_report["object"]["overall"]["recall_at_1"]
    obj_r5 = category_report["object"]["overall"]["recall_at_5"]
    evt_r1 = category_report["event"]["overall"]["recall_at_1"]
    evt_r5 = category_report["event"]["overall"]["recall_at_5"]
    cmb_r1 = category_report["combo"]["overall"]["recall_at_1"]
    cmb_r5 = category_report["combo"]["overall"]["recall_at_5"]

    thresholds = [
        ("Object Recall R@1", obj_r1, 0.60, obj_r1 >= 0.60),
        ("Object Recall R@5", obj_r5, 0.80, obj_r5 >= 0.80),
        ("Event Accuracy R@1", evt_r1, 0.50, evt_r1 >= 0.50),
        ("Event Accuracy R@5", evt_r5, 0.75, evt_r5 >= 0.75),
        ("Combo Recall R@1", cmb_r1, 0.45, cmb_r1 >= 0.45),
        ("Combo Recall R@5", cmb_r5, 0.70, cmb_r5 >= 0.70),
    ]

    all_pass = True
    for name, value, threshold, passed in thresholds:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<25} {value:.1%} (目标 {threshold:.0%}) [{status}]")

    print(f"\n  总体判定: {'全部达标' if all_pass else '未达标'}")

    # Save report
    output_path = PROJECT_ROOT / "tests" / "phase4_eval_report.json"
    full_report = {
        "total_queries": len(query_reports),
        "overall": overall,
        "categories": category_report,
        "thresholds": {name: {"value": val, "target": thr, "passed": p} for name, val, thr, p in thresholds},
        "all_pass": all_pass,
        "query_reports": query_reports,
    }
    output_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
