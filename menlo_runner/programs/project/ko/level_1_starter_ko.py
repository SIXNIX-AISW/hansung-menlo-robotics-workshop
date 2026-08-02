from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 1 프로젝트 시작 파일입니다."""

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.llm import ask_vlm, call_llm
from menlo_runner.perception import detect_color_blobs
from menlo_runner.scene import delivered_cube_ids, held_cube_info

TASK = "Find and sort cubes from the source area into their matching destination pads."

APPROVED_LLM_MODELS = (
    "minimaxai/minimax-m3",
    "qwen/qwen3.6-35b-a3b",
)
LLM_MODEL = os.environ.setdefault("MENLO_LLM_MODEL", "qwen/qwen3.6-35b-a3b")
VLM_MODEL = os.environ.setdefault("MENLO_VLM_MODEL", "minimaxai/minimax-m3")

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
COLOR_NAMES_KO = {
    "red": "빨강",
    "green": "초록",
    "blue": "파랑",
    "yellow": "노랑",
}
ACTION_NAMES_KO = {
    "search_cube": "큐브 찾기",
    "navigate_to_cube": "큐브로 이동",
    "pick_cube": "큐브 집기",
    "search_pad": "목적지 패드 찾기",
    "navigate_to_pad": "목적지 패드로 이동",
    "place_cube": "큐브 내려놓기",
    "recover": "복구 동작",
    "skip_target": "대상 건너뛰기",
    "stop": "정지",
}
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)

ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}


@dataclass
class AgentDecision:
    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    delivered_count: int = 0
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    last_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    known_source_location: dict[str, Any] | None = None
    known_pad_locations: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_vlm_signage: str = ""
    target_sign_position: str | None = None
    target_sign_confidence: float = 0.0
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        return self.angle_deg + math.degrees(self.head_yaw)


def parse_agent_decision(text: str) -> AgentDecision | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    target_color = data.get("target_color")
    if target_color is not None and not isinstance(target_color, str):
        return None
    if isinstance(target_color, str) and target_color not in VALID_COLORS:
        return None

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visible = [
        {
            "color": detection.color,
            "angle_deg": detection.angle_deg,
            "full_bearing_deg": round(getattr(detection, "full_bearing_deg", detection.angle_deg), 1),
            "blob_area": detection.blob_area,
            "bbox": detection.bbox,
            "is_pad": getattr(detection, "is_pad", False),
        }
        for detection in observation.detections
    ]
    destination_target = None
    if memory.held_color in DESTINATION_SIGN_RULES:
        destination_target = {
            "held_color": memory.held_color,
            "sign_letter": DESTINATION_SIGN_RULES[memory.held_color],
            "rule": f"{memory.held_color} cube must go to sign {DESTINATION_SIGN_RULES[memory.held_color]}",
            "known_location": memory.known_pad_locations.get(memory.held_color),
        }

    return {
        "task": task,
        "visible_targets": visible,
        "absent_cube_colors": sorted(ABSENT_CUBE_COLORS),
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "destination_target": destination_target,
        "delivered_count": memory.delivered_count,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "known_source_location": memory.known_source_location,
        "known_pad_locations": memory.known_pad_locations,
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
    }


# ---------------------------------------------------------------------------
# 지원 코드: SDK wrapper
# ---------------------------------------------------------------------------
async def get_robot_status(ctx: Any) -> Any:
    return await ctx.state("robot_status")

async def get_camera_frame(ctx: Any) -> bytes:
    return await ctx.get_vision("pov")

async def get_delivered_count(ctx: Any) -> int:
    return len(await delivered_cube_ids(ctx))

async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    held = await held_cube_info(ctx)
    return {"entity_id": held[0], "color": held[1]} if held else None

def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" Robot이 {held_color} cube를 들고 있으므로 target destination sign은 {DESTINATION_SIGN_RULES[held_color]}입니다."
    return (
        "이 robot camera frame에 보이는 warehouse sign을 읽으세요. "
        f"{SIGNAGE_NOTE} "
        "A sign은 source이므로 destination 후보에서 제외하세요. "
        "보이는 sign letter, color, 대략적인 left/center/right 위치, confidence를 JSON으로 반환하세요. "
        "반드시 signs 배열을 사용하고 각 항목은 letter, color, position, confidence key를 포함하세요."
        + target
    )

async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)

async def perceive(ctx: Any) -> list[Any]:
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)

async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    args: dict[str, float] = {}
    if yaw is not None:
        args["yaw"] = yaw
    if pitch is not None:
        args["pitch"] = pitch
    return await ctx.invoke("set_head", args, timeout_s=10)

async def perceive_scanned_view(ctx: Any, *, yaw: float = 0.0, pitch: float = 0.15) -> list[Any]:
    await set_head(ctx, yaw=yaw, pitch=pitch)
    await asyncio.sleep(0.4)
    detections = await perceive(ctx)
    return [
        ScannedDetection(
            color=detection.color,
            angle_deg=detection.angle_deg,
            blob_area=detection.blob_area,
            centroid=detection.centroid,
            bbox=detection.bbox,
            head_yaw=yaw,
            head_pitch=pitch,
        )
        for detection in detections
    ]

async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """set_velocity를 완전한 fire-and-forget으로 실행합니다.

    [핵심 수정] 기존엔 invoke 응답을 최대 6초까지 기다린 뒤 duration_s만큼
    추가로 sleep했음 — RPC가 계속 timeout나는 상황에서 한 번 이동에
    6초 이상 낭비되어 cycle이 극도로 느려졌음(로봇이 거의 안 움직이는 것처럼 보임).
    이제 invoke를 background task로 던지고 local에서는 duration_s만큼만 대기.
    """
    async def _fire_and_ignore() -> None:
        try:
            await ctx.invoke(
                "set_velocity",
                {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
                timeout_s=10,
            )
        except Exception:
            pass  # 응답 없어도 무시 — 시뮬레이터는 이미 명령을 실행 중일 수 있음

    asyncio.create_task(_fire_and_ignore())
    await asyncio.sleep(duration_s)
    return None

async def cancel_action(ctx: Any) -> Any:
    return await ctx.invoke("cancel", {})

async def pick_nearest_cube(ctx: Any) -> Any:
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )

async def place_nearest_zone(ctx: Any) -> Any:
    return await ctx.invoke("place_entity", {}, timeout_s=300)

def result_summary(result: Any) -> dict[str, Any]:
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }

def format_decision_ko(decision: AgentDecision) -> str:
    action = ACTION_NAMES_KO.get(decision.next_action, decision.next_action)
    color = COLOR_NAMES_KO.get(decision.target_color or "", decision.target_color or "없음")
    parts = [f"다음 행동={action}", f"대상 색={color}"]
    if decision.reason:
        parts.append(f"이유={decision.reason}")
    if decision.recovery_strategy:
        parts.append(f"복구 전략={decision.recovery_strategy}")
    return ", ".join(parts)

async def scan_head(
    ctx: Any,
    *,
    yaws: tuple[float, ...] = (-0.8, 0.0, 0.8),
    pitch: float = 0.15,
    per_view_timeout_s: float = 8.0,
) -> list[Any]:
    all_detections: list[Any] = []
    for yaw in yaws:
        try:
            await asyncio.wait_for(set_head(ctx, yaw=yaw, pitch=pitch), timeout=per_view_timeout_s)
            await asyncio.sleep(0.4)
            detections = await asyncio.wait_for(perceive(ctx), timeout=per_view_timeout_s)
        except asyncio.TimeoutError:
            print(f"scan_head timeout at yaw={yaw:+.2f}; skipping.")
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"scan_head failed at yaw={yaw:+.2f}: {exc}")
            continue
        for detection in detections:
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    head_yaw=yaw,
                    head_pitch=pitch,
                )
            )
    return all_detections


VALID_COLORS = set(DESTINATION_SIGN_RULES)
# 특정 라운드에 실제로 없는 색이 있으면 여기 채워서 그 색 큐브 타겟팅을 막을 수
# 있지만, cube_color_order_key가 라운드마다 달라 어떤 색이 나올지 고정돼 있지
# 않으므로 기본값은 비워둡니다(하드코딩하면 그 색이 등장하는 라운드에서
# 영원히 못 집게 됨).
ABSENT_CUBE_COLORS: set[str] = set()
CUBE_DETECTION_MIN_AREA = 350
PICK_READY_MIN_AREA = 850
SOURCE_EXCLUSION_RADIUS_M = 1.8

# ── 패드 판별 기준 ─────────────────────────────────────────────────────────
# 패드 위 표지판(C/D/B)은 큐브와 같은 색 blob으로 감지되지만
# 위치(화면 위쪽)와 가로/세로 비율(넓음)로 큐브와 구별 가능
PAD_MIN_AREA = 400           # 패드 표지판 최소 blob 크기 (멀리 있어도 감지되도록 완화)
PAD_MAX_CENTROID_Y = 260     # 화면 y좌표 260 이하 = 위쪽 = 패드 (큐브는 아래쪽)
PAD_MIN_ASPECT = 0.6         # bbox 가로/세로 비율 (완화)
CUBE_MAX_CENTROID_Y = 280    # 화면 y좌표 280 이상 = 아래쪽 = 큐브일 가능성 높음


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def is_pad_detection(detection: Any) -> bool:
    """blob이 패드 표지판인지 판별합니다.

    패드 표지판 특징:
    - 화면 위쪽에 위치 (centroid_y 작음)
    - 큐브보다 blob_area가 크고 가로로 넓음
    - 큐브와 달리 화면 상단에 고정 위치
    """
    x, y, w, h = detection.bbox
    area = detection.blob_area
    centroid_y = detection.centroid[1]
    aspect = w / max(h, 1)

    return (
        area >= PAD_MIN_AREA
        and centroid_y <= PAD_MAX_CENTROID_Y
        and aspect >= PAD_MIN_ASPECT
    )

def is_cube_detection(detection: Any) -> bool:
    """blob이 큐브인지 판별합니다."""
    centroid_y = detection.centroid[1]
    area = detection.blob_area
    return (
        area >= CUBE_DETECTION_MIN_AREA
        and centroid_y >= CUBE_MAX_CENTROID_Y - 50  # 큐브는 화면 아래쪽
    )

def robot_xy_yaw_deg(robot_status: Any) -> tuple[float, float, float] | None:
    try:
        pose = robot_status.robot.pose
        return float(pose.position[0]), float(pose.position[1]), float(pose.yaw_deg)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

def detection_score(detection: Any) -> float:
    bearing = abs(float(getattr(detection, "full_bearing_deg", detection.angle_deg)))
    center_bonus = max(0.0, 1.0 - bearing / 90.0)
    return float(detection.blob_area) * (0.75 + 0.25 * center_bonus)

def estimate_xy_for_detection(
    robot_status: Any,
    detection: Any,
    *,
    target_kind: str = "cube",
) -> tuple[float, float] | None:
    pose = robot_xy_yaw_deg(robot_status)
    if pose is None or detection.blob_area <= 0:
        return None
    robot_x, robot_y, yaw_deg = pose
    bearing_deg = yaw_deg + float(getattr(detection, "full_bearing_deg", detection.angle_deg))
    if target_kind == "pad":
        scale, min_distance, max_distance = 95.0, 0.6, 3.0
    else:
        scale, min_distance, max_distance = 85.0, 0.55, 5.5
    distance = clamp(scale / math.sqrt(float(detection.blob_area)), min_distance, max_distance)
    bearing_rad = math.radians(bearing_deg)
    x = robot_x + math.cos(bearing_rad) * distance
    y = robot_y + math.sin(bearing_rad) * distance
    return round(x, 3), round(y, 3)

def xy_is_near_known_source(memory: AgentMemory, xy: tuple[float, float]) -> bool:
    source = memory.known_source_location
    if source is None or "xy" not in source:
        return False
    return xy_distance(tuple(source["xy"]), xy) < SOURCE_EXCLUSION_RADIUS_M

def color_failed_attempts(memory: AgentMemory, color: str | None, *, action: str | None = None) -> int:
    if not color:
        return 0
    prefix = f"{action}:" if action else ""
    return sum(
        count
        for key, count in memory.failed_attempts.items()
        if key.startswith(prefix) and key.endswith(f":{color}")
    )

def color_is_blocked_as_cube(memory: AgentMemory, color: str | None) -> bool:
    if not color:
        return False
    return (
        color in ABSENT_CUBE_COLORS
        or color in memory.completed_colors
        or color in memory.skipped_colors
        or color_failed_attempts(memory, color, action="pick_cube") >= 2
    )

def fallback_decision(decision_context: dict[str, Any], memory: AgentMemory) -> AgentDecision:
    visible = decision_context["visible_targets"]

    def visible_score(item: dict[str, Any]) -> float:
        bearing = abs(float(item.get("full_bearing_deg", item.get("angle_deg", 0.0))))
        center_bonus = max(0.0, 1.0 - bearing / 90.0)
        return float(item["blob_area"]) * (0.75 + 0.25 * center_bonus)

    if memory.held_color:
        target_color = memory.held_color
        if memory.stage == "near_pad":
            return AgentDecision(next_action="place_cube", target_color=target_color,
                                 reason="Fallback: pad 근처이므로 place 시도.")
        if target_color in memory.known_pad_locations:
            return AgentDecision(next_action="navigate_to_pad", target_color=target_color,
                                 reason="Fallback: 저장된 pad 위치 사용.")
        return AgentDecision(next_action="search_pad", target_color=target_color,
                             reason="Fallback: pad 탐색.")

    if memory.stage == "near_cube" and memory.active_color:
        return AgentDecision(next_action="pick_cube", target_color=memory.active_color,
                             reason="Fallback: cube 근처이므로 pick 시도.")

    cube_candidates = [
        item for item in visible
        if not color_is_blocked_as_cube(memory, item.get("color"))
        and not item.get("is_pad", False)
    ]
    if cube_candidates:
        target = max(cube_candidates, key=visible_score)
        return AgentDecision(next_action="navigate_to_cube", target_color=target["color"],
                             reason="Fallback: visible cube 선택.")
    if memory.known_source_location is not None:
        return AgentDecision(next_action="navigate_to_cube", target_color=memory.active_color,
                             reason="Fallback: source 캐시로 이동.")
    return AgentDecision(next_action="search_cube", reason="Fallback: 보이는 target 없음.")

def enforce_stage_decision(decision: AgentDecision, memory: AgentMemory) -> AgentDecision:
    cube_actions = {"search_cube", "navigate_to_cube", "pick_cube"}
    pad_actions = {"search_pad", "navigate_to_pad", "place_cube"}

    if memory.held_color and decision.next_action in cube_actions:
        target_color = memory.held_color
        if memory.stage == "near_pad":
            return AgentDecision(next_action="place_cube", target_color=target_color,
                                 reason=f"Guard: place 시도.")
        if target_color in memory.known_pad_locations:
            return AgentDecision(next_action="navigate_to_pad", target_color=target_color,
                                 reason=f"Guard: cached pad로 이동.")
        return AgentDecision(next_action="search_pad", target_color=target_color,
                             reason=f"Guard: pad 탐색.")

    # [핵심 수정] held_color가 있는데 pad_actions의 target_color가 다르면
    # LLM이 잘못 고른 색(예: green)이 아니라 반드시 held_color(예: blue)로 강제
    if memory.held_color and decision.next_action in pad_actions:
        if decision.target_color != memory.held_color:
            return AgentDecision(next_action=decision.next_action, target_color=memory.held_color,
                                 reason=f"Guard: pad target_color를 held_color({memory.held_color})로 강제 교정. "
                                        f"원래={decision.target_color}")

    if not memory.held_color and decision.next_action in pad_actions:
        return AgentDecision(next_action="search_cube", target_color=memory.active_color,
                             reason=f"Guard: cube 없이 pad 행동 차단.")

    if not memory.held_color and decision.next_action in cube_actions and color_is_blocked_as_cube(memory, decision.target_color):
        return AgentDecision(next_action="search_cube", target_color=None,
                             reason=f"Guard: {decision.target_color} cube 차단됨.")
    return decision


# ---------------------------------------------------------------------------
# 학생 TODO 함수들
# ---------------------------------------------------------------------------
async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    decision_context = build_decision_context(task, observation, memory, last_result)
    api_key = os.environ.get("TOKAMAK_API_KEY", "").strip()
    if not api_key:
        return enforce_stage_decision(fallback_decision(decision_context, memory), memory)

    system_prompt = (
        "You are a high-level supervisor for a Level 1 warehouse robot. "
        "Return ONLY one JSON object. Do not include markdown. "
        "Allowed next_action values are: "
        + ", ".join(sorted(ALLOWED_NEXT_ACTIONS))
        + ". Use target_color as one of red, green, blue, yellow, or null. "
        "Do not choose absent_cube_colors as cube targets. "
        "Destination rules: red->B, green->C, blue->D, yellow->E. "
        "Pads are identified by colored signs (blobs) at the top of the camera frame (low centroid_y, is_pad=true). "
        "Cubes are at the bottom of the frame (high centroid_y, is_pad=false). "
        "When holding a cube, find and navigate to the matching pad sign blob. "
        "Do not output raw velocity commands, entity IDs, scene_state, or ground-truth coordinates."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(decision_context, ensure_ascii=False)},
    ]
    try:
        response = call_llm(messages, api_key=api_key, model=LLM_MODEL)
    except Exception as exc:
        fallback = fallback_decision(decision_context, memory)
        fallback.reason = f"{fallback.reason} LLM 호출 실패: {exc}"
        return enforce_stage_decision(fallback, memory)

    parsed = parse_agent_decision(response)
    if parsed is None:
        fallback = fallback_decision(decision_context, memory)
        fallback.reason = f"{fallback.reason} LLM JSON 검증 실패."
        return enforce_stage_decision(fallback, memory)
    return enforce_stage_decision(parsed, memory)


async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    held = await get_held_cube_info(ctx)
    memory.held_color = held["color"] if held else None
    if memory.held_color:
        memory.stage = "holding_cube" if memory.stage in {"need_cube", "near_cube"} else memory.stage
        memory.active_color = memory.held_color
    elif memory.stage == "holding_cube":
        memory.stage = "need_cube"
        memory.target_sign_position = None
        memory.target_sign_confidence = 0.0

    robot_status = await get_robot_status(ctx)

    # 큐브를 들고 있으면 패드 찾기용 pitch(위), 아니면 큐브 찾기용 pitch(아래)
    pitch = 0.08 if memory.held_color else 0.18
    detections_raw = await scan_head(ctx, yaws=(-0.8, 0.0, 0.8), pitch=pitch)

    # 각 detection에 is_pad 태그 추가
    detections = []
    for d in detections_raw:
        obj = ScannedDetection(
            color=d.color,
            angle_deg=d.angle_deg,
            blob_area=d.blob_area,
            centroid=d.centroid,
            bbox=d.bbox,
            head_yaw=d.head_yaw,
            head_pitch=d.head_pitch,
        )
        # frozen dataclass라 직접 속성 추가 불가 → dict 래퍼 형태로 처리
        detections.append(obj)

    observation = Observation(
        robot_status=robot_status,
        detections=detections,
        note=(
            f"stage={memory.stage}, held_color={memory.held_color}, "
            f"destination_sign={DESTINATION_SIGN_RULES.get(memory.held_color or '')}"
        ),
    )

    # 큐브 source 위치 기억 (큐브 탐색 중에만)
    if not memory.held_color:
        for d in detections:
            if color_is_blocked_as_cube(memory, d.color):
                continue
            if is_pad_detection(d):
                continue  # 패드 표지판은 source로 저장하지 않음
            xy = estimate_xy_for_detection(robot_status, d, target_kind="cube")
            if xy is None:
                continue
            previous = memory.known_source_location
            if previous is None or d.blob_area >= int(previous.get("blob_area", 0)):
                memory.known_source_location = {
                    "xy": xy,
                    "blob_area": d.blob_area,
                    "source": "visual_cube_source_candidate",
                }

    return observation


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    held = await get_held_cube_info(ctx)
    delivered_count = await get_delivered_count(ctx)
    verified = {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": delivered_count,
        "delivery_increased": delivered_count > int(action_result.get("previous_delivered_count", -1)),
        "held_cube": held,
        "held_color": held["color"] if held else None,
    }
    if action_result.get("action") == "pick_cube":
        verified["pick_success"] = held is not None
        if held is None:
            verified["failure_reason"] = "pick_cube 후 held cube 없음."
    if action_result.get("action") == "place_cube":
        verified["place_success"] = verified["delivery_increased"]
        if not verified["place_success"]:
            verified["failure_reason"] = "place_cube 후 delivered_count 미증가."
    return verified


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    previous_delivered_count = memory.delivered_count
    if "delivered_count" in verified:
        memory.delivered_count = int(verified["delivered_count"])
    memory.held_color = verified.get("held_color")
    action_result = verified.get("action_result", {})
    action = action_result.get("action")
    status = action_result.get("status")

    if memory.delivered_count > previous_delivered_count:
        if memory.active_color and memory.active_color not in memory.completed_colors:
            memory.completed_colors.append(memory.active_color)
        memory.stage = "need_cube"
        memory.active_color = None
        memory.held_color = None
        memory.search_turns = 0
        memory.target_sign_position = None
        memory.target_sign_confidence = 0.0
        # 배달 성공한 색상의 pad 캐시 제거 (다음 큐브와 혼동 방지)
        if decision.target_color in memory.known_pad_locations:
            del memory.known_pad_locations[decision.target_color]
    elif action == "pick_cube" and verified.get("pick_success"):
        memory.stage = "holding_cube"
        memory.active_color = memory.held_color
        memory.search_turns = 0
    elif action in ("search_cube", "search_pad"):
        # [핵심 수정] held_color 유무와 무관하게 search_turns 증가.
        # 기존엔 held_color가 있으면 이 분기 전에 걸려서 search_pad 실패 횟수가
        # 절대 누적되지 않아 "3회 실패 시 후퇴" 로직이 죽어있었음.
        memory.search_turns += 1
        if memory.held_color:
            memory.stage = "holding_cube"
            memory.active_color = memory.held_color
    elif memory.held_color:
        if action == "navigate_to_pad" and status == "moved":
            memory.stage = "near_pad"
        else:
            memory.stage = "holding_cube"
        memory.active_color = memory.held_color
    elif action == "navigate_to_cube" and status == "moved":
        memory.stage = "near_cube"
        memory.active_color = action_result.get("target_color") or decision.target_color or memory.active_color
    elif action == "pick_cube" and memory.held_color is None:
        memory.stage = "near_cube" if status == "failed" and memory.active_color else "need_cube"
    elif action == "recover":
        memory.stage = "holding_cube" if memory.held_color else "need_cube"

    if (
        status == "failed"
        or (action == "pick_cube" and not verified.get("pick_success", False))
        or (action == "place_cube" and not verified.get("place_success", False))
    ):
        key = f"{action}:{decision.target_color or memory.active_color or memory.held_color or 'none'}"
        memory.failed_attempts[key] = memory.failed_attempts.get(key, 0) + 1
        failed_color = decision.target_color or memory.active_color
        if action == "pick_cube" and color_failed_attempts(memory, failed_color, action="pick_cube") >= 2:
            if failed_color and failed_color not in memory.skipped_colors:
                memory.skipped_colors.append(failed_color)
            if memory.active_color == failed_color:
                memory.active_color = None
            memory.stage = "need_cube"

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
            "stage": memory.stage,
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })


# ---------------------------------------------------------------------------
# LEVEL 1: coordinate-guided action + visual servoing
# ---------------------------------------------------------------------------
def estimate_target_xy_from_observation(
    observation: Observation,
    target_color: str | None,
    *,
    target_kind: str = "cube",
    memory: AgentMemory | None = None,
) -> tuple[float, float] | None:
    candidates = []
    for detection in observation.detections:
        if detection.color != target_color:
            continue
        if target_kind == "pad":
            if not is_pad_detection(detection):
                continue
            xy = estimate_xy_for_detection(observation.robot_status, detection, target_kind="pad")
            if xy and memory and xy_is_near_known_source(memory, xy):
                continue
        else:
            if is_pad_detection(detection):
                continue  # 패드 표지판을 큐브로 착각하지 않도록
            if memory and color_is_blocked_as_cube(memory, detection.color):
                continue
            xy = estimate_xy_for_detection(observation.robot_status, detection, target_kind="cube")
        if xy is None:
            continue
        candidates.append((detection, xy))

    if not candidates:
        return None

    best_det, best_xy = max(candidates, key=lambda c: detection_score(c[0]))
    return best_xy


async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    return await ctx.invoke(
        "go_to",
        {"target": {"kind": "pose", "pose": {"frame_id": "world", "position": [x, y, 0]}}},
        timeout_s=300,
    )


async def _visual_servo_to_pad(
    ctx: Any,
    target_color: str,
    memory: AgentMemory,
    *,
    max_steps: int = 15,
    center_threshold_deg: float = 12.0,
    arrival_area: int = 60000,
    arrival_steps: int = 2,
) -> dict[str, Any]:
    """패드 blob을 직접 보면서 visual servoing으로 접근합니다.

    패드 표지판 blob이 화면 중앙에 오도록 회전하고,
    중앙에 왔으면 전진합니다.
    blob_area가 arrival_area 이상이면 도달 판정.

    [수정 1] 실제 로그에서 관측된 area 범위(2만~13만)를 보면 8000은 너무 낮아서
    실제로는 멀리 있는데도 도달로 오판했음. 60000으로 상향.
    [수정 2] angle이 여러 step 동안 거의 안 바뀌는 경우(컨베이어에 막혀
    회전이 안 되는 상황)를 감지해 후진+옆이동(strafe)으로 탈출.
    """
    centered_count = 0
    best_area_seen = 0
    prev_angle: float | None = None
    stuck_count = 0

    for step in range(max_steps):
        try:
            # 패드는 위쪽에 있으므로 pitch를 올려서 봄
            dets = await asyncio.wait_for(
                perceive_scanned_view(ctx, yaw=0.0, pitch=0.05),
                timeout=8.0,
            )
        except Exception as e:
            print(f"  [servo] perceive 실패 step {step}: {e}")
            await move_velocity(ctx, vx=0.2, duration_s=0.5)
            continue

        # 목표 색상의 패드 blob 찾기
        pad_blobs = [
            d for d in dets
            if d.color == target_color and is_pad_detection(d)
        ]

        if not pad_blobs:
            # 패드가 안 보이면 조금 전진하며 탐색
            print(f"  [servo] step {step}: {target_color} 패드 blob 미감지 → 전진")
            await move_velocity(ctx, vx=0.25, duration_s=0.5)
            centered_count = 0
            prev_angle = None
            stuck_count = 0
            continue

        # 가장 큰 blob 선택
        best = max(pad_blobs, key=lambda d: d.blob_area)
        angle = best.angle_deg
        area = best.blob_area
        best_area_seen = max(best_area_seen, area)
        print(f"  [servo] step {step}: angle={angle:+.1f}°, area={area} (최대 관측={best_area_seen})")

        # 도달 판정 (상향된 임계값 + 연속 확인)
        if area >= arrival_area and abs(angle) <= center_threshold_deg:
            centered_count += 1
            if centered_count >= arrival_steps:
                print(f"  [servo] 패드 도달! area={area}")
                return {"status": "arrived", "steps": step, "area": area}
            await move_velocity(ctx, vx=0.15, duration_s=0.4)
            prev_angle = angle
            stuck_count = 0
            continue

        # [핵심 수정] 회전으로 angle이 거의 안 바뀌면 고착 상태로 판단
        # (컨베이어 벨트 등 장애물에 막혀 몸 회전이 안 되는 경우)
        if abs(angle) > center_threshold_deg:
            if prev_angle is not None and abs(angle - prev_angle) < 2.0:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= 2:
                # 후진 + 옆으로 이동(strafe) + 회전을 동시에 시도해 장애물 회피
                strafe = 0.35 if angle > 0 else -0.35
                wz = -0.5 if angle > 0 else 0.5
                print(f"  [servo] step {step}: 회전 고착 감지 → 후진+strafe 탈출 시도")
                await move_velocity(ctx, vx=-0.3, vy=strafe, wz=wz, duration_s=0.6)
                stuck_count = 0
            else:
                # 방향 조정: angle 양수=오른쪽 → wz 음수(오른쪽 회전)
                wz = -0.4 if angle > 0 else 0.4
                await move_velocity(ctx, wz=wz, duration_s=0.4)
            centered_count = 0
            prev_angle = angle
        else:
            # 중앙이지만 아직 멀다 → 전진
            centered_count = 0
            prev_angle = angle
            stuck_count = 0
            await move_velocity(ctx, vx=0.3, duration_s=0.5)

    # max_steps 도달 시, 최대로 가까웠던 적이 있으면 그래도 place 시도해볼 가치 있음
    reached_close_enough = best_area_seen >= arrival_area * 0.7
    return {"status": "max_steps", "steps": max_steps, "best_area_seen": best_area_seen,
            "reached_close_enough": reached_close_enough}


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    decision = enforce_stage_decision(decision, memory)

    # ── 큐브 탐색 ──────────────────────────────────────────────────────────
    if decision.next_action == "search_cube":
        detections = await scan_head(ctx, yaws=(-0.8, 0.0, 0.8), pitch=0.18)
        if not detections:
            await move_velocity(ctx, wz=0.4, duration_s=0.5)
        # source 위치 기억
        for d in detections:
            if color_is_blocked_as_cube(memory, d.color) or is_pad_detection(d):
                continue
            xy = estimate_xy_for_detection(observation.robot_status, d, target_kind="cube")
            if xy:
                prev = memory.known_source_location
                if prev is None or d.blob_area >= int(prev.get("blob_area", 0)):
                    memory.known_source_location = {"xy": xy, "blob_area": d.blob_area,
                                                     "source": "visual_cube"}
        return {"action": "search_cube", "status": "scanned", "visible_count": len(detections)}

    # ── 패드 탐색: 고개 스캔 먼저 → 안 보이면 몸 회전 ────────────────────
    if decision.next_action == "search_pad":
        target_color = decision.target_color or memory.held_color
        if not target_color:
            return {"action": "search_pad", "status": "failed", "reason": "target_color 없음"}

        print(f"  [search_pad] {target_color} 패드 blob 탐색 시작")

        # 1단계: 고개 돌려서 패드 blob 탐색 (pitch 위로)
        found_detection = None
        for yaw in (-1.2, -0.6, 0.0, 0.6, 1.2):
            try:
                dets = await asyncio.wait_for(
                    perceive_scanned_view(ctx, yaw=yaw, pitch=0.05),
                    timeout=8.0,
                )
                pad_blobs = [d for d in dets if d.color == target_color and is_pad_detection(d)]
                if pad_blobs:
                    found_detection = max(pad_blobs, key=lambda d: d.blob_area)
                    print(f"  [search_pad] 패드 blob 발견! yaw={yaw:+.1f}°, area={found_detection.blob_area}")
                    break
            except Exception as e:
                print(f"  [search_pad] 스캔 실패 yaw={yaw:+.1f}: {e}")

        # 2단계: 고개로 못 찾으면 몸 회전 (30도씩)
        if found_detection is None:
            print("  [search_pad] 고개 스캔 실패 → 몸 회전")
            for _ in range(2):
                # 두 번 나눠 호출하지 않고 한 번에 60도 회전
                await move_velocity(ctx, wz=0.6, duration_s=0.5)
                try:
                    dets = await asyncio.wait_for(
                        perceive_scanned_view(ctx, yaw=0.0, pitch=0.05),
                        timeout=8.0,
                    )
                    pad_blobs = [d for d in dets if d.color == target_color and is_pad_detection(d)]
                    if pad_blobs:
                        found_detection = max(pad_blobs, key=lambda d: d.blob_area)
                        print(f"  [search_pad] 회전 후 패드 발견! area={found_detection.blob_area}")
                        break
                except Exception as e:
                    print(f"  [search_pad] 회전 후 스캔 실패: {e}")

        # 패드 발견 시 좌표 저장
        if found_detection is not None:
            xy = estimate_xy_for_detection(observation.robot_status, found_detection, target_kind="pad")
            if xy and not xy_is_near_known_source(memory, xy):
                memory.known_pad_locations[target_color] = {
                    "xy": xy,
                    "blob_area": found_detection.blob_area,
                    "source": "visual_pad_blob",
                }
                print(f"  [search_pad] pad 좌표 저장: {xy}")
        else:
            # [수정] 실패 반복 시 컨베이어에서 멀어지도록 후퇴
            # search_turns가 쌓일수록(3회 이상) 전진 대신 후퇴 + 좌우 회전으로
            # 컨베이어를 벗어나 넓은 시야를 확보
            if memory.search_turns >= 3:
                print(f"  [search_pad] {memory.search_turns}회 연속 실패 → 컨베이어에서 후퇴")
                # 후퇴 방향은 컨베이어 반대쪽(왼쪽, source에서 먼 쪽)으로
                await move_velocity(ctx, vx=-0.35, wz=0.3, duration_s=0.5)
            else:
                print("  [search_pad] 패드 미발견 → 전진 후 재시도")
                await move_velocity(ctx, vx=0.4, duration_s=0.5)

        return {
            "action": "search_pad",
            "status": "found" if found_detection else "not_found",
            "target_color": target_color,
            "known_pad_locations": memory.known_pad_locations,
        }

    # ── 큐브로 이동 ────────────────────────────────────────────────────────
    if decision.next_action == "navigate_to_cube":
        target_xy = estimate_target_xy_from_observation(
            observation, decision.target_color, target_kind="cube", memory=memory,
        )
        used_source_cache = False
        if target_xy is None and memory.known_source_location is not None:
            target_xy = tuple(memory.known_source_location["xy"])
            used_source_cache = True
        if target_xy is None:
            return {"action": "navigate_to_cube", "status": "failed", "reason": "coordinate estimate 없음"}
        result = await go_to_xy(ctx, *target_xy)
        return {
            "action": "navigate_to_cube",
            "status": "moved",
            "target_color": decision.target_color,
            "target_xy": target_xy,
            "used_source_cache": used_source_cache,
            "result": result_summary(result),
        }

    # ── 패드로 이동: Visual Servoing ───────────────────────────────────────
    if decision.next_action == "navigate_to_pad":
        target_color = decision.target_color or memory.held_color
        if not target_color:
            return {"action": "navigate_to_pad", "status": "failed", "reason": "target_color 없음"}

        print(f"  [navigate_to_pad] {target_color} 패드로 visual servoing 시작")

        # 캐시에 좌표 있으면 먼저 go_to로 대략 이동
        cached = memory.known_pad_locations.get(target_color)
        if cached and "xy" in cached:
            print(f"  [navigate_to_pad] 캐시 좌표로 go_to: {cached['xy']}")
            try:
                await go_to_xy(ctx, *cached["xy"])
            except Exception as e:
                print(f"  [navigate_to_pad] go_to 실패: {e}")

        # Visual servoing으로 정밀 접근
        servo_result = await _visual_servo_to_pad(ctx, target_color, memory)
        print(f"  [navigate_to_pad] servo 결과: {servo_result}")

        # [핵심 수정] 실제로 도달(arrived)했을 때만 status=moved 반환.
        # 기존엔 max_steps로 끝나도 무조건 moved를 반환해서
        # near_pad로 넘어가 place_cube를 시도해버렸음 → 엉뚱한 곳에 place.
        arrived = servo_result.get("status") == "arrived"
        return {
            "action": "navigate_to_pad",
            "status": "moved" if arrived else "failed",
            "target_color": target_color,
            "servo_result": servo_result,
            "reason": None if arrived else "패드 근처까지 도달하지 못함 (max_steps)",
        }

    # ── 큐브 집기 ──────────────────────────────────────────────────────────
    if decision.next_action == "pick_cube":
        target_color = decision.target_color or memory.active_color
        robot_status = await get_robot_status(ctx)
        detections = await scan_head(ctx, yaws=(-0.35, 0.0, 0.35), pitch=0.18)

        ready_detection = None
        for d in detections:
            if d.color == target_color and not is_pad_detection(d) and d.blob_area >= PICK_READY_MIN_AREA:
                if ready_detection is None or d.blob_area > ready_detection.blob_area:
                    ready_detection = d

        if ready_detection is None:
            # 가까운 큐브 blob 찾아서 접근
            approach_det = None
            for d in detections:
                if d.color == target_color and not is_pad_detection(d) and d.blob_area >= CUBE_DETECTION_MIN_AREA:
                    if approach_det is None or d.blob_area > approach_det.blob_area:
                        approach_det = d

            if approach_det is None:
                return {"action": "pick_cube", "status": "failed",
                        "reason": "pick 전 재확인에서 cube 없음.", "visible_count": len(detections)}

            target_xy = estimate_xy_for_detection(robot_status, approach_det, target_kind="cube")
            if target_xy is None:
                return {"action": "pick_cube", "status": "failed", "reason": "좌표 추정 실패."}

            result = await go_to_xy(ctx, *target_xy)
            return {
                "action": "navigate_to_cube",
                "status": "moved",
                "target_color": target_color,
                "target_xy": target_xy,
                "pre_pick_reposition": True,
                "result": result_summary(result),
            }

        result = await pick_nearest_cube(ctx)
        return {
            "action": "pick_cube",
            "status": "attempted",
            "target_color": target_color,
            "ready_blob_area": ready_detection.blob_area,
            "result": result_summary(result),
        }

    # ── 큐브 내려놓기 ──────────────────────────────────────────────────────
    if decision.next_action == "place_cube":
        target_color = memory.held_color or decision.target_color
        result = await place_nearest_zone(ctx)
        return {
            "action": "place_cube",
            "status": "attempted",
            "target_color": target_color,
            "destination_sign": DESTINATION_SIGN_RULES.get(target_color or ""),
            "result": result_summary(result),
        }

    # ── 복구 ───────────────────────────────────────────────────────────────
    if decision.next_action == "recover":
        await move_velocity(ctx, vx=-0.1, wz=0.4, duration_s=0.5)
        return {"action": "recover", "status": "done"}

    # ── 건너뛰기 ───────────────────────────────────────────────────────────
    if decision.next_action == "skip_target":
        skipped = memory.active_color or memory.held_color or decision.target_color
        if skipped and skipped not in memory.skipped_colors:
            memory.skipped_colors.append(skipped)
        memory.active_color = None
        if not memory.held_color:
            memory.stage = "need_cube"
        return {"action": "skip_target", "status": "skipped", "target_color": skipped}

    return {"action": decision.next_action, "status": "no_op"}


async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 1] Cycle {cycle}")
        try:
            if tracker is not None:
                first_cycle = tracker.started_at is None
                tracker.start_first_cycle()
                if first_cycle:
                    tracker.print_start()
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached before cycle action: {reason}.")
                    break

            observation = await observe_world(ctx, memory)
            decision = await decide_next_action(TASK, observation, memory, last_result)
            print("LLM 판단:", format_decision_ko(decision))

            if decision.next_action == "stop":
                break

            action_result = await execute_decision(ctx, decision, observation, memory)
            action_result.setdefault("previous_delivered_count", memory.delivered_count)
            verified = await verify_outcome(ctx, decision, action_result)
            update_memory(memory, observation, decision, verified)
            last_result = verified
            if tracker is not None:
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached after cycle action: {reason}.")
                    break
        except Exception as exc:
            # [핵심 수정] 이 cycle 안에서 예상 못한 예외(예: SDK가 이상한 형태의
            # 응답을 줌)가 나면 예전엔 run_agent 전체가 그대로 죽어서 남은
            # 시간/배달 기회를 전부 날렸음. 한 cycle 실패로 전체 라운드를
            # 끝내지 않고, 이 cycle만 건너뛰고 다음 cycle에서 다시 시도합니다.
            print(f"Cycle {cycle} 오류(건너뜀): {exc}")
            await asyncio.sleep(1.0)

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 1 adaptive-navigation project starter 실행")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=1, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)