"""掉落物磁吸与拾取回滚。

本模块实现掉落物被磁吸、异步提交拾取请求、以及失败回滚时的六条不变量：

1. :func:`swept_attract` 用扫掠判定（相对位移线段 vs. 磁吸圆）代替逐帧点判定，
   避免玩家单帧位移超过磁吸半径、或掉落物在两帧之间来回穿越边界时漏判。
2. :func:`object_generation` / :func:`submit_pickup` 让异步请求携带对象代际
   （generation），提交与结算时都做代际校验，旧请求不会作用到同 id 的新对象上。
3. :func:`capacity_first` / :func:`grant` 先校验背包容量，容量不足时绝不扣除物体。
4. :func:`stable_slot_order` 按稳定 ID 排序，多个掉落物同帧入包时槽位与字典遍历
   顺序无关，重放结果一致。
5. :func:`stop_sfx_on_destroy` 在对象销毁的同一时刻停止磁吸音效，不播放尾音。
6. :func:`rollback_restore` 在失败回滚时恢复位置与可见性，且绝不会把已销毁对象
   重新激活。

模块内的函数都接受「结构化输入」（映射／对象）并返回字典或列表；传入无法识别的
值（例如标量）时退化为空操作结果，保证契约函数始终返回非 ``None``。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Optional

__all__ = [
    "swept_attract",
    "object_generation",
    "capacity_first",
    "stable_slot_order",
    "stop_sfx_on_destroy",
    "rollback_restore",
    "attract_step",
    "submit_pickup",
    "grant",
    "reject",
    "validate_capacity",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]


# ---------------------------------------------------------------------------
# 通用读取工具：同时兼容 dict 与普通对象，且对标量输入保持安全。
# ---------------------------------------------------------------------------

def _record(value: Any) -> Mapping:
    """把输入解释为只读记录；无法解释时返回空映射。"""
    if isinstance(value, Mapping):
        return value
    if value is None or isinstance(value, (int, float, bool, str, bytes)):
        return {}
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return data
    return {}


def _get(record: Any, *names: str, default: Any = None) -> Any:
    data = _record(record)
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _set(record: Any, name: str, value: Any) -> bool:
    """尽力写回字段；不可写时静默失败，由调用方决定是否降级。"""
    if isinstance(record, MutableMapping):
        record[name] = value
        return True
    try:
        setattr(record, name, value)
    except (AttributeError, TypeError):
        return False
    return True


def _unpack(value: Any, keys: Sequence[str]) -> Any:
    """单个位置参数若是参数字典（含 ``keys`` 中的键），返回该字典。"""
    if isinstance(value, Mapping) and any(key in value for key in keys):
        return value
    return None


def _sequence(value: Any, *names: str) -> list:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for name in names:
            inner = value.get(name)
            if isinstance(inner, (list, tuple)):
                return list(inner)
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _point(value: Any) -> Optional[tuple]:
    """把 (x, y) / [x, y] / {"x": .., "y": ..} 归一为二维点。"""
    if isinstance(value, Mapping):
        x = _number(value.get("x"))
        y = _number(value.get("y"))
        if x is not None and y is not None:
            return (x, y)
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _number(value[0])
        y = _number(value[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def _natural_key(value: Any) -> tuple:
    """稳定 ID 的自然排序键：``item2`` < ``item10``，None 排最前。"""
    if value is None:
        return ((0, ""),)
    text = str(value)
    key = []
    for part in re.split(r"(\d+)", text):
        if part.isdigit():
            key.append((1, int(part)))
        elif part:
            key.append((0, part))
    return tuple(key) or ((0, ""),)


def _signature(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. 扫掠吸附
# ---------------------------------------------------------------------------


def _segment_entry(d0: tuple, v: tuple, radius: float) -> Optional[float]:
    """相对位移线段 ``d0 + t*v, t ∈ [0, 1]`` 首次进入半径圆的时间参数。

    返回 ``None`` 表示整段都在圆外；返回 ``0.0`` 表示帧首已在圆内。
    两端点都在圆外、但线段中途穿过圆的情形会返回进入时刻，这正是逐帧点判定
    会漏掉的「穿越」。
    """
    dx, dy = d0
    vx, vy = v
    a = vx * vx + vy * vy
    c = dx * dx + dy * dy - radius * radius
    if c <= 0.0:  # 帧首已经在磁吸半径内
        return 0.0
    if a == 0.0:  # 相对静止且在圆外
        return None
    b = 2.0 * (dx * vx + dy * vy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t_enter = (-b - root) / (2.0 * a)
    if 0.0 <= t_enter <= 1.0:
        return t_enter
    return None


def swept_attract(drop: Any = None, player: Any = None, radius: Any = None, **kwargs: Any) -> dict:
    """对单个掉落物做扫掠吸附判定。

    判定使用「掉落物相对玩家的位移线段」与磁吸圆的相交关系，因此：

    * 玩家单帧位移超过半径时，只要相对轨迹穿过圆就能命中；
    * 掉落物在两帧之间从圆外穿到圆外另一侧时也不会漏判。
    """
    bundle = _unpack(drop, ("drop", "player"))
    if bundle is not None:
        player = bundle.get("player", player)
        radius = bundle.get("radius", radius)
        drop = bundle.get("drop")

    data = _record(drop)
    player_data = _record(player)

    drop_prev = _point(
        _get(data, "prev_pos", "previous_position", "old_pos", "last_pos", default=_get(data, "pos", "position"))
    )
    drop_cur = _point(_get(data, "pos", "position", "current_pos"))
    player_prev = _point(
        _get(player_data, "prev_pos", "previous_position", "old_pos", default=_get(player_data, "pos", "position"))
    )
    player_cur = _point(_get(player_data, "pos", "position", "current_pos"))

    resolved_radius = _number(radius)
    if resolved_radius is None:
        resolved_radius = _number(_get(data, "radius", "magnet_radius"))
    if resolved_radius is None:
        resolved_radius = _number(_get(player_data, "radius", "magnet_radius"))
    if resolved_radius is None:
        resolved_radius = _number(kwargs.get("radius"))
    if resolved_radius is None:
        resolved_radius = 0.0
    resolved_radius = max(0.0, resolved_radius)

    object_id = _get(data, "object_id", "id")
    generation = _integer(_get(data, "generation", default=0), 0)
    base = {
        "object_id": object_id,
        "generation": generation,
        "radius": resolved_radius,
        "swept": True,
    }

    if drop_cur is None:
        return {**base, "hit": False, "attracted": False, "t": None, "point": None, "reason": "no_drop"}
    if player_cur is None:
        player_cur = (0.0, 0.0)
    if drop_prev is None:
        drop_prev = drop_cur
    if player_prev is None:
        player_prev = player_cur

    d0 = (drop_prev[0] - player_prev[0], drop_prev[1] - player_prev[1])
    d1 = (drop_cur[0] - player_cur[0], drop_cur[1] - player_cur[1])
    v = (d1[0] - d0[0], d1[1] - d0[1])

    t = _segment_entry(d0, v, resolved_radius)
    start_distance = math.hypot(*d0)
    end_distance = math.hypot(*d1)

    if t is None:
        return {
            **base,
            "hit": False,
            "attracted": False,
            "t": None,
            "point": None,
            "start_distance": start_distance,
            "end_distance": end_distance,
            "reason": "out_of_range",
        }

    contact = (drop_prev[0] + (drop_cur[0] - drop_prev[0]) * t, drop_prev[1] + (drop_cur[1] - drop_prev[1]) * t)
    return {
        **base,
        "hit": True,
        "attracted": True,
        "t": t,
        "point": contact,
        "start_distance": start_distance,
        "end_distance": end_distance,
        "crossed": start_distance > resolved_radius and end_distance > resolved_radius,
        "reason": "inside_radius" if t == 0.0 else "swept_crossing",
    }


# ---------------------------------------------------------------------------
# 2. 对象代际
# ---------------------------------------------------------------------------


def object_generation(obj: Any = None, request: Any = None, **kwargs: Any) -> dict:
    """读取对象代际，并在提供请求时校验 id / 代际 / 销毁状态。

    代际在对象被销毁、回池或复用时递增；异步请求提交时记录当时的代际，
    返回时若对象已被另一路径销毁并复用，代际不等即判定为过期请求。
    """
    bundle = _unpack(obj, ("object", "request"))
    if bundle is not None:
        request = bundle.get("request", request)
        obj = bundle.get("object", bundle.get("drop", obj))
    if request is None:
        request = kwargs.get("request")

    data = _record(obj)
    object_id = _get(data, "object_id", "id")
    generation = _integer(_get(data, "generation", default=0), 0)
    destroyed = bool(_get(data, "destroyed", "dead", default=False))

    req = _record(request)
    req_id = _get(req, "object_id", "id")
    req_generation = _integer(_get(req, "generation"))

    valid = True
    reason = "no_request"
    if req:
        if req_id is not None and object_id is not None and str(req_id) != str(object_id):
            valid, reason = False, "id_mismatch"
        elif req_generation is not None and req_generation != generation:
            valid, reason = False, "stale_generation"
        elif destroyed:
            valid, reason = False, "destroyed"
        else:
            reason = "ok"

    return {
        "object_id": object_id,
        "generation": generation,
        "destroyed": destroyed,
        "request_generation": req_generation,
        "valid": valid,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 3. 容量前置
# ---------------------------------------------------------------------------


def _capacity_of(inventory: Any) -> Optional[int]:
    if isinstance(inventory, bool):
        return None
    if isinstance(inventory, int):
        return inventory
    data = _record(inventory)
    return _integer(_get(data, "capacity", "max_slots", "slot_count", "size"))


def _items_of(inventory: Any) -> Any:
    data = _record(inventory)
    items = _get(data, "items", "contents", "inventory")
    if items is None:
        items = _get(data, "slots")
    return items


def _used_slots(inventory: Any) -> int:
    items = _items_of(inventory)
    if isinstance(items, Mapping):
        return sum(1 for value in items.values() if value is not None)
    if isinstance(items, (list, tuple)):
        return sum(1 for value in items if value is not None)
    return 0


def validate_capacity(inventory: Any = None, item: Any = None, count: Any = 1, **kwargs: Any) -> dict:
    """只读地校验背包能否再容纳 ``count`` 个物品，不做任何扣除。"""
    bundle = _unpack(inventory, ("inventory", "item"))
    if bundle is not None:
        item = bundle.get("item", item)
        count = bundle.get("count", count)
        inventory = bundle.get("inventory")

    needed = _integer(count, 1) or 0
    if needed <= 1 and isinstance(item, (list, tuple, set)):
        needed = len(item)

    capacity = _capacity_of(inventory)
    used = _used_slots(inventory)

    if capacity is None:
        return {
            "ok": True,
            "capacity": None,
            "used": used,
            "free": None,
            "needed": needed,
            "reason": "unbounded",
        }

    free = capacity - used
    ok = free >= needed
    return {
        "ok": ok,
        "capacity": capacity,
        "used": used,
        "free": free,
        "needed": needed,
        "reason": "ok" if ok else "capacity_exceeded",
    }


def capacity_first(inventory: Any = None, item: Any = None, count: Any = 1, **kwargs: Any) -> dict:
    """容量前置校验：结果与 :func:`validate_capacity` 一致，并显式标注「未扣除」。"""
    result = validate_capacity(inventory, item, count, **kwargs)
    return {
        **result,
        "capacity_checked_before_deduct": True,
        "deducted": False,
    }


# ---------------------------------------------------------------------------
# 4. 稳定槽位排序
# ---------------------------------------------------------------------------


class SlotOrder(list):
    """按稳定 ID 排序后的物品列表，附带 id 顺序与槽位计划。"""

    def __init__(self, items: list, entries: list) -> None:
        super().__init__(items)
        self.entries = entries
        self.ids = [entry["object_id"] for entry in entries]
        self.stable = True


def _ordering(sequence: Sequence) -> list:
    entries = []
    for index, item in enumerate(sequence):
        data = _record(item)
        if not data and isinstance(item, (str, int, float)):
            data = {"id": item}
        object_id = _get(data, "object_id", "id")
        generation = _integer(_get(data, "generation", default=0), 0)
        entries.append(
            {
                "object_id": object_id,
                "generation": generation,
                "source_index": index,
                "item": item,
            }
        )
    entries.sort(
        key=lambda entry: (
            _natural_key(entry["object_id"]),
            entry["generation"],
            entry["source_index"],
        )
    )
    for slot, entry in enumerate(entries):
        entry["slot"] = slot
    return entries


def stable_slot_order(items: Any = None, key: Any = None, **kwargs: Any) -> SlotOrder:
    """按稳定 ID 排序，使入包顺序与输入遍历顺序无关。

    同帧多个掉落物被吸附时，调用方应使用本函数给出的顺序分配槽位；这样
    重放同一批事件时物品槽位保持一致。
    """
    bundle = _unpack(items, ("items", "drops", "objects"))
    if bundle is not None:
        items = bundle.get("items", bundle.get("drops", bundle.get("objects")))
    if items is None:
        items = kwargs.get("items")

    sequence = _sequence(items, "items", "drops", "objects")
    if key is not None:
        # 允许自定义稳定键：先按该键归一化，再交给默认排序。
        remapped = []
        for item in sequence:
            value = _get(_record(item), str(key))
            if value is None:
                value = item
            remapped.append({"id": value, "item": item})
        entries = _ordering(remapped)
        ordered_items = [entry["item"]["item"] for entry in entries]
        for entry in entries:
            entry["object_id"] = entry["item"]["id"]
            entry.pop("item", None)
        return SlotOrder(ordered_items, entries)

    entries = _ordering(sequence)
    ordered_items = [entry["item"] for entry in entries]
    for entry in entries:
        entry.pop("item", None)
    return SlotOrder(ordered_items, entries)


# ---------------------------------------------------------------------------
# 5. 销毁即停音效
# ---------------------------------------------------------------------------


def stop_sfx_on_destroy(obj: Any = None, audio: Any = None, **kwargs: Any) -> dict:
    """对象已销毁时立即停止其磁吸音效，不留尾音。幂等。"""
    bundle = _unpack(obj, ("object", "audio", "drop"))
    if bundle is not None:
        audio = bundle.get("audio", audio)
        obj = bundle.get("object", bundle.get("drop", obj))
    if audio is None:
        audio = kwargs.get("audio")

    data = _record(obj)
    destroyed = bool(_get(data, "destroyed", "dead", default=False))
    sfx = _get(data, "sfx", "sfx_id", "sound", "sound_id")
    was_playing = bool(_get(data, "sfx_playing", "sound_playing", default=False))

    stopped = destroyed and (was_playing or sfx is not None)

    if destroyed:
        # 只停播放，保留 sfx id 便于审计。
        _set(obj, "sfx_playing", False)
        _set(obj, "sound_playing", False)

    if isinstance(audio, MutableMapping):
        stopped_list = audio.setdefault("stopped", [])
        if destroyed and sfx is not None and sfx not in stopped_list:
            stopped_list.append(sfx)
        audio["playing"] = False if destroyed else audio.get("playing", was_playing)

    return {
        "object_id": _get(data, "object_id", "id"),
        "sfx": sfx,
        "destroyed": destroyed,
        "was_playing": was_playing,
        "stopped": stopped,
        "tail_cut": stopped,
        "playing": False if destroyed else was_playing,
    }


# ---------------------------------------------------------------------------
# 6. 回滚恢复
# ---------------------------------------------------------------------------


def _snapshot_position(snapshot: Any) -> Optional[tuple]:
    return _point(
        _get(
            snapshot,
            "position",
            "pos",
            "prev_pos",
            "previous_position",
            "original_position",
            "saved_position",
        )
    )


def _snapshot_visible(snapshot: Any) -> Optional[bool]:
    value = _get(snapshot, "visible", "original_visible", "saved_visible")
    if value is None:
        return None
    return bool(value)


def rollback_restore(obj: Any = None, snapshot: Any = None, **kwargs: Any) -> dict:
    """回滚一次失败的拾取：恢复位置与可见性，但不复活已销毁对象。

    这是「回滚把已销毁对象重新激活」这一缺陷的修复点：销毁是单向来路，
    回滚只负责把存活对象放回原位并重新显示，不会重置 ``destroyed``。
    """
    bundle = _unpack(obj, ("object", "snapshot", "drop"))
    if bundle is not None:
        snapshot = bundle.get("snapshot", snapshot)
        obj = bundle.get("object", bundle.get("drop", obj))
    if snapshot is None:
        snapshot = kwargs.get("snapshot")

    data = _record(obj)
    destroyed = bool(_get(data, "destroyed", "dead", default=False))
    current_position = _point(_get(data, "pos", "position"))
    current_visible = bool(_get(data, "visible", default=True))

    base = {
        "object_id": _get(data, "object_id", "id"),
        "destroyed": destroyed,
        "position": current_position,
        "visible": current_visible,
    }

    if destroyed:
        return {
            **base,
            "restored": False,
            "reactivated": False,
            "reason": "destroyed_do_not_reactivate",
        }

    target_position = _snapshot_position(snapshot)
    target_visible = _snapshot_visible(snapshot)
    restored_position = False
    restored_visible = False

    if target_position is not None:
        restored_position = _set(obj, "pos", target_position)
        _set(obj, "position", target_position)
        _set(obj, "prev_pos", target_position)
    if target_visible is not None:
        restored_visible = _set(obj, "visible", target_visible)
        current_visible = target_visible

    restored = restored_position or restored_visible
    return {
        **base,
        "position": target_position if target_position is not None else current_position,
        "visible": current_visible,
        "restored": restored,
        "restored_position": restored_position,
        "restored_visible": restored_visible,
        "reactivated": False,
        "reason": "restored" if restored else "no_snapshot",
    }


# ---------------------------------------------------------------------------
# 7. 单帧吸附
# ---------------------------------------------------------------------------


def attract_step(drops: Any = None, player: Any = None, radius: Any = None, **kwargs: Any) -> dict:
    """推进一帧磁吸：扫掠判定 + 逐帧拉向玩家 + 按稳定 ID 去重排序。

    命中判定使用扫掠线段，因此玩家或掉落物单帧位移超过磁吸半径也不会漏判。
    命中后掉落物沿「掉落物 → 玩家」方向移动最多 ``pull_speed`` 的距离
    （默认取磁吸半径，即一帧内可吸到玩家；显式给出较小速度时为逐帧渐进牵引），
    并保留移动前的位置作为下一帧的 ``prev_pos``。

    返回的 ``attracted`` 顺序与输入遍历顺序无关，可直接用于同帧入包。
    """
    bundle = _unpack(drops, ("drops", "objects", "player"))
    if bundle is not None:
        player = bundle.get("player", player)
        radius = bundle.get("radius", radius)
        drops = bundle.get("drops", bundle.get("objects"))
    if drops is None:
        drops = kwargs.get("drops")

    move = bool(kwargs.get("move", True))
    sequence = _sequence(drops, "drops", "objects", "items")
    player_data = _record(player)
    player_pos = _point(_get(player_data, "pos", "position")) or (0.0, 0.0)

    best: dict = {}
    for index, drop in enumerate(sequence):
        result = swept_attract(drop, player, radius)
        if not result.get("hit"):
            continue
        object_id = result.get("object_id")
        key = str(object_id) if object_id is not None else f"__index_{index}"
        candidate = {
            "object_id": object_id,
            "generation": result.get("generation"),
            "t": result.get("t"),
            "point": result.get("point"),
            "radius": result["radius"],
            "source_index": index,
            "crossed": result.get("crossed", False),
            "drop": drop,
        }
        previous = best.get(key)
        if previous is None or (candidate["t"], candidate["source_index"]) < (
            previous["t"],
            previous["source_index"],
        ):
            best[key] = candidate

    ordered = sorted(
        best.values(),
        key=lambda hit: (_natural_key(hit["object_id"]), hit["t"], hit["source_index"]),
    )

    updates = []
    if move:
        for hit in ordered:
            drop = hit["drop"]
            data = _record(drop)
            start = _point(_get(data, "pos", "position"))
            if start is None:
                continue

            speed = _number(kwargs.get("pull_speed", kwargs.get("speed")))
            if speed is None:
                speed = _number(_get(data, "pull_speed", "speed"))
            if speed is None:
                speed = hit["radius"]
            speed = max(0.0, speed or 0.0)

            dx = player_pos[0] - start[0]
            dy = player_pos[1] - start[1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-12 or speed <= 0.0:
                target = start
            else:
                ratio = min(1.0, speed / distance)
                target = (start[0] + dx * ratio, start[1] + dy * ratio)

            if _set(drop, "prev_pos", start):
                _set(drop, "pos", target)
                _set(drop, "position", target)
            _set(drop, "attracted", True)
            updates.append({"object_id": hit["object_id"], "from": start, "to": target})

    public = [
        {
            "object_id": hit["object_id"],
            "generation": hit["generation"],
            "t": hit["t"],
            "point": hit["point"],
            "crossed": hit["crossed"],
        }
        for hit in ordered
    ]

    return {
        "attracted": public,
        "ids": [hit["object_id"] for hit in ordered],
        "updates": updates,
        "count": len(public),
        "swept": True,
        "order_stable": True,
    }


# ---------------------------------------------------------------------------
# 8. 异步拾取：提交 / 结算 / 拒绝
# ---------------------------------------------------------------------------


def _request_snapshot(req: Mapping, drop: Any) -> dict:
    snapshot = _get(req, "snapshot")
    if snapshot is None:
        data = _record(drop)
        snapshot = {
            "position": _point(_get(data, "pos", "position")),
            "visible": bool(_get(data, "visible", default=True)),
        }
    return snapshot


def submit_pickup(request: Any = None, inventory: Any = None, drop: Any = None, **kwargs: Any) -> dict:
    """提交一次异步拾取请求，并固定提交时刻的对象代际与视觉快照。

    请求在真正结算前不会改动掉落物；容量在这里只做前置校验（只读），
    以便尽早拒绝。
    """
    bundle = _unpack(request, ("request", "inventory", "drop"))
    if bundle is not None:
        inventory = bundle.get("inventory", inventory)
        drop = bundle.get("drop", drop)
        request = bundle.get("request", bundle)

    req = _record(request)
    payload = _record(_get(req, "payload", default=req))
    data = _record(drop)

    object_id = _get(payload, "object_id", "id")
    if object_id is None:
        object_id = _get(data, "object_id", "id")
    generation = _integer(_get(payload, "generation"))
    if generation is None:
        generation = _integer(_get(data, "generation", default=0), 0)
    request_id = _get(payload, "request_id")
    if request_id is None:
        request_id = f"pickup:{object_id}:{generation}"

    snapshot = _request_snapshot(payload, drop)
    item = _get(payload, "item", default=_get(data, "item"))
    capacity = capacity_first(inventory, item=item)

    status = "pending" if capacity["ok"] else "rejected"
    reason = None if capacity["ok"] else capacity["reason"]

    return {
        "request_id": request_id,
        "object_id": object_id,
        "generation": generation,
        "status": status,
        "reason": reason,
        "snapshot": snapshot,
        "capacity": capacity,
        "capacity_checked": True,
        "mutated_object": False,
    }


def _first_free_slot(used: set, capacity: Optional[int]) -> Optional[int]:
    limit = capacity if capacity is not None else max(used, default=-1) + 1
    for slot in range(max(0, limit)):
        if slot not in used:
            return slot
    if capacity is None:
        return max(used, default=-1) + 1
    return None


def _add_to_inventory(inventory: Any, object_id: Any) -> Optional[int]:
    if not isinstance(inventory, MutableMapping):
        return _used_slots(inventory)

    capacity = _capacity_of(inventory)
    items = inventory.get("items")

    if isinstance(items, MutableMapping):
        used = {_integer(key) for key in items}
        used.discard(None)
        slot = _first_free_slot(used, capacity)
        if slot is None:
            return None
        items[slot] = object_id
        return slot

    slots = inventory.get("slots")
    if items is None and isinstance(slots, MutableMapping):
        used = {_integer(key) for key, value in slots.items() if value is not None}
        used.discard(None)
        slot = _first_free_slot(used, capacity if capacity is not None else len(slots))
        if slot is None:
            return None
        slots[slot] = object_id
        return slot

    if items is None:
        items = []
        inventory["items"] = items
    if not isinstance(items, list):
        return None
    slot = len(items)
    items.append(object_id)
    return slot


def _remove_object(drop: Any) -> bool:
    if drop is None or not _record(drop):
        return False
    removed = _set(drop, "picked_up", True)
    _set(drop, "destroyed", True)
    _set(drop, "visible", False)
    return removed


def grant(request: Any = None, inventory: Any = None, drop: Any = None, **kwargs: Any) -> dict:
    """结算拾取请求：代际校验 → 容量前置 → 扣物体 → 入包。

    任一步失败都不会扣除物体；容量失败或对象失效时执行回滚，恢复位置与
    可见性，且不复活已销毁对象。
    """
    bundle = _unpack(request, ("request", "inventory", "drop"))
    if bundle is not None:
        inventory = bundle.get("inventory", inventory)
        drop = bundle.get("drop", drop)
        request = bundle.get("request", bundle)

    req = _record(request)
    payload = _record(_get(req, "payload", default=req))
    if drop is None:
        drop = _get(payload, "drop")

    object_id = _get(payload, "object_id", "id")
    if object_id is None:
        object_id = _get(drop, "object_id", "id")
    generation = _integer(_get(payload, "generation"), _integer(_get(drop, "generation", default=0), 0))
    request_id = _get(payload, "request_id", default=f"pickup:{object_id}:{generation}")
    snapshot = _request_snapshot(payload, drop)

    generation_check = object_generation(drop if drop is not None else payload, payload)

    def _reject(reason: str, capacity: Optional[dict] = None) -> dict:
        rollback = rollback_restore(drop, snapshot) if drop is not None else None
        return {
            "granted": False,
            "status": "rejected",
            "request_id": request_id,
            "object_id": object_id,
            "generation": generation,
            "reason": reason,
            "removed": False,
            "slot": None,
            "generation_check": generation_check,
            "capacity": capacity,
            "rollback": rollback,
        }

    if not generation_check["valid"]:
        return _reject(generation_check["reason"])

    if drop is None:
        return _reject("missing_object")

    capacity = capacity_first(inventory, item=_get(payload, "item", default=_get(drop, "item")))
    if not capacity["ok"]:
        return _reject(capacity["reason"], capacity)

    # 顺序关键：容量已确认可以容纳，才允许扣除物体。
    removed = _remove_object(drop)
    slot = _add_to_inventory(inventory, object_id)
    sfx = stop_sfx_on_destroy(drop)

    return {
        "granted": True,
        "status": "granted",
        "request_id": request_id,
        "object_id": object_id,
        "generation": generation,
        "reason": "ok",
        "removed": removed,
        "slot": slot,
        "generation_check": generation_check,
        "capacity": capacity,
        "rollback": None,
        "sfx": sfx,
    }


def reject(request: Any = None, drop: Any = None, snapshot: Any = None, **kwargs: Any) -> dict:
    """拒绝一次拾取请求，并回滚其副作用。

    回滚只恢复存活对象的位置与可见性；若对象已被另一路径销毁，则保持销毁态，
    只停止残留音效。
    """
    bundle = _unpack(request, ("request", "drop", "snapshot"))
    if bundle is not None:
        drop = bundle.get("drop", drop)
        snapshot = bundle.get("snapshot", snapshot)
        request = bundle.get("request", bundle)

    req = _record(request)
    if drop is None:
        drop = _get(req, "drop")
    if snapshot is None:
        snapshot = _get(req, "snapshot")

    generation_check = object_generation(drop if drop is not None else req, req)
    rollback = rollback_restore(drop, snapshot) if drop is not None else None
    sfx = stop_sfx_on_destroy(drop) if drop is not None else None

    reason = _get(req, "reason", default=kwargs.get("reason", "rejected"))
    return {
        "status": "rejected",
        "request_id": _get(req, "request_id"),
        "object_id": _get(req, "object_id", "id", default=_get(_record(drop), "object_id", "id")),
        "reason": reason,
        "reactivated": False,
        "rollback": rollback,
        "generation_check": generation_check,
        "sfx": sfx,
    }


# ---------------------------------------------------------------------------
# 9. 观测与稳定性
# ---------------------------------------------------------------------------


_METRIC_DEFAULTS = {
    "attracted": 0,
    "pickup_requests": 0,
    "granted": 0,
    "rejected": 0,
    "rollbacks": 0,
    "sfx_stopped": 0,
    "generation_rejects": 0,
    "capacity_rejects": 0,
    "reactivated_destroyed": 0,
}


def metrics(state: Any = None, **kwargs: Any) -> dict:
    """返回计数器快照；缺省值全部归零，便于确定性对比。"""
    data = _record(state)
    source = _record(_get(data, "counters", "metrics", default={}))
    result = dict(_METRIC_DEFAULTS)
    for key in list(result):
        value = _integer(source.get(key, data.get(key)))
        if value is not None:
            result[key] = value
    return result


def audit(events: Any = None, **kwargs: Any) -> list:
    """把审计事件按序号稳定排序后返回，保证重放顺序一致。"""
    bundle = _unpack(events, ("events", "log"))
    if bundle is not None:
        events = bundle.get("events", bundle.get("log"))
    if events is None:
        events = kwargs.get("events")

    entries = []
    for index, event in enumerate(_sequence(events, "events", "log")):
        data = _record(event)
        if not data and isinstance(event, str):
            data = {"type": event}
        sequence = _integer(_get(data, "seq", "sequence"), index)
        entries.append(
            {
                "seq": sequence,
                "source_index": index,
                "type": _get(data, "type", "action", "event", default="event"),
                "object_id": _get(data, "object_id", "id"),
                "generation": _integer(_get(data, "generation")),
                "request_id": _get(data, "request_id"),
                "ok": bool(_get(data, "ok", default=True)),
                "reason": _get(data, "reason"),
            }
        )
    entries.sort(key=lambda entry: (entry["seq"], entry["source_index"]))
    return entries


def fallback(request: Any = None, drop: Any = None, **kwargs: Any) -> dict:
    """请求无法正常结算时的安全退路：丢弃请求，绝不复活已销毁对象。"""
    bundle = _unpack(request, ("request", "drop"))
    if bundle is not None:
        drop = bundle.get("drop", drop)
        request = bundle.get("request", bundle)

    req = _record(request)
    if drop is None:
        drop = _get(req, "drop")
    data = _record(drop)

    generation_check = object_generation(drop if drop is not None else req, req)
    destroyed = bool(_get(data, "destroyed", "dead", default=False))

    rollback = None
    if drop is not None and not destroyed:
        rollback = rollback_restore(drop, _get(req, "snapshot", default=kwargs.get("snapshot")))

    if destroyed:
        reason = "destroyed"
    elif not generation_check["valid"]:
        reason = generation_check["reason"]
    else:
        reason = "pickup_unavailable"

    return {
        "action": "discard",
        "status": "fallback",
        "object_id": _get(req, "object_id", "id", default=_get(data, "object_id", "id")),
        "reason": reason,
        "retryable": False,
        "reactivated": False,
        "kept_destroyed": destroyed,
        "rollback": rollback,
        "generation_check": generation_check,
    }


def determinism(items: Any = None, **kwargs: Any) -> dict:
    """对同一批掉落物做顺序无关的槽位规划，并给出可复现签名。

    实现上会把输入正序与逆序各排序一次：只有按稳定 ID 排序，两次得到的
    id 序列才会完全一致。签名使用 sha256，不受 Python hash 随机化影响。
    """
    bundle = _unpack(items, ("items", "drops"))
    if bundle is not None:
        items = bundle.get("items", bundle.get("drops"))
    if items is None:
        items = kwargs.get("items")

    sequence = _sequence(items, "items", "drops")
    forward = _ordering(sequence)
    backward = _ordering(list(reversed(sequence)))

    forward_ids = [str(entry["object_id"]) for entry in forward]
    backward_ids = [str(entry["object_id"]) for entry in backward]
    stable = forward_ids == backward_ids

    offset = _integer(kwargs.get("start", kwargs.get("slot_offset")), 0) or 0
    slots = [
        {
            "object_id": entry["object_id"],
            "generation": entry["generation"],
            "slot": offset + position,
        }
        for position, entry in enumerate(forward)
    ]

    return {
        "stable": stable,
        "input_order_independent": stable,
        "slots": slots,
        "signature": _signature(slots),
        "count": len(slots),
    }
