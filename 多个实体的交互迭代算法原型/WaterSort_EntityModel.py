#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构说明（与 DICE.py 同构）：
  f1..fn  — 每个"试管"是一个独立的实体，fn() 返回其初始内容
  h(M)    — 多实体交互规则：输入当前所有试管，返回所有合法倾倒后的后继状态
  P       — 参数集：容积、规范化函数、目标检测、迭代上限
  g()     — BFS 引擎，枚举全部可达状态，输出完整状态空间

用法：
  python WaterSort_EntityModel.py WaterTest.json              → BFS + 控制台输出
  python WaterSort_EntityModel.py WaterTest.json --html       → 生成 HTML 相图
  python WaterSort_EntityModel.py WaterTest.json --trajectory longest  → 单条轨迹模式
══════════════════════════════════════════════════════════════════════
"""

import json
import sys
import os
import base64
from collections import deque
from typing import Tuple, List, Optional


# ══════════════════════════════════════════════════════════════════
# 1. 实体定义：fn(Wn) — 每个试管是一个实体
# ══════════════════════════════════════════════════════════════════

def auto_generate_tube_fns(json_path: str) -> Tuple[List, dict]:
    """从 JSON 自动生成 f1()..fn() 函数。返回 (tube_fns, raw_data)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    Vol = data['Vol']
    keys = sorted(data['TtList'].keys(), key=lambda k: (len(k), k))
    tube_tuples = [tuple(data['TtList'][k]) for k in keys]

    tube_fns = []
    for tube in tube_tuples:

        def make_fn(t):
            return lambda: t

        tube_fns.append(make_fn(tube))

    return tube_fns, {"Vol": Vol, "tubes": tube_tuples, "keys": keys}


# 也可以手动定义：
# def f1(): return (0, 0, 0, 1, 2)
# def f2(): return (1, 2, 3, 1, 1)
# ...


# ══════════════════════════════════════════════════════════════════
# 2. 核心函数
# ══════════════════════════════════════════════════════════════════

def get_top_layer(tube: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
    """返回 (颜色, 连续层体积)；空管返回 None"""
    if not tube:
        return None
    color = tube[-1]
    count = 0
    for i in range(len(tube) - 1, -1, -1):
        if tube[i] == color:
            count += 1
        else:
            break
    return (color, count)


def is_solution(tubes: Tuple[Tuple[int, ...], ...], Vol: int) -> bool:
    """目标状态：每管要么为空，要么单色满管"""
    for tube in tubes:
        if not tube:
            continue
        if len(tube) != Vol or len(set(tube)) != 1:
            return False
    return True


def canonicalize(state: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    """试管排序消除置换对称"""
    return tuple(sorted(state))


# ══════════════════════════════════════════════════════════════════
# 3. 交互规则：h(M) — 多实体交互（后继生成器）
#   输入当前所有试管，返回所有合法倾倒的后继状态列表
# ══════════════════════════════════════════════════════════════════

def h(*tubes: Tuple[int, ...], P: dict) -> List[Tuple[Tuple[int, ...], ...]]:
    """多实体交互规则：枚举所有合法 (源管, 目标管) 倾倒，返回后继状态列表"""
    Vol = P["Vol"]
    successors = []
    h_len = len(tubes)

    for i in range(h_len):
        if not tubes[i]:
            continue
        top = get_top_layer(tubes[i])
        if top is None:
            continue
        color, count = top

        for j in range(h_len):
            if i == j:
                continue
            space = Vol - len(tubes[j])
            if space == 0:
                continue
            if tubes[j] and tubes[j][-1] != color:
                continue
            pour = min(count, space)
            if pour == 0:
                continue

            tubes_list = list(tubes)
            src_list = list(tubes_list[i])
            dst_list = list(tubes_list[j])
            for _ in range(pour):
                src_list.pop()
            for _ in range(pour):
                dst_list.append(color)
            tubes_list[i] = tuple(src_list)
            tubes_list[j] = tuple(dst_list)

            successors.append(tuple(tubes_list))

    return successors


# ══════════════════════════════════════════════════════════════════
# 4. 参数定义：P
# ══════════════════════════════════════════════════════════════════

def make_P(Vol: int, max_states: int = 50000) -> dict:
    return {
        "Vol": Vol,
        "max_states": max_states,
        "canonicalize": canonicalize,
        "is_solution": lambda tubes: is_solution(tubes, Vol),
    }


# ══════════════════════════════════════════════════════════════════
# 5. 主函数：g(f1, f2, ..., fn, h, P) — BFS 枚举模式
# ══════════════════════════════════════════════════════════════════

def g(*fns, h, P: dict) -> dict:
    """
    g(f1(W1), f2(W2), ..., fn(Wn), h(M), P)

    BFS 枚举全状态空间。返回 dict:
      states, edges, init_id, init_raw, sol_ids, Vol
    """
    init_raw = tuple(fn() for fn in fns)
    init = P["canonicalize"](init_raw)
    Vol = P["Vol"]

    sid = {init: 0}
    states = [init]
    edges = []
    q = deque([0])

    while q:
        cur_idx = q.popleft()
        cur_state = states[cur_idx]
        next_states = h(*cur_state, P=P)

        for ns in next_states:
            ns_canon = P["canonicalize"](ns)
            if ns_canon not in sid:
                if len(states) >= P.get("max_states", 50000):
                    print(
                        "[WARN] 达到状态上限 %d，状态空间可能不完整"
                        % P.get("max_states", 50000),
                        file=sys.stderr,
                    )
                    break
                sid[ns_canon] = len(states)
                states.append(ns_canon)
                q.append(len(states) - 1)
            edges.append((cur_idx, sid[ns_canon]))
        else:
            continue
        break

    init_id = sid[init]
    sol_ids = {i for i, s in enumerate(states) if is_solution(s, Vol)}

    return dict(
        states=states, edges=edges, init_id=init_id,
        init_raw=init_raw, sol_ids=sol_ids, Vol=Vol,
    )


# ══════════════════════════════════════════════════════════════════
# 6. 单轨迹模式：g_trajectory
# ══════════════════════════════════════════════════════════════════

class _PourStep:
    """h() 的单步选择变体：按策略从所有后继中选一个"""

    def __init__(self, strategy="first_valid", seed=None):
        self.strategy = strategy
        if seed is not None:
            import random
            random.seed(seed)

    def _pick(self, cur_tubes, successors):
        if not successors:
            return cur_tubes
        if self.strategy == "first_valid":
            return successors[0]
        elif self.strategy == "last_valid":
            return successors[-1]
        elif self.strategy == "random":
            import random
            return random.choice(successors)
        elif self.strategy == "longest":

            def _gain(after, before):
                for a, b in zip(after, before):
                    d = len(a) - len(b)
                    if d > 0:
                        return d
                return 0

            return max(successors, key=lambda s: _gain(s, cur_tubes))
        return successors[0]

    def __call__(self, *tubes, P):
        successors = h(*tubes, P=P)
        return self._pick(tubes, successors)


def g_trajectory(*fns, h_step, P: dict, max_steps: int = 80) -> list:
    """单轨迹模式：逐步输出演化路径（演示动力系统行为）"""
    entities = tuple(fn() for fn in fns)
    traces = []

    print(f"初始: {_fmt_tubes(entities)}")
    print(
        f"策略: {P.get('pour_strategy', 'first_valid')}  "
        f"容积: {P['Vol']}  最大步数: {max_steps}"
    )
    print("=" * 60)

    for rnd in range(1, max_steps + 1):
        prev = entities
        entities = h_step(*entities, P=P)
        pour_desc = _describe_pour(prev, entities)
        traces.append((rnd, pour_desc, entities))
        print(
            f"步 {rnd:3d}: {pour_desc:40s} | {_fmt_tubes(entities)}"
        )

        if P["is_solution"](entities):
            print("=" * 60)
            print(f"[SOLVED] 在第 {rnd} 步达到目标状态！")
            return traces

    print("=" * 60)
    print(f"[MAX] 达到最大步数 {max_steps}，未找到解")
    return traces


# ══════════════════════════════════════════════════════════════════
# 7. 辅助展示函数
# ══════════════════════════════════════════════════════════════════

def _fmt_tubes(tubes: tuple) -> str:
    parts = []
    for t in tubes:
        if not t:
            parts.append("[]")
        else:
            parts.append("[" + "".join(str(x) for x in t) + "]")
    return " ".join(parts)


def _describe_pour(before: tuple, after: tuple) -> str:
    diffs = []
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            if len(b) < len(a):
                diffs.append(f"管{i+1}<-{a[-1]}x{len(a)-len(b)}")
            elif len(b) > len(a):
                diffs.append(f"管{i+1}->{len(b)-len(a)}")
    if diffs:
        return "倾倒: " + ", ".join(diffs)
    return "无操作 (死锁)"


def _print_bfs_summary(result: dict) -> None:
    print(f"\n状态空间摘要:")
    print(f"  总状态数: {len(result['states'])}")
    print(f"  转移边数: {len(result['edges'])}")
    print(f"  解状态:   {len(result['sol_ids'])} 个")
    print(f"  有解:      {'是' if result['sol_ids'] else '否'}")
    print(f"  试管容积:  {result['Vol']}")
    print(f"  试管数:    {len(result['init_raw'])}")
    print()


# ══════════════════════════════════════════════════════════════════
# 8. SCC 压缩 (Tarjan)
# ══════════════════════════════════════════════════════════════════

def _tarjan(n: int, adj: List[List[int]]) -> List[List[int]]:
    """返回强连通分量列表"""
    idx = 0
    indices = [-1] * n
    lowlink = [0] * n
    onstack = [False] * n
    stack = []
    sccs = []

    def strongconnect(v):
        nonlocal idx
        indices[v] = idx
        lowlink[v] = idx
        idx += 1
        stack.append(v)
        onstack[v] = True
        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif onstack[w]:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                onstack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)
    return sccs


def _condense(states, edges, sccs):
    """构建 SCC 凝聚图（DAG），返回 (dag_adj, state_to_scc)"""
    state_to_scc = {}
    for sid, scc in enumerate(sccs):
        for v in scc:
            state_to_scc[v] = sid
    dag_adj = [set() for _ in sccs]
    for u, v in edges:
        su, sv = state_to_scc[u], state_to_scc[v]
        if su != sv:
            dag_adj[su].add(sv)
    return [sorted(s) for s in dag_adj], state_to_scc


# ══════════════════════════════════════════════════════════════════
# 9. 分类 & 距离
# ══════════════════════════════════════════════════════════════════

_CLASS_LABEL = {
    "init": "初始", "solution": "解", "cycle": "环",
    "dead_cycle": "死循环", "dead_end": "死胡同", "intermediate": "中间",
}


def _classify_nodes(sccs, state_to_scc, init_id, Vol, states, dag_adj):
    init_scc = state_to_scc[init_id]

    sol_sccs = set()
    for sid, scc in enumerate(sccs):
        if any(is_solution(states[v], Vol) for v in scc):
            sol_sccs.add(sid)

    cycle_sccs = {sid for sid, scc in enumerate(sccs) if len(scc) > 1}
    dead_sccs = set()
    for sid in range(len(sccs)):
        if not dag_adj[sid] and sid not in sol_sccs:
            dead_sccs.add(sid)

    cls = {}
    for sid in range(len(sccs)):
        if sid in sol_sccs:
            cls[sid] = "solution"
        elif sid == init_scc:
            cls[sid] = "init"
        elif sid in dead_sccs and sid in cycle_sccs:
            cls[sid] = "dead_cycle"
        elif sid in cycle_sccs:
            cls[sid] = "cycle"
        elif sid in dead_sccs:
            cls[sid] = "dead_end"
        else:
            cls[sid] = "intermediate"

    return cls, init_scc, sol_sccs


def _compute_distances(dag_adj, init_scc, sol_sccs):
    d_init = {init_scc: 0}
    q = deque([init_scc])
    while q:
        u = q.popleft()
        for v in dag_adj[u]:
            if v not in d_init:
                d_init[v] = d_init[u] + 1
                q.append(v)

    rev_adj = [[] for _ in dag_adj]
    for u in range(len(dag_adj)):
        for v in dag_adj[u]:
            rev_adj[v].append(u)

    d_sol = {}
    q = deque()
    for s in sol_sccs:
        d_sol[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in rev_adj[u]:
            if v not in d_sol:
                d_sol[v] = d_sol[u] + 1
                q.append(v)

    return d_init, d_sol


# ══════════════════════════════════════════════════════════════════
# 10. 边颜色插值
# ══════════════════════════════════════════════════════════════════

def _edge_color(src_scc, dst_scc, d_init, d_sol, sol_exists, init_scc, sol_sccs):
    if sol_exists and dst_scc in sol_sccs:
        return "#00FFFF"
    if src_scc == init_scc:
        return "#00FF00"

    di = d_init.get(dst_scc)
    ds = d_sol.get(dst_scc) if sol_exists else None

    if di is None:
        return "#666666"

    if sol_exists:
        if ds is None:
            return "#CC3333"

        max_di = max(d_init.values()) if d_init else 1
        max_ds = max(d_sol.values()) if d_sol else 1
        if max_di == 0 and max_ds == 0:
            return "#00FF00"

        ni = di / max_di if max_di > 0 else 0
        ns = ds / max_ds if max_ds > 0 else 0
        w_g = (1.0 - ni) ** 1.5
        w_c = (1.0 - ns) ** 1.5
        w_r = (ni * ns) ** 0.7

        total = w_g + w_c + w_r
        if total < 0.001:
            total = 1.0

        r = max(0, min(255, int(255 * w_r / total)))
        g = max(0, min(255, int(255 * (w_g + w_c) / total)))
        b = max(0, min(255, int(255 * w_c / total)))
        return "#%02X%02X%02X" % (r, g, b)
    else:
        max_di = max(d_init.values()) if d_init else 1
        ni = di / max_di if max_di > 0 else 0
        r = int(255 * ni)
        g = int(255 * (1.0 - ni))
        return "#%02X%02X00" % (r, g)


def _dead_cycle_svg_uri():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44">'
        '<circle cx="22" cy="22" r="20" fill="#FFB74D" stroke="#F57C00" stroke-width="2.5"/>'
        '<circle cx="22" cy="22" r="7" fill="#EF5350" stroke="#C62828" stroke-width="1"/>'
        '</svg>'
    )
    b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return 'data:image/svg+xml;base64,' + b64


# ══════════════════════════════════════════════════════════════════
# 11. HTML 生成
# ══════════════════════════════════════════════════════════════════

_NODE_STYLE = {
    "init": {"bg": "#448AFF", "border": "#1565C0", "font": "#ffffff"},
    "intermediate": {"bg": "#F5F5F5", "border": "#BDBDBD", "font": "#333333"},
    "cycle": {"bg": "#FFB74D", "border": "#F57C00", "font": "#333333"},
    "dead_cycle": {"bg": "#FFB74D", "border": "#EF5350", "font": "#333333",
                    "image": True},
    "dead_end": {"bg": "#EF5350", "border": "#C62828", "font": "#ffffff"},
    "solution": {"bg": "#66BB6A", "border": "#2E7D32", "font": "#ffffff"},
}


def _build_html(sccs, dag_adj, cls, d_init, d_sol,
                init_scc, sol_sccs, Vol, states, s2scc, initial_raw):
    sol_exists = bool(sol_sccs)
    num_sccs = len(sccs)
    dead_cycle_img = _dead_cycle_svg_uri()

    # ── 节点数据 ──
    nodes_json = []
    for sid in range(num_sccs):
        c = cls[sid]
        sc = len(sccs[sid])
        sty = _NODE_STYLE[c]
        is_key = c in ("init", "solution")
        level = d_init.get(sid, 999) if d_init.get(sid) is not None else 999

        label = (
            "S%d" % sid if c == "intermediate"
            else "死循环\nS%d" % sid if c == "dead_cycle"
            else "%s\nS%d" % (_CLASS_LABEL[c], sid)
        )

        di_val = d_init.get(sid)
        ds_val = d_sol.get(sid) if sol_exists else None
        title = "SCC%d | %s | 含%d状态 | d_init=%s | d_sol=%s" % (
            sid, _CLASS_LABEL[c], sc,
            str(di_val) if di_val is not None else "inf",
            str(ds_val) if ds_val is not None
            else ("N/A" if not sol_exists else "inf"),
        )

        node_obj = {
            "id": sid, "label": label, "title": title, "level": level,
            "color": {
                "background": sty["bg"], "border": sty["border"],
                "highlight": {"background": sty["bg"], "border": "#000000"},
                "hover": {"background": sty["bg"], "border": "#000000"},
            },
            "font": {
                "size": 11 if is_key else 9,
                "color": sty["font"],
                "face": "Microsoft YaHei, PingFang SC, sans-serif",
                "bold": is_key,
            },
            "shape": "box" if is_key else "dot",
            "size": 28 if is_key else 16,
            "borderWidth": 3 if is_key else 1,
        }

        if c == "dead_cycle":
            node_obj["shape"] = "image"
            node_obj["image"] = dead_cycle_img
            node_obj["size"] = 24
            node_obj["borderWidth"] = 0
            node_obj["color"]["highlight"] = {
                "background": "#FFB74D", "border": "#EF5350"}
            node_obj["color"]["hover"] = {
                "background": "#FFB74D", "border": "#EF5350"}

        nodes_json.append(node_obj)

    # ── 边数据 ──
    edges_json = []
    seen_edges = set()
    for u in range(num_sccs):
        for v in dag_adj[u]:
            if (u, v) in seen_edges:
                continue
            seen_edges.add((u, v))
            ec = _edge_color(u, v, d_init, d_sol, sol_exists, init_scc, sol_sccs)
            edges_json.append({
                "from": u, "to": v,
                "color": {"color": ec, "highlight": ec, "hover": ec,
                           "inherit": False},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                "width": 2.5,
                "smooth": {"type": "curvedCCW", "roundness": 0.18},
            })

    # ── 统计 ──
    counts = {}
    for c in ["init", "intermediate", "cycle", "dead_cycle",
              "dead_end", "solution"]:
        counts[c] = sum(1 for x in cls.values() if x == c)

    stat_html = (
        '<div class="stat-row"><span>总状态数</span><span>%d</span></div>\n'
        '<div class="stat-row"><span>SCC 节点</span><span>%d</span></div>\n'
        '<div class="stat-row"><span>DAG 边数</span><span>%d</span></div>\n'
        '<div class="stat-row"><span>试管容积 N</span><span>%d</span></div>\n'
        '<div class="stat-row"><span>试管数 h</span><span>%d</span></div>\n'
        '<div class="stat-row"><span>存在解</span>'
        '<span style="color:%s">%s</span></div>'
    ) % (
        len(states), num_sccs, len(edges_json),
        Vol, len(initial_raw),
        "#66BB6A" if sol_exists else "#EF5350",
        "是" if sol_exists else "否",
    )

    legend_items = []
    for c in ["init", "intermediate", "cycle", "dead_cycle",
              "dead_end", "solution"]:
        sty = _NODE_STYLE[c]
        label = _CLASS_LABEL[c]
        cnt = counts[c]
        if c == "dead_cycle":
            legend_items.append(
                '<div class="legend-item">'
                '<img src="%s" width="22" height="22" style="flex-shrink:0" alt="">'
                '<span>%s (%d)</span>'
                '</div>' % (dead_cycle_img, label, cnt)
            )
        else:
            legend_items.append(
                '<div class="legend-item">'
                '<div class="dot" style="background:%s;border-color:%s"></div>'
                '<span>%s (%d)</span>'
                '</div>' % (sty["bg"], sty["border"], label, cnt)
            )
    legend_html = "\n".join(legend_items)

    display = {}
    for i, t in enumerate(initial_raw):
        display["tt%d" % (i + 1)] = list(t) if t else []
    init_json_str = json.dumps(display, ensure_ascii=False, indent=2)

    init_scc_js = init_scc
    sol_scc_js = min(sol_sccs) if sol_sccs else -1

    nodes_json_str = json.dumps(nodes_json, ensure_ascii=False)
    edges_json_str = json.dumps(edges_json, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Water Sort — 状态空间图 (多实体架构)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js">
</script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Microsoft YaHei','PingFang SC','Noto Sans SC',sans-serif;
  background:#0c0c1d;color:#c8c8d4;display:flex;height:100vh;overflow:hidden;
}
#graph{flex:1;background:#12122a;min-width:0}
#side{
  width:320px;padding:20px;background:#16162e;
  overflow-y:auto;font-size:13px;border-left:2px solid #0f3460;
  display:flex;flex-direction:column;gap:14px;
}
h2{color:#e94560;font-size:20px;border-bottom:2px solid #e94560;
   padding-bottom:10px;margin-bottom:2px}
h3{color:#f0a500;font-size:14px;margin:0 0 6px}
.stat-row{display:flex;justify-content:space-between;padding:5px 0;
          border-bottom:1px solid #2a2a4a}
.stat-row span:last-child{font-weight:bold;color:#e0e0e0}
.legend-item{display:flex;align-items:center;gap:10px;margin:5px 0}
.dot{width:20px;height:20px;border-radius:50%;border:2px solid transparent;flex-shrink:0}
.line{width:42px;height:4px;border-radius:2px;flex-shrink:0}
.init-box{
  background:#0d1b36;border-radius:6px;padding:10px;
  font:11px 'Cascadia Code','Courier New',monospace;
  max-height:200px;overflow-y:auto;white-space:pre-wrap;
  word-break:break-all;color:#8ab4f8;border:1px solid #1a3a5c;
}
.btn-row{display:flex;flex-wrap:wrap;gap:6px}
.btn{
  flex:1;min-width:70px;padding:8px 10px;
  border:none;border-radius:6px;cursor:pointer;
  font-size:12px;font-weight:bold;font-family:inherit;
  transition:transform .1s,box-shadow .1s;
}
.btn:active{transform:scale(0.96)}
.btn-init{background:#448AFF;color:#fff}
.btn-sol{background:#66BB6A;color:#fff}
.btn-layout{background:#f0a500;color:#0c0c1d}
.btn-physics{background:#7c4dff;color:#fff}
.btn:hover{box-shadow:0 0 12px rgba(255,255,255,.25)}
.note{font-size:11px;color:#888;line-height:1.5}
</style>
</head>
<body>
<div id="graph"></div>
<div id="side">
<h2>状态空间图 (多实体架构)</h2>
<div class="note">SCC 凝聚图 · 拖拽节点 · 滚轮缩放 · 按钮定位</div>

<div class="btn-row">
  <button class="btn btn-init" onclick="focusInit()" title="定位到初始状态节点">初始节点</button>
  <button class="btn btn-sol" id="btnSol" onclick="focusSol()" title="定位到解节点">解节点</button>
</div>
<div class="btn-row">
  <button class="btn btn-layout" id="btnLayout" onclick="toggleLayout()">层级布局</button>
  <button class="btn btn-physics" id="btnPhysics" onclick="togglePhysics()">暂停物理</button>
</div>

<h3>统计</h3>
""" + stat_html + """

<h3>节点图例</h3>
""" + legend_html + """

<h3>边颜色</h3>
<div class="legend-item"><div class="line" style="background:#00FF00"></div><span>近初始 / 出自初始</span></div>
<div class="legend-item"><div class="line" style="background:#00FFFF"></div><span>近解 / 直连解</span></div>
<div class="legend-item"><div class="line" style="background:#FF0000"></div><span>远离两者</span></div>
<div class="legend-item"><div class="line" style="background:#CC3333"></div><span>不可达解 (死支)</span></div>
<div class="legend-item"><div class="line" style="background:#666666"></div><span>不可达</span></div>

<h3>初始条件</h3>
<div class="init-box">""" + init_json_str + """</div>
</div>
<script>
/* -- 参数 -- */
var INIT_SCC = """ + str(init_scc_js) + """;
var SOL_SCC  = """ + str(sol_scc_js) + """;
var SOL_EXISTS = """ + ("true" if sol_exists else "false") + """;

/* -- 数据 -- */
var nodes = new vis.DataSet(""" + nodes_json_str + """);
var edges = new vis.DataSet(""" + edges_json_str + """);

var forceOptions = {
  physics: {
    enabled: true, solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -55, centralGravity: 0.012,
      springLength: 160, springConstant: 0.06, damping: 0.45
    },
    stabilization: { iterations: 250, updateInterval: 10 }
  },
  layout: { hierarchical: false },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    smooth: { type: 'curvedCCW', roundness: 0.18 }
  },
  interaction: {
    hover: true, tooltipDelay: 120,
    zoomView: true, dragView: true, navigationButtons: true
  },
  nodes: { borderWidthSelected: 4 }
};

var hierOptions = {
  physics: { enabled: false },
  layout: {
    hierarchical: {
      enabled: true, direction: 'LR', sortMethod: 'directed',
      nodeSpacing: 140, levelSeparation: 220, treeSpacing: 40,
      blockShifting: true, edgeMinimization: true,
      parentCentralization: false
    }
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    smooth: { type: 'curvedCCW', roundness: 0.18 }
  },
  interaction: {
    hover: true, tooltipDelay: 120,
    zoomView: true, dragView: true, navigationButtons: true
  },
  nodes: { borderWidthSelected: 4 }
};

var container = document.getElementById('graph');
var network = new vis.Network(container, { nodes: nodes, edges: edges }, forceOptions);
var usingHierarchical = false;
var physicsEnabled = true;

function focusInit() {
  network.focus(INIT_SCC, { scale: 0.9, animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
  setTimeout(function(){ network.selectNodes([INIT_SCC]); }, 750);
}
function focusSol() {
  if (!SOL_EXISTS) { alert('no solution'); return; }
  network.focus(SOL_SCC, { scale: 0.9, animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
  setTimeout(function(){ network.selectNodes([SOL_SCC]); }, 750);
}
function toggleLayout() {
  var btn = document.getElementById('btnLayout');
  if (usingHierarchical) {
    network.setOptions(forceOptions);
    usingHierarchical = false;
    btn.textContent = '层级布局';
    if (!physicsEnabled) { network.setOptions({ physics: { enabled: false } }); }
  } else {
    network.setOptions(hierOptions);
    usingHierarchical = true;
    btn.textContent = '力导向布局';
    physicsEnabled = false;
    document.getElementById('btnPhysics').textContent = '启动物理';
  }
}
function togglePhysics() {
  var btn = document.getElementById('btnPhysics');
  if (physicsEnabled) {
    network.setOptions({ physics: { enabled: false } });
    physicsEnabled = false;
    btn.textContent = '启动物理';
  } else {
    if (usingHierarchical) { toggleLayout(); }
    network.setOptions({ physics: { enabled: true } });
    physicsEnabled = true;
    btn.textContent = '暂停物理';
  }
}
if (!SOL_EXISTS) { document.getElementById('btnSol').style.display = 'none'; }
document.addEventListener('keydown', function(e) {
  if (e.key === 'i' || e.key === 'I') { focusInit(); }
  if ((e.key === 's' || e.key === 'S') && SOL_EXISTS) { focusSol(); }
  if (e.key === 'l' || e.key === 'L') { toggleLayout(); }
  if (e.key === 'p' || e.key === 'P') { togglePhysics(); }
});
</script>
</body>
</html>"""

    return html


def _generate_html(result: dict, output_path: str) -> None:
    """内部管线：BFS 结果 → SCC → HTML"""
    states = result["states"]
    edges = result["edges"]
    Vol = result["Vol"]
    init_raw = result["init_raw"]
    init_id = result["init_id"]

    n = len(states)
    raw_adj = [[] for _ in range(n)]
    for u, v in edges:
        raw_adj[u].append(v)
    sccs = _tarjan(n, raw_adj)
    dag_adj, s2scc = _condense(states, edges, sccs)
    cls, init_scc, sol_sccs = _classify_nodes(
        sccs, s2scc, init_id, Vol, states, dag_adj
    )
    d_init, d_sol = _compute_distances(dag_adj, init_scc, sol_sccs)

    html = _build_html(
        sccs, dag_adj, cls, d_init, d_sol,
        init_scc, sol_sccs, Vol, states, s2scc, init_raw,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML 相图已生成: {output_path}")
    print(f"文件大小: {len(html.encode('utf-8'))} 字节")


# ══════════════════════════════════════════════════════════════════
# 12. 入口
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]
    json_path = None
    mode_bfs = True
    mode_html = False
    mode_trajectory = False
    trajectory_strategy = "first_valid"

    i = 0
    while i < len(args):
        if args[i] == "--html":
            mode_html = True
            mode_bfs = True
        elif args[i] == "--trajectory":
            mode_trajectory = True
            mode_bfs = False
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 1
                trajectory_strategy = args[i]
        elif not args[i].startswith("--") and json_path is None:
            json_path = args[i]
        i += 1

    if json_path is None:
        print("用法: python WaterSort_EntityModel.py <input.json> [选项]")
        print()
        print("选项:")
        print("  --html                  生成 HTML 相图（自包含，无外部依赖）")
        print("  --trajectory [strategy] 单轨迹模式（默认: first_valid）")
        print()
        print("轨迹策略: first_valid / last_valid / random / longest")
        print()
        print("示例:")
        print("  python WaterSort_EntityModel.py WaterTest.json")
        print("  python WaterSort_EntityModel.py WaterTest.json --html")
        print(
            "  python WaterSort_EntityModel.py WaterTest.json"
            " --trajectory longest"
        )
        sys.exit(1)

    # ── 加载输入 ──
    tube_fns, raw_data = auto_generate_tube_fns(json_path)
    Vol = raw_data["Vol"]
    h_len = len(tube_fns)

    print("=" * 60)
    print("Water Sort Puzzle — 多实体交互迭代架构")
    print("g(f1, f2, ..., fn, h, P)")
    print("=" * 60)
    print(f"试管容量: {Vol}")
    print(f"试管数量: {h_len}")
    print()

    if mode_trajectory:
        P = make_P(Vol)
        P["pour_strategy"] = trajectory_strategy
        h_step = _PourStep(
            strategy=trajectory_strategy,
            seed=42 if trajectory_strategy == "random" else None,
        )
        g_trajectory(*tube_fns, h_step=h_step, P=P, max_steps=80)
    else:
        P = make_P(Vol, max_states=50000)
        result = g(*tube_fns, h=h, P=P)
        _print_bfs_summary(result)
        if mode_html:
            base = os.path.splitext(json_path)[0]
            output_path = base + "_entity_graph.html"
            _generate_html(result, output_path)
        else:
            print("提示: 加 --html 参数可生成交互式 HTML 相图")
