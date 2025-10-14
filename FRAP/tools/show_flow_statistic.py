import json
from collections import Counter
from typing import List, Tuple
import random

def route2phaseID(route:str):
    route = route.strip()
    translate = {
        "road_2_1_2 road_1_1_2":'E->W',
        "road_0_1_0 road_1_1_0":'W->E',
        "road_1_2_3 road_1_1_3":'N->S',
        "road_1_0_1 road_1_1_1":'S->N',
        "road_2_1_2 road_1_1_3":'E->S(L)',
        "road_0_1_0 road_1_1_1":'W->N(L)',
        "road_1_2_3 road_1_1_0":'N->E(L)',
        "road_1_0_1 road_1_1_2":'S->W(L)',
    }
    return translate[route]

def count_route_combinations(json_path: str) -> List[Tuple[Tuple[str, ...], int]]:
    """
    读取 json_path（顶层为数组），统计每条记录的 "route" 组合出现次数。
    返回按次数降序排序的 (route_tuple, count) 列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    ctr = Counter()
    for rec in data:
        route = rec.get("route", None)
        if route is None:
            continue

        if isinstance(route, list):
            key = tuple(str(p) for p in route)
        else:
            assert False, 'route should be  a list!'

        ctr[key] += 1

    counts = sorted(ctr.items(), key=lambda x: x[1], reverse=True)
    for route_tuple, cnt in counts:
        phase_id = route2phaseID(f'{route_tuple[0]} {route_tuple[1]}')
        print(f"{cnt:6d}  {route_tuple} {phase_id}")

def save_as_morning_rushhour(dest_file: str, src_file:str):

    with open(src_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    for rec in data:
        route = rec.get("route", None)
        if route is None:
            continue

        if isinstance(route, list):
            key = tuple(str(p) for p in route)
        else:
            assert False, 'route should be  a list!'
        if key == ('road_1_0_1', 'road_1_1_1'): #S->N
            if random.random() < 0.7: # 改成 N->S
                route[0] = 'road_1_2_3'
                route[1] = 'road_1_1_3'
        if key == ('road_0_1_0', 'road_1_1_0'): #W->E
            if random.random() < 0.7: # 改成 E->W
                route[0], route[1] = 'road_2_1_2', 'road_1_1_2'
    #另存为dest_file文件
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



# 使用示例
if __name__ == "__main__":
    count_route_combinations("../env/flow.json")
    save_as_morning_rushhour("../env/rushour_flow.json", "../env/flow.json")
    print()
    count_route_combinations("../env/rushour_flow.json")
