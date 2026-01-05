# matcher.py
import networkx as nx
from shapely.geometry import LineString
from geopy.distance import geodesic
from typing import List, Tuple, Optional

# ==================================================================
# 🚨 [수정 1]: Import 순서 변경 (모듈 간 호출 안정성 확보)
# ==================================================================
try:
    # 1. 패키지 모드 (상대 경로 우선)
    from . import config as cfg
    from . import utils
    from . import data_loader as dl
    from . import graph_manager as gm

except ImportError:
    # 2. 독립 실행 모드 (절대 경로 예외 처리)
    import config as cfg
    import utils
    import data_loader as dl
    import graph_manager as gm


# ==================================================================
# 🗺️ 1. 그래프 위에서 최단 경로 찾기 (Core Routing Logic)
# ==================================================================
def route_on_graph_with_waypoints(G: nx.MultiDiGraph, waypoints: List[Tuple[float, float]]) -> Optional[LineString]:
    """
    주어진 좌표들(Waypoints)을 그래프(G) 상의 노드에 매칭하고, 
    노드와 노드 사이를 최단 경로 알고리즘(Dijkstra)으로 연결하여 하나의 선(LineString)으로 만듭니다.
    """
    if len(waypoints) < 2: return None
    
    # 1. 모든 Waypoint를 그래프 상의 가장 가까운 노드로 스내핑(Snap)
    all_nodes = []
    try: 
        for lon, lat in waypoints:
            all_nodes.append(gm.snap_nodes_hybrid(G, lon, lat))
    except: return None
    
    # 2. 연속으로 중복된 노드 제거 (제자리걸음 방지)
    unique_nodes = [all_nodes[0]]
    for node in all_nodes[1:]:
        if node != unique_nodes[-1]: unique_nodes.append(node)
    
    # 점 하나만 남으면 그냥 점으로 반환
    if len(unique_nodes) < 2: return gm._safe_same_node_linestring(G, unique_nodes[0])
    
    full_coords = []
    
    # 3. 노드 사이사이의 경로 탐색
    for i in range(len(unique_nodes) - 1):
        try:
            # Dijkstra 알고리즘으로 최단 경로(노드 리스트) 탐색
            route_nodes = nx.shortest_path(G, unique_nodes[i], unique_nodes[i+1], weight="length")
            
            # 4. 노드만 잇는 게 아니라, 실제 도로의 곡선(Geometry) 정보를 가져옴
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                edge_data = G.get_edge_data(u, v)
                if not edge_data: continue
                
                # 멀티그래프(같은 노드 간 여러 도로) 중 가장 짧은 도로 선택
                best_key = min(edge_data, key=lambda k: edge_data[k].get("length", float("inf")))
                edge = edge_data[best_key]
                
                # 도로 형상(Geometry)이 있으면 그걸 쓰고, 없으면 직선 연결
                if "geometry" in edge:
                    xs, ys = edge["geometry"].xy
                    edge_coords = list(zip(xs, ys))
                else:
                    u_n, v_n = G.nodes[u], G.nodes[v]
                    edge_coords = [(u_n["x"], u_n["y"]), (v_n["x"], v_n["y"])]
                
                # 좌표 이어 붙이기 (중복점 제거)
                if not full_coords: full_coords.extend(edge_coords)
                elif full_coords[-1] == edge_coords[0]: full_coords.extend(edge_coords[1:])
                else: full_coords.extend(edge_coords)
        except: return None
        
    return LineString(full_coords) if full_coords else None

# ==================================================================
# 🔄 2. 폴백(Fallback) 지원 경로 탐색
# ==================================================================
def route_between_points_with_fallback(reg_cache, region_id, waypoints):
    """
    경로 탐색을 시도하되, 실패하면 다양한 방법(네트워크 변경, BBox 다운로드 등)으로 재시도합니다.
    """
    # 1. 리전 그래프 캐시 사용 (Walk -> Drive)
    for net in [cfg.NETWORK_TYPE_PRIMARY, cfg.NETWORK_TYPE_SECONDARY]:
        try:
            # config.py의 상수를 cfg.로 참조하도록 변경
            G = reg_cache.get_with_expand(region_id, net) 
            line = route_on_graph_with_waypoints(G, waypoints)
            if line and len(line.coords) > 1: return line
        except Exception: 
            pass
    
    # 2. 리전 범위를 벗어난 경우, 해당 구간 BBox로 실시간 다운로드 시도
    s_lon, s_lat = waypoints[0]
    e_lon, e_lat = waypoints[-1]
    for net in [cfg.NETWORK_TYPE_PRIMARY, cfg.NETWORK_TYPE_SECONDARY]:
        try:
            G_bbox = gm.graph_from_segment_bbox(s_lat, s_lon, e_lat, e_lon, pad_m=800, network_type=net)
            line = route_on_graph_with_waypoints(G_bbox, waypoints)
            if line and len(line.coords) > 1: return line
        except: pass
        
    return None

# ==================================================================
# 📦 3. 전체 데이터 맵매칭 실행기
# ==================================================================
def perform_map_matching_by_region(reg_cache, features_df, all_waypoints, seg_region_ids):
    """
    모든 세그먼트에 대해 순차적으로 맵매칭을 수행합니다.
    """
    matched = []
    for _, row in features_df.iterrows():
        sid = row['segment_id']
        waypoints = all_waypoints.get(sid)
        
        # Waypoint가 없거나 너무 적으면 스킵
        if not waypoints or len(waypoints) < 2:
            matched.append(None)
            continue
            
        # 경로 탐색 수행
        line = route_between_points_with_fallback(reg_cache, seg_region_ids.get(sid), waypoints)
        matched.append(line)
        
    return matched

# ==================================================================
# 🧵 4. 경로 스티칭 (Stitching) 및 병합 (Merging)
# ==================================================================
def stitch_and_merge_paths(reg_cache, matched_lines, seg_region_ids):
    """
    끊어진 세그먼트들을 하나로 잇습니다(Stitching).
    - 단순 연결: 갭이 작으면 그냥 잇습니다.
    - 브릿징(Bridging): 갭이 적당히 크면(GAP_BREAK_M 이상) 경로 탐색으로 메꿉니다.
    - 분할(Split): 갭이 너무 크면(MAX_BRIDGE_TRY_M 이상) 잇지 않고 새로운 청크로 시작합니다.
    """
    chunks = []            # 최종 결과물 (LineString 리스트)
    merged_coords = []     # 현재 작업 중인 청크의 좌표들
    prev_end = None        # 이전 세그먼트의 끝점
    prev_rid = None        # 이전 세그먼트의 리전 ID
    EPS_CONNECT_M = 50.0   # ⬅️ [주의]: 이 상수는 함수 내부에 그대로 둠

    def flush_chunk():
        """현재까지 모인 좌표를 LineString으로 만들고 저장"""
        nonlocal merged_coords
        if len(merged_coords) >= 2: chunks.append(LineString(merged_coords))
        merged_coords = []

    for i, line in enumerate(matched_lines):
        rid = seg_region_ids[i]
        
        # 1. 매칭 실패한 라인 처리
        if line is None or len(line.coords) < 2:
            flush_chunk(); prev_end = None; prev_rid = None; continue
        
        # 🚨 [수정 2]: 변수 정의 오류 해결 및 좌표 추출
        start_lon, start_lat = line.coords[0]
        curr_end_lon, curr_end_lat = line.coords[-1]
        
        # 2. 첫 번째 라인 (새 청크 시작) 처리
        if prev_end is None:
            merged_coords.extend(list(line.coords))
            prev_end = line.coords[-1]
            prev_rid = rid
            continue
        
        # ----------------------------------------------------
        # 이전 경로가 있었다면, 연결 여부 판단
        # ----------------------------------------------------
        
        gap_m = geodesic((prev_end[1], prev_end[0]), (line.coords[0][1], line.coords[0][0])).meters
        end_to_end_m = geodesic((prev_end[1], prev_end[0]), (line.coords[-1][1], line.coords[-1][0])).meters
        
        # 3. 누적 길이가 너무 길 때 강제 분할
        if end_to_end_m >= cfg.MAX_END_TO_END_DIST_M:
            print(f"🔵 FORCE SPLIT (E2E): {end_to_end_m:.1f}m")
            flush_chunk()
            merged_coords.extend(list(line.coords)) # 새 청크 시작
            prev_end = line.coords[-1]; prev_rid = rid
            continue

        # 4. 갭이 임계치(GAP_BREAK_M)보다 클 때
        if gap_m > cfg.GAP_BREAK_M:
            # 4-1. 너무 멀면(MAX_BRIDGE_TRY_M 이상) 포기하고 끊음 (Force Split)
            if gap_m > cfg.MAX_BRIDGE_TRY_M:
                print(f"🚫 FORCE SPLIT (Too Far): Gap {gap_m:.1f}m")
                flush_chunk()
                merged_coords.extend(list(line.coords)) # 새 청크 시작
                prev_end = line.coords[-1]; prev_rid = rid
                continue
            else:
                # 4-2. 적당히 멀면 브릿징 시도
                rid_gap = prev_rid if prev_rid else rid
                gap_waypoints = [(prev_end[0], prev_end[1]), (start_lon, start_lat)]
                bridge = route_between_points_with_fallback(reg_cache, rid_gap, gap_waypoints)
                
                if bridge and len(bridge.coords) > 1:
                    # 브릿징 성공: 다리 놓고 이어감
                    if merged_coords[-1] == bridge.coords[0]: merged_coords.extend(list(bridge.coords)[1:])
                    else: merged_coords.extend(list(bridge.coords))
                else:
                    # 🚫 브릿징 실패: 강제 분리 후 새 청크 시작 (직선 연결 방지)
                    print(f"🔴 BRIDGING FAILED: Gap {gap_m:.1f}m. Flushed Chunk.")
                    flush_chunk()
                    merged_coords.extend(list(line.coords)) # 새 청크 시작
                    prev_end = line.coords[-1]; prev_rid = rid
                    continue # ⬅️ 핵심: 실패 시 무조건 다음 라인을 새 청크로 시작하도록 건너뜀

        # 5. 갭이 작을 때 (50m 초과 ~ 300m 이하)
        elif gap_m > EPS_CONNECT_M:
             # 5-1. 간단한 브릿징 시도 (직선 연결 방지)
            rid_gap = prev_rid if prev_rid else rid
            gap_waypoints = [(prev_end[0], prev_end[1]), (start_lon, start_lat)]
            bridge = route_between_points_with_fallback(reg_cache, rid_gap, gap_waypoints)
            
            if bridge and len(bridge.coords) > 1:
                if merged_coords[-1] == bridge.coords[0]: merged_coords.extend(list(bridge.coords)[1:])
                else: merged_coords.extend(list(bridge.coords))
            else:
                # 🚫 짧은 갭 브릿징 실패: 강제 분리 후 새 청크 시작 (직선 연결 방지)
                print(f"🟡 SHORT GAP BRIDGE FAILED: {gap_m:.1f}m. Split Chunk.")
                flush_chunk()
                merged_coords.extend(list(line.coords)) # 새 청크 시작
                prev_end = line.coords[-1]; prev_rid = rid
                continue # ⬅️ 핵심: 실패 시 무조건 다음 라인을 새 청크로 시작하도록 건너뜀
        
        # 6. 갭이 50m 이하일 때: pass. (Implicitly connected)

        # ----------------------------------------------------
        # 🚨 이 지점은 '연결이 결정된 경우(성공 또는 50m 이하)'에만 도달함.
        # ----------------------------------------------------
        
        # 현재 라인 좌표 추가 (이전 로직에서 끊지 않았으므로 이어 붙임)
        if merged_coords[-1] == line.coords[0]: merged_coords.extend(list(line.coords)[1:])
        else: merged_coords.extend(list(line.coords)) # ⬅️ 50m 이하 갭이 여기서 직선 연결됨
        
        prev_end = line.coords[-1]; prev_rid = rid

    flush_chunk() # 마지막 청크 저장
    return chunks