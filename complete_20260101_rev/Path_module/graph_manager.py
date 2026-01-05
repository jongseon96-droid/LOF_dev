# graph_manager.py
import osmnx as ox
import networkx as nx
import pandas as pd
from typing import Dict, Tuple
from geopy.distance import geodesic
from shapely.geometry import LineString
try:
    import config as cfg
except ImportError:
    from . import config as cfg

# ==================================================================
# 💾 1. 리전(Region) 단위 그래프 캐시 관리자
# ==================================================================
class RegionGraphCache:
    """
    특정 지역(Region)의 도로망 그래프를 다운로드하고 메모리에 저장(Cache)하는 클래스입니다.
    - 매번 API로 다운로드하면 느리기 때문에, 한 번 받은 지역은 self.cache에 저장해둡니다.
    - '반경 확장(Expand)' 기능을 통해, 기본 반경에서 경로를 못 찾으면 더 넓은 범위를 로드합니다.
    """
    def __init__(self, reg_df: pd.DataFrame):
        self.reg_df = reg_df
        # 캐시 저장소: {(리전ID, 네트워크타입, 반경): 그래프객체}
        self.cache: Dict[Tuple[int, str, int], nx.MultiDiGraph] = {}

    def get_with_expand(self, region_id: int, network_type: str) -> nx.MultiDiGraph:
        """
        주어진 리전 ID와 네트워크 타입(walk/drive)에 맞는 그래프를 반환합니다.
        설정된 확장 단계(DIST_EXPANDS)를 순회하며 다운로드를 시도합니다.
        """
        last_exc = None
        
        # 리전 정보(중심 좌표) 가져오기
        row = self.reg_df[self.reg_df['region_id'] == region_id]
        if row.empty:
            raise ValueError(f"region_id {region_id} 없음")
        lat = float(row.iloc[0]['mean_lat'])
        lon = float(row.iloc[0]['mean_lon'])
        
        # 🔄 반경 확장 루프 (예: 5km -> 8km -> 11km 순으로 시도)
        # 작은 반경에서 로드에 실패하거나 그래프가 불완전할 경우, 더 큰 반경을 시도함
        for extra in cfg.DIST_EXPANDS:
            dist = int(cfg.REGION_GRAPH_DIST_M + extra)
            key = (region_id, network_type, dist)
            
            # 1. 캐시에 있으면 바로 반환 (가장 빠름)
            if key in self.cache:
                return self.cache[key]
            
            # 2. 캐시에 없으면 OSMnx API로 다운로드 시도
            try:
                # print(f"🌍 그래프 로딩: region={region_id}, type={network_type}, dist={dist}m")
                G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
                self.cache[key] = G # 성공 시 캐시에 저장
                return G
            except Exception as e:
                # 다운로드 실패 시 에러 저장하고 다음 확장 반경으로 넘어감
                last_exc = e
                
        # 모든 확장 반경에서도 실패하면 에러 발생
        raise last_exc if last_exc else RuntimeError("graph load failed")

# ==================================================================
# 📦 2. 세그먼트 전용 BBox(Bounding Box) 그래프 생성 (Fallback)
# ==================================================================
def graph_from_segment_bbox(s_lat, s_lon, e_lat, e_lon, pad_m=800, network_type="walk"):
    """
    미리 정의된 리전(Region)에 속하지 않는 경로가 나왔을 때 사용하는 '비상용' 함수입니다.
    - 출발점과 도착점을 포함하는 사각형(BBox) 영역만큼만 지도를 다운로드합니다.
    - pad_m: 여유 공간(패딩)을 줘서 경로가 잘리지 않게 함
    """
    pad_deg = pad_m / 111_000.0 # 미터를 위도/경도 도(degree) 단위로 대략 변환
    north = max(s_lat, e_lat) + pad_deg
    south = min(s_lat, e_lat) - pad_deg
    east  = max(s_lon, e_lon) + pad_deg
    west  = min(s_lon, e_lon) - pad_deg
    
    return ox.graph_from_bbox(north, south, east, west, network_type=network_type)

# ==================================================================
# 🧲 3. 하이브리드 노드 스내핑 (Hybrid Node Snapping)
# ==================================================================
def snap_nodes_hybrid(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    """
    GPS 좌표(lon, lat)를 그래프 상의 가장 가까운 '노드(Node) ID'로 매칭합니다.
    - 1차 시도: ox.distance.nearest_nodes (가장 빠르고 정확함)
    - 2차 시도: 실패 시 nearest_edges를 찾아 그 엣지의 양 끝점 중 가까운 점 선택
    - 이렇게 하는 이유: 가끔 그래프가 희소할 때 nearest_nodes가 실패하는 경우를 방지하기 위함
    """
    try:
        # 1. 가장 가까운 노드 찾기
        return ox.distance.nearest_nodes(G, lon, lat)
    except Exception:
        pass
    
    try:
        # 2. (실패 시) 가장 가까운 엣지(도로)를 찾은 뒤, 그 엣지의 시작점(u)/끝점(v) 중 선택
        u, v, key = ox.distance.nearest_edges(G, lon, lat, return_dist=False)
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        
        du = geodesic((lat, lon), (uy, ux)).meters
        dv = geodesic((lat, lon), (vy, vx)).meters
        
        return u if du <= dv else v
    except Exception:
        # 최후의 수단: 다시 nearest_nodes 시도 (혹은 에러 전파)
        return ox.distance.nearest_nodes(G, lon, lat)

# ==================================================================
# 🛠️ 4. 안전한 LineString 생성 유틸
# ==================================================================
def _safe_same_node_linestring(G, node):
    """
    출발 노드와 도착 노드가 같을 때(이동 거리 0), 
    오류를 방지하기 위해 해당 노드 위치에 점(Point) 같은 길이 0짜리 LineString을 만듭니다.
    """
    x, y = G.nodes[node]["x"], G.nodes[node]["y"]
    return LineString([(x, y), (x, y)])