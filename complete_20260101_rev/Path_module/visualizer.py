# visualizer.py
import folium
from folium import FeatureGroup
import itertools
from shapely.geometry import LineString
import networkx as nx
from typing import List, Tuple, Optional

# ==================================================================
# 🧩 1. 공간적 연결성 기반 그룹핑 (Spatial Connectivity Grouping)
# ==================================================================
def group_lines_by_connectivity(lines: List[LineString], tol_deg: float = 1e-6) -> List[List[LineString]]:
    """
    여러 개의 끊어진 경로(LineString)들을 받아, 서로 공간적으로 맞닿아 있거나
    가까운 것들끼리 묶어서 그룹(Group)으로 만듭니다.
    
    - 목적: 서울, 부산 등 멀리 떨어진 경로들이 하나의 색으로 칠해지는 것을 방지하고,
            서로 연결된 '덩어리' 단위로 시각화하기 위함입니다.
    - 원리: NetworkX의 연결 요소(Connected Components) 알고리즘을 사용합니다.
    """
    if not lines: return []
    
    # 그래프 생성 (노드 = 각 라인의 인덱스)
    G = nx.Graph()
    G.add_nodes_from(range(len(lines)))
    
    # 모든 라인 쌍을 비교하여 거리가 가까우면 엣지(연결) 추가
    # (tol_deg 1e-6은 약 10cm 정도의 매우 가까운 거리)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i].distance(lines[j]) < tol_deg:
                G.add_edge(i, j)
                
    # 연결된 노드들끼리 그룹핑하여 반환
    # 예: [[line1, line2], [line3], [line4, line5, line6]]
    return [[lines[i] for i in comp] for comp in nx.connected_components(G)]

# ==================================================================
# 🗺️ 2. 계층형 지도 시각화 (Layered Map Plotting)
# ==================================================================
def plot_map_layered_by_group(base_center, matched_lines, grouped_lines, interp_points, zoom_start=14):
    """
    분석된 데이터를 레이어(Layer)별로 나누어 지도에 그립니다.
    사용자는 지도 우측 상단의 레이어 컨트롤을 통해 보고 싶은 정보만 켜고 끌 수 있습니다.
    """
    # 1. 지도 객체 초기화 (배경: 밝은 지도)
    m = folium.Map(location=list(base_center), zoom_start=zoom_start, tiles='OpenStreetMap')
    folium.TileLayer('cartodbpositron', name="Light").add_to(m)

    # ---------------------------------------------------------
    # 레이어 1: Raw Segments (기본적으로 꺼둠)
    # 맵매칭된 개별 세그먼트들을 회색 실선으로 표시합니다.
    # ---------------------------------------------------------
    grp_matched = FeatureGroup(name="Raw Segments", show=False, overlay=True)
    for line in matched_lines:
        if line and len(line.coords) > 1:
            # Folium은 (lat, lon) 순서, Shapely는 (x=lon, y=lat) 순서임에 주의
            folium.PolyLine(
                [(y, x) for x, y in line.coords], 
                color='gray', weight=1, opacity=0.3
            ).add_to(grp_matched)
    grp_matched.add_to(m)

    # ---------------------------------------------------------
    # 레이어 2: Spatial Groups (메인 결과)
    # 공간적으로 연결된 경로 그룹별로 다른 색상을 입혀서 표시합니다.
    # ---------------------------------------------------------
    if grouped_lines:
        grp_merged = FeatureGroup(name=f"Spatial Groups ({len(grouped_lines)})", show=True, overlay=True)
        
        # 색상 팔레트 (선명한 색상 순환)
        colors = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#46F0F0"]
        cycle = itertools.cycle(colors)
        
        for i, group in enumerate(grouped_lines):
            c = next(cycle) # 그룹마다 색상 변경
            for line in group:
                folium.PolyLine(
                    [(lat, lon) for lon, lat in line.coords], 
                    color=c, weight=6, opacity=0.7,
                    popup=f"Group {i}"
                ).add_to(grp_merged)
        grp_merged.add_to(m)

    # ---------------------------------------------------------
    # 레이어 3: LOF Points (보간된 점)
    # LOF 모델에 입력으로 들어갈 점들을 검은색 원으로 표시합니다.
    # ---------------------------------------------------------
    if interp_points:
        grp_interp = FeatureGroup(name="LOF Points", show=True, overlay=True)
        for (lat, lon) in interp_points:
            folium.CircleMarker(
                [lat, lon], 
                radius=3, 
                color='black',       # 테두리
                fill=True, 
                fill_color='black',  # 내부 채움
                fill_opacity=1
            ).add_to(grp_interp)
        grp_interp.add_to(m)
    
    # 레이어 컨트롤 추가 (체크박스로 껐다 켰다 가능하게 함)
    folium.LayerControl().add_to(m)
    
    return m