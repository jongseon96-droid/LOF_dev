#integrated_viz.py
import folium
import json
from tqdm import tqdm
import pandas as pd

def get_lof_color_standard(score):
    """LOF Score 1.2를 기준으로 색상 반환"""
    if score <= 1.2:
        return '#008000' # Green
    else:
        # score가 커질수록 빨간색에 가깝게 변함
        ratio = min(1.0, (score - 1.2) / 1.8) 
        G = int(255 * (1 - ratio))
        return '#%02x%02x%02x' % (255, G, 0)

class IntegratedVisualizer:
    def __init__(self, center_lat, center_lon):
        # 기본 맵 생성 (OpenStreetMap 고정)
        self.m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=14,
            tiles='OpenStreetMap',
            control_scale=True
        )

    def add_raw_points(self, df):
        """1. Actual GPS Points (Raw) 레이어 - 들여쓰기 및 컬럼 대응 수정"""
        fg = folium.FeatureGroup(name="Actual GPS Points (Raw)", show=False) 
        
        # 컬럼명 유연하게 대응
        lat_col = 'centroid_lat' if 'centroid_lat' in df.columns else 'latitude'
        lon_col = 'centroid_lon' if 'centroid_lon' in df.columns else 'longitude'
        
        for _, row in df.iterrows():
            is_hull = row.get('is_hull', False)
            color = 'red' if is_hull else 'green'
            
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]], 
                radius=2 if is_hull else 1.5,
                color=color,
                fill=True,
                fill_opacity=0.8,
                popup="Hull Point" if is_hull else "Inner Point"
            ).add_to(fg)
        fg.add_to(self.m)

    def add_regions_from_module(self, region_df):
        """2. Stay Regions 레이어"""
        fg = folium.FeatureGroup(name="Stay Regions", show=True)
        for _, r in region_df.iterrows():
            coords = json.loads(r['polygon_latlon'])
            folium.Polygon(
                locations=coords, color='#3388ff', fill=True, fill_opacity=0.2, weight=1,
                popup=f"Region ID: {r['region_id']}"
            ).add_to(fg)
        fg.add_to(self.m)

    def add_final_path_chunks(self, grouped_lines, layer_name="Path Matching (Blue)"):
            """도로 노드를 따라 연결된 청크 선들을 그리는 핵심 함수"""
            fg = folium.FeatureGroup(name=layer_name, show=True)
            
            if not grouped_lines:
                print(f"⚠️ {layer_name}: 표시할 선 데이터가 없습니다.")
                return

            # 리스트 깊이에 상관없이 모든 LineString을 추출하여 그리기
            def recursive_add_line(data):
                if isinstance(data, list):
                    for item in data:
                        recursive_add_line(item)
                elif data is not None and not data.is_empty:
                    # 🚨 핵심: (lon, lat) -> (lat, lon) 변환은 필수입니다.
                    coords = [(lat, lon) for lon, lat in data.coords]
                    folium.PolyLine(
                        locations=coords, 
                        color='#0074D9', 
                        weight=5,  # 굵게 표시
                        opacity=0.8,
                        line_cap='round'
                    ).add_to(fg)

            recursive_add_line(grouped_lines)
            fg.add_to(self.m)

    def add_sample_points(self, points_list, layer_name="All Path LOF Scores", default_show=True, lof_scores=None): 
        """4. All Path LOF Scores / Region Area Samples 레이어"""
        display_name = "All Path LOF Scores" if "LOF" in layer_name else "Region Area Samples (Background)"
        fg = folium.FeatureGroup(name=display_name, show=default_show) 

        is_lof = lof_scores is not None and "LOF" in layer_name
        for i, (lat, lon) in enumerate(points_list):
            if is_lof:
                score = lof_scores[i]
                color = get_lof_color_standard(score)
                radius, opacity = 4, 1.0
                tooltip = f"LOF Score: {score:.2f}"
            else:
                color, radius, opacity = 'red', 2, 0.4
                tooltip = "Region Sample"

            folium.CircleMarker(
                location=[lat, lon], radius=radius, color=color, 
                fill=True, fill_color=color, fill_opacity=opacity, 
                tooltip=tooltip
            ).add_to(fg)
        fg.add_to(self.m)

    def save(self, path):
        folium.LayerControl(collapsed=False).add_to(self.m) 
        self.m.save(path)
        print(f"✅ Integrated Map saved: {path}")