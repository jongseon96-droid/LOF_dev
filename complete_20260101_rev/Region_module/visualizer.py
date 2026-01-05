# visualizer.py
import folium
import json

# 👇 [여기 수정]
try:
    from . import config
except ImportError:
    import config
class MapVisualizer:
    def __init__(self, center_lat, center_lon):
        self.m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
    def add_stay_points(self, df):
        """
        [수정] 외곽점(Hull)은 빨간색, 내부점은 검은색으로 표시
        """
        fg = folium.FeatureGroup(name="Stay Points (Raw)")
        
        # 'is_hull' 컬럼이 없으면 에러 방지를 위해 False로 채움
        if 'is_hull' not in df.columns:
            df['is_hull'] = False
            
        for _, r in df.iterrows():
            # 외곽점이면 빨간색, 아니면 검은색
            color = 'red' if r['is_hull'] else 'black'
            # 외곽점은 조금 더 눈에 띄게 (반경 3), 내부는 작게 (반경 2)
            radius = 3 if r['is_hull'] else 2
            fill_opacity = 0.8 if r['is_hull'] else 0.3
            
            folium.CircleMarker(
                [r['centroid_lat'], r['centroid_lon']], 
                radius=radius, 
                color=color,     # 색상 차별화
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                popup=f"Hull Point" if r['is_hull'] else "Inner Point"
            ).add_to(fg)
            
        fg.add_to(self.m)
        
    def add_regions(self, df):
        """
        [수정] 툴팁에 버퍼 확장 정보 표시
        """
        fg = folium.FeatureGroup(name="Regions (Polygon)")
        
        for _, r in df.iterrows():
            coords = json.loads(r['polygon_latlon'])
            
            # 툴팁 내용 구성 (HTML 태그 사용 가능)
            # <br>은 줄바꿈입니다.
            popup_content = (
                f"<b>Region ID: {r['region_id']}</b><br>"
                f"Visits: {r['visit_count']}<br>"
                f"-----------------<br>"
                f"Base Buffer: {config.EXTRA_BUFFER}m<br>"
                f"Visit Bonus: +{r.get('buffer_added_m', 0)}m<br>"
                f"<b>Total Buffer: {r.get('total_buffer_m', 0)}m</b>"
            )
            
            folium.Polygon(
                locations=coords, color='blue', fill=True, fill_opacity=0.3, stroke=False,
                popup=folium.Popup(popup_content, max_width=250) # 툴팁 적용
            ).add_to(fg)
            
            folium.Marker(
                [r['mean_lat'], r['mean_lon']], icon=folium.Icon(color='red', icon='flag')
            ).add_to(fg)
            
        fg.add_to(self.m)

    def add_samples(self, df):
        fg = folium.FeatureGroup(name="Sample Points")
        for _, r in df.iterrows():
            folium.CircleMarker(
                [r['latitude'], r['longitude']], radius=2, color='blue', fill=True, fill_opacity=0.5
            ).add_to(fg)
        fg.add_to(self.m)
        
    def save(self, path):
        folium.LayerControl().add_to(self.m)
        self.m.save(path)
        print(f"Map saved to {path}")