# sampler.py
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Polygon, Point

# 👇 [여기 수정]
try:
    from . import config
except ImportError:
    import config

class PointSampler:
    def sample_from_polygons(self, poly_df):
        """
        [핵심 기능]
        입력받은 폴리곤(영역)의 면적에 비례하여,
        영역 내부에 무작위 점(Random Points)들을 생성합니다.
        """
        geoms = []
        
        # 1. JSON 문자열 -> Shapely Polygon 객체로 변환
        for _, r in poly_df.iterrows():
            # CSV에는 '[[lat, lon], [lat, lon]...]' 형태의 문자열로 저장되어 있음
            coords = json.loads(r['polygon_latlon'])
            
            # [중요] 좌표 순서 변경 (Lat, Lon -> Lon, Lat)
            # - 우리가 쓰는 GPS는 (위도, 경도) 순서지만,
            # - GIS 라이브러리(GeoPandas/Shapely)는 수학적 (x, y) 순서인 (경도, 위도)를 사용합니다.
            lonlat = [(c[1], c[0]) for c in coords] 
            geoms.append(Polygon(lonlat))
            
        # 2. GeoDataFrame 생성 및 좌표계 변환 (EPSG:4326 -> EPSG:3857)
        # - EPSG:4326 (위경도): 각도 단위라 '면적(m²)' 계산이 불가능합니다.
        # - EPSG:3857 (Web Mercator): 미터(m) 단위라 정확한 '면적' 계산이 가능합니다.
        gdf = gpd.GeoDataFrame(poly_df, geometry=geoms, crs="EPSG:4326").to_crs(epsg=3857)
        
        sample_rows = []
        for _, row in gdf.iterrows():
            poly = row.geometry
            
            # 3. 생성할 점의 개수 결정 (Density)
            # config.AREA_PER_POINT(예: 800m²) 당 1개의 점을 찍도록 계산
            # 영역이 넓을수록 더 많은 점을 생성하여 '밀도'를 유지함
            n_points = max(1, int(poly.area / config.AREA_PER_POINT))
            
            # 4. 리젝션 샘플링 (Rejection Sampling) - 일명 '다트 던지기'
            # 불규칙한 다각형 내부에 랜덤 점을 찍기 위한 가장 단순하고 확실한 방법
            minx, miny, maxx, maxy = poly.bounds # 다각형을 감싸는 사각형(Box) 범위 추출
            added = 0
            
            while added < n_points:
                # 4-1. 사각형 범위 내에서 랜덤 좌표(x, y) 생성
                rx = np.random.uniform(minx, maxx)
                ry = np.random.uniform(miny, maxy)
                p = Point(rx, ry)
                
                # 4-2. 생성된 점이 실제 다각형 '안에' 있는지 검사
                # 사각형 안에는 있지만 다각형 밖인 빈 공간에 찍힌 점은 버림 (Reject)
                if poly.contains(p):
                    # 5. 좌표계 복원 (EPSG:3857 -> EPSG:4326)
                    # 저장할 때는 다시 표준 위경도(Lat, Lon)로 바꿔야 지도에 찍을 수 있음
                    p_ll = gpd.GeoSeries([p], crs=3857).to_crs(4326).iloc[0]
                    
                    sample_rows.append({
                        "region_id": row['region_id'],
                        "latitude": p_ll.y,  # 위도
                        "longitude": p_ll.x  # 경도
                    })
                    added += 1
                    
        return pd.DataFrame(sample_rows)