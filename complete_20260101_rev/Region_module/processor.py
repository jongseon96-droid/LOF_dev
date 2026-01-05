# [Region_module/processor.py]

import numpy as np
import pandas as pd
import json
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPoint

# 👇 [여기 수정] config와 utils를 불러오는 방식을 안전하게 변경
try:
    # app.py에서 실행할 때 (패키지로 인식됨)
    from . import config
    from . import utils
except ImportError:
    # 직접 실행할 때 (같은 폴더 파일로 인식됨)
    import config
    import utils

class RegionProcessor:
    def __init__(self, df):
        self.df = df
        self.df['is_hull'] = False 
        
    def run_dbscan(self):
        """DBSCAN을 실행하고, 노이즈 점을 단일 Region으로 처리합니다."""
        coords = self.df[["centroid_lat", "centroid_lon"]].values
        kms_per_rad = 6371.0088
        eps_rad = (config.EPSILON_M / 1000) / kms_per_rad
        
        # 1. DBSCAN 실행
        db = DBSCAN(eps=eps_rad, min_samples=config.MIN_SAMPLES, metric='haversine', algorithm='ball_tree')
        self.df['region_id'] = db.fit_predict(np.radians(coords))
        
        # 2. 💡 [핵심 수정] 노이즈(-1)로 분류된 점을 고유 Region으로 재할당
        noise_df = self.df[self.df['region_id'] == -1]
        
        if not noise_df.empty:
            # 현재 가장 큰 region_id를 찾아 다음 ID부터 할당합니다.
            max_rid = self.df['region_id'].max()
            new_rid = max(0, max_rid + 1)
                
            # 각 노이즈 포인트를 고유한 region_id로 재할당
            for idx in noise_df.index:
                self.df.at[idx, 'region_id'] = new_rid
                new_rid += 1
                
        return self.df

    def create_polygons(self):
        out_rows = []
        # 노이즈(-1)는 이미 run_dbscan에서 처리되었으므로, 모든 region_id >= 0 입니다.
        groups = self.df[self.df['region_id'] != -1].groupby('region_id')
        
        for rid, g in groups:
            mean_lat = g['centroid_lat'].mean()
            mean_lon = g['centroid_lon'].mean()
            visits = len(g)
            
            # 버퍼 계산
            v_buf = 5.0 * np.log1p(visits)
            total_buf = config.EXTRA_BUFFER + v_buf
            
            pts_local = [Point(utils.ll_to_local_m(r.centroid_lat, r.centroid_lon, mean_lat, mean_lon)) 
                          for _, r in g.iterrows()]
            
            poly = None
            
            if visits >= 3:
                # 3점 이상: Convex Hull 구하고 버퍼 적용
                hull_geom = MultiPoint(pts_local).convex_hull
                poly = hull_geom.buffer(total_buf)
                
                # 외곽점 식별
                for idx, pt in zip(g.index, pts_local):
                    if hull_geom.boundary.distance(pt) < 1e-3:
                        self.df.at[idx, 'is_hull'] = True
                        
            elif visits == 2:
                # 2점: 타원 생성 및 버퍼 적용
                poly = utils.create_ellipse(pts_local[0], pts_local[1], total_buf)
                self.df.loc[g.index, 'is_hull'] = True
                
            else: # visits == 1 (단일 점 처리)
                # 1점: 원형 버퍼 생성 (total_buf만 적용)
                poly = pts_local[0].buffer(total_buf)
                self.df.loc[g.index, 'is_hull'] = True
                
            if poly and not poly.is_empty:
                # 폴리곤 좌표를 Lat/Lon으로 역투영
                coords_local = list(poly.exterior.coords)
                coords_ll = [utils.local_m_to_ll(x, y, mean_lat, mean_lon) for x, y in coords_local]
                out_rows.append({
                    "region_id": rid,
                    "mean_lat": mean_lat, "mean_lon": mean_lon,
                    "visit_count": visits,
                    "polygon_latlon": json.dumps([list(c) for c in coords_ll]),
                    
                    "buffer_added_m": round(v_buf, 2), 
                    "total_buffer_m": round(total_buf, 2) 
                })
                
        return pd.DataFrame(out_rows)