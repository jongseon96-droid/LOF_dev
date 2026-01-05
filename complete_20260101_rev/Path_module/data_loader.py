# data_loader.py
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List, Dict, Tuple
try:
    import config as cfg
except ImportError:
    from . import config as cfg

# ==================================================================
# 📂 1. 리전(Region) 데이터 로드
# ==================================================================
def load_regions(reg_path: str) -> pd.DataFrame:
    """
    미리 정의된 '머무름 지역(Stay Regions)' CSV 파일을 로드합니다.
    - OSMnx 그래프를 전역에서 다 로드하지 않고, 필요한 지역만 부분 로드하기 위해 사용됩니다.
    - 반환값: region_id, mean_lat, mean_lon 컬럼을 가진 데이터프레임
    """
    df = pd.read_csv(reg_path)
    df.columns = df.columns.str.strip() # 컬럼명 공백 제거 (안전장치)
    
    # 필수 컬럼만 선택하고 결측치 제거
    return df[['region_id', 'mean_lat', 'mean_lon']].dropna().reset_index(drop=True)

# ==================================================================
# 📍 2. 가장 가까운 리전 매핑
# ==================================================================
def assign_nearest_region_id(reg_df: pd.DataFrame, lat: float, lon: float) -> int:
    """
    주어진 좌표(lat, lon)에서 가장 가까운 리전(Region)의 ID를 찾습니다.
    - 경로의 시작점이 어느 리전에 속하는지 판별하여, 해당 리전의 지도만 로드합니다.
    - geopy.distance.geodesic을 사용하여 지구 곡면 거리를 계산합니다.
    """
    # 현재 좌표와 모든 리전 중심점 간의 거리 계산
    dists = reg_df.apply(lambda r: geodesic((lat, lon), (r['mean_lat'], r['mean_lon'])).meters, axis=1)
    
    # 가장 거리가 짧은 리전의 인덱스 찾기
    idx = int(dists.idxmin())
    return int(reg_df.loc[idx, 'region_id'])

# ==================================================================
# 👣 3. 경로(Waypoint) 추출 및 중복 제거
# ==================================================================
def extract_waypoints_by_segment(df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """
    Raw GPS 데이터에서 각 세그먼트별로 이동 경로(좌표 리스트)를 추출합니다.
    - 연속된 중복 좌표(정지 상태 등)를 제거하여 맵매칭 노이즈를 줄입니다.
    - 반환값: {'segment_id': [(lon, lat), (lon, lat), ...]} 형태의 딕셔너리
    """
    all_waypoints = {}
    grouped = df.groupby('segment_id', sort=False)
    
    for seg_id, seg_df in grouped:
        # (경도, 위도) 순서로 추출 (OSMnx/GeoJSON 표준)
        waypoints_raw = seg_df[['longitude', 'latitude']].values.tolist()
        unique = []
        
        if waypoints_raw:
            unique.append(tuple(waypoints_raw[0]))
            # 이전 좌표와 동일하면 건너뛰고, 다를 때만 추가 (Dedup)
            for pt in waypoints_raw[1:]:
                if tuple(pt) != unique[-1]:
                    unique.append(tuple(pt))
                    
        all_waypoints[str(seg_id)] = unique
    return all_waypoints

# ==================================================================
# ⚙️ 4. 데이터 전처리, 속도 계산 및 필터링 (핵심 로직)
# ==================================================================
def process_data_and_extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    세그먼트별 메타데이터(거리, 속도 등)를 계산하고, 이상 데이터(GPS 튐)를 필터링합니다.
    """
    # 각 세그먼트의 첫 번째 행(Start Point)만 가져와서 메타데이터 분석
    df_meta = df.sort_values('timestamp').groupby('segment_id').head(1).reset_index(drop=True)
    features_list = []
    valid_segment_ids = []

    for _, row in df_meta.iterrows():
        start_coord = (row['from_lat'], row['from_lon'])
        end_coord   = (row['to_lat'], row['to_lon'])
        
        # 시작점-끝점 직선 거리(Geodesic) 계산
        distance_m  = geodesic(start_coord, end_coord).meters
        duration_sec = row.get('duration_sec', np.nan)
        
        # 0으로 나누기 오류 방지 (시간이 0.01초 미만이면 0.01로 보정)
        duration_sec_calc = duration_sec if (not pd.isna(duration_sec) and duration_sec > 0.01) else 0.01
        
        # 시속(km/h) 환산: (거리 / 시간) * 3.6
        speed_kmh = (distance_m / duration_sec_calc) * 3.6

        # 🚨 필터링 로직:
        # 1. 속도가 비정상적으로 빠르거나 (MAX_GROUND_SPEED_KPH 초과)
        # 2. 이동 거리가 지역 범위를 벗어날 정도로 길면 (MIN_NON_LOCAL_DISTANCE_M 초과)
        # -> GPS 오류(튀는 현상)로 간주하고 제외함.
        if (speed_kmh > cfg.MAX_GROUND_SPEED_KPH) and (distance_m > cfg.MIN_NON_LOCAL_DISTANCE_M):
            print(f"⚠️ 세그먼트 {row['segment_id']} 제외: 비정상 속도({speed_kmh:.1f}km/h) 혹은 거리({distance_m:.0f}m)")
            continue

        valid_segment_ids.append(row['segment_id'])
        features_list.append({
            'segment_id': row['segment_id'], 
            'from_lat': row['from_lat'],
            'from_lon': row['from_lon'],
            'to_lat': row['to_lat'],
            'to_lon': row['to_lon'],
            'calculated_distance_m': distance_m,
            'speed_kmh': speed_kmh,
        })

    print(f"   > 유효 세그먼트 ID 수: {len(valid_segment_ids)}")
    # 필터링된 정보와 원본 데이터프레임 반환
    return pd.DataFrame(features_list), df.copy()

# ==================================================================
# 📥 5. Raw 데이터 로드 및 정렬
# ==================================================================
def get_sorted_paths(path_file):
    """
    원본 경로 CSV 파일을 읽어서 세그먼트 ID와 시간 순서대로 정렬합니다.
    - 정렬이 되어 있어야 Waypoint 추출이나 보간 작업이 올바르게 수행됩니다.
    """
    df = pd.read_csv(path_file)
    df.columns = df.columns.str.strip()
    
    # ID는 문자열로 통일 (매칭 오류 방지)
    df['segment_id'] = df['segment_id'].astype(str)
    
    # duration_sec 컬럼이 없으면 0.0으로 초기화
    if 'duration_sec' not in df.columns:
        df['duration_sec'] = 0.0
        
    # segment_id(그룹별) -> timestamp(시간순) 정렬
    return df.sort_values(['segment_id', 'timestamp']).reset_index(drop=True)