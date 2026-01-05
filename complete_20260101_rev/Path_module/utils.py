# utils.py
import os
import time
import math
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge
from typing import List, Tuple, Optional
try:
    import config as cfg
except ImportError:
    from . import config as cfg

# ==================================================================
# 🕒 1. 시간 및 파일 입출력 유틸
# ==================================================================
def get_current_time_str():
    """현재 시간을 파일명에 붙이기 좋은 형태(_YYYYMMDD_HHMM)로 반환"""
    return datetime.now().strftime("_%Y%m%d_%H%M")

def safe_write_csv(df: pd.DataFrame, path: str, retries: int = 7, base_delay: float = 0.3) -> None:
    """
    Windows 파일 잠금(Lock) 문제를 회피하기 위한 안전한 CSV 저장 함수입니다.
    - 파일을 바로 덮어쓰지 않고 임시 파일(.tmp)을 만든 뒤 교체(Replace)합니다.
    - PermissionError 발생 시 지수 백오프(Exponential Backoff)로 재시도합니다.
    """
    dir_ok = os.path.dirname(path)
    if dir_ok and not os.path.exists(dir_ok):
        os.makedirs(dir_ok, exist_ok=True)
    
    tmp_path = f"{path}.tmp.{os.getpid()}"
    last_err = None
    
    for i in range(retries):
        try:
            df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            os.replace(tmp_path, path) # 원자적(Atomic) 교체
            return
        except PermissionError as e:
            last_err = e
            time.sleep(base_delay * (2 ** i)) # 0.3s, 0.6s, 1.2s ... 대기 시간 증가
        except Exception:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            raise
    raise last_err

# ==================================================================
# 📏 2. 선형 보간(Linear Interpolation) - 단일 라인용
# ==================================================================
def interpolate_linestring_every(line: LineString, step_m: float) -> List[Tuple[float, float]]:
    """
    하나의 LineString을 입력받아, 지정된 간격(step_m)마다 점을 찍습니다.
    - 주로 맵매칭된 개별 경로(세그먼트)를 보간할 때 사용합니다.
    - 직선 거리(Geodesic)를 기준으로 비율을 계산하여 새 좌표를 생성합니다.
    """
    if line.is_empty or len(line.coords) < 2:
        return []
        
    pts = list(line.coords)
    # (lat, lon) 순서로 저장 (Shapely는 lon, lat 순서임에 주의)
    samples = [(pts[0][1], pts[0][0])] 
    
    dist_since = 0.0 # 마지막으로 점을 찍은 후 이동한 거리
    last_lon, last_lat = pts[0]
    
    for i in range(1, len(pts)):
        cur_lon, cur_lat = pts[i]
        # 현재 구간(Segment)의 길이 계산
        seg_len = geodesic((last_lat, last_lon), (cur_lat, cur_lon)).meters
        
        # 현재 구간 내에서 step_m 간격으로 점을 찍을 수 있는지 확인
        while dist_since + seg_len >= step_m:
            need = step_m - dist_since # 다음 점까지 남은 거리
            ratio = (need / seg_len) if seg_len > 0 else 0.0
            
            # 선형 보간 공식: P_new = P_start + (P_end - P_start) * ratio
            interp_lon = last_lon + (cur_lon - last_lon) * ratio
            interp_lat = last_lat + (cur_lat - last_lat) * ratio
            
            samples.append((interp_lat, interp_lon))
            
            # 기준점 업데이트
            last_lon, last_lat = interp_lon, interp_lat
            seg_len -= need
            dist_since = 0.0 # 방금 점을 찍었으므로 누적 거리 초기화
            
        dist_since += seg_len
        last_lon, last_lat = cur_lon, cur_lat
    
    # 마지막 점도 포함 (부동소수점 오차 고려)
    last_pt_lon, last_pt_lat = pts[-1]
    last_sample_lat, last_sample_lon = samples[-1]
    if (abs(last_sample_lon - last_pt_lon) > 1e-7 or abs(last_sample_lat - last_pt_lat) > 1e-7):
        samples.append((last_pt_lat, last_pt_lon))
        
    return samples

# ==================================================================
# 🧬 3. 지오메트리 변환 유틸
# ==================================================================
def _coords_from_lines(geom_list: List[LineString]) -> List[List[Tuple[float, float]]]:
    """
    Shapely LineString 객체 리스트를 순수 파이썬 좌표 리스트로 변환합니다.
    - Shapely: (lon, lat) -> Output: (lat, lon)
    - 연속된 좌표의 중복을 제거하여 깔끔하게 정리합니다.
    """
    all_lines_coords = []
    for ln in geom_list:
        if ln and len(ln.coords) > 1:
            # 위도(lat), 경도(lon) 순서로 뒤집음
            coords = [(coord[1], coord[0]) for coord in ln.coords]
            
            # 연속 중복 제거 (Clean up)
            cleaned = [coords[0]]
            for (lat, lon) in coords[1:]:
                if (lat, lon) != cleaned[-1]:
                    cleaned.append((lat, lon))
            
            if len(cleaned) > 1:
                all_lines_coords.append(cleaned)
    return all_lines_coords

# ==================================================================
# 🔗 4. 연속 보간 (Global Continuous Interpolation) - 통합 모드용
# ==================================================================
def interpolate_continuous_coords_global(lines_coords: List[List[Tuple[float, float]]], step_m: float) -> List[Tuple[float, float]]:
    """
    여러 개의 끊어진 경로(Chunk)들을 받아 보간하되, 
    '연결된 구간' 내에서는 길이를 누적하여 부드럽게 점을 찍습니다.
    - 중요: 끊어진 구간(Gap)은 건너뛰고, 억지로 잇지 않습니다.
    """
    print("🔥 [Continuous Mode] 끊어진 구간은 건너뛰고, 연결된 구간 내에서만 보간합니다...")
    
    final_points = []
    
    # 각 경로 덩어리(Island)별로 순회
    for segment in lines_coords:
        if not segment or len(segment) < 2:
            continue
            
        island_points = [segment[0]] # 시작점
        dist_since = 0.0
        
        # 덩어리 내부를 순회하며 보간
        for i in range(len(segment) - 1):
            p1 = segment[i]
            p2 = segment[i+1]
            
            seg_dist = geodesic(p1, p2).meters
            current_seg_traveled = 0.0
            
            while dist_since + (seg_dist - current_seg_traveled) >= step_m:
                need = step_m - dist_since
                ratio = (current_seg_traveled + need) / seg_dist if seg_dist > 0 else 0
                
                new_lat = p1[0] + (p2[0] - p1[0]) * ratio
                new_lon = p1[1] + (p2[1] - p1[1]) * ratio
                
                island_points.append((new_lat, new_lon))
                current_seg_traveled += need
                dist_since = 0.0
            
            dist_since += (seg_dist - current_seg_traveled)
            
        final_points.extend(island_points)
        
    print(f"   > 총 생성된 포인트: {len(final_points)}개")
    return final_points

# ==================================================================
# 🧩 5. 경로 병합 및 단순화 (Merge & Simplify)
# ==================================================================
def merge_and_simplify_lines(lines: List[LineString]) -> List[List[Tuple[float, float]]]:
    """
    수백 개의 자잘한 경로 조각들을 기하학적으로 병합(Union -> Merge)합니다.
    - 겹치는 구간은 하나로 합쳐지고(Unary Union),
    - 끊어진 구간 중 맞닿아 있는 곳은 하나의 긴 선으로 연결됩니다(Line Merge).
    - 결과: 깔끔하게 정리된 몇 개의 긴 경로(Long Paths)가 반환됩니다.
    """
    print("🔥 중복 경로 제거(Melting) 및 접합(Stitching) 작업 시작...")
    
    # 1. 겹치는 선들을 하나로 융합 (기하학적 합집합)
    merged_geom = unary_union(lines)
    
    # 2. 끝점이 맞닿은 선들을 하나의 긴 선으로 병합
    merged_geom = linemerge(merged_geom)
    
    unique_lines = []
    if isinstance(merged_geom, LineString):
        unique_lines = [merged_geom]
    elif isinstance(merged_geom, MultiLineString):
        unique_lines = list(merged_geom.geoms)
    
    print(f"🔥 통합 완료: {len(lines)}개 조각 -> {len(unique_lines)}개의 긴 경로로 최적화됨.")
    
    # 좌표 리스트 형태로 변환하여 반환
    return _coords_from_lines(unique_lines)