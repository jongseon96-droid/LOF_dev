#utils.py
import numpy as np
from shapely.geometry import Point
from shapely import affinity # 도형의 회전, 크기 조절, 이동을 위한 모듈

# 지구 반지름 (단위: 미터) - 좌표 변환 공식에 사용됨
R_EARTH = 6371008.8

def ll_to_local_m(lat, lon, lat0, lon0):
    """
    [좌표 변환: 위경도(Lat/Lon) -> 로컬 미터(x, y)]
    
    지구는 둥글기 때문에 위경도 좌표계에서는 정확한 거리(m) 계산이나 
    도형 그리기(원, 타원 등)가 어렵습니다. (위도에 따라 경도 1도의 거리가 달라짐)
    
    따라서 특정 기준점(lat0, lon0)을 원점(0,0)으로 잡고,
    해당 지점 주변을 평면이라고 가정하여 미터(m) 단위 좌표로 변환합니다.
    """
    # 1. 각도(도)를 라디안으로 변환 (수학 공식을 위해)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    
    # 2. 변환 공식 (Equirectangular Projection 근사)
    # x축(동서 방향): 경도 차이 * 코사인(위도) * 지구반지름
    # -> 위도가 높아질수록 경도 간 거리가 좁아지는 것을 cos(lat)로 보정
    x = (lon_r - lon0_r) * np.cos(lat0_r) * R_EARTH
    
    # y축(남북 방향): 위도 차이 * 지구반지름
    y = (lat_r - lat0_r) * R_EARTH
    
    return x, y

def local_m_to_ll(x, y, lat0, lon0):
    """
    [좌표 변환: 로컬 미터(x, y) -> 위경도(Lat/Lon)]
    
    평면 좌표계에서 계산(Buffer, 회전 등)이 끝난 도형을
    다시 지도(Map)에 올리기 위해 위경도 좌표로 되돌립니다.
    (위의 함수와 정확히 반대되는 연산 수행)
    """
    lat0_r = np.radians(lat0)
    
    # y(m)를 다시 위도(도)로 변환
    lat = lat0 + np.degrees(y / R_EARTH)
    
    # x(m)를 다시 경도(도)로 변환 (역산 시에도 cos 보정 필요)
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(lat0_r)))
    
    return lat, lon

def create_ellipse(p1, p2, buffer_m, resolution=32):
    """
    [도형 생성: 두 점을 잇는 타원(Ellipse)]
    
    점이 2개일 때, 단순 원(Circle)보다는 두 점의 방향성을 반영한 
    '길쭉한 타원'을 만드는 것이 더 합리적입니다.
    
    Shapely에는 '타원 그리기' 함수가 없으므로,
    원(Circle)을 만든 뒤 -> 늘리고(Scale) -> 회전시켜서(Rotate) 만듭니다.
    """
    # 1. 두 점의 좌표 추출
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    
    # 2. 중심점(Midpoint) 계산: 타원의 중심이 될 위치
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    # 3. 두 점 사이의 거리 및 방향 계산
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy) # 피타고라스 정리로 거리 구함
    
    # 4. 타원의 긴 반지름(a)과 짧은 반지름(b) 설정
    # a(장축): 두 점 거리의 절반 + 여유분(10m) + 버퍼
    a = dist / 2.0 + 10.0 + buffer_m 
    # b(단축): 기본 두께(30m) + 버퍼 (너무 얇아지지 않도록 최소폭 보장)
    b = 30.0 + buffer_m              
    
    # 5. 아핀 변환(Affine Transformation)으로 타원 생성
    # Step A: (0,0)에 반지름 1짜리 기본 원 생성
    circ = Point(0, 0).buffer(1.0, resolution=resolution)
    
    # Step B: x축으로 a배, y축으로 b배 늘림 (찌그러뜨려서 타원 만듦)
    ell = affinity.scale(circ, xfact=a, yfact=b)
    
    # Step C: 두 점이 기울어진 각도(angle)만큼 회전시킴
    angle = np.arctan2(dy, dx)
    ell = affinity.rotate(ell, angle, use_radians=True)
    
    # Step D: 계산해둔 중심점(mx, my)으로 이동시킴
    ell = affinity.translate(ell, xoff=mx, yoff=my)
    
    return ell