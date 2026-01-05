import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import sys
import streamlit.components.v1 as components 
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
# =========================================================
# 🛠️ 경로 설정 & 모듈 로드
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

region_module_dir = os.path.join(parent_dir, "Region_module")
path_module_dir = os.path.join(parent_dir, "Path_module")

# 시스템 경로에 추가 (모듈을 찾기 위함)
if parent_dir not in sys.path: sys.path.append(parent_dir)
if region_module_dir not in sys.path: sys.path.append(region_module_dir)
if path_module_dir not in sys.path: sys.path.append(path_module_dir)

try:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    
    # [Region Module Imports]
    from Region_module.processor import RegionProcessor
    from Region_module.sampler import PointSampler
    
    # [Path Module Imports]
    import Path_module.data_loader as path_dl
    import Path_module.graph_manager as path_gm
    import Path_module.matcher as path_matcher
    
    
except ImportError as e:
    st.error(f"❌ 모듈 로드 실패: {e}\n\n폴더 구조와 __init__.py 파일을 확인해주세요.")
    st.stop()

# =========================================================
# 🔑 카카오 API 키
# =========================================================
KAKAO_API_KEY = "c35cce22633084bc711c74ef0696d1cc"

# =========================================================
# ⚙️ 설정 및 데이터 로드
# =========================================================
st.set_page_config(page_title="LOF Dashboard (Real Map Matching)", layout="wide")

path_case_1 = os.path.join(current_dir, "analysis_cache.pkl")
path_case_2 = os.path.join(parent_dir, "analysis_cache.pkl")
CACHE_FILE = path_case_1 if os.path.exists(path_case_1) else (path_case_2 if os.path.exists(path_case_2) else None)

@st.cache_data
def load_analysis_data(cache_path):
    if not cache_path: return None
    with open(cache_path, 'rb') as f: data = pickle.load(f)
    return data

# =========================================================
# 🔄 [핵심] 모듈 실행 함수들 (Region + Path)
# =========================================================

def run_region_module_update(raw_stay_df, new_points):
    """Region 모듈: DBSCAN 재연산"""
    new_df = pd.DataFrame(new_points, columns=['centroid_lat', 'centroid_lon'])
    target_df = raw_stay_df.copy()
    if 'latitude' in target_df.columns:
        target_df.rename(columns={'latitude': 'centroid_lat', 'longitude': 'centroid_lon'}, inplace=True)
    combined_df = pd.concat([target_df[['centroid_lat', 'centroid_lon']], new_df], ignore_index=True)
    
    proc = RegionProcessor(combined_df)
    proc.run_dbscan()
    poly_df = proc.create_polygons()
    sampler = PointSampler()
    sample_df = sampler.sample_from_polygons(poly_df)
    return poly_df, sample_df

def run_path_module_realtime(regions_df, new_points_latlon):
    """
    [Path Module 연동]
    사용자가 찍은 점(Lat, Lon)을 입력받아
    Path_module의 로직(Region찾기 -> 그래프다운 -> 맵매칭)을 수행하여
    도로 위에 매칭된 LineString을 반환합니다.
    """
    if len(new_points_latlon) < 2:
        return None

    # 1. Waypoint 변환: (Lat, Lon) -> (Lon, Lat)
    waypoints = [(p[1], p[0]) for p in new_points_latlon]
    
    # 2. 시작점이 속한 Region 찾기
    start_lat, start_lon = new_points_latlon[0]
    try:
        rid = path_dl.assign_nearest_region_id(regions_df, start_lat, start_lon)
    except Exception as e:
        st.warning(f"리전 찾기 실패 (기본값 사용): {e}")
        rid = regions_df.iloc[0]['region_id'] # Fallback

    # 3. 그래프 캐시 초기화 및 로드
    reg_cache = path_gm.RegionGraphCache(regions_df)
    
    # 4. 맵매칭 수행
    try:
        matched_line = path_matcher.route_between_points_with_fallback(reg_cache, rid, waypoints)
        return matched_line # Shapely LineString 반환
    except Exception as e:
        st.error(f"맵매칭 실패: {e}")
        return None

# =========================================================
# 🇰🇷 카카오지도 HTML 생성
# =========================================================
def generate_kakao_html(center_lat, center_lon, 
                        regions_df, raw_stay_df, 
                        path_lines, new_matched_line, 
                        lof_points, 
                        new_path_points, new_stay_points,
                        vis_options):
    
    # 1. Regions (Polygon)
    regions_data = []
    if vis_options['show_regions'] and not regions_df.empty:
        for _, r in regions_df.iterrows():
            coords = json.loads(r['polygon_latlon'])
            regions_data.append({"path": coords, "info": f"Region {r['region_id']}"})

    # 2. Existing Stay Points
    existing_stay_data = []
    if vis_options['show_exist_stay'] and not raw_stay_df.empty:
        if 'is_hull' not in raw_stay_df.columns: raw_stay_df['is_hull'] = False
        for _, r in raw_stay_df.iterrows():
            lat = r.get('centroid_lat', r.get('latitude'))
            lon = r.get('centroid_lon', r.get('longitude'))
            existing_stay_data.append({"lat": lat, "lon": lon, "is_hull": r['is_hull']})

    # ==============================================================================
    # 🛠️ [기하학적 해결] 3. Path Lines - Union 적용 (겹침 제거)
    # ==============================================================================
    lines_data = []
    if vis_options['show_lines']:
        all_lines_to_merge = []
        
        # (1) 기존 Path Lines 수집 (path_lines는 그룹화된 리스트 구조임)
        if path_lines:
            for group in path_lines:
                for line in group:
                    if line and not line.is_empty:
                        all_lines_to_merge.append(line)
        
        # (2) 새로 매칭된 라인(Simulation)이 있으면 추가
        if new_matched_line and not new_matched_line.is_empty:
            all_lines_to_merge.append(new_matched_line)
            
        # (3) 기하학적 병합 (Unary Union) 수행
        # -> 수천 개의 선을 겹치지 않는 최소한의 선으로 통합합니다.
        if all_lines_to_merge:
            merged_geom = unary_union(all_lines_to_merge)
            
            # 결과가 LineString 하나일 수도 있고, MultiLineString(여러 개)일 수도 있음
            final_lines = []
            if isinstance(merged_geom, LineString):
                final_lines = [merged_geom]
            elif isinstance(merged_geom, MultiLineString):
                final_lines = list(merged_geom.geoms)
            
            # (4) 좌표 변환 (Shapely: lon,lat -> Kakao: lat,lon)
            for line in final_lines:
                coords = [[lat, lon] for lon, lat in line.coords]
                lines_data.append(coords)

    # 4. LOF Points
    points_data = lof_points if vis_options['show_lof'] else []

    # 5. New Stay Points
    new_stay_data = new_stay_points if vis_options['show_new_stay'] else []

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; }}</style>
    </head>
    <body>
        <div id="map"></div>
        <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_API_KEY}"></script>
        <script>
            var container = document.getElementById('map');
            var options = {{ center: new kakao.maps.LatLng({center_lat}, {center_lon}), level: 4 }};
            var map = new kakao.maps.Map(container, options);
            
            // 1. Regions
            var regions = {json.dumps(regions_data)};
            regions.forEach(function(r) {{
                var path = r.path.map(c => new kakao.maps.LatLng(c[0], c[1]));
                new kakao.maps.Polygon({{
                    map: map, path: path,
                    strokeWeight: 1, strokeColor: '#55A546', strokeOpacity: 1, 
                    fillColor: '#55A546', fillOpacity: 0.2 
                }});
            }});
            
            // 2. Existing Stay
            var existStay = {json.dumps(existing_stay_data)};
            existStay.forEach(function(p) {{
                var color = p.is_hull ? '#FF0000' : '#000000';
                var radius = p.is_hull ? 3 : 2;
                new kakao.maps.Circle({{
                    map: map, center: new kakao.maps.LatLng(p.lat, p.lon), radius: radius,
                    strokeColor: color, strokeOpacity: 0.8, fillColor: color, fillOpacity: 0.6
                }});
            }});

            // 3. New Stay
            var newStay = {json.dumps(new_stay_data)};
            newStay.forEach(function(p) {{
                 var marker = new kakao.maps.Marker({{ position: new kakao.maps.LatLng(p[0], p[1]) }});
                 marker.setMap(map);
            }});
            
            // 4. Path Lines
            var lines = {json.dumps(lines_data)};
            lines.forEach(function(linePath) {{
                var path = linePath.map(c => new kakao.maps.LatLng(c[0], c[1]));
                new kakao.maps.Polyline({{
                    map: map, path: path,
                    strokeWeight: 3, strokeColor: '#55A546', strokeOpacity: 0.4 
                }});
            }});
            
            // 5. LOF Points
            var points = {json.dumps(points_data)};
            points.forEach(function(p) {{
                var circle = new kakao.maps.Circle({{
                    map: map, center: new kakao.maps.LatLng(p.lat, p.lon), radius: 5,
                    strokeColor: '#000000', strokeOpacity: 0.5, fillColor: p.color, fillOpacity: 0.9
                }});
            }});
            
            // 6. User Input Points (Red Dashed)
            var newPathPoints = {json.dumps(new_path_points)};
            if (newPathPoints.length > 1) {{
                var path = newPathPoints.map(c => new kakao.maps.LatLng(c[0], c[1]));
                new kakao.maps.Polyline({{
                    map: map, path: path,
                    strokeWeight: 5, strokeColor: '#FF0000', strokeOpacity: 0.8, strokeStyle: 'shortdash'
                }});
            }}
            
            var mapTypeControl = new kakao.maps.MapTypeControl();
            map.addControl(mapTypeControl, kakao.maps.ControlPosition.TOPRIGHT);
            var zoomControl = new kakao.maps.ZoomControl();
            map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);
        </script>
    </body>
    </html>
    """
    return html

# =========================================================
# ⚙️ 기타 로직 (LOF 등)
# =========================================================
def get_lof_color_hex(score, threshold):
    if score <= threshold: return '#008000'
    else:
        ratio = min(1.0, (score - threshold) / 2.0)
        G = int(255 * (1 - ratio))
        return '#%02x%02x%02x' % (255, G, 0)

def calculate_realtime_lof(train_df, test_points, n_neighbors):
    if len(train_df) == 0 or len(test_points) == 0: return np.array([])
    X_train = train_df[['latitude', 'longitude']].values
    X_test = np.array(test_points)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train_scaled)
    return 1.0 - lof.decision_function(X_test_scaled)

# =========================================================
# 🖥️ 메인 앱 UI
# =========================================================
def main():
    st.title("🛰️ LOF Dashboard (Real Map Matching)")
    
    # Session State
    if 'new_path_data' not in st.session_state: st.session_state.new_path_data = [] 
    if 'new_stay_data' not in st.session_state: st.session_state.new_stay_data = [] 
    if 'updated_poly_df' not in st.session_state: st.session_state.updated_poly_df = None
    if 'updated_sample_df' not in st.session_state: st.session_state.updated_sample_df = None
    if 'new_matched_line' not in st.session_state: st.session_state.new_matched_line = None 

    # 데이터 로드
    if CACHE_FILE is None:
        st.error("❌ analysis_cache.pkl 파일을 찾을 수 없습니다.")
        return
    data = load_analysis_data(CACHE_FILE)
    
    original_results = data.get('original_path_results', {})
    region_data = data.get('region_data', {})
    
    # ---------------------------------------------------------
    # 🛠️ [수정됨] CSV 파일 경로 찾기 로직
    # ---------------------------------------------------------
    # Path_module 기준으로 상위(complete) -> 상위(LOF_dev) -> common_csv 순으로 이동해야 함
    # 경로: .../LOF_dev/common_csv/stay_regions.csv
    csv_path = os.path.abspath(os.path.join(path_module_dir, "..", "..", "common_csv", "stay_regions.csv"))
    
    if os.path.exists(csv_path):
        regions_df = path_dl.load_regions(csv_path)
    else:
        # Fallback: 혹시 구조가 다를 경우 한 단계 위도 체크
        csv_path_fallback = os.path.abspath(os.path.join(path_module_dir, "..", "common_csv", "stay_regions.csv"))
        if os.path.exists(csv_path_fallback):
            regions_df = path_dl.load_regions(csv_path_fallback)
        else:
            st.warning(f"⚠️ 'stay_regions.csv' 파일을 찾을 수 없습니다.\n검색 경로: {csv_path}")
            # 파일이 없으면 pkl에서 가져오거나 빈 DF 생성
            if 'regions_df' in original_results:
                regions_df = original_results['regions_df']
            else:
                regions_df = pd.DataFrame()

    init_poly_df = region_data.get('poly_df', pd.DataFrame())
    init_sample_df = region_data.get('region_sample_df', pd.DataFrame())
    raw_stay_df = region_data.get('raw_df', pd.DataFrame())
    
    path_points = original_results.get('interp_points', [])
    center_coords = [37.557846, 127.045549]
    grouped_lines = original_results.get('final_grouped_lines', [])

    # -----------------------------------------------------
    # 👁️ Visibility Settings
    # -----------------------------------------------------
    st.sidebar.header("👁️ Visibility Settings")
    vis_options = {
        'show_regions': st.sidebar.checkbox("Show Regions", True),
        'show_lines': st.sidebar.checkbox("Show Path Lines", True),
        'show_lof': st.sidebar.checkbox("Show LOF Points", False),
        'show_exist_stay': st.sidebar.checkbox("Show Existing Stay Points", False),
        'show_new_stay': st.sidebar.checkbox("Show New Stay Points", False)
    }

    # -----------------------------------------------------
    # 🔧 Parameters & Simulation
    # -----------------------------------------------------
    st.sidebar.markdown("---")
    n_neighbors = st.sidebar.slider("LOF Neighbors (k)", 5, 100, 30, 5)
    lof_threshold = st.sidebar.slider("Anomaly Threshold", 1.0, 2.0, 1.2, 0.05)

    st.sidebar.header("➕ Data Simulation")
    with st.sidebar.form("sim_form"):
        sim_type = st.radio("추가할 데이터 타입", ["Path Point (이동)", "Stay Point (정상구역)"])
        last_pt = path_points[-1] if len(path_points) > 0 else center_coords
        lat_in = st.number_input("Latitude", value=center_coords[0], format="%.5f")
        lon_in = st.number_input("Longitude", value=center_coords[1], format="%.5f")
        
        if st.form_submit_button("데이터 추가"):
            if sim_type == "Path Point (이동)":
                        st.session_state.new_path_data.append([lat_in, lon_in])
                        
                        # 🔥 [수정됨] 기존 경로의 끝점과 새 점들을 연결!
                        if len(path_points) > 0:
                            # 1. 기존 데이터의 마지막 점 가져오기 (End Point)
                            last_existing_point = path_points[-1] # [lat, lon]
                            
                            # 2. [마지막 점] + [새로 찍은 점들]을 합쳐서 매칭 요청
                            # 이렇게 해야 끊어지지 않고 이어집니다.
                            points_to_route = [last_existing_point] + st.session_state.new_path_data
                        else:
                            # 기존 데이터가 하나도 없으면 그냥 새 점만 사용
                            points_to_route = st.session_state.new_path_data

                        # 🔥 [Path Module 사용] 실시간 맵매칭
                        if len(points_to_route) >= 2:
                            with st.spinner("Running OSMnx Map Matching..."):
                                matched = run_path_module_realtime(regions_df, points_to_route)
                                st.session_state.new_matched_line = matched
                                if matched: st.success("기존 경로와 연결 성공! (연두색 실선)")
                                else: st.warning("매칭 실패 (도로를 찾을 수 없음)")
                        else:
                            st.info("점을 하나 더 추가해야 경로가 생성됩니다.")
                            
                        st.rerun()

            else:
                st.session_state.new_stay_data.append([lat_in, lon_in])
                # [Region Module 사용]
                with st.spinner("Running Region Update..."):
                    new_poly, new_sample = run_region_module_update(raw_stay_df, st.session_state.new_stay_data)
                    st.session_state.updated_poly_df = new_poly
                    st.session_state.updated_sample_df = new_sample
                st.success("Region 재계산 완료")
                st.rerun()

    if st.sidebar.button("데이터 초기화"):
        st.session_state.new_path_data = []
        st.session_state.new_stay_data = []
        st.session_state.updated_poly_df = None
        st.session_state.updated_sample_df = None
        st.session_state.new_matched_line = None
        st.rerun()

    # 🔄 데이터 준비
    current_poly_df = st.session_state.updated_poly_df if st.session_state.updated_poly_df is not None else init_poly_df
    current_sample_df = st.session_state.updated_sample_df if st.session_state.updated_sample_df is not None else init_sample_df
    
    # LOF 계산
    base_train = current_sample_df[['latitude', 'longitude']].values.tolist() + path_points
    train_df = pd.DataFrame(base_train, columns=['latitude', 'longitude'])
    target_points = path_points + st.session_state.new_path_data
    scores = calculate_realtime_lof(train_df, target_points, n_neighbors)
    
    lof_points_data = []
    if len(target_points) > 0:
        for i, (lat, lon) in enumerate(target_points):
            s = scores[i]
            lof_points_data.append({"lat": lat, "lon": lon, "score": float(s), "color": get_lof_color_hex(s, lof_threshold)})

    # 🇰🇷 HTML 생성
    html_code = generate_kakao_html(
        center_coords[0], center_coords[1],
        current_poly_df,
        raw_stay_df,
        grouped_lines,
        st.session_state.new_matched_line, # 새로 매칭된 라인 전달
        lof_points_data,
        st.session_state.new_path_data,
        st.session_state.new_stay_data,
        vis_options
    )
    
    components.html(html_code, height=800)
    st.markdown("### 📊 Analysis Stats")
    cnt = np.sum(scores > lof_threshold)
    st.info(f"Total Points: {len(scores)} | Anomalies: {cnt} (Threshold: {lof_threshold})")

if __name__ == "__main__":
    main()