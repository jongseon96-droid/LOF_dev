import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import streamlit.components.v1 as components
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from shapely.ops import unary_union
from shapely.geometry import MultiLineString, LineString

# =========================================================
# 🔑 카카오 API 키 설정
# =========================================================
KAKAO_API_KEY = "c35cce22633084bc711c74ef0696d1cc"

# =========================================================
# ⚙️ 설정 및 캐시 로드
# =========================================================
st.set_page_config(page_title="LOF Trajectory Dashboard (Kakao)", layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
path_case_1 = os.path.join(CURRENT_DIR, "analysis_cache.pkl")
path_case_2 = os.path.join(CURRENT_DIR, "Integration_module", "analysis_cache.pkl")

if os.path.exists(path_case_1):
    CACHE_FILE = path_case_1
elif os.path.exists(path_case_2):
    CACHE_FILE = path_case_2
else:
    CACHE_FILE = path_case_2 

@st.cache_data
def load_analysis_data(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data

# =========================================================
# 🎨 시각화 유틸리티
# =========================================================
def get_lof_color_hex(score, threshold):
    if score <= threshold:
        return '#008000' # Green
    else:
        ratio = min(1.0, (score - threshold) / 2.0)
        G = int(255 * (1 - ratio))
        return '#%02x%02x%02x' % (255, G, 0)

# =========================================================
# 🧠 LOF 재계산 로직
# =========================================================
def calculate_realtime_lof(train_df, test_points, n_neighbors):
    if len(train_df) == 0 or len(test_points) == 0:
        return np.array([])

    X_train = train_df[['latitude', 'longitude']].values
    X_test = np.array(test_points)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train_scaled)

    decision_scores = lof.decision_function(X_test_scaled)
    lof_scores = 1.0 - decision_scores
    
    return lof_scores

# =========================================================
# 🇰🇷 카카오지도 HTML 생성기
# =========================================================
def generate_kakao_html(center_lat, center_lon, regions_json, lines_json, points_json, new_path_json):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_API_KEY}"></script>
        <script>
            var container = document.getElementById('map');
            var options = {{
                center: new kakao.maps.LatLng({center_lat}, {center_lon}),
                level: 4
            }};
            var map = new kakao.maps.Map(container, options);

            var regions = {regions_json};
            var lines = {lines_json};
            var points = {points_json};
            var newPath = {new_path_json}; // 새로 추가된 경로

            // 1. Regions //머무름 지역 (Regions) 색상 바꾸기
            if (regions) {{
                regions.forEach(function(reg) {{
                    var path = [];
                    reg.coords.forEach(function(c) {{
                        path.push(new kakao.maps.LatLng(c[0], c[1])); 
                    }});

                    var polygon = new kakao.maps.Polygon({{
                        path: path,
                        strokeWeight: 1,
                        strokeColor: '#55A546', // [변경] 테두리 색 (현재: 파랑) -> 기본 색상: #3388ff
                        strokeOpacity: 1,
                        strokeStyle: 'solid',
                        fillColor: '#55A546', // [변경] 채우기 색 (현재: 파랑) -> 기본 색상: #3388ff
                        fillOpacity: 0.2 // [변경] 투명도 (0~1)
                    }});
                    polygon.setMap(map);
                }});
            }}

            // 2. Lines (Merged Geometry) //이동 경로 (Path Lines) 색상 바꾸기
            if (lines) {{
                lines.forEach(function(lineGroup) {{
                    lineGroup.forEach(function(segment) {{
                        var path = [];
                        segment.forEach(function(c) {{
                            path.push(new kakao.maps.LatLng(c[0], c[1]));
                        }});

                        var polyline = new kakao.maps.Polyline({{
                            path: path,
                            strokeWeight: 3, // [변경] 선 두께
                            strokeColor: '#55A546', // [변경] 선 색상 (현재: 파랑) -> 기본 색상: #0074D9
                            strokeOpacity: 0.5,
                            strokeStyle: 'solid'
                        }});
                        polyline.setMap(map);
                    }});
                }});
            }}
            
            // 🆕 새로 추가된 경로 (빨간 점선으로 표시)
            if (newPath && newPath.length > 1) {{
                 var path = [];
                 newPath.forEach(function(c) {{
                    path.push(new kakao.maps.LatLng(c[0], c[1]));
                 }});
                 
                 var newPolyline = new kakao.maps.Polyline({{
                    path: path,
                    strokeWeight: 5,
                    strokeColor: '#FF0000', // [변경] 추가된 경로 색 (현재: 빨강)
                    strokeOpacity: 0.8,
                    strokeStyle: 'shortdash' // [변경] 선 스타일 ('solid', 'dashed', 'shortdash' 등)
                }});
                newPolyline.setMap(map);
            }}

            // 3. Points (LOF Scores)
            if (points) {{
                points.forEach(function(p) {{
                    var circle = new kakao.maps.Circle({{
                        center : new kakao.maps.LatLng(p.lat, p.lon),
                        radius: 3, 
                        strokeWeight: 1,
                        strokeColor: '#000000',
                        strokeOpacity: 0.5,
                        fillColor: p.color,
                        fillOpacity: p.is_anomaly ? 0.9 : 0.6 
                    }});
                    circle.setMap(map);

                    var infowindow = new kakao.maps.InfoWindow({{
                        content: '<div style="padding:5px;font-size:12px;">Score: ' + p.score.toFixed(2) + '</div>'
                    }});
                    kakao.maps.event.addListener(circle, 'mouseover', function() {{
                        infowindow.open(map, circle);
                    }});
                    kakao.maps.event.addListener(circle, 'mouseout', function() {{
                        infowindow.close();
                    }});
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
    return html_code

# =========================================================
# 🖥️ 메인 앱 UI
# =========================================================
def main():
    st.title("🛰️ LOF Trajectory Dashboard (Kakao Map)")

    # 0. Session State 초기화 (새 데이터 저장용)
    if 'new_path_data' not in st.session_state:
        st.session_state.new_path_data = [] # [[lat, lon], ...]
    if 'new_stay_data' not in st.session_state:
        st.session_state.new_stay_data = [] # [[lat, lon], ...]

    data = load_analysis_data(CACHE_FILE)
    if data is None:
        st.error(f"❌ 분석 캐시 파일({CACHE_FILE})을 찾을 수 없습니다.")
        return

    original_results = data.get('original_path_results', {})
    region_data = data.get('region_data', {})
    
    poly_df = region_data.get('poly_df', pd.DataFrame())
    region_sample_df = region_data.get('region_sample_df', pd.DataFrame())
    path_points = original_results.get('interp_points', [])
    #center_coords = original_results.get('center_coords', [37.5, 127.0])
    center_coords = [37.557846, 127.045549]
    grouped_lines = original_results.get('final_grouped_lines', [])

    # =================================================
    # 📝 [신규 기능] 데이터 시뮬레이션 패널
    # =================================================
    st.sidebar.header("➕ Add New Data (Simulation)")
    
    with st.sidebar.form("add_data_form"):
        data_type = st.radio("데이터 타입 선택", ["Path Point (이동 경로)", "Stay Point (정상 구역)"])
        # 기본값을 지도 중심점으로 설정
        in_lat = st.number_input("위도 (Lat)", value=center_coords[0], format="%.6f")
        in_lon = st.number_input("경도 (Lon)", value=center_coords[1], format="%.6f")
        
        submitted = st.form_submit_button("데이터 추가")
        if submitted:
            if data_type == "Path Point (이동 경로)":
                st.session_state.new_path_data.append([in_lat, in_lon])
                st.success("Path Point 추가됨!")
            else:
                st.session_state.new_stay_data.append([in_lat, in_lon])
                st.success("Stay Point (Normal) 추가됨!")

    # 데이터 초기화 버튼
    if st.sidebar.button("추가 데이터 초기화"):
        st.session_state.new_path_data = []
        st.session_state.new_stay_data = []
        st.rerun()

    # =================================================
    # 🔧 파라미터 설정
    # =================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Parameters")
    n_neighbors = st.sidebar.slider("LOF Neighbors (k)", 5, 100, 30, 5)
    lof_threshold = st.sidebar.slider("Anomaly Threshold", 1.0, 2.0, 1.2, 0.05)
    
    st.sidebar.markdown("---")
    show_regions = st.sidebar.checkbox("Show Regions", True)
    show_lines = st.sidebar.checkbox("Show Path Lines", True)   
    show_points = st.sidebar.checkbox("Show LOF Points", False)

    # =================================================
    # 🚀 데이터 병합 및 LOF 계산
    # =================================================
    
    # 1. 학습 데이터 (Train): 기존 Region + 기존 Path + [NEW] 신규 Stay Data
    #    (Stay Point를 추가하면 해당 지역이 '정상'으로 학습됨 -> LOF 점수가 낮아짐)
    base_train = region_sample_df[['latitude', 'longitude']].values.tolist() + path_points
    if st.session_state.new_stay_data:
        base_train += st.session_state.new_stay_data
    
    train_df = pd.DataFrame(base_train, columns=['latitude', 'longitude'])

    # 2. 분석 대상 (Test): 기존 Path + [NEW] 신규 Path Data
    #    (Path Point를 추가하면 그 점이 이상치인지 평가받음)
    target_points = path_points + st.session_state.new_path_data
    
    with st.spinner('Calculating LOF Scores (with new data)...'):
        scores = calculate_realtime_lof(train_df, target_points, n_neighbors)

    # --- 통계 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Stats")
    st.sidebar.info(f"**Max Score:** {np.max(scores):.2f}")
    cnt = np.sum(scores > lof_threshold)
    st.sidebar.warning(f"**Anomalies:** {cnt} / {len(scores)}")
    
    if len(st.session_state.new_path_data) > 0:
        st.sidebar.markdown(f"📍 **Added Path Points:** {len(st.session_state.new_path_data)}")
    if len(st.session_state.new_stay_data) > 0:
        st.sidebar.markdown(f"🏠 **Added Stay Points:** {len(st.session_state.new_stay_data)}")

    # =================================================
    # 🗺️ 시각화 데이터 준비
    # =================================================
    regions_data = []
    if show_regions and not poly_df.empty:
        for _, r in poly_df.iterrows():
            coords = json.loads(r['polygon_latlon'])
            regions_data.append({"id": r['region_id'], "coords": coords})
    
    lines_data = []
    if show_lines and grouped_lines:
        all_lines_flat = [line for group in grouped_lines for line in group]
        merged_geom = unary_union(all_lines_flat)
        
        final_lines_list = []
        if isinstance(merged_geom, MultiLineString):
            final_lines_list = list(merged_geom.geoms)
        elif isinstance(merged_geom, LineString):
            final_lines_list = [merged_geom]
            
        merged_group_coords = []
        for line in final_lines_list:
            coords = [[lat, lon] for lon, lat in line.coords]
            merged_group_coords.append(coords)
        lines_data.append(merged_group_coords)

    points_data = []
    if show_points and len(target_points) > 0:
        for i, (lat, lon) in enumerate(target_points):
            s = scores[i]
            points_data.append({
                "lat": lat, "lon": lon,
                "score": float(s),
                "color": get_lof_color_hex(s, lof_threshold),
                "is_anomaly": bool(s > lof_threshold)
            })

    regions_json = json.dumps(regions_data)
    lines_json = json.dumps(lines_data)
    points_json = json.dumps(points_data)
    new_path_json = json.dumps(st.session_state.new_path_data) # 신규 경로 시각화용

    # --- 화면 표시 ---
    html_map = generate_kakao_html(
        center_coords[0], center_coords[1], 
        regions_json, lines_json, points_json, new_path_json
    )
    components.html(html_map, height=800)

if __name__ == "__main__":
    main()