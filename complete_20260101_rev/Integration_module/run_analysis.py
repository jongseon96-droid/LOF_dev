#run_analysis.py
import sys
import os
import pandas as pd
import pickle
import time
from datetime import datetime
from tqdm import tqdm 
import numpy as np

# =========================================================
# 1. 모듈 경로 설정
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 외부 모듈 임포트
from Region_module.processor import RegionProcessor
from Region_module import config as region_config
from Region_module.sampler import PointSampler
from Path_module.main import run_path_analysis 
from Path_module import config as path_config
from integrated_viz import IntegratedVisualizer
from lof_processor import calculate_lof_scores

# 💡 분석 상수
LOF_NORMAL_THRESHOLD = 1.2
CACHE_FILE = "analysis_cache.pkl"

def analyze_path_data(file_name, original_name):
    """Path_module 분석을 실행합니다."""
    path_config.FILE_NAME_PATHS = file_name
    path_results = run_path_analysis() 
    path_config.FILE_NAME_PATHS = original_name
    return path_results

def main():
    total_start_time = time.time()
    original_file_name = path_config.FILE_NAME_PATHS 
    
    # ------------------------------------------------------------------
    # 1. 캐시 확인 및 데이터 로드
    # ------------------------------------------------------------------
    cache_path = os.path.join(current_dir, CACHE_FILE)
    use_cache = False
    
    cached_data = {}
    if os.path.exists(cache_path):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 캐시 파일을 발견했습니다. 데이터를 로드합니다...")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            use_cache = True
            print(" -> 캐시 로드 성공.")
        except Exception as e:
            print(f" -> 캐시 로드 실패: {e}. 전체 분석을 시작합니다.")

    # ------------------------------------------------------------------
    # 2. 데이터 준비 (Original Path & Regions)
    # ------------------------------------------------------------------
    
    # A. 원본 경로 분석
    if use_cache and 'original_path_results' in cached_data:
        original_path_results = cached_data['original_path_results']
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 1️⃣-A. 기존 경로 분석 중 (Full Run)...")
        original_path_results = analyze_path_data(original_file_name, original_file_name)
    
    center_lat, center_lon = original_path_results['center_coords']

    # B. 리전 분석 및 샘플링
    if use_cache and 'region_data' in cached_data:
        poly_df = cached_data['region_data']['poly_df']
        region_sample_df = cached_data['region_data']['region_sample_df']
        raw_df = cached_data['region_data']['raw_df']
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 2️⃣ 리전 및 폴리곤 생성 중...")
        raw_df = pd.read_csv(region_config.STAY_POINT_FILE)
        raw_df.rename(columns={'latitude': 'centroid_lat', 'longitude': 'centroid_lon'}, inplace=True, errors='ignore')
        
        proc = RegionProcessor(raw_df.copy())
        proc.run_dbscan()
        poly_df = proc.create_polygons() 
        raw_df = proc.df # 업데이트된 정보(is_hull 등) 저장
        
        sampler = PointSampler()
        region_sample_df = sampler.sample_from_polygons(poly_df)

    # 캐시 저장
    if not use_cache:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 💾 분석 결과를 캐시에 저장합니다...")
        cache_data_to_save = {
            'original_path_results': original_path_results,
            'region_data': {
                'poly_df': poly_df,
                'region_sample_df': region_sample_df,
                'raw_df': raw_df
            }
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data_to_save, f)

    # ------------------------------------------------------------------
    # 3. 테스트 경로 분석 (Test Path)
    # ------------------------------------------------------------------
    test_file_name = "LOF_score_test.csv"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 1️⃣-B. 테스트 경로(Analysis Target) 분석 중...")
    test_path_results = analyze_path_data(test_file_name, original_file_name) 

    # ------------------------------------------------------------------
    # 4. LOF 스코어 계산
    # ------------------------------------------------------------------
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 3️⃣ LOF 배회 지수 분석 중...")
    LOF_K_NEIGHBORS = getattr(path_config, 'LOF_K_NEIGHBORS', 60) 
    
    all_path_interp_points = (
        original_path_results.get('interp_points', []) + 
        test_path_results.get('interp_points', [])
    )
    
    region_coords = region_sample_df[['latitude', 'longitude']].values.tolist()
    safe_path_points = original_path_results.get('interp_points', [])
    combined_safe_coords = region_coords + safe_path_points
    safe_df = pd.DataFrame(combined_safe_coords, columns=['latitude', 'longitude'])

    lof_scores = calculate_lof_scores(
        path_points=all_path_interp_points, 
        region_points_df=safe_df, 
        k_neighbors=LOF_K_NEIGHBORS
    )

    # ------------------------------------------------------------------
    # 5. 통합 시각화 (이미지 2번처럼 선이 나오게 하는 핵심 섹션)
    # ------------------------------------------------------------------
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 통합 지도를 생성합니다...")
    
    viz = IntegratedVisualizer(center_lat, center_lon)

    # 1. GPS 원본 점 (Stay Points)
    if not raw_df.empty:
        viz.add_raw_points(raw_df)

    # 2. 머무름 지역 (Regions)
    if not poly_df.empty:
        viz.add_regions_from_module(poly_df)

    # 🚨 [핵심] 3. 원본 매칭 경로 (OSMnx 노드를 따라가는 파란색 선)
    # 3-A. 기존의 안전 경로 매칭 선
    if 'final_grouped_lines' in original_path_results:
        viz.add_final_path_chunks(
            original_path_results['final_grouped_lines'], 
            layer_name="Original Path (OSMnx Merged)"
        )

    # 4. Test Path Matching (테스트 경로의 머지된 선)
    # 이미지 2번처럼 노드 따라 이어진 선을 보고 싶다면 이 데이터가 핵심입니다.
    if 'final_grouped_lines' in test_path_results:
        viz.add_final_path_chunks(
            test_path_results['final_grouped_lines'], 
            layer_name="Test Path (OSMnx Merged)"
        )

    # 4. LOF 결과 점 시각화
    viz.add_sample_points(
        all_path_interp_points, 
        layer_name="All Path LOF Scores", 
        default_show=True,
        lof_scores=lof_scores
    )
    
    # 5. 배경 샘플링 데이터
    if not region_sample_df.empty:
        viz.add_sample_points(
            region_sample_df[['latitude', 'longitude']].values.tolist(), 
            layer_name="Region Area Samples (Background)",
            default_show=False 
        )
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(current_dir, f"Integrated_Map_{ts}.html")
    viz.save(save_path)

    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 분석 완료! 소요 시간: {total_elapsed:.2f}초")

if __name__ == "__main__":
    main()