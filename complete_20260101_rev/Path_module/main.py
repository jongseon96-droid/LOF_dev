# main.py
import os
import time
import numpy as np
import pandas as pd
import sys
from datetime import datetime # datetime이 없어서 추가

# 진행률 표시 라이브러리 (설치되어 있으면 사용, 없으면 패스)
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    # print("ℹ️ tqdm 라이브러리가 없습니다. 텍스트 로그로 진행상황을 표시합니다.") # 주석 처리하여 깔끔하게 만듦

# 모듈 임포트 (Path_module 내의 형제 파일들)
try:
    # 1. 단독 실행 모드
    import config as cfg
    import utils
    import data_loader as dl
    import graph_manager as gm
    import matcher
    import visualizer as viz
except ImportError:
    # 2. 통합 모드
    from . import config as cfg
    from . import utils
    from . import data_loader as dl
    from . import graph_manager as gm
    from . import matcher
    from . import visualizer as viz

def print_step(step_num, total_steps, message):
    """현재 단계와 메시지를 예쁘게 출력"""
    print(f"\n[{step_num}/{total_steps}] {message}")
    print("=" * 60)

# =========================================================
# 🌟 전체 로직을 재사용 가능한 함수로 분리 (핵심)
# =========================================================
def run_path_analysis():
    """
    Path_module의 1단계부터 5단계까지의 복잡한 분석을 수행하고,
    시각화에 필요한 모든 데이터를 반환합니다.
    """
    # 프로그램 전체 시작 시간 측정 (내부 시간 측정용)
    total_start_time = time.time()
    TOTAL_STEPS = 6

    # 결과 파일명에 붙을 시간 접미사 생성
    time_suffix = utils.get_current_time_str()

    path_file = os.path.join(cfg.COMMON_CSV_DIR, cfg.FILE_NAME_PATHS)
    reg_file = os.path.join(cfg.COMMON_CSV_DIR, cfg.FILE_NAME_REGS)

    # =========================================================
    # 1. 데이터 로드 및 전처리
    # =========================================================
    print_step(1, TOTAL_STEPS, "데이터 로드 및 전처리 시작")
    t_start = time.time()

    print(f"   📂 경로 파일 로드 중... ({cfg.FILE_NAME_PATHS})")
    df_paths_sorted = dl.get_sorted_paths(path_file)

    print(f"   📂 리전 파일 로드 중... ({cfg.FILE_NAME_REGS})")
    regions_df = dl.load_regions(reg_file) # 👈 Region Graph Cache 초기화에 사용

    print("   🔍 세그먼트별 Waypoint 추출 중...")
    all_waypoints = dl.extract_waypoints_by_segment(df_paths_sorted)

    print("   🚀 속도/거리 기반 이상치 필터링 수행 중...")
    full_features_df, raw_df_after_sort = dl.process_data_and_extract_features(df_paths_sorted)

    valid_ids = [sid for sid in full_features_df['segment_id'].unique() if len(all_waypoints.get(sid, [])) >= 2]
    full_features_df = full_features_df[full_features_df['segment_id'].isin(valid_ids)].reset_index(drop=True)

    print(f"   ✅ 1단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 2. 리전 매핑
    # =========================================================
    print_step(2, TOTAL_STEPS, "리전 매핑 (Region Assignment)")
    t_start = time.time()

    iterator = full_features_df.iterrows()
    if USE_TQDM:
        iterator = tqdm(full_features_df.iterrows(), total=len(full_features_df), desc="   Processing Regions", unit="seg")

    seg_region_ids = {}
    for _, row in iterator:
        # 이 부분은 Region_module이 아닌 Path_module 내의 dl.assign_nearest_region_id 함수를 사용합니다.
        rid = dl.assign_nearest_region_id(regions_df, row['from_lat'], row['from_lon'])
        seg_region_ids[row['segment_id']] = rid

    # 지도 시각화의 중심점 계산
    center_lat = np.mean(full_features_df[['from_lat', 'to_lat']].values)
    center_lon = np.mean(full_features_df[['from_lon', 'to_lon']].values)

    print(f"   ✅ 2단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 3. 맵매칭
    # =========================================================
    print_step(3, TOTAL_STEPS, "OSMnx 맵매칭 (Map Matching)")
    t_start = time.time()

    reg_cache = gm.RegionGraphCache(regions_df)

    matched_lines = matcher.perform_map_matching_by_region(reg_cache, full_features_df, all_waypoints, seg_region_ids)

    print(f"   ✅ 3단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 4. 경로 스티칭
    # =========================================================
    print_step(4, TOTAL_STEPS, "경로 스티칭 및 병합 (Stitching)")
    t_start = time.time()

    seg_rid_list = [seg_region_ids.get(sid) for sid in full_features_df['segment_id']]
    merged_chunks = matcher.stitch_and_merge_paths(reg_cache, matched_lines, seg_rid_list)

    valid_chunks = [ch for ch in merged_chunks if ch and not ch.is_empty]

    final_grouped_lines = viz.group_lines_by_connectivity(valid_chunks)
    all_lines_flat = [line for group in final_grouped_lines for line in group]

    print(f"   ✅ 4단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 5. 보간(Interpolation)
    # =========================================================
    print_step(5, TOTAL_STEPS, f"보간 (Step: {cfg.INTERP_STEP_M}m)")
    t_start = time.time()

    interp_points = []

    if cfg.DO_INTERPOLATE and cfg.INTERP_MODE == "merged_global" and all_lines_flat:
        lines_coords = utils.merge_and_simplify_lines(all_lines_flat)
        interp_points = utils.interpolate_continuous_coords_global(lines_coords, cfg.INTERP_STEP_M)
    else:
        print("   ⚠️ 보간 옵션이 꺼져있거나 데이터가 없어 보간을 건너뜁니다.")

    print(f"   ✅ 5단계 완료 (소요시간: {time.time() - t_start:.2f}초)")


    # 6. 최종 결과 반환 (시각화 및 저장은 run_analysis.py가 처리)
    return {
        'center_coords': (center_lat, center_lon),
        'final_grouped_lines': final_grouped_lines, 
        'interp_points': interp_points, 
        'regions_df': regions_df, 
        'total_start_time': total_start_time,
        'raw_path_df': df_paths_sorted, # ⬅️ 이미 정렬된 Raw 데이터(필터링 전)를 반환
        'filtered_features_df': full_features_df # ⬅️ 필터링 후 사용된 메타데이터
    }


# =========================================================
# 🌟 밖에서 실행할 때의 메인 블록 (이전과 달리 로직이 간단해짐)
# =========================================================
if __name__ == "__main__":
    
    # 1. Path 분석 전체 실행
    path_results = run_path_analysis()
    
    # 2. Step 6: 지도 시각화 및 종료 (Standalone 모드에서는 파일 저장까지 담당)
    print_step(6, 6, "지도 시각화 및 종료")
    
    print("   🎨 HTML 지도 생성 중...")
    
    # viz 모듈의 기존 시각화 함수를 사용하여 지도 생성
    m_final = viz.plot_map_layered_by_group(
        path_results['center_coords'], 
        path_results['matched_lines'] if 'matched_lines' in path_results else None, 
        path_results['final_grouped_lines'], 
        path_results['interp_points']
    )
    
    # 결과 CSV 저장 (Step 5에서 처리되었지만, 필요 시 여기서 추가 처리 가능)
    
    # HTML 저장
    time_suffix = utils.get_current_time_str() # 다시 생성
    out_html = os.path.join(cfg.OUTPUT_PATH, f"Result_Map_Standalone_{time_suffix}.html")
    m_final.save(out_html)
    print(f"      -> 지도 저장됨: {out_html}")
    
    # 최종 시간 출력
    total_elapsed = time.time() - path_results['total_start_time']
    print("\n" + "=" * 60)
    print(f"🎉 모든 작업이 성공적으로 완료되었습니다!")
    print(f"⏱️  총 실행 시간: {total_elapsed // 60:.0f}분 {total_elapsed % 60:.2f}초")
    print("=" * 60)