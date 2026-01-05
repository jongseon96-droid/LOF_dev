import os
import time
import numpy as np
import pandas as pd
import sys

# 진행률 표시 라이브러리 (설치되어 있으면 사용, 없으면 패스)
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("ℹ️ tqdm 라이브러리가 없습니다. 텍스트 로그로 진행상황을 표시합니다.")

# 모듈 임포트
import config as cfg
import utils
import data_loader as dl
import graph_manager as gm
import matcher
import visualizer as viz

def print_step(step_num, total_steps, message):
    """현재 단계와 메시지를 예쁘게 출력"""
    print(f"\n[{step_num}/{total_steps}] {message}")
    print("=" * 60)

if __name__ == "__main__":
    # 프로그램 전체 시작 시간 측정
    total_start_time = time.time()
    
    # 결과 파일명에 붙을 시간 접미사 생성 (예: _20231127_2030)
    time_suffix = utils.get_current_time_str()
    
    path_file = os.path.join(cfg.COMMON_CSV_DIR, cfg.FILE_NAME_PATHS)
    reg_file  = os.path.join(cfg.COMMON_CSV_DIR, cfg.FILE_NAME_REGS)
    
    TOTAL_STEPS = 6

    # =========================================================
    # 1. 데이터 로드 및 전처리
    # =========================================================
    print_step(1, TOTAL_STEPS, "데이터 로드 및 전처리 시작")
    t_start = time.time()
    
    print(f"   📂 경로 파일 로드 중... ({cfg.FILE_NAME_PATHS})")
    df_paths_sorted = dl.get_sorted_paths(path_file)
    
    print(f"   📂 리전 파일 로드 중... ({cfg.FILE_NAME_REGS})")
    regions_df = dl.load_regions(reg_file)
    
    print("   🔍 세그먼트별 Waypoint 추출 중...")
    all_waypoints = dl.extract_waypoints_by_segment(df_paths_sorted)
    
    print("   🚀 속도/거리 기반 이상치 필터링 수행 중...")
    full_features_df, _ = dl.process_data_and_extract_features(df_paths_sorted)
    
    # Waypoint가 2개 미만인(점 하나만 찍힌) 의미 없는 세그먼트 제거
    valid_ids = [sid for sid in full_features_df['segment_id'].unique() if len(all_waypoints.get(sid, [])) >= 2]
    full_features_df = full_features_df[full_features_df['segment_id'].isin(valid_ids)].reset_index(drop=True)
    
    print(f"   ✅ 1단계 완료 (소요시간: {time.time() - t_start:.2f}초)")
    print(f"      - 처리된 세그먼트 수: {len(full_features_df)}개")

    # =========================================================
    # 2. 리전 매핑 (각 경로가 어느 지역에 속하는지 판별)
    # =========================================================
    print_step(2, TOTAL_STEPS, "리전 매핑 (Region Assignment)")
    t_start = time.time()
    
    print("   🌍 각 경로의 시작점을 기준으로 가장 가까운 리전 ID를 찾습니다...")
    
    # tqdm을 사용하여 진행률 표시 (데이터가 많을 경우 오래 걸림)
    iterator = full_features_df.iterrows()
    if USE_TQDM:
        iterator = tqdm(full_features_df.iterrows(), total=len(full_features_df), desc="   Processing Regions", unit="seg")
    
    seg_region_ids = {}
    for _, row in iterator:
        rid = dl.assign_nearest_region_id(regions_df, row['from_lat'], row['from_lon'])
        seg_region_ids[row['segment_id']] = rid
        
    # 지도 시각화의 중심점 계산
    center_lat = np.mean(full_features_df[['from_lat', 'to_lat']].values)
    center_lon = np.mean(full_features_df[['from_lon', 'to_lon']].values)

    print(f"   ✅ 2단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 3. 맵매칭 (가장 시간이 오래 걸리는 작업)
    # =========================================================
    print_step(3, TOTAL_STEPS, "OSMnx 맵매칭 (Map Matching)")
    t_start = time.time()
    
    print("   🗺️  OSM 도로망 그래프를 다운로드하고 경로를 매칭합니다.")
    print("       (네트워크 상태에 따라 시간이 오래 걸릴 수 있습니다...)")
    
    # 그래프 캐시 초기화
    reg_cache = gm.RegionGraphCache(regions_df)
    
    # 맵매칭 수행 (matcher 모듈 내부에서 수행)
    # 팁: matcher.py 내부 루프에 tqdm을 달면 더 좋지만, 외부에서 감싸기는 어려우므로 시간만 측정
    matched_lines = matcher.perform_map_matching_by_region(reg_cache, full_features_df, all_waypoints, seg_region_ids)
    
    success_count = sum(1 for ln in matched_lines if ln is not None)
    print(f"   ✅ 3단계 완료 (소요시간: {time.time() - t_start:.2f}초)")
    print(f"      - 매칭 성공률: {success_count} / {len(matched_lines)} ({success_count/len(matched_lines)*100:.1f}%)")

    # =========================================================
    # 4. 경로 스티칭 (끊어진 경로 잇기)
    # =========================================================
    print_step(4, TOTAL_STEPS, "경로 스티칭 및 병합 (Stitching)")
    t_start = time.time()
    
    print(f"   🧵 끊어진 경로를 연결합니다 (Gap Limit: {cfg.GAP_BREAK_M}m, Max Bridge: {cfg.MAX_BRIDGE_TRY_M}m)")
    
    seg_rid_list = [seg_region_ids.get(sid) for sid in full_features_df['segment_id']]
    merged_chunks = matcher.stitch_and_merge_paths(reg_cache, matched_lines, seg_rid_list)
    
    # 유효한 청크만 남기기
    valid_chunks = [ch for ch in merged_chunks if ch and not ch.is_empty]
    
    # 시각화를 위해 공간적으로 연결된 그룹끼리 묶기
    print("   🧩 공간적 연결성 분석 중...")
    final_grouped_lines = viz.group_lines_by_connectivity(valid_chunks)
    all_lines_flat = [line for group in final_grouped_lines for line in group]
    
    print(f"   ✅ 4단계 완료 (소요시간: {time.time() - t_start:.2f}초)")
    print(f"      - 최종 생성된 경로 청크: {len(valid_chunks)}개")
    print(f"      - 형성된 공간 그룹: {len(final_grouped_lines)}개")

    # =========================================================
    # 5. 보간(Interpolation) 및 CSV 저장
    # =========================================================
    print_step(5, TOTAL_STEPS, f"보간 및 결과 저장 (Step: {cfg.INTERP_STEP_M}m)")
    t_start = time.time()
    
    interp_points = []
    
    if cfg.DO_INTERPOLATE and cfg.INTERP_MODE == "merged_global" and all_lines_flat:
        print("   🔥 경로 통합(Union/Merge) 및 보간 수행 중...")
        
        # 1. 겹치는 경로 하나로 녹이기 (Melting)
        lines_coords = utils.merge_and_simplify_lines(all_lines_flat)
        
        # 2. 지정된 간격으로 점 찍기
        interp_points = utils.interpolate_continuous_coords_global(lines_coords, cfg.INTERP_STEP_M)
        
        # 3. 보간된 점 CSV 저장
        out_csv = os.path.join(cfg.OUTPUT_PATH, f"mergedGLOBAL_interpolated_{int(cfg.INTERP_STEP_M)}m{time_suffix}.csv")
        df_interp = pd.DataFrame(interp_points, columns=['lat', 'lon'])
        utils.safe_write_csv(df_interp, out_csv)
        print(f"      -> 보간 데이터 저장됨: {out_csv}")
        
        # 4. LOF 입력용 데이터 저장
        # (여기서 나중에 중복 제거(dedup)나 로그 샘플링 로직을 추가할 수 있음)
        lof_csv = os.path.join(cfg.OUTPUT_PATH, f"lof_input_{int(cfg.INTERP_STEP_M)}m{time_suffix}.csv")
        
        # 예시: 좌표 반올림 후 중복 제거 로직 (선택사항)
        # df_interp['lat_r'] = df_interp['lat'].round(cfg.DEDUP_PRECISION)
        # df_interp['lon_r'] = df_interp['lon'].round(cfg.DEDUP_PRECISION)
        # df_interp = df_interp.drop_duplicates(subset=['lat_r', 'lon_r'])
        
        utils.safe_write_csv(df_interp[['lat', 'lon']], lof_csv)
        print(f"      -> LOF 입력 데이터 저장됨: {lof_csv}")
        print(f"      -> 총 포인트 수: {len(df_interp)}개")

    else:
        print("   ⚠️ 보간 옵션이 꺼져있거나 데이터가 없어 건너뜁니다.")

    print(f"   ✅ 5단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # =========================================================
    # 6. 지도 시각화 및 종료
    # =========================================================
    print_step(6, TOTAL_STEPS, "지도 시각화 (Visualization)")
    t_start = time.time()
    
    print("   🎨 HTML 지도 생성 중...")
    m_final = viz.plot_map_layered_by_group(
        (center_lat, center_lon), matched_lines, final_grouped_lines, interp_points
    )
    
    out_html = os.path.join(cfg.OUTPUT_PATH, f"Result_Map{time_suffix}.html")
    m_final.save(out_html)
    print(f"      -> 지도 저장됨: {out_html}")
    
    print(f"   ✅ 6단계 완료 (소요시간: {time.time() - t_start:.2f}초)")

    # 프로그램 종료
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"🎉 모든 작업이 성공적으로 완료되었습니다!")
    print(f"⏱️  총 실행 시간: {total_elapsed // 60:.0f}분 {total_elapsed % 60:.2f}초")
    print("=" * 60)