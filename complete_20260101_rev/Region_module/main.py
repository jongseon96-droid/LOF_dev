#main.py
import pandas as pd
import os
import time
from datetime import datetime
import config
from processor import RegionProcessor
from sampler import PointSampler
from visualizer import MapVisualizer

def main():
    # 전체 시작 시간 기록
    start_time = time.time()

    # =========================================================
    # 1. 데이터 로드 (Data Loading)
    # =========================================================
    # 현재 시각과 함께 로그 출력
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📂 Loading Data...")
    
    raw_df = pd.read_csv(config.STAY_POINT_FILE)
    
    if 'latitude' in raw_df.columns: 
        raw_df.rename(columns={'latitude': 'centroid_lat', 'longitude': 'centroid_lon'}, inplace=True)

    # 💡 로드된 데이터 개수 확인
    print(f"   ㄴ Loaded {len(raw_df):,} rows.") 

    # =========================================================
    # 2. 리전 생성 (Region Processing)
    # =========================================================
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧩 Processing Regions (DBSCAN & Polygon)...")
    
    proc = RegionProcessor(raw_df)
    proc.run_dbscan()
    poly_df = proc.create_polygons()
    
    # 💡 생성된 리전(구역) 개수 확인
    print(f"   ㄴ Created {len(poly_df):,} regions.")

    # =========================================================
    # 3. 샘플링 (Point Sampling)
    # =========================================================
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎲 Sampling Points...")
    
    sampler = PointSampler()
    sample_df = sampler.sample_from_polygons(poly_df)

    # 💡 생성된 샘플 포인트 개수 확인
    print(f"   ㄴ Generated {len(sample_df):,} sample points.")

    # =========================================================
    # 4. 결과 저장 (Saving Results)
    # =========================================================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 타임스탬프 파일로 저장 (보존용)
    poly_path = os.path.join(config.OUTPUT_DIR, f"regions_{ts}.csv")
    sample_path = os.path.join(config.OUTPUT_DIR, f"samples_{ts}.csv")
    
    poly_df.to_csv(poly_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    
    # 🚨 [수정된 핵심 로직]: Path 모듈이 참조할 고정 파일명으로 common_csv에 추가 저장
    # Path_module/config.py의 BASE_PATH를 사용해야 하지만, Region_module config에는 BASE_PATH가 common_csv 디렉토리 내부를 가리킴.
    # 안전하게 상위 디렉토리인 LOF_dev (2)\LOF_dev\common_csv에 저장하도록 경로를 조정합니다.
    
    # 💡 [가정] config.BASE_PATH가 common_csv를 가리킨다고 가정하고 저장합니다.
    fixed_reg_dir = os.path.dirname(config.STAY_POINT_FILE)
    fixed_reg_path = os.path.join(fixed_reg_dir, "stay_regions.csv")
    
    poly_df.to_csv(fixed_reg_path, index=False) 
    print(f"   ㄴ Saved fixed region file for Path Module: {fixed_reg_path}")


    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Saved CSVs completed.")

    # =========================================================
    # 5. 시각화 (Visualization)
    # =========================================================
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🗺️ Generating Map...")
    
    center_lat = raw_df['centroid_lat'].mean()
    center_lon = raw_df['centroid_lon'].mean()
    
    viz = MapVisualizer(center_lat, center_lon)
    
    # 레이어 추가 작업이 오래 걸릴 수 있으므로 안내 메시지
    print("   ㄴ Adding layers to map... (This might take a while)")
    
    viz.add_stay_points(raw_df)
    viz.add_regions(poly_df)
    viz.add_samples(sample_df)
    
    map_path = os.path.join(config.OUTPUT_DIR, f"region_map_{ts}.html")
    viz.save(map_path)

    # =========================================================
    # 종료: 총 소요 시간 계산
    # =========================================================
    elapsed = time.time() - start_time
    print(f"\n✅ All Done! Total execution time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()