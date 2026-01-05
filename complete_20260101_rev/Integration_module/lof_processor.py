# 파일 위치: lof_processor.py

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def calculate_lof_scores(path_points: list, region_points_df: pd.DataFrame, k_neighbors: int):
    """
    [최종 수정]
    고객님 요청대로 논문 정의에 맞춘 직관적 스코어 반환.
    - 결과 < 1.0 : 정상 (안전)
    - 결과 > 1.0 : 이상치 (배회) -> 값이 클수록 위험
    """
    
    if region_points_df.empty or len(path_points) == 0:
        return np.array([])
        
    X_train = region_points_df[['latitude', 'longitude']].values
    X_test = np.array(path_points)

    # 차원 검사
    if X_train.ndim != 2 or X_test.ndim != 2:
        return np.array([])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 

    # 모델 학습
    lof = LocalOutlierFactor(n_neighbors=k_neighbors, novelty=True)
    lof.fit(X_train_scaled) 

    # 점수 계산
    decision_scores = lof.decision_function(X_test_scaled)
    
    # 💡 [정답 수식]: 1.0 - (Raw Score)
    # (+0.5 정상) -> 1.0 - 0.5 = 0.5 (매우 정상, 초록색 대상)
    # (-2.0 이상) -> 1.0 - (-2.0) = 3.0 (강력한 이상치, 빨간색 대상)
    lof_scores = 1.0 - decision_scores

    print(f"✅ LOF 완료. (기준: 1.0, 현재 최대값: {np.max(lof_scores):.2f})")
    
    return lof_scores