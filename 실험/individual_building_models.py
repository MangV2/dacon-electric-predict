# 📊 개별 건물 모델링 (Individual Building Models)
# Dacon 1위 솔루션의 핵심 전략 구현

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 개별 건물 모델링 시작!")
print("=" * 50)

# 1. 데이터 로드
print("📊 데이터 로드 중...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
building_info = pd.read_csv('data/building_info.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

print(f"✅ Train 데이터: {train_df.shape}")
print(f"✅ Test 데이터: {test_df.shape}")
print(f"✅ Building info: {building_info.shape}")
print(f"✅ 총 건물 수: {len(building_info)}")

# 2. Feature Engineering 함수
def create_features(df, building_info):
    """
    Feature Engineering 함수 - 시간, 기상, 건물 관련 피처 생성
    """
    df = df.copy()
    
    # 건물 정보 병합
    building_info_processed = building_info.copy()
    
    # 불필요한 컬럼 제거 (결측치가 많은 컬럼들)
    drop_columns = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in drop_columns:
        if col in building_info_processed.columns:
            building_info_processed = building_info_processed.drop(col, axis=1)
    
    # 건물 정보 병합
    df = df.merge(building_info_processed, on='건물번호', how='left')
    
    # 원본 데이터에서 불필요한 컬럼 제거
    drop_original_columns = ['일조(hr)', '일사(MJ/m2)']
    for col in drop_original_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # 날짜/시간 변환
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    
    # 시간 관련 피처
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    df['요일'] = df['일시'].dt.dayofweek
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    
    # 계절 피처
    df['계절'] = df['월'].apply(lambda x: 0 if x in [12, 1, 2] else 
                              1 if x in [3, 4, 5] else 
                              2 if x in [6, 7, 8] else 3)
    
    # 시간대 구분
    df['시간대'] = df['시간'].apply(lambda x: 0 if 6 <= x < 12 else 
                                1 if 12 <= x < 18 else 
                                2 if 18 <= x < 24 else 3)
    
    # 온도 관련 피처
    df['온도_제곱'] = df['기온(°C)'] ** 2
    
    # CDH (Cooling Degree Hours) - 냉방도시
    df['CDH'] = np.maximum(df['기온(°C)'] - 26, 0)
    
    # THI (Temperature Humidity Index) - 온습도지수
    df['THI'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)'] / 100) * (9/5 * df['기온(°C)'] - 26) + 32
    
    # 기상 상호작용
    df['습도_온도'] = df['습도(%)'] * df['기온(°C)']
    df['바람세기'] = df['풍속(m/s)'] * df['기온(°C)']
    
    # 건물 관련 피처
    df['냉방면적_비율'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    
    return df

print("\n🔧 Feature Engineering 적용 중...")
train_processed = create_features(train_df, building_info)
test_processed = create_features(test_df, building_info)

print(f"✅ 처리된 train 데이터: {train_processed.shape}")
print(f"✅ 처리된 test 데이터: {test_processed.shape}")

# 3. 피처 준비
exclude_columns = ['num_date_time', '일시', '전력소비량(kWh)']
feature_columns = [col for col in train_processed.columns if col not in exclude_columns]

print(f"✅ 사용할 피처 수: {len(feature_columns)}")
print(f"피처 목록: {feature_columns[:10]}...")  # 첫 10개만 출력

# 카테고리 변수 인코딩
categorical_features = ['건물유형']
label_encoders = {}

for cat_col in categorical_features:
    if cat_col in train_processed.columns:
        le = LabelEncoder()
        # 전체 데이터(train + test)의 카테고리를 기반으로 인코더 학습
        all_categories = pd.concat([train_processed[cat_col], test_processed[cat_col]]).astype(str).unique()
        le.fit(all_categories)
        
        train_processed[cat_col] = le.transform(train_processed[cat_col].astype(str))
        test_processed[cat_col] = le.transform(test_processed[cat_col].astype(str))
        label_encoders[cat_col] = le

# 결측치 처리
# train과 test에서 공통으로 존재하는 numeric 컬럼만 처리
train_numeric_columns = train_processed.select_dtypes(include=[np.number]).columns
test_numeric_columns = test_processed.select_dtypes(include=[np.number]).columns
common_numeric_columns = list(set(train_numeric_columns) & set(test_numeric_columns))

train_processed[train_numeric_columns] = train_processed[train_numeric_columns].fillna(train_processed[train_numeric_columns].mean())
test_processed[test_numeric_columns] = test_processed[test_numeric_columns].fillna(test_processed[test_numeric_columns].mean())

print("✅ 전처리 완료!")

# 4. 개별 건물 모델 훈련
print("\n🏢 개별 건물 모델 훈련 시작...")
print("=" * 50)

# 모델 저장용 딕셔너리
building_models = {}
building_predictions = {}
building_scores = {}

# 각 건물별로 모델 훈련
unique_buildings = sorted(train_processed['건물번호'].unique())
total_buildings = len(unique_buildings)

for idx, building_num in enumerate(unique_buildings, 1):
    print(f"\n🏢 건물 {building_num} 모델 훈련 중... ({idx}/{total_buildings})")
    
    # 해당 건물 데이터 필터링
    train_building = train_processed[train_processed['건물번호'] == building_num].copy()
    test_building = test_processed[test_processed['건물번호'] == building_num].copy()
    
    print(f"   📊 훈련 데이터: {len(train_building)}개")
    print(f"   📊 테스트 데이터: {len(test_building)}개")
    
    # 피처와 타겟 분리
    X_building = train_building[feature_columns].copy()
    y_building = train_building['전력소비량(kWh)'].copy()
    X_test_building = test_building[feature_columns].copy()
    
    # 훈련/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_building, y_building, test_size=0.2, random_state=42
    )
    
    # XGBoost 모델 설정
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # 모델 훈련
    model = xgb.XGBRegressor(**xgb_params)
    
    # 모델 훈련 (단순화)
    model.fit(X_train, y_train)
    
    # 검증 성능 평가
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"   📈 검증 RMSE: {val_rmse:.2f} kWh")
    print(f"   📈 검증 MAE: {val_mae:.2f} kWh")
    
    # 테스트 데이터 예측
    test_pred = model.predict(X_test_building)
    
    # 음수값 처리 (전력소비량은 0 이상이어야 함)
    test_pred = np.maximum(test_pred, 0)
    
    # 결과 저장
    building_models[building_num] = model
    building_predictions[building_num] = test_pred
    building_scores[building_num] = {'rmse': val_rmse, 'mae': val_mae}

print(f"\n✅ 총 {len(building_models)}개 건물 모델 훈련 완료!")

# 5. 예측 결과 조합 및 제출 파일 생성
print("\n📝 제출 파일 생성 중...")

# 테스트 데이터 순서대로 예측값 배열
final_predictions = []

for idx, row in test_processed.iterrows():
    building_num = row['건물번호']
    # 해당 건물의 테스트 데이터에서의 순서 찾기
    building_test_data = test_processed[test_processed['건물번호'] == building_num]
    building_idx = list(building_test_data.index).index(idx)
    
    # 해당 건물 모델의 예측값 가져오기
    prediction = building_predictions[building_num][building_idx]
    final_predictions.append(prediction)

# 제출 파일 생성
submission = sample_submission.copy()
submission['answer'] = final_predictions

# 결과 저장
submission.to_csv('individual_building_submission.csv', index=False)

print("✅ 제출 파일 저장: individual_building_submission.csv")

# 6. 결과 분석
print("\n📊 결과 분석")
print("=" * 50)

# 전체 성능 통계
all_rmse = [score['rmse'] for score in building_scores.values()]
all_mae = [score['mae'] for score in building_scores.values()]

print(f"🏆 전체 건물 평균 성능:")
print(f"   📈 평균 RMSE: {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f} kWh")
print(f"   📈 평균 MAE: {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} kWh")
print(f"   📈 최고 RMSE: {np.min(all_rmse):.2f} kWh (건물 {min(building_scores.keys(), key=lambda x: building_scores[x]['rmse'])})")
print(f"   📈 최저 RMSE: {np.max(all_rmse):.2f} kWh (건물 {max(building_scores.keys(), key=lambda x: building_scores[x]['rmse'])})")

# 예측 결과 통계
print(f"\n🎯 예측 결과 통계:")
print(f"   📊 예측 평균: {np.mean(final_predictions):.2f} kWh")
print(f"   📊 예측 중앙값: {np.median(final_predictions):.2f} kWh")
print(f"   📊 예측 범위: {np.min(final_predictions):.2f} ~ {np.max(final_predictions):.2f} kWh")
print(f"   📊 예측 표준편차: {np.std(final_predictions):.2f} kWh")

# 건물별 성능이 좋은/나쁜 상위 5개
print(f"\n🏆 성능이 가장 좋은 건물 TOP 5:")
best_buildings = sorted(building_scores.items(), key=lambda x: x[1]['rmse'])[:5]
for building, score in best_buildings:
    print(f"   건물 {building}: RMSE {score['rmse']:.2f} kWh")

print(f"\n⚠️  성능 개선이 필요한 건물 TOP 5:")
worst_buildings = sorted(building_scores.items(), key=lambda x: x[1]['rmse'], reverse=True)[:5]
for building, score in worst_buildings:
    print(f"   건물 {building}: RMSE {score['rmse']:.2f} kWh")

# 모델 저장
print(f"\n💾 모델 저장 중...")
os.makedirs('individual_models', exist_ok=True)

# 각 건물 모델 저장
for building_num, model in building_models.items():
    model_path = f'individual_models/building_{building_num}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# 전체 결과 저장
results_summary = {
    'building_scores': building_scores,
    'feature_columns': feature_columns,
    'label_encoders': label_encoders,
    'summary_stats': {
        'mean_rmse': np.mean(all_rmse),
        'std_rmse': np.std(all_rmse),
        'mean_mae': np.mean(all_mae),
        'std_mae': np.std(all_mae)
    }
}

with open('individual_models/results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("✅ 모델 저장 완료: individual_models/ 폴더")

print(f"\n🎉 개별 건물 모델링 완료!")
print("=" * 50)
print("📁 생성된 파일:")
print("   ✅ individual_building_submission.csv - 제출 파일")
print("   ✅ individual_models/ - 훈련된 모델들")
print("   ✅ individual_models/results_summary.pkl - 결과 요약")
print(f"\n💡 다음 단계: 이 결과를 다른 모델(전체 모델, 건물유형별 모델)과 앙상블해보세요!") 