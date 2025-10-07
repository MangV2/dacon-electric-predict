"""
전력 사용량 예측 최종 모델
- 건물 타입별 모델 + 개별 건물 모델 앙상블
- 백화점 휴일 규칙 및 고급 피처 엔지니어링 적용
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# 설정 및 평가 함수


RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

def smape(y_true, y_pred):
    """SMAPE (Symmetric Mean Absolute Percentage Error) 계산"""
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def weighted_mse(alpha=3):
    """과소 예측에 더 큰 페널티를 부여하는 손실 함수"""
    def loss(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * alpha, 2.0)
        return grad, hess
    return loss

def custom_smape(preds, dtrain):
    """XGBoost용 SMAPE 평가 함수"""
    labels = dtrain.get_label()
    return 'custom_smape', smape(labels, preds)

# 데이터 로드 및 기본 전처리

def load_and_preprocess_data():
    """데이터 로드 및 기본 전처리"""
    print("데이터 로드 중...")
    
    # 데이터 로드
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    building_info = pd.read_csv('data/building_info.csv')
    
    # 컬럼명 영어로 변경
    train = train.rename(columns={
        '건물번호': 'building_number', '일시': 'date_time',
        '기온(°C)': 'temperature', '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed', '습도(%)': 'humidity',
        '일조(hr)': 'sunshine', '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    }).drop('num_date_time', axis=1)
    
    test = test.rename(columns={
        '건물번호': 'building_number', '일시': 'date_time',
        '기온(°C)': 'temperature', '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed', '습도(%)': 'humidity',
        '일조(hr)': 'sunshine', '일사(MJ/m2)': 'solar_radiation'
    }).drop('num_date_time', axis=1)
    
    building_info = building_info.rename(columns={
        '건물번호': 'building_number', '건물유형': 'building_type',
        '연면적(m2)': 'total_area', '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'solar_power_capacity',
        'ESS저장용량(kWh)': 'ess_capacity', 'PCS용량(kW)': 'pcs_capacity'
    })
    
    # 건물 유형 영어로 변경
    building_info['building_type'] = building_info['building_type'].replace({
        '건물기타': 'Other Buildings', '공공': 'Public', '학교': 'University',
        '백화점': 'Department Store', '병원': 'Hospital', '상용': 'Commercial',
        '아파트': 'Apartment', '연구소': 'Research Institute',
        'IDC(전화국)': 'IDC', '호텔': 'Hotel'
    })
    
    # 설비 유무 피처
    building_info['solar_power_utility'] = (building_info.solar_power_capacity != '-').astype(int)
    building_info['ess_utility'] = (building_info.ess_capacity != '-').astype(int)
    
    # 건물 정보 병합
    train = train.merge(building_info, on='building_number', how='left')
    test = test.merge(building_info, on='building_number', how='left')
    
    # 날짜 변환
    train['date_time'] = pd.to_datetime(train['date_time'], format='%Y%m%d %H')
    test['date_time'] = pd.to_datetime(test['date_time'], format='%Y%m%d %H')
    
    # 이상치 제거 (전력소비량 = 0)
    train = train[train['power_consumption'] > 0].reset_index(drop=True)
    
    print(f"데이터 로드 완료 - Train: {train.shape}, Test: {test.shape}")
    return train, test, building_info


# 피처 엔지니어링


def create_time_features(df):
    """시간 관련 피처 생성"""
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['day_of_week'] = df['date_time'].dt.dayofweek
    
    # 주기성 피처 (sin/cos 변환)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 23.0)
    df['sin_date'] = -np.sin(2 * np.pi * (df['month'] + df['day'] / 31) / 12)
    df['cos_date'] = -np.cos(2 * np.pi * (df['month'] + df['day'] / 31) / 12)
    df['sin_month'] = -np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = -np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_dayofweek'] = -np.sin(2 * np.pi * (df['day_of_week'] + 1) / 7.0)
    df['cos_dayofweek'] = -np.cos(2 * np.pi * (df['day_of_week'] + 1) / 7.0)
    
    return df

def create_temperature_features(df):
    """일별 온도 통계 피처"""
    temp_stats = df.groupby(['building_number', 'month', 'day'])['temperature'].agg(
        day_max_temperature='max',
        day_mean_temperature='mean',
        day_min_temperature='min'
    ).reset_index()
    
    df = df.merge(temp_stats, on=['building_number', 'month', 'day'], how='left')
    df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
    
    return df

def create_weather_features(df):
    """기상 관련 파생 피처 (CDH, THI, WCT)"""
    # CDH (Cooling Degree Hours) 계산
    cdh_list = []
    for building_num in range(1, 101):
        temp = df[df['building_number'] == building_num]['temperature'].values
        cumsum = np.cumsum(temp - 26)
        cdh = np.concatenate((cumsum[:11], cumsum[11:] - cumsum[:-11]))
        cdh_list.append(cdh)
    df['CDH'] = np.concatenate(cdh_list)
    
    # THI (Temperature Humidity Index)
    df['THI'] = (9/5 * df['temperature'] - 
                 0.55 * (1 - df['humidity']/100) * (9/5 * df['temperature'] - 26) + 32)
    
    # WCT (Wind Chill Temperature)
    df['WCT'] = (13.12 + 0.6125 * df['temperature'] - 
                 11.37 * (df['windspeed']**0.16) + 
                 0.3965 * (df['windspeed']**0.16) * df['temperature'])
    
    return df

def create_holiday_features(df, building_info):
    """휴일 피처 생성 (백화점 특별 규칙 포함)"""
    # 기본 공휴일
    national_holidays = ['2024-06-06', '2024-08-15']
    df['holiday'] = ((df['day_of_week'] >= 5) | 
                     (df['date_time'].dt.strftime('%Y-%m-%d').isin(national_holidays))).astype(int)
    
    # 백화점 건물 추출
    dept_buildings = building_info[building_info['building_type'] == 'Department Store']['building_number'].tolist()
    
    # 백화점 휴일 초기화
    df.loc[df['building_number'].isin(dept_buildings), 'holiday'] = 0
    
    # 달력 기준 주차 계산
    def get_week_of_month(dates):
        result = []
        for date in dates:
            first_day = date.replace(day=1)
            days_to_week_start = (first_day.weekday() + 1) % 7
            week_start = first_day - pd.Timedelta(days=days_to_week_start)
            week_num = ((date - week_start).days // 7) + 1
            result.append(week_num)
        return result
    
    df['week_of_month'] = get_week_of_month(df['date_time'])
    
    # 백화점별 특별 휴일 규칙
    # 18번: 매주 일요일
    df.loc[(df['building_number'] == 18) & (df['day_of_week'] == 6), 'holiday'] = 1
    
    # 27, 40, 59, 63번: 홀수 주 일요일
    for bldg in [27, 40, 59, 63]:
        df.loc[(df['building_number'] == bldg) & 
               (df['day_of_week'] == 6) & 
               (df['week_of_month'] % 2 == 1), 'holiday'] = 1
    
    # 29번: 매달 10일 + 5번째 주 일요일
    df.loc[(df['building_number'] == 29) & (df['day'] == 10), 'holiday'] = 1
    df.loc[(df['building_number'] == 29) & 
           (df['day_of_week'] == 6) & 
           (df['week_of_month'] == 5), 'holiday'] = 1
    
    # 32번: 홀수 주 월요일
    df.loc[(df['building_number'] == 32) & 
           (df['day_of_week'] == 0) & 
           (df['week_of_month'] % 2 == 1), 'holiday'] = 1
    
    df.drop('week_of_month', axis=1, inplace=True)
    return df

def create_power_features(train, test):
    """전력 소비량 기반 통계 피처"""
    # 건물별 시간대/요일별 통계
    power_stats = train.groupby(['building_number', 'hour', 'day_of_week'])['power_consumption'].agg(
        day_hour_mean='mean',
        day_hour_std='std'
    ).reset_index()
    
    # 건물별 시간대별 통계
    hour_stats = train.groupby(['building_number', 'hour'])['power_consumption'].agg(
        hour_mean='mean',
        hour_std='std'
    ).reset_index()
    
    # 통계 피처 병합
    train = train.merge(power_stats, on=['building_number', 'hour', 'day_of_week'], how='left')
    train = train.merge(hour_stats, on=['building_number', 'hour'], how='left')
    test = test.merge(power_stats, on=['building_number', 'hour', 'day_of_week'], how='left')
    test = test.merge(hour_stats, on=['building_number', 'hour'], how='left')
    
    return train, test

def feature_engineering(train, test, building_info):
    """전체 피처 엔지니어링 파이프라인"""
    print("🔧 피처 엔지니어링 시작...")
    
    train = create_time_features(train)
    test = create_time_features(test)
    
    train = create_temperature_features(train)
    test = create_temperature_features(test)
    
    train = create_weather_features(train)
    test = create_weather_features(test)
    
    train = create_holiday_features(train, building_info)
    test = create_holiday_features(test, building_info)
    
    train, test = create_power_features(train, test)
    
    print(f"피처 엔지니어링 완료 - Train: {train.shape}, Test: {test.shape}")
    return train, test


# 모델 학습


def prepare_model_data(train, test):
    """모델링용 데이터 준비"""
    drop_columns = [
        'solar_power_capacity', 'ess_capacity', 'pcs_capacity',
        'power_consumption', 'rainfall', 'sunshine', 'solar_radiation',
        'hour', 'day', 'month', 'day_of_week', 'date_time'
    ]
    
    X = train.drop(drop_columns, axis=1)
    y = train['power_consumption']
    test_X = test.drop([col for col in drop_columns if col in test.columns], axis=1)
    
    return X, y, test_X

def train_type_models(X, y, test_X):
    """건물 타입별 모델 학습"""
    print("건물 타입별 모델 학습 시작...")
    
    kf = KFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED)
    type_preds = pd.DataFrame(index=test_X.index, columns=['pred'], dtype=float)
    
    for building_type in X['building_type'].unique():
        print(f"  - {building_type} 학습 중...")
        
        # 타입별 데이터 필터링
        type_mask = X['building_type'] == building_type
        X_type = pd.get_dummies(X[type_mask].drop('building_type', axis=1), 
                                columns=['building_number'])
        y_type = y[type_mask]
        test_type = pd.get_dummies(test_X[test_X['building_type'] == building_type].drop('building_type', axis=1),
                                   columns=['building_number'])
        test_type = test_type.reindex(columns=X_type.columns, fill_value=0)
        
        # K-Fold 학습
        fold_preds = []
        for tr_idx, va_idx in kf.split(X_type):
            model = XGBRegressor(
                learning_rate=0.05, n_estimators=5000, max_depth=10,
                subsample=0.7, colsample_bytree=0.5, min_child_weight=3,
                random_state=RANDOM_SEED, objective=weighted_mse(),
                early_stopping_rounds=100
            )
            model.fit(
                X_type.iloc[tr_idx], np.log(y_type.iloc[tr_idx]),
                eval_set=[(X_type.iloc[va_idx], np.log(y_type.iloc[va_idx]))],
                eval_metric=custom_smape, verbose=False
            )
            fold_preds.append(np.exp(model.predict(test_type)))
        
        type_preds.loc[test_type.index, 'pred'] = np.mean(fold_preds, axis=0)
    
    print("타입별 모델 학습 완료")
    return type_preds

def train_individual_models(X, y, test_X):
    """개별 건물 모델 학습"""
    print("개별 건물 모델 학습 시작...")
    
    kf = KFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED)
    individual_preds = pd.DataFrame(index=test_X.index, columns=['pred'], dtype=float)
    
    for building_num in sorted(X['building_number'].unique()):
        if building_num % 10 == 0:
            print(f"  - 건물 {building_num}/100 진행 중...")
        
        # 건물별 데이터
        bldg_mask = X['building_number'] == building_num
        X_bldg = X[bldg_mask].drop(['building_number', 'building_type'], axis=1)
        y_bldg = y[bldg_mask]
        test_bldg = test_X[test_X['building_number'] == building_num].drop(
            ['building_number', 'building_type'], axis=1)
        
        # K-Fold 학습
        fold_preds = []
        for tr_idx, va_idx in kf.split(X_bldg):
            model = XGBRegressor(
                learning_rate=0.05, n_estimators=5000, max_depth=10,
                subsample=0.7, colsample_bytree=0.5, min_child_weight=3,
                random_state=RANDOM_SEED, objective=weighted_mse(),
                early_stopping_rounds=100
            )
            model.fit(
                X_bldg.iloc[tr_idx], np.log(y_bldg.iloc[tr_idx]),
                eval_set=[(X_bldg.iloc[va_idx], np.log(y_bldg.iloc[va_idx]))],
                eval_metric=custom_smape, verbose=False
            )
            fold_preds.append(np.exp(model.predict(test_bldg)))
        
        individual_preds.loc[test_bldg.index, 'pred'] = np.mean(fold_preds, axis=0)
    
    print("개별 모델 학습 완료")
    return individual_preds


# 메인 실행

def main():
    """메인 실행 함수"""
    print("="*60)
    print("전력 사용량 예측 최종 모델 실행")
    print("="*60)
    
    # 1. 데이터 로드 및 전처리
    train, test, building_info = load_and_preprocess_data()
    
    # 2. 피처 엔지니어링
    train, test = feature_engineering(train, test, building_info)
    
    # 3. 모델링 데이터 준비
    X, y, test_X = prepare_model_data(train, test)
    print(f"모델링 데이터 - 피처 수: {X.shape[1]}, 건물 수: {X['building_number'].nunique()}")
    
    # 4. 모델 학습
    type_preds = train_type_models(X, y, test_X)
    individual_preds = train_individual_models(X, y, test_X)
    
    # 5. 앙상블 (70% 개별 + 30% 타입)
    print("앙상블 수행 중...")
    ensemble_preds = 0.7 * individual_preds['pred'] + 0.3 * type_preds['pred']
    
    # 6. 제출 파일 생성
    submission = pd.read_csv('data/sample_submission.csv')
    submission['answer'] = np.maximum(ensemble_preds.values, 0)  # 음수 방지
    submission.to_csv('submission.csv', index=False)
    
    print("="*60)
    print("모델 학습 및 예측 완료!")
    print(f"제출 파일 저장: submission.csv")
    print(f"예측값 통계:")
    print(f"   평균: {submission['answer'].mean():.2f} kWh")
    print(f"   범위: {submission['answer'].min():.2f} ~ {submission['answer'].max():.2f} kWh")
    print("="*60)

if __name__ == "__main__":
    main()