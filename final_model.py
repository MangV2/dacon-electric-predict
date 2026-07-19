"""
전력 사용량 예측 최종 모델

주요 변경 사항
1. 무작위 K-Fold 대신 날짜 기준 Expanding Window 검증 사용
2. 각 검증 구간을 연속된 168시간(기본 7일)으로 구성
3. 검증 시점보다 미래인 데이터는 학습에 포함하지 않음
4. 전력 사용량 기반 통계 피처는 각 fold의 학습 데이터만으로 생성
5. 학습 행에는 Leave-One-Out 통계 피처를 적용해 자기 타깃 누수 완화
6. CV에서 얻은 최적 boosting round의 중앙값으로 전체 학습 데이터를 재학습
7. 건물 타입별 모델과 개별 건물 모델을 30:70으로 앙상블
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 설정
# ============================================================

RANDOM_SEED = 2025
N_SPLITS = 7
VALID_HOURS = 24 * 7  # 실제 테스트 기간과 같은 1주일
MIN_TRAIN_HOURS = 24 * 14  # 첫 fold에서 최소 2주 학습
EARLY_STOPPING_ROUNDS = 100
UNDER_PREDICTION_WEIGHT = 3.0

DATA_DIR = Path("data")
OUTPUT_PATH = Path("submission.csv")

np.random.seed(RANDOM_SEED)


# ============================================================
# 평가 및 XGBoost 설정
# ============================================================


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """SMAPE(Symmetric Mean Absolute Percentage Error)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denominator, eps)) * 100)


def weighted_mse(alpha: float = UNDER_PREDICTION_WEIGHT):
    """로그 타깃 공간에서 과소예측에 더 큰 페널티를 주는 목적함수."""

    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        weight = np.where(residual > 0, alpha, 1.0)
        grad = -2.0 * weight * residual
        hess = 2.0 * weight
        return grad, hess

    return objective


def log_smape(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """XGBoost early stopping용 SMAPE. 입력은 로그 변환된 타깃이다."""
    y_true = np.exp(np.clip(np.asarray(y_true_log, dtype=float), -20, 20))
    y_pred = np.exp(np.clip(np.asarray(y_pred_log, dtype=float), -20, 20))
    return smape(y_true, y_pred)


def build_model(n_estimators: int, use_early_stopping: bool) -> XGBRegressor:
    """프로젝트에서 사용한 XGBoost 설정을 공통 생성한다."""
    params = {
        "learning_rate": 0.05,
        "n_estimators": int(max(1, n_estimators)),
        "max_depth": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "min_child_weight": 3,
        "random_state": RANDOM_SEED,
        "objective": weighted_mse(),
        "eval_metric": log_smape,
        "n_jobs": -1,
    }
    if use_early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBRegressor(**params)


def to_log_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """0 이하 값으로 인한 log 오류를 방지한다."""
    return np.log(np.clip(np.asarray(y, dtype=float), 1e-6, None))


def from_log_prediction(prediction: np.ndarray) -> np.ndarray:
    """로그 예측값을 원래 전력 단위로 복원한다."""
    return np.exp(np.clip(np.asarray(prediction, dtype=float), -20, 20))


# ============================================================
# 데이터 로드 및 비타깃 피처 엔지니어링
# ============================================================


def load_and_preprocess_data(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터를 로드하고 기본 컬럼을 정리한다."""
    print("데이터 로드 중...")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    building_info = pd.read_csv(data_dir / "building_info.csv")

    train = train.rename(
        columns={
            "건물번호": "building_number",
            "일시": "date_time",
            "기온(°C)": "temperature",
            "강수량(mm)": "rainfall",
            "풍속(m/s)": "windspeed",
            "습도(%)": "humidity",
            "일조(hr)": "sunshine",
            "일사(MJ/m2)": "solar_radiation",
            "전력소비량(kWh)": "power_consumption",
        }
    )
    test = test.rename(
        columns={
            "건물번호": "building_number",
            "일시": "date_time",
            "기온(°C)": "temperature",
            "강수량(mm)": "rainfall",
            "풍속(m/s)": "windspeed",
            "습도(%)": "humidity",
            "일조(hr)": "sunshine",
            "일사(MJ/m2)": "solar_radiation",
        }
    )
    building_info = building_info.rename(
        columns={
            "건물번호": "building_number",
            "건물유형": "building_type",
            "연면적(m2)": "total_area",
            "냉방면적(m2)": "cooling_area",
            "태양광용량(kW)": "solar_power_capacity",
            "ESS저장용량(kWh)": "ess_capacity",
            "PCS용량(kW)": "pcs_capacity",
        }
    )

    for frame in (train, test):
        if "num_date_time" in frame.columns:
            frame.drop(columns="num_date_time", inplace=True)

    building_info["building_type"] = building_info["building_type"].replace(
        {
            "건물기타": "Other Buildings",
            "공공": "Public",
            "학교": "University",
            "백화점": "Department Store",
            "병원": "Hospital",
            "상용": "Commercial",
            "아파트": "Apartment",
            "연구소": "Research Institute",
            "IDC(전화국)": "IDC",
            "호텔": "Hotel",
        }
    )

    # '-'로 기록된 용량은 모델에 직접 넣지 않고 설비 유무만 사용한다.
    building_info["solar_power_utility"] = (
        building_info["solar_power_capacity"].astype(str) != "-"
    ).astype(int)
    building_info["ess_utility"] = (
        building_info["ess_capacity"].astype(str) != "-"
    ).astype(int)

    train = train.merge(building_info, on="building_number", how="left")
    test = test.merge(building_info, on="building_number", how="left")

    train["date_time"] = pd.to_datetime(train["date_time"], format="%Y%m%d %H")
    test["date_time"] = pd.to_datetime(test["date_time"], format="%Y%m%d %H")

    # 로그 타깃을 사용하므로 0 이하 값은 제외한다.
    train = train.loc[train["power_consumption"] > 0].copy()

    # 제출 파일의 원래 테스트 행 순서를 마지막에 복원하기 위해 보존한다.
    train["_original_order"] = np.arange(len(train))
    test["_original_order"] = np.arange(len(test))

    # 원본 행 순서와 무관하게 시간 계산이 안정적으로 동작하도록 정렬한다.
    train.sort_values(["building_number", "date_time"], inplace=True)
    test.sort_values(["building_number", "date_time"], inplace=True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    print(f"데이터 로드 완료 - Train: {train.shape}, Test: {test.shape}")
    return train, test, building_info


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 및 주기성 피처를 생성한다."""
    df = df.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day"] = df["date_time"].dt.day
    df["month"] = df["date_time"].dt.month
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["day_of_year"] = df["date_time"].dt.dayofyear

    # 시간 0시와 23시, 일요일과 월요일 사이의 순환 관계를 표현한다.
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_date"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["cos_date"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    return df


def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """건물·일자별 온도 통계 피처를 생성한다."""
    df = df.copy()
    daily_group = df.groupby(["building_number", "month", "day"])["temperature"]
    df["day_max_temperature"] = daily_group.transform("max")
    df["day_mean_temperature"] = daily_group.transform("mean")
    df["day_min_temperature"] = daily_group.transform("min")
    df["day_temperature_range"] = (
        df["day_max_temperature"] - df["day_min_temperature"]
    )
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """CDH, THI, WCT 등 기상 파생 피처를 생성한다."""
    df = df.copy()

    # 각 건물의 직전 12시간 냉방 부하를 인덱스에 맞게 계산한다.
    df["CDH"] = (
        df.groupby("building_number", group_keys=False)["temperature"]
        .transform(lambda series: (series - 26.0).rolling(12, min_periods=1).sum())
    )

    df["THI"] = (
        9.0 / 5.0 * df["temperature"]
        - 0.55
        * (1.0 - df["humidity"] / 100.0)
        * (9.0 / 5.0 * df["temperature"] - 26.0)
        + 32.0
    )

    safe_wind = df["windspeed"].clip(lower=0)
    df["WCT"] = (
        13.12
        + 0.6215 * df["temperature"]
        - 11.37 * (safe_wind**0.16)
        + 0.3965 * (safe_wind**0.16) * df["temperature"]
    )
    return df


def create_holiday_features(
    df: pd.DataFrame,
    building_info: pd.DataFrame,
) -> pd.DataFrame:
    """공휴일과 백화점별 휴무 규칙을 반영한다."""
    df = df.copy()
    national_holidays = {"2024-06-06", "2024-08-15"}

    df["holiday"] = (
        (df["day_of_week"] >= 5)
        | df["date_time"].dt.strftime("%Y-%m-%d").isin(national_holidays)
    ).astype(int)

    department_stores = building_info.loc[
        building_info["building_type"] == "Department Store", "building_number"
    ].tolist()
    df.loc[df["building_number"].isin(department_stores), "holiday"] = 0

    first_day = df["date_time"].dt.to_period("M").dt.start_time
    days_to_sunday = (first_day.dt.dayofweek + 1) % 7
    week_start = first_day - pd.to_timedelta(days_to_sunday, unit="D")
    df["week_of_month"] = ((df["date_time"] - week_start).dt.days // 7) + 1

    df.loc[
        (df["building_number"] == 18) & (df["day_of_week"] == 6), "holiday"
    ] = 1

    for building_num in [27, 40, 59, 63]:
        df.loc[
            (df["building_number"] == building_num)
            & (df["day_of_week"] == 6)
            & (df["week_of_month"] % 2 == 1),
            "holiday",
        ] = 1

    df.loc[
        (df["building_number"] == 29) & (df["day"] == 10), "holiday"
    ] = 1
    df.loc[
        (df["building_number"] == 29)
        & (df["day_of_week"] == 6)
        & (df["week_of_month"] == 5),
        "holiday",
    ] = 1
    df.loc[
        (df["building_number"] == 32)
        & (df["day_of_week"] == 0)
        & (df["week_of_month"] % 2 == 1),
        "holiday",
    ] = 1

    df.drop(columns="week_of_month", inplace=True)
    return df


def create_non_target_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    building_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """타깃을 사용하지 않는 피처만 미리 생성한다."""
    print("비타깃 피처 엔지니어링 시작...")

    train = create_time_features(train)
    test = create_time_features(test)
    train = create_temperature_features(train)
    test = create_temperature_features(test)
    train = create_weather_features(train)
    test = create_weather_features(test)
    train = create_holiday_features(train, building_info)
    test = create_holiday_features(test, building_info)

    print(f"피처 엔지니어링 완료 - Train: {train.shape}, Test: {test.shape}")
    return train, test


# ============================================================
# 시간 분할
# ============================================================


@dataclass(frozen=True)
class TimeFold:
    fold: int
    train_mask: np.ndarray
    valid_mask: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def expanding_window_splits(
    date_time: pd.Series,
    n_splits: int = N_SPLITS,
    valid_hours: int = VALID_HOURS,
    min_train_hours: int = MIN_TRAIN_HOURS,
) -> Iterator[TimeFold]:
    """
    연속된 검증 window를 사용하는 expanding-window split.

    각 fold는 검증 시작 시점보다 과거인 행만 학습에 포함한다.
    마지막 n_splits개의 검증 구간이 서로 겹치지 않도록 구성한다.
    """
    timestamps = pd.to_datetime(date_time)
    unique_times = pd.DatetimeIndex(timestamps.drop_duplicates().sort_values())

    if len(unique_times) <= min_train_hours + n_splits:
        raise ValueError(
            "시간 분할에 필요한 데이터가 부족합니다. "
            f"고유 시점 수={len(unique_times)}, 최소 학습 시점={min_train_hours}"
        )

    max_valid_hours = (len(unique_times) - min_train_hours) // n_splits
    actual_valid_hours = min(valid_hours, max_valid_hours)
    if actual_valid_hours < 1:
        raise ValueError("검증 window를 구성할 수 없습니다.")

    first_valid_position = len(unique_times) - n_splits * actual_valid_hours

    for fold_index in range(n_splits):
        valid_start_position = first_valid_position + fold_index * actual_valid_hours
        valid_end_position = valid_start_position + actual_valid_hours
        valid_times = unique_times[valid_start_position:valid_end_position]

        valid_start = valid_times[0]
        valid_end = valid_times[-1]
        train_mask = (timestamps < valid_start).to_numpy()
        valid_mask = timestamps.isin(valid_times).to_numpy()

        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Fold {fold_index + 1}에 빈 학습/검증 데이터가 있습니다.")

        train_end = timestamps.loc[train_mask].max()
        yield TimeFold(
            fold=fold_index + 1,
            train_mask=train_mask,
            valid_mask=valid_mask,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )


# ============================================================
# 타깃 통계 피처: fold 내부에서만 계산
# ============================================================


FINE_GROUP = ["building_number", "hour", "day_of_week"]
HOUR_GROUP = ["building_number", "hour"]
BUILDING_GROUP = ["building_number"]

POWER_FEATURE_COLUMNS = [
    "day_hour_mean",
    "day_hour_std",
    "hour_mean",
    "hour_std",
]


@dataclass
class PowerStatistics:
    day_hour: pd.DataFrame
    hour: pd.DataFrame
    building: pd.DataFrame
    global_mean: float
    global_std: float


def fit_power_statistics(train_frame: pd.DataFrame) -> PowerStatistics:
    """현재 fold의 학습 데이터만 사용해 전력 통계를 계산한다."""
    target = "power_consumption"

    day_hour = (
        train_frame.groupby(FINE_GROUP, as_index=False)[target]
        .agg(day_hour_mean="mean", day_hour_std="std")
    )
    hour = (
        train_frame.groupby(HOUR_GROUP, as_index=False)[target]
        .agg(hour_mean="mean", hour_std="std")
    )
    building = (
        train_frame.groupby(BUILDING_GROUP, as_index=False)[target]
        .agg(building_mean="mean", building_std="std")
    )

    global_mean = float(train_frame[target].mean())
    global_std = float(train_frame[target].std())
    if not np.isfinite(global_std):
        global_std = 0.0

    return PowerStatistics(
        day_hour=day_hour,
        hour=hour,
        building=building,
        global_mean=global_mean,
        global_std=global_std,
    )


def apply_power_statistics(
    frame: pd.DataFrame,
    statistics: PowerStatistics,
) -> pd.DataFrame:
    """학습 fold에서 계산된 전력 통계를 검증/테스트 데이터에 적용한다."""
    result = frame.copy()
    result = result.merge(statistics.day_hour, on=FINE_GROUP, how="left")
    result = result.merge(statistics.hour, on=HOUR_GROUP, how="left")
    result = result.merge(statistics.building, on=BUILDING_GROUP, how="left")

    result["hour_mean"] = result["hour_mean"].fillna(result["building_mean"])
    result["hour_mean"] = result["hour_mean"].fillna(statistics.global_mean)
    result["hour_std"] = result["hour_std"].fillna(result["building_std"])
    result["hour_std"] = result["hour_std"].fillna(statistics.global_std)

    result["day_hour_mean"] = result["day_hour_mean"].fillna(result["hour_mean"])
    result["day_hour_std"] = result["day_hour_std"].fillna(result["hour_std"])

    result.drop(columns=["building_mean", "building_std"], inplace=True)
    return result


def _leave_one_out_group_stats(
    frame: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    """그룹별 Leave-One-Out 평균과 표준편차를 계산한다."""
    group = frame.groupby(group_columns)[target_column]
    count = group.transform("count").astype(float)
    total = group.transform("sum").astype(float)
    total_square = frame[target_column].pow(2).groupby(
        [frame[column] for column in group_columns]
    ).transform("sum")

    remaining_count = count - 1.0
    remaining_sum = total - frame[target_column]
    mean = remaining_sum / remaining_count.replace(0, np.nan)

    # 자기 자신을 제외한 표본분산. 남은 표본이 2개 미만이면 NaN으로 둔다.
    remaining_square_sum = total_square - frame[target_column].pow(2)
    variance_numerator = remaining_square_sum - (
        remaining_sum.pow(2) / remaining_count.replace(0, np.nan)
    )
    variance = variance_numerator / (remaining_count - 1.0).replace(0, np.nan)
    std = np.sqrt(variance.clip(lower=0))
    return mean, std


def add_leave_one_out_power_features(train_frame: pd.DataFrame) -> pd.DataFrame:
    """학습 행 자신의 타깃을 제외한 전력 통계 피처를 생성한다."""
    result = train_frame.copy()
    target = "power_consumption"

    day_hour_mean, day_hour_std = _leave_one_out_group_stats(
        result, FINE_GROUP, target
    )
    hour_mean, hour_std = _leave_one_out_group_stats(result, HOUR_GROUP, target)
    building_mean, building_std = _leave_one_out_group_stats(
        result, BUILDING_GROUP, target
    )

    global_count = float(len(result))
    global_sum = float(result[target].sum())
    global_square_sum = float(result[target].pow(2).sum())
    remaining_count = global_count - 1.0

    if remaining_count > 0:
        global_mean = (global_sum - result[target]) / remaining_count
    else:
        global_mean = pd.Series(result[target].mean(), index=result.index)

    if remaining_count > 1:
        remaining_sum = global_sum - result[target]
        remaining_square_sum = global_square_sum - result[target].pow(2)
        global_variance = (
            remaining_square_sum - remaining_sum.pow(2) / remaining_count
        ) / (remaining_count - 1.0)
        global_std = np.sqrt(global_variance.clip(lower=0))
    else:
        global_std = pd.Series(0.0, index=result.index)

    result["hour_mean"] = hour_mean.fillna(building_mean).fillna(global_mean)
    result["hour_std"] = hour_std.fillna(building_std).fillna(global_std)
    result["day_hour_mean"] = day_hour_mean.fillna(result["hour_mean"])
    result["day_hour_std"] = day_hour_std.fillna(result["hour_std"])

    return result


# ============================================================
# 모델 입력 변환
# ============================================================


DROP_COLUMNS = [
    "solar_power_capacity",
    "ess_capacity",
    "pcs_capacity",
    "power_consumption",
    "rainfall",
    "sunshine",
    "solar_radiation",
    "hour",
    "day",
    "month",
    "day_of_week",
    "day_of_year",
    "date_time",
    "_original_order",
]


def _drop_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[column for column in DROP_COLUMNS if column in frame.columns]
    )


def prepare_type_matrices(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """건물 타입 모델용 입력 행렬을 생성하고 컬럼을 정렬한다."""
    train_x = _drop_model_columns(train_frame).drop(columns="building_type")
    other_x = _drop_model_columns(other_frame).drop(columns="building_type")

    train_x = pd.get_dummies(
        train_x, columns=["building_number"], dtype=float
    )
    other_x = pd.get_dummies(
        other_x, columns=["building_number"], dtype=float
    )
    other_x = other_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x.astype(float), other_x.astype(float)


def prepare_individual_matrices(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """개별 건물 모델용 입력 행렬을 생성한다."""
    train_x = _drop_model_columns(train_frame).drop(
        columns=["building_number", "building_type"]
    )
    other_x = _drop_model_columns(other_frame).drop(
        columns=["building_number", "building_type"]
    )
    other_x = other_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x.astype(float), other_x.astype(float)


def get_best_iteration(model: XGBRegressor) -> int:
    """early stopping 결과를 안전하게 boosting round 수로 변환한다."""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return int(model.get_params()["n_estimators"])
    return int(best_iteration) + 1


# ============================================================
# 건물 타입별 모델
# ============================================================


def train_type_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """건물 유형별 CV 검증 후 전체 데이터로 최종 모델을 학습한다."""
    print("건물 타입별 모델 학습 시작...")
    type_predictions = pd.Series(index=test.index, dtype=float)

    for building_type in sorted(train["building_type"].dropna().unique()):
        print(f"\n  [{building_type}]")
        train_type = train.loc[train["building_type"] == building_type].copy()
        test_type = test.loc[test["building_type"] == building_type].copy()

        cv_scores: list[float] = []
        best_rounds: list[int] = []

        for time_fold in expanding_window_splits(train_type["date_time"]):
            fold_train = train_type.loc[time_fold.train_mask].copy()
            fold_valid = train_type.loc[time_fold.valid_mask].copy()

            fold_statistics = fit_power_statistics(fold_train)
            fold_train_features = add_leave_one_out_power_features(fold_train)
            fold_valid_features = apply_power_statistics(fold_valid, fold_statistics)

            x_train, x_valid = prepare_type_matrices(
                fold_train_features, fold_valid_features
            )
            y_train_log = to_log_target(fold_train["power_consumption"])
            y_valid_log = to_log_target(fold_valid["power_consumption"])

            model = build_model(n_estimators=5000, use_early_stopping=True)
            model.fit(
                x_train,
                y_train_log,
                eval_set=[(x_valid, y_valid_log)],
                verbose=False,
            )

            valid_prediction = from_log_prediction(model.predict(x_valid))
            fold_score = smape(
                fold_valid["power_consumption"].to_numpy(), valid_prediction
            )
            cv_scores.append(fold_score)
            best_rounds.append(get_best_iteration(model))

            print(
                f"    Fold {time_fold.fold}: "
                f"train ~ {time_fold.train_end:%m-%d %H시}, "
                f"valid {time_fold.valid_start:%m-%d %H시}~"
                f"{time_fold.valid_end:%m-%d %H시}, "
                f"SMAPE={fold_score:.4f}, rounds={best_rounds[-1]}"
            )

        final_rounds = max(50, int(np.median(best_rounds)))
        print(
            f"    CV SMAPE={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}, "
            f"최종 rounds={final_rounds}"
        )

        full_statistics = fit_power_statistics(train_type)
        full_train_features = add_leave_one_out_power_features(train_type)
        full_test_features = apply_power_statistics(test_type, full_statistics)
        x_full, x_test = prepare_type_matrices(
            full_train_features, full_test_features
        )

        final_model = build_model(
            n_estimators=final_rounds,
            use_early_stopping=False,
        )
        final_model.fit(
            x_full,
            to_log_target(train_type["power_consumption"]),
            verbose=False,
        )
        type_predictions.loc[test_type.index] = from_log_prediction(
            final_model.predict(x_test)
        )

    if type_predictions.isna().any():
        missing = int(type_predictions.isna().sum())
        raise RuntimeError(f"타입별 예측에서 {missing}개 행이 누락되었습니다.")

    print("\n건물 타입별 모델 학습 완료")
    return type_predictions


# ============================================================
# 개별 건물 모델
# ============================================================


def train_individual_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """건물별 CV 검증 후 전체 데이터로 최종 모델을 학습한다."""
    print("\n개별 건물 모델 학습 시작...")
    individual_predictions = pd.Series(index=test.index, dtype=float)

    building_numbers = sorted(train["building_number"].dropna().unique())
    for position, building_number in enumerate(building_numbers, start=1):
        train_building = train.loc[
            train["building_number"] == building_number
        ].copy()
        test_building = test.loc[
            test["building_number"] == building_number
        ].copy()

        cv_scores: list[float] = []
        best_rounds: list[int] = []

        for time_fold in expanding_window_splits(train_building["date_time"]):
            fold_train = train_building.loc[time_fold.train_mask].copy()
            fold_valid = train_building.loc[time_fold.valid_mask].copy()

            fold_statistics = fit_power_statistics(fold_train)
            fold_train_features = add_leave_one_out_power_features(fold_train)
            fold_valid_features = apply_power_statistics(fold_valid, fold_statistics)

            x_train, x_valid = prepare_individual_matrices(
                fold_train_features, fold_valid_features
            )
            y_train_log = to_log_target(fold_train["power_consumption"])
            y_valid_log = to_log_target(fold_valid["power_consumption"])

            model = build_model(n_estimators=5000, use_early_stopping=True)
            model.fit(
                x_train,
                y_train_log,
                eval_set=[(x_valid, y_valid_log)],
                verbose=False,
            )

            valid_prediction = from_log_prediction(model.predict(x_valid))
            fold_score = smape(
                fold_valid["power_consumption"].to_numpy(), valid_prediction
            )
            cv_scores.append(fold_score)
            best_rounds.append(get_best_iteration(model))

        final_rounds = max(50, int(np.median(best_rounds)))

        full_statistics = fit_power_statistics(train_building)
        full_train_features = add_leave_one_out_power_features(train_building)
        full_test_features = apply_power_statistics(test_building, full_statistics)
        x_full, x_test = prepare_individual_matrices(
            full_train_features, full_test_features
        )

        final_model = build_model(
            n_estimators=final_rounds,
            use_early_stopping=False,
        )
        final_model.fit(
            x_full,
            to_log_target(train_building["power_consumption"]),
            verbose=False,
        )
        individual_predictions.loc[test_building.index] = from_log_prediction(
            final_model.predict(x_test)
        )

        print(
            f"  건물 {int(building_number):3d} ({position:3d}/{len(building_numbers)}): "
            f"CV SMAPE={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}, "
            f"rounds={final_rounds}"
        )

    if individual_predictions.isna().any():
        missing = int(individual_predictions.isna().sum())
        raise RuntimeError(f"개별 건물 예측에서 {missing}개 행이 누락되었습니다.")

    print("개별 건물 모델 학습 완료")
    return individual_predictions


# ============================================================
# 메인 실행
# ============================================================


def main() -> None:
    print("=" * 70)
    print("전력 사용량 예측 최종 모델 - Time-Series Safe Validation")
    print("=" * 70)

    train, test, building_info = load_and_preprocess_data()
    train, test = create_non_target_features(train, test, building_info)

    print(
        f"모델링 데이터 - 피처 후보 수: {train.shape[1]}, "
        f"건물 수: {train['building_number'].nunique()}"
    )

    type_predictions = train_type_models(train, test)
    individual_predictions = train_individual_models(train, test)

    print("\n앙상블 수행 중...")
    ensemble_predictions = (
        0.7 * individual_predictions + 0.3 * type_predictions
    ).clip(lower=0)

    # 전처리를 위해 정렬했던 테스트 데이터를 원래 제출 순서로 복원한다.
    prediction_frame = pd.DataFrame(
        {
            "_original_order": test["_original_order"].to_numpy(),
            "prediction": ensemble_predictions.to_numpy(),
        }
    ).sort_values("_original_order")

    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if len(submission) != len(prediction_frame):
        raise ValueError(
            "sample_submission과 테스트 예측 행 수가 일치하지 않습니다: "
            f"{len(submission)} != {len(prediction_frame)}"
        )

    submission["answer"] = prediction_frame["prediction"].to_numpy()
    submission.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("모델 학습 및 예측 완료")
    print(f"제출 파일 저장: {OUTPUT_PATH}")
    print(f"예측 평균: {submission['answer'].mean():.2f} kWh")
    print(
        f"예측 범위: {submission['answer'].min():.2f} ~ "
        f"{submission['answer'].max():.2f} kWh"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
