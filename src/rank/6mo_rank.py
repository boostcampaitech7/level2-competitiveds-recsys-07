import cupy as cp
import pandas as pd
from haversine import Unit, haversine
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm


# haversine 라이브러리를 활용하여 CuPy 배열 간의 거리 계산
def haversine_distance_cupy(lat1, lon1, lat2, lon2):
    """
    CuPy 배열을 사용해 각 지점 간의 거리를 계산하고 haversine 공식을 적용하는 함수.
    Args:
    - lat1, lon1: 첫 번째 지점들의 위도와 경도 배열
    - lat2, lon2: 두 번째 지점들의 위도와 경도 배열

    Returns:
    - 거리 (미터 단위, CuPy 배열로 반환)
    """
    lat1_np, lon1_np = cp.asnumpy(lat1), cp.asnumpy(lon1)
    lat2_np, lon2_np = cp.asnumpy(lat2), cp.asnumpy(lon2)

    distances = [
        haversine((lat1_np[i], lon1_np[i]), (lat2_np[i], lon2_np[i]), unit=Unit.METERS) for i in range(len(lat1_np))
    ]

    return cp.array(distances)


def find_closest_facility_and_six_months_ago_rank(facility_data, contract_year_month, apartment_lat, apartment_lon):
    """
    위도와 경도를 기준으로 가장 가까운 시설을 찾고, 계약 연월에서 6개월 전의 rank를 반환.
    Args:
    - facility_data: 시설 데이터 (위도, 경도, rank 포함)
    - contract_year_month: 아파트 계약 연월 (예: '202106')
    - apartment_lat, apartment_lon: 아파트의 위도, 경도

    Returns:
    - rank 값
    - rank가 속한 연월
    - 가장 가까운 시설과의 거리
    """
    # 모든 시설과 아파트의 거리를 계산
    facility_lat = cp.array(facility_data["latitude"])
    facility_lon = cp.array(facility_data["longitude"])
    distances = haversine_distance_cupy(apartment_lat, apartment_lon, facility_lat, facility_lon)

    # 가장 가까운 시설의 인덱스 찾기
    closest_idx = cp.argmin(distances)

    # 계약 연월에서 6개월을 뺀 연월 계산
    contract_date = pd.to_datetime(contract_year_month, format="%Y%m")
    six_months_ago_date = contract_date - DateOffset(months=6)
    six_months_ago_str = six_months_ago_date.strftime("%Y%m")

    # 6개월 전 연월의 rank 가져오기
    if six_months_ago_str in facility_data.columns:
        closest_rank = facility_data.iloc[int(closest_idx)][six_months_ago_str]
        rank_month = six_months_ago_str
    else:
        return cp.nan, None, None  # 해당 연월에 데이터가 없으면 NaN 반환

    # 가장 가까운 시설과의 거리 계산
    closest_distance = cp.asnumpy(distances[int(closest_idx)])

    return closest_rank, rank_month, closest_distance


def assign_closest_rank(train_data, subway_data, park_data, school_data):
    """
    아파트 계약 연월에 맞춰 가장 가까운 시설의 rank 값을 할당하고, 몇 개월 전의 데이터인지와 거리를 추가.
    Args:
    - train_data: 아파트 계약 데이터 (위도, 경도 포함)
    - subway_data: 지하철 시설 데이터 (위도, 경도, rank 포함)
    - park_data: 공원 시설 데이터 (위도, 경도, rank 포함)
    - school_data: 학교 시설 데이터 (위도, 경도, rank 포함)

    Returns:
    - train_data: 각 시설의 rank, 몇 개월 전 rank인지, 시설과의 거리가 포함된 train_data
    """
    for idx, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Assigning closest rank based on lat/lon"):
        contract_year_month = row["contract_year_month"]
        apartment_lat = cp.array([row["latitude"]])
        apartment_lon = cp.array([row["longitude"]])

        # 지하철 rank 계산
        subway_rank, subway_months_ago, subway_distance = find_closest_facility_and_six_months_ago_rank(
            subway_data, contract_year_month, apartment_lat, apartment_lon
        )

        # 공원 rank 계산
        park_rank, park_months_ago, park_distance = find_closest_facility_and_six_months_ago_rank(
            park_data, contract_year_month, apartment_lat, apartment_lon
        )

        # 학교 rank 계산
        school_rank, school_months_ago, school_distance = find_closest_facility_and_six_months_ago_rank(
            school_data, contract_year_month, apartment_lat, apartment_lon
        )

        # 결과를 train_data에 할당
        train_data.at[idx, "subway_rank"] = subway_rank
        train_data.at[idx, "subway_rank_distance"] = subway_distance

        train_data.at[idx, "park_rank"] = park_rank
        train_data.at[idx, "park_rank_distance"] = park_distance

        train_data.at[idx, "school_rank"] = school_rank
        train_data.at[idx, "school_rank_distance"] = school_distance

    return train_data


def remove_rows_based_on_missing_data(train_data, threshold=0.5):
    """
    데이터프레임에서 결측치 비율에 따라 행을 제거하는 함수.

    Args:
    - train_data: 데이터프레임 (삭제할 행이 포함된 데이터)
    - threshold: 허용할 최대 결측치 비율 (기본값 0.5, 즉 50%)

    Returns:
    - 결측치가 기준보다 높은 행이 제거된 데이터프레임
    """
    # 각 행의 결측치 비율 계산 (모든 열 기준)
    train_data["missing_ratio"] = train_data.isnull().mean(axis=1)

    # 결측치 비율이 threshold 이상인 행 제거
    cleaned_data = train_data[train_data["missing_ratio"] < threshold].copy()

    # 'missing_ratio' 컬럼 제거 (결과 데이터에 필요하지 않음)
    cleaned_data.drop(columns=["missing_ratio"], inplace=True)

    return cleaned_data


train_data = pd.read_csv("/content/train.csv")
test_data = pd.read_csv("/content/test.csv")
subway_data = pd.read_csv("/content/subway_df.csv")
park_data = pd.read_csv("/content/park_df.csv")
school_data = pd.read_csv("/content/school_df.csv")

threshold = 0.8  # 결측치가 50% 이상인 경우 제거

# 결측치 비율에 따라 행 제거
cleaned_sub = remove_rows_based_on_missing_data(subway_data, threshold)
cleaned_park = remove_rows_based_on_missing_data(park_data, threshold)
cleaned_sch = remove_rows_based_on_missing_data(school_data, threshold)

sub_df = cleaned_sub.copy()
park_df = cleaned_park.copy()
sch_df = cleaned_sch.copy()

train_df = assign_closest_rank(train_data, sub_df, park_df, sch_df)
test_df = assign_closest_rank(test_data, sub_df, park_df, sch_df)

start_date = 201910  # 시작 기준: 2019년 10월
end_date = 202312  # 종료 기준: 2023년 12월

# contract_year_month을 기준으로 데이터 필터링
train_data = train_df[(train_df["contract_year_month"] >= start_date) & (train_df["contract_year_month"] <= end_date)]

train_data["subway_wr"] = train_data["subway_rank"] / (train_data["subway_rank_distance"])
train_data["park_wr"] = train_data["park_rank"] / (train_data["park_rank_distance"])
train_data["school_wr"] = train_data["school_rank"] / (train_data["school_rank_distance"])
