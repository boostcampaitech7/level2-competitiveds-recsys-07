# 필요한 라이브러리 설치 및 불러오기
import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")
submission_df = pd.read_csv("/content/sample_submission.csv")
subway_info = pd.read_csv("/content/subway.csv")
school_info = pd.read_csv("/content/school.csv")
park_info = pd.read_csv("/content/park.csv")
ir_info = pd.read_csv("/content/interestRate.csv")

train_df["year"] = pd.to_datetime(train_df["contract_year_month"], format="%Y%m").dt.year
train_2022 = train_df[train_df["year"] == 2022]
train_2023 = train_df[train_df["year"] == 2023]

# 1. 지구 반지름 상수 정의 (미터 단위)
R = 6371000  # 지구 반지름 (단위: 미터)


# 2. Haversine 공식을 사용한 거리 계산 함수 (CuPy를 사용하여 거리 계산)
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = cp.radians(lat1), cp.radians(lon1), cp.radians(lat2), cp.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = cp.sin(dlat / 2.0) ** 2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon / 2.0) ** 2
    c = 2 * cp.arcsin(cp.sqrt(a))

    return R * c  # 미터 단위 거리 반환


# 3. 반경 내 아파트들의 평균 전세가 계산 (2023년도만 고려)
def calculate_avg_deposit(facility_lat, facility_lon, apt_lats, apt_lons, deposits, radius=2000):
    """반경 내 2023년 아파트들의 평균 전세가 계산 (CuPy 사용)"""
    # 거리 계산
    distances = haversine_distance(facility_lat, facility_lon, apt_lats, apt_lons)
    within_radius = distances <= radius

    if cp.sum(within_radius) > 0:
        return cp.mean(deposits[within_radius])  # 반경 내 아파트들의 평균 전세가
    else:
        return cp.nan  # 해당 반경 내 아파트가 없으면 NaN 반환


# 4. 데이터의 위도, 경도, 전세가를 CuPy 배열로 변환
apt_lats_2022 = cp.array(train_2022["latitude"].values)
apt_lons_2022 = cp.array(train_2022["longitude"].values)
deposits_2022 = cp.array(train_2022["deposit"].values)
apt_lats_2023 = cp.array(train_2023["latitude"].values)
apt_lons_2023 = cp.array(train_2023["longitude"].values)
deposits_2023 = cp.array(train_2023["deposit"].values)


# 5. tqdm을 사용하여 각 시설의 평균 전세가 계산 (2022년 기준)
def calculate_avg_deposit_for_facilities_2022(facility_info, facility_type):
    facility_avg_deposits_2022 = []

    for _, row in tqdm(
        facility_info.iterrows(),
        total=len(facility_info),
        desc=f"Calculating average deposits for {facility_type} (2022 only)",
    ):
        facility_lat = cp.array(row["latitude"])  # 시설의 위도 (CuPy 배열로 변환)
        facility_lon = cp.array(row["longitude"])  # 시설의 경도 (CuPy 배열로 변환)

        # Calculate average deposit
        avg_deposit = calculate_avg_deposit(facility_lat, facility_lon, apt_lats_2022, apt_lons_2022, deposits_2022)

        # Convert to NumPy if it's a CuPy array, otherwise keep as is (handling NaN)
        facility_avg_deposits_2022.append(avg_deposit.get() if isinstance(avg_deposit, cp.ndarray) else avg_deposit)

    return facility_avg_deposits_2022


# 5. tqdm을 사용하여 각 시설의 평균 전세가 계산 (2023년 기준)
def calculate_avg_deposit_for_facilities_2023(facility_info, facility_type):
    facility_avg_deposits_2023 = []

    for _, row in tqdm(
        facility_info.iterrows(),
        total=len(facility_info),
        desc=f"Calculating average deposits for {facility_type} (2023 only)",
    ):
        facility_lat = cp.array(row["latitude"])  # 시설의 위도 (CuPy 배열로 변환)
        facility_lon = cp.array(row["longitude"])  # 시설의 경도 (CuPy 배열로 변환)

        # Calculate average deposit
        avg_deposit = calculate_avg_deposit(facility_lat, facility_lon, apt_lats_2023, apt_lons_2023, deposits_2023)

        # Convert to NumPy if it's a CuPy array, otherwise keep as is (handling NaN)
        facility_avg_deposits_2023.append(avg_deposit.get() if isinstance(avg_deposit, cp.ndarray) else avg_deposit)

    return facility_avg_deposits_2023


# 각 시설에 대해 계산된 평균 전세가 추가
subway_info["avg_deposit_2022"] = calculate_avg_deposit_for_facilities_2022(subway_info, "subway")
school_info["avg_deposit_2022"] = calculate_avg_deposit_for_facilities_2022(school_info, "school")
park_info["avg_deposit_2022"] = calculate_avg_deposit_for_facilities_2022(park_info, "park")

subway_info["avg_deposit_2023"] = calculate_avg_deposit_for_facilities_2023(subway_info, "subway")
school_info["avg_deposit_2023"] = calculate_avg_deposit_for_facilities_2023(school_info, "school")
park_info["avg_deposit_2023"] = calculate_avg_deposit_for_facilities_2023(park_info, "park")

# 확인

print(subway_info[["latitude", "longitude", "avg_deposit_2022", "avg_deposit_2023"]].head())
print(school_info[["latitude", "longitude", "avg_deposit_2022", "avg_deposit_2023"]].head())
print(park_info[["latitude", "longitude", "avg_deposit_2022", "avg_deposit_2023"]].head())


# 6. 각 시설 중 가장 가까운 시설의 2022년 평균 전세가를 찾아주는 함수(train)
def find_closest_facility_avg_deposit(apt_lat, apt_lon, facilities_lats, facilities_lons, facilities_avg_deposits):
    distances = haversine_distance(apt_lat, apt_lon, facilities_lats, facilities_lons)
    closest_index = cp.argmin(distances)
    closest_index_np = closest_index.get()
    return facilities_avg_deposits[closest_index_np]


# 7. 시설의 위도, 경도 및 2022년 평균 전세가 정보를 CuPy로 변환
facilities_lats_subway = cp.array(subway_info["latitude"].values)
facilities_lons_subway = cp.array(subway_info["longitude"].values)
facilities_avg_deposits_subway = subway_info["avg_deposit_2022"].values  # CuPy 변환 불필요

facilities_lats_school = cp.array(school_info["latitude"].values)
facilities_lons_school = cp.array(school_info["longitude"].values)
facilities_avg_deposits_school = school_info["avg_deposit_2022"].values  # CuPy 변환 불필요

facilities_lats_park = cp.array(park_info["latitude"].values)
facilities_lons_park = cp.array(park_info["longitude"].values)
facilities_avg_deposits_park = park_info["avg_deposit_2022"].values  # CuPy 변환 불필요

# 8. train 데이터에서 가장 가까운 지하철, 학교, 공원의 2022년 평균 전세가를 Feature로 추가
test_avg_deposits_subway = []
test_avg_deposits_school = []
test_avg_deposits_park = []

for _, row in tqdm(train_2023.iterrows(), total=len(train_2023), desc="Assigning 2023 avg deposit to test data"):
    apt_lat = cp.array(row["latitude"])  # 아파트의 위도 (CuPy 배열로 변환)
    apt_lon = cp.array(row["longitude"])  # 아파트의 경도 (CuPy 배열로 변환)

    avg_deposit_subway = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_subway, facilities_lons_subway, facilities_avg_deposits_subway
    )
    avg_deposit_school = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_school, facilities_lons_school, facilities_avg_deposits_school
    )
    avg_deposit_park = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_park, facilities_lons_park, facilities_avg_deposits_park
    )

    test_avg_deposits_subway.append(avg_deposit_subway)
    test_avg_deposits_school.append(avg_deposit_school)
    test_avg_deposits_park.append(avg_deposit_park)

# 9. train 데이터에 Feature 추가
train_2023["subway"] = test_avg_deposits_subway
train_2023["school"] = test_avg_deposits_school
train_2023["park"] = test_avg_deposits_park

# 결과 확인
print(train_2023[["latitude", "longitude", "subway", "school", "park"]].head())


# 6. 각 시설 중 가장 가까운 시설의 2023년 평균 전세가를 찾아주는 함수
def find_closest_facility_avg_deposit(apt_lat, apt_lon, facilities_lats, facilities_lons, facilities_avg_deposits):
    distances = haversine_distance(apt_lat, apt_lon, facilities_lats, facilities_lons)
    closest_index = cp.argmin(distances)
    closest_index_np = closest_index.get()
    return facilities_avg_deposits[closest_index_np]


# 7. 시설의 위도, 경도 및 2023년 평균 전세가 정보를 CuPy로 변환
facilities_lats_subway = cp.array(subway_info["latitude"].values)
facilities_lons_subway = cp.array(subway_info["longitude"].values)
facilities_avg_deposits_subway = subway_info["avg_deposit_2023"].values  # CuPy 변환 불필요

facilities_lats_school = cp.array(school_info["latitude"].values)
facilities_lons_school = cp.array(school_info["longitude"].values)
facilities_avg_deposits_school = school_info["avg_deposit_2023"].values  # CuPy 변환 불필요

facilities_lats_park = cp.array(park_info["latitude"].values)
facilities_lons_park = cp.array(park_info["longitude"].values)
facilities_avg_deposits_park = park_info["avg_deposit_2023"].values  # CuPy 변환 불필요

# 8. Test 데이터에서 가장 가까운 지하철, 학교, 공원의 2023년 평균 전세가를 Feature로 추가
test_avg_deposits_subway = []
test_avg_deposits_school = []
test_avg_deposits_park = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Assigning 2023 avg deposit to test data"):
    apt_lat = cp.array(row["latitude"])  # 아파트의 위도 (CuPy 배열로 변환)
    apt_lon = cp.array(row["longitude"])  # 아파트의 경도 (CuPy 배열로 변환)

    avg_deposit_subway = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_subway, facilities_lons_subway, facilities_avg_deposits_subway
    )
    avg_deposit_school = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_school, facilities_lons_school, facilities_avg_deposits_school
    )
    avg_deposit_park = find_closest_facility_avg_deposit(
        apt_lat, apt_lon, facilities_lats_park, facilities_lons_park, facilities_avg_deposits_park
    )

    test_avg_deposits_subway.append(avg_deposit_subway)
    test_avg_deposits_school.append(avg_deposit_school)
    test_avg_deposits_park.append(avg_deposit_park)

# 9. Test 데이터에 Feature 추가
test_df["subway"] = test_avg_deposits_subway
test_df["school"] = test_avg_deposits_school
test_df["park"] = test_avg_deposits_park

# 결과 확인
print(test_df[["latitude", "longitude", "subway", "school", "park"]].head())
