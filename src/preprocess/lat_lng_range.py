import numpy as np
import pandas as pd
from geopy.distance import geodesic

def get_extended_lat_lng_range(train_path, test_path, distance_km=10):
    """
    주어진 train, test CSV 파일의 위도/경도 데이터를 바탕으로, 각 경계에서 distance_km만큼 확장된 위도/경도 값을 계산.
    - train_path: train.csv 파일 경로
    - test_path: test.csv 파일 경로
    - distance_km: 경계에서 확장할 거리 (km), 기본값은 10km
    """
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 두 데이터를 합치기
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 위도와 경도의 최소/최대값을 계산
    lat_min, lat_max = combined_df['latitude'].min(), combined_df['latitude'].max()
    lon_min, lon_max = combined_df['longitude'].min(), combined_df['longitude'].max()

    # 위도 경도에 10km 떨어진 지점 계산 함수
    def shift_lat_lon(lat, lon, distance_km, direction):
        if direction == 'lat':  # 위도를 이동할 경우
            return geodesic(kilometers=distance_km).destination((lat, lon), 0).latitude
        elif direction == 'lon':  # 경도를 이동할 경우
            return geodesic(kilometers=distance_km).destination((lat, lon), 90).longitude

    # 최소값에서 10km 떨어진 위도/경도 계산
    lat_min_10km = shift_lat_lon(lat_min, lon_min, -distance_km, 'lat')
    lon_min_10km = shift_lat_lon(lat_min, lon_min, -distance_km, 'lon')

    # 최대값에서 10km 떨어진 위도/경도 계산
    lat_max_10km = shift_lat_lon(lat_max, lon_max, distance_km, 'lat')
    lon_max_10km = shift_lat_lon(lat_max, lon_max, distance_km, 'lon')

    # 반환
    return lat_min_10km, lat_max_10km, lon_min_10km, lon_max_10km
