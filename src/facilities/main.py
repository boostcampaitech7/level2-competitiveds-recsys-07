import pandas as pd
from data_loader import load_data
from dataset_utils import split_and_save_datasets
from feature_engineering import \
    add_radius_count_features_to_unique_coords_haversine


def main():
    # 데이터 로드
    combined_df, park, school, subway = load_data()

    # 고유한 위도/경도 좌표 추출
    unique_coords = combined_df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

    # 반경 내 시설 개수 추가
    unique_coords_with_features = add_radius_count_features_to_unique_coords_haversine(
        unique_coords, park, school, subway
    )

    # 원래 combined_df에 결과 매핑 (facilities_added_df로 저장)
    facilities_added_df = pd.merge(combined_df, unique_coords_with_features, on=["latitude", "longitude"], how="left")

    # 결과 확인 및 저장
    print(facilities_added_df.head())
    facilities_added_df.to_csv("../data/facilities_added.csv", index=False)

    # 데이터셋 분리 및 저장
    split_and_save_datasets(facilities_added_df)


if __name__ == "__main__":
    main()
