from haversine_utils import count_within_radius_haversine


# 거리 계산 실행 및 결과 추가 (haversine 사용)
def add_radius_count_features_to_unique_coords_haversine(unique_coords, park, school, subway):
    radii = [500, 1000, 2000, 3000, 4000, 5000]

    elementary_schools = school[school["schoolLevel"] == "elementary"][["latitude", "longitude"]].values
    middle_schools = school[school["schoolLevel"] == "middle"][["latitude", "longitude"]].values
    high_schools = school[school["schoolLevel"] == "high"][["latitude", "longitude"]].values

    parks = park[["latitude", "longitude"]].values
    subways = subway[["latitude", "longitude"]].values

    # 새로운 데이터프레임에 결과 추가
    for radius in radii:
        unique_coords[f"park_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(row["latitude"], row["longitude"], parks, radius), axis=1
        )

        unique_coords[f"subway_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(row["latitude"], row["longitude"], subways, radius), axis=1
        )

        unique_coords[f"school_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(
                row["latitude"], row["longitude"], school[["latitude", "longitude"]].values, radius
            ),
            axis=1,
        )

        unique_coords[f"elementary_school_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(row["latitude"], row["longitude"], elementary_schools, radius),
            axis=1,
        )

        unique_coords[f"middle_school_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(row["latitude"], row["longitude"], middle_schools, radius), axis=1
        )

        unique_coords[f"high_school_count_{radius}m"] = unique_coords.apply(
            lambda row: count_within_radius_haversine(row["latitude"], row["longitude"], high_schools, radius), axis=1
        )

    return unique_coords
