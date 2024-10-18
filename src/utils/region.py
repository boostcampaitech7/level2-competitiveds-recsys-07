from haversine import haversine


def group_by_distance(df, k=20):

    # median point
    median_long = df.longitude.median()
    median_lat = df.latitude.median()

    # median point에서 아파트(위도, 경도)까지의 거리
    df["distance"] = df.apply(
        lambda x: haversine((median_lat, median_long), (x["latitude"], x["longitude"]), unit="km"), axis=1
    )

    # 소수점 4자리로 반올림
    df["distance"] = df["distance"].round(4)

    # 클래스 추가(0: 반경 이내, 1: 반경 이외)
    df["distance_class"] = 1
    df.loc[df["distance"] <= k, "distance_class"] = 0

    return df
