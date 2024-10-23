from haversine import Unit, haversine


# 특정 좌표에 대한 반경 내 개수 계산 함수 (haversine 사용)
def count_within_radius_haversine(lat, lon, locations, radius_m):
    count = 0
    for loc in locations:
        distance = haversine((lat, lon), (loc[0], loc[1]), unit=Unit.METERS)
        if distance <= radius_m:
            count += 1
    return count
