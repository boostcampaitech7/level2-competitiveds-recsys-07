
def filter_data_by_loc(data, train_data, lat_offset=0.019, lon_offset=0.0228):
    '''
    아파트 위치를 기준으로 특정 범위 내의 데이터를 필터링하는 함수.
    
    Args:
    - data (DataFrame): 필터링할 데이터 (latitude, longitude 컬럼 포함).
    - train_data (DataFrame): 아파트 또는 기준 위치 데이터 (latitude, longitude 컬럼 포함).
    - lat_offset (float): 위도 경계에 더할 오프셋 값 (기본값: 0.019).
    - lon_offset (float): 경도 경계에 더할 오프셋 값 (기본값: 0.0228).
    - 대략 2km 마진을 줌
    Returns:
    - DataFrame: 지정한 범위 내의 데이터로 필터링된 결과.
    '''
    
    min_lat = train_data['latitude'].min() - lat_offset
    max_lat = train_data['latitude'].max() + lat_offset
    min_lon = train_data['longitude'].min() - lon_offset
    max_lon = train_data['longitude'].max() + lon_offset
    
    filtered_data = data[
        (data['latitude'] >= min_lat) & 
        (data['latitude'] <= max_lat) & 
        (data['longitude'] >= min_lon) & 
        (data['longitude'] <= max_lon)
    ]
    
    return filtered_data
