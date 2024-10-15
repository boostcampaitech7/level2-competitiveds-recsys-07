import pandas as pd
from haversine import haversine, Unit
from tqdm import tqdm


def calculate_total_park_area_within_radius(train_data, park_data, radius=1000):
    '''
    특정 반경 내의 총 공원 면적을 계산하는 함수.
    
    Args:
    - train_data (DataFrame): 아파트 또는 건물의 위치 데이터 (latitude, longitude 컬럼 포함).
    - park_data (DataFrame): 공원 위치 및 면적 데이터 (latitude, longitude, area 컬럼 포함).
    - radius (float): 반경 거리 (기본값: 1000미터).
    
    Returns:
    - DataFrame: 각 아파트 위치에 대해 반경 내의 총 공원 면적을 포함하는 데이터프레임.
    '''
    
    unique_locations = train_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    
    total_park_areas = []
    
    for idx, location in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Calculating total park areas"):
        apartment_location = (location['latitude'], location['longitude'])
        total_area = 0
        
        for _, park in park_data.iterrows():
            park_location = (park['latitude'], park['longitude'])
            distance = haversine(apartment_location, park_location, unit=Unit.METERS)
            
            if distance <= radius:
                total_area += park['area']
        
        total_park_areas.append((location['latitude'], location['longitude'], total_area))
    
    park_area_df = pd.DataFrame(total_park_areas, columns=['latitude', 'longitude', 'total_park_area_within_radius'])
    return park_area_df



def calculate_subway_distance(train_data, subway_data):
    '''
    아파트 위치와 지하철역 간의 거리를 계산하고, 특정 거리 내 지하철역 개수를 집계하는 함수.
    
    Args:
    - train_data (DataFrame): 아파트 또는 건물의 위치 데이터 (latitude, longitude 컬럼 포함).
    - subway_data (DataFrame): 지하철역 위치 데이터 (latitude, longitude 컬럼 포함).
    
    Returns:
    - DataFrame: 각 아파트 위치에 대해 거리별(250m, 500m, 1km, 2km) 지하철역 개수를 포함한 데이터프레임.
    '''
    
    unique_locations = train_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    
    distances = []
    
    for idx, location in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Calculating distances"):
        apartment_location = (location['latitude'], location['longitude'])
        
        for _, subway in subway_data.iterrows():
            subway_location = (subway['latitude'], subway['longitude'])
            distance = haversine(apartment_location, subway_location, unit=Unit.METERS)
            distances.append((location['latitude'], location['longitude'], distance))
    
    distance_df = pd.DataFrame(distances, columns=['latitude', 'longitude', 'distance'])
    
    counts = distance_df.groupby(['latitude', 'longitude']).agg(
        subway_within_250m=('distance', lambda x: (x<=250).sum()),
        subway_within_500m=('distance', lambda x: (x<=500).sum()),
        subway_within_1km=('distance', lambda x: (x<=1000).sum()),
        subway_within_2km=('distance', lambda x: (x<=2000).sum())
    ).reset_index()
    
    return counts



def calculate_school_distance(train_data, school_data):
    '''
    아파트 위치와 학교 간의 거리를 계산하고, 특정 거리 내 학교 개수를 집계하는 함수.
    
    Args:
    - train_data (DataFrame): 아파트 또는 건물의 위치 데이터 (latitude, longitude 컬럼 포함).
    - school_data (DataFrame): 학교 위치 데이터 (latitude, longitude 컬럼 포함).
    
    Returns:
    - DataFrame: 각 아파트 위치에 대해 거리별(250m, 500m, 1km, 2km) 학교 개수를 포함한 데이터프레임.
    '''
    
    unique_locations = train_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

    distances = []
    
    for idx, location in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Calculating distances"):
        apartment_location = (location['latitude'], location['longitude'])
        
        for _, school in school_data.iterrows():
            school_location = (school['latitude'], school['longitude'])
            distance = haversine(apartment_location, school_location, unit=Unit.METERS)
            distances.append((location['latitude'], location['longitude'], distance))
    
    distance_df = pd.DataFrame(distances, columns=['latitude', 'longitude', 'distance'])
    
    counts = distance_df.groupby(['latitude', 'longitude']).agg(
        school_within_250m=('distance', lambda x: (x<=250).sum()),
        school_within_500m=('distance', lambda x: (x<=500).sum()),
        school_within_1km=('distance', lambda x: (x<=1000).sum()),
        school_within_2km=('distance', lambda x: (x<=2000).sum())
    ).reset_index()
    
    return counts



def calculate_park_distance(train_data, park_data):
    '''
    아파트 위치와 공원 간의 거리를 계산하고, 특정 거리 내 공원 개수를 집계하는 함수.
    
    Args:
    - train_data (DataFrame): 아파트 또는 건물의 위치 데이터 (latitude, longitude 컬럼 포함).
    - park_data (DataFrame): 공원 위치 데이터 (latitude, longitude 컬럼 포함).
    
    Returns:
    - DataFrame: 각 아파트 위치에 대해 거리별(250m, 500m, 1km, 2km) 공원 개수를 포함한 데이터프레임.
    '''
    
    unique_locations = train_data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

    distances = []
    
    for idx, location in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Calculating distances"):
        apartment_location = (location['latitude'], location['longitude'])
        
        for _, park in park_data.iterrows():
            park_location = (park['latitude'], park['longitude'])
            distance = haversine(apartment_location, park_location, unit=Unit.METERS)
            distances.append((location['latitude'], location['longitude'], distance))
    
    distance_df = pd.DataFrame(distances, columns=['latitude', 'longitude', 'distance'])
    
    counts = distance_df.groupby(['latitude', 'longitude']).agg(
        park_within_250m=('distance', lambda x: (x<=250).sum()),
        park_within_500m=('distance', lambda x: (x<=500).sum()),
        park_within_1km=('distance', lambda x: (x<=1000).sum()),
        park_within_2km=('distance', lambda x: (x<=2000).sum())
    ).reset_index() 
    
    return counts
