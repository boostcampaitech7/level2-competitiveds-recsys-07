
def categorize_season(month):
    '''
    월을 기준으로 계절을 분류하는 함수.
    
    Args:
    - month (int): 월 (1~12).
    
    Returns:
    - str: 해당 월에 따른 계절 ('Spring', 'Summer', 'Fall', 'Winter').
    '''
    
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'



def categorize_interest(x):
    '''
    관심도를 구간에 따라 분류하는 함수.
    
    example:
    train_data['interest_category'] = train_data['interest_rate'].apply(categorize_interest)
    
    Args:
    - x (float): 관심도 값 (0~5 범위).
    
    Returns:
    - str: 해당 관심도 구간에 따른 분류 ('Very Low', 'Low', 'Medium', 'High', 'Very High').
    '''
    
    if 0 <= x < 1:
        return 'Very Low'
    elif 1 <= x < 2:
        return 'Low'
    elif 2 <= x < 3:
        return 'Medium'
    elif 3 <= x < 4:
        return 'High'
    elif 4 <= x < 5:
        return 'Very High'


def categorize_area(x):
    '''
    면적을 50 단위로 나누어 범위 카테고리로 변환하는 함수.
    
    Args:
    - x (float): 면적 값.
    
    Returns:
    - str: 50 단위로 구분된 범위 (예: "0~40", "50~90").
    '''
    
    range_start = (x // 50) * 50
    range_end = range_start + 40
    return f"{range_start}~{range_end}"



def categorize_date(x):
    '''
    날짜를 10일 단위로 구분하는 함수.
    
    Args:
    - x (int): 일(day) 값 (1~31).
    
    Returns:
    - int: 10일 단위로 구분된 값 (10, 20, 30).
    '''
    
    if 1 <= x <= 10:
        return 10
    elif 11 <= x <= 20:
        return 20
    else:
        return 30



def categorize_price(x):
    '''
    가격을 10,000 단위로 구분하여 범위 카테고리로 변환하는 함수.
    
    Args:
    - x (int): 가격 값.
    
    Returns:
    - str: 10,000 단위로 구분된 범위 (예: "0~9999", "10000~19999").
    '''
    
    scale = 10000
    range_start = (x // scale) * scale
    range_end = range_start + scale - 1
    return f"{range_start}~{range_end}"
