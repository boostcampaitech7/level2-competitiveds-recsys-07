'''
example:
train_data = add_previous_interest_rate(train_data, interest_rate, months=3)

주의:
Train data 시작: 2019-04-01 00:00:00
Train data 끝: 2023-12-01 00:00:00
Interest rate 시작: 2018-12-01 00:00:00
Interest rate 끝: 2024-05-01 00:00:00

->> 따라서 5개월 전 금리부터는 nan값 생김.
'''

import pandas as pd

def add_previous_interest_rate(train_data, interest_rate_data, months):
    '''
    계약 날짜를 기준으로 이전 개월의 이자율을 추가하는 함수.
    
    Args:
    - train_data (DataFrame): 계약 데이터가 포함된 데이터프레임 (year_month 컬럼 포함).
    - interest_rate_data (DataFrame): 이자율 데이터가 포함된 데이터프레임 (year_month 및 interest_rate 컬럼 포함).
    - months (int): 이전 개월의 이자율을 조회할 개월 수.
    
    Returns:
    - DataFrame: 업데이트된 train_data, 이전 이자율 컬럼 추가.
    '''
    
    train_data['year_month'] = pd.to_datetime(train_data['contract_year_month'], format='%Y%m')
    interest_rate_data['year_month'] = pd.to_datetime(interest_rate_data['year_month'], format='%Y%m')
    
    def get_previous_interest(contract_date, interest_rate_data, months_before):
        previous_date = contract_date - pd.DateOffset(months=months_before)
        rate = interest_rate_data.loc[interest_rate_data['year_month'] == previous_date, 'interest_rate']
        return rate.values[0] if not rate.empty else None
    
    column_name = f'interest_{months}_months_ago'
    train_data[column_name] = train_data['year_month'].apply(lambda x: get_previous_interest(x, interest_rate_data, months))
    
    return train_data
