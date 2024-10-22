import pandas as pd


def load_data():
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")
    park = pd.read_csv("../data/filtered_park.csv")
    school = pd.read_csv("../data/filtered_school.csv")
    subway = pd.read_csv("../data/subway.csv")

    # 두 데이터를 합치기
    combined_df = pd.concat([train_data, test_data], ignore_index=True)
    return combined_df, park, school, subway
