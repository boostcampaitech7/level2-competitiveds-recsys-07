# 데이터셋 분리 및 저장 함수
def split_and_save_datasets(facilities_added_df):
    # train과 test로 나누기 및 dataset 칼럼 삭제
    facilities_added_train = facilities_added_df[facilities_added_df["dataset"] == "train"].drop(columns=["dataset"])
    facilities_added_test = facilities_added_df[facilities_added_df["dataset"] == "test"].drop(
        columns=["dataset", "deposit"]
    )

    # CSV 파일로 저장
    facilities_added_train.to_csv("../data/facilities_added_train.csv", index=False)
    facilities_added_test.to_csv("../data/facilities_added_test.csv", index=False)
