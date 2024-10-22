import pandas as pd
from haversine import haversine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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


def knn_clustering(df_train, df_test, features=["latitude", "longitude"], k=2):
    x_train = df_train[features]
    y_train = df_train["deposit"]

    x_test = df_test[features]

    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(x_train, y_train)

    # Train dataset
    df_train["pred_deposit"] = knn_regressor.predict(x_train)
    df_train["cluster"] = pd.cut(df_train["pred_deposit"], bins=[0, 50000, float("inf")], labels=[0, 1]).astype(int)
    df_train["cluster"] = df_train["cluster"].astype("category")

    # Test dataset
    df_test["pred_deposit"] = knn_regressor.predict(x_test)
    df_test["cluster"] = pd.cut(df_test["pred_deposit"], bins=[0, 50000, float("inf")], labels=[0, 1]).astype(int)
    df_test["cluster"] = df_test["cluster"].astype("category")

    return df_train, df_test


def knn_clf_clustering(df_train, df_test, bins, labels, features=["latitude", "longitude"], k=10):
    # 위도,경도 그룹화(평균 전세가 생성)
    df_train_mean = df_train.groupby(["latitude", "longitude"])[["deposit"]].mean()
    df_train_mean.reset_index(inplace=True)

    # train cluster 생성
    df_train_mean["cluster"] = pd.cut(df_train_mean["deposit"], bins=bins, labels=labels).astype(int)

    # KNN
    x_train = df_train_mean[features]
    y_train = df_train_mean["cluster"]

    knn_regressor = KNeighborsClassifier(n_neighbors=k)
    knn_regressor.fit(x_train, y_train)

    # predict cluster
    df_test_cluster = df_test.copy()
    x_test = df_test[features]
    y_pred = knn_regressor.predict(x_test)
    df_test_cluster["cluster"] = y_pred

    df_train_cluster = pd.merge(
        df_train, df_train_mean[["latitude", "longitude", "cluster"]], on=["latitude", "longitude"]
    )

    # Change type
    df_train_cluster["cluster"] = df_train_cluster["cluster"].astype("category")
    df_test_cluster["cluster"] = df_test_cluster["cluster"].astype("category")

    return df_train_cluster, df_test_cluster
