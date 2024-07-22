import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from scripts.plots import generate_plots, generate_cluster_score_plots, generate_cluster_plots


GENERATE_FIGURES = False
FIND_OPTIMAL_CLUSTERS = False
GENERATE_CLUSTERS = False
TRAIN_NEW_MODEL = False

OPTIMAL_CLUSTERS = 4

RAW_DATA = r"./data/flight_delay.csv"
CLEAN_DATA = r"./data/cleaned_data.csv"
CLUSTER_CENTER_DATA = r"./data/cluster_centers.csv"
MODEL_DATA = r"./models/random_forest_regressor.pkl"
PREDICTION_DATA = r"./data/prediction.csv"

DAY_MAP = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI", 6: "SAT", 7: "SUN"}


def __process_time_column(df, col_name, date_col):
    df[col_name] = df[col_name].apply(lambda x: str(x).zfill(4))
    df[col_name] = df[date_col].astype(str) + " " + df[col_name]
    df[col_name] = pd.to_datetime(df[col_name], format="%Y-%m-%d %H%M", errors="coerce")

    return df[col_name]


def __poly_transform(df, poly_features, columns: list):
    interaction_features = poly_features.fit_transform(df[columns])
    interaction_df = pd.DataFrame(interaction_features, columns=poly_features.get_feature_names_out(columns))
    interaction_df.rename(columns={f"{columns[0]} {columns[1]}": f"{columns[0]}_{columns[1]}"}, inplace=True)

    return interaction_df[f"{columns[0]}_{columns[1]}"]


def _preprocess():
    label_encoder = LabelEncoder()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    data_df = pd.read_csv(RAW_DATA)

    data_df["DepDate"] = pd.to_datetime(data_df["Date"], dayfirst=True)
    data_df = data_df.drop(columns=["Date"])

    data_df["DepTime"] = __process_time_column(data_df, "DepTime", "DepDate")
    data_df["ArrTime"] = data_df.apply(lambda x: x["DepTime"] + pd.offsets.Minute(x["ActualElapsedTime"]), axis=1)
    data_df["ArrDate"] = data_df["ArrTime"].dt.date
    data_df["CRSArrTime"] = __process_time_column(data_df, "CRSArrTime", "ArrDate")
    data_df["CRSDepTime"] = data_df.apply(lambda x: x["CRSArrTime"] - pd.offsets.Minute(x["CRSElapsedTime"]), axis=1)

    data_df["DepHour"] = data_df["DepTime"].dt.hour
    data_df["DepTimeMinutes"] = data_df["DepTime"].dt.hour * 60 + data_df["DepTime"].dt.minute
    data_df["CRSDepHour"] = data_df["CRSDepTime"].dt.hour
    data_df["CRSDepTimeMinutes"] = data_df["CRSDepTime"].dt.hour * 60 + data_df["CRSDepTime"].dt.minute
    data_df["ArrHour"] = data_df["ArrTime"].dt.hour
    data_df["ArrTimeMinutes"] = data_df["ArrTime"].dt.hour * 60 + data_df["ArrTime"].dt.minute
    data_df["CRSArrHour"] = data_df["CRSArrTime"].dt.hour
    data_df["CRSArrTimeMinutes"] = data_df["CRSArrTime"].dt.hour * 60 + data_df["CRSArrTime"].dt.minute

    data_df["DayOfWeekEncoded"] = data_df["DayOfWeek"].replace(to_replace=7, value=0)
    data_df["DayOfWeek"] = data_df["DayOfWeek"].replace(DAY_MAP)

    data_df["Month"] = data_df["DepDate"].dt.month

    data_df["NonCarrierDelay"] = data_df["WeatherDelay"] + data_df["NASDelay"] + data_df["SecurityDelay"] + data_df[
        "LateAircraftDelay"]
    data_df["FlightID"] = data_df["UniqueCarrier"] + data_df["FlightNum"].astype(str)
    data_df["FlightIDEncoded"] = label_encoder.fit_transform(data_df["FlightID"])

    data_df["Origin_Dep_Count"] = data_df.groupby(["Origin", "DepDate", "CRSDepHour"])["FlightNum"].transform("count")
    data_df["Dest_Arr_Count"] = data_df.groupby(["Dest", "ArrDate", "CRSArrHour"])["FlightNum"].transform("count")

    data_df = data_df.dropna(subset=["DepTime", "ArrTime", "DepHour", "ArrHour", "Origin_Dep_Count", "Dest_Arr_Count"])

    dep_day_df = __poly_transform(data_df, poly, ["DepTimeMinutes", "DayOfWeekEncoded"])
    arr_day_df = __poly_transform(data_df, poly, ["CRSArrTimeMinutes", "DayOfWeekEncoded"])
    data_df = pd.concat([data_df, dep_day_df, arr_day_df], axis=1)

    data_df = data_df.drop_duplicates(inplace=False)

    data_df = data_df.dropna(subset=["Org_Airport", "Dest_Airport", "DepTime", "ArrTime", "DepHour", "ArrHour",
                                     "DepTimeMinutes_DayOfWeekEncoded", "CRSArrTimeMinutes_DayOfWeekEncoded",
                                     "Origin_Dep_Count", "Dest_Arr_Count"], inplace=False)

    column_order = ["DayOfWeek", "DayOfWeekEncoded", "Month", "DepDate", "DepTime", "DepTimeMinutes", "DepHour",
                    "CRSDepTime", "CRSDepTimeMinutes", "CRSDepHour", "ArrDate", "ArrTime", "ArrTimeMinutes", "ArrHour",
                    "CRSArrTime", "CRSArrTimeMinutes", "CRSArrHour", "Distance", "ActualElapsedTime", "CRSElapsedTime",
                    "AirTime", "UniqueCarrier", "FlightNum", "FlightID", "FlightIDEncoded", "Airline", "TailNum",
                    "Origin",
                    "Org_Airport", "Dest", "Dest_Airport", "TaxiOut", "TaxiIn", "Cancelled", "CancellationCode",
                    "Diverted",
                    "DepDelay", "ArrDelay", "CarrierDelay", "NonCarrierDelay", "WeatherDelay", "NASDelay",
                    "SecurityDelay",
                    "LateAircraftDelay", "DepTimeMinutes_DayOfWeekEncoded", "CRSArrTimeMinutes_DayOfWeekEncoded",
                    "Origin_Dep_Count", "Dest_Arr_Count"]
    ordered_df = data_df[column_order]

    ordered_df.to_csv(CLEAN_DATA, index=False)
    print("> > > Preprocessing complete.")


def _describe(generate_figures: bool = GENERATE_FIGURES):
    data_df = pd.read_csv(CLEAN_DATA)

    numerical_features = ["DayOfWeekEncoded", "Month", "DepTimeMinutes", "DepHour", "CRSDepTimeMinutes", "CRSDepHour",
                          "ArrTimeMinutes", "ArrHour", "CRSArrTimeMinutes", "CRSArrHour", "Distance",
                          "ActualElapsedTime",
                          "CRSElapsedTime", "AirTime", "TaxiOut", "TaxiIn", "Origin_Dep_Count", "Dest_Arr_Count"]
    delay_features = ["DepDelay", "ArrDelay", "CarrierDelay", "NonCarrierDelay", "WeatherDelay", "NASDelay",
                      "SecurityDelay", "LateAircraftDelay"]
    categorical_features = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "FlightID"]

    print("\nNumerical features description:")
    print(data_df[numerical_features].describe())

    print("\nDelay features description:")
    print(data_df[delay_features].describe())

    print("\nCategorical features description:")
    print(data_df[categorical_features].describe())

    if generate_figures:
        generate_plots(data_df, numerical_features, delay_features)

    print("> > > Description complete.")


def _cluster(find_k: bool = FIND_OPTIMAL_CLUSTERS, generate_figures: bool = GENERATE_FIGURES):
    data_df = pd.read_csv(CLEAN_DATA)
    features = ["DayOfWeekEncoded", "Month", "DepTimeMinutes", "DepHour", "CRSDepTimeMinutes", "CRSDepHour", "ArrTimeMinutes", "ArrHour", "CRSArrTimeMinutes", "CRSArrHour", "Distance", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "TaxiOut", "TaxiIn", "Origin_Dep_Count", "Dest_Arr_Count"]
    features += [col for col in data_df.columns if col.startswith("UniqueCarrier_")]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_df[features])

    if find_k:
        wcss = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=777)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            print(f"WCSS for {i} clusters: ", kmeans.inertia_)

        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=777)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels, random_state=777)
            silhouette_scores.append(silhouette_avg)
            print(f"Silhouette Score for {i} clusters: ", silhouette_avg)

        generate_cluster_score_plots(wcss, silhouette_scores)

    kmeans = KMeans(n_clusters=OPTIMAL_CLUSTERS, random_state=777)
    cluster_labels = kmeans.fit_predict(X_scaled)
    data_df["Cluster"] = cluster_labels

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

    cluster_centers_df.to_csv(CLUSTER_CENTER_DATA, index=False)

    if generate_figures:
        generate_cluster_plots(data_df)

    print("> > > Clustering complete.")


def _predict(train_new_model: bool = TRAIN_NEW_MODEL):
    data_df = pd.read_csv(CLEAN_DATA)
    data_df = pd.get_dummies(data_df, columns=["UniqueCarrier"], drop_first=True)

    features = ["DayOfWeekEncoded", "Month", "CRSDepTimeMinutes", "CRSArrTimeMinutes", "Distance", "CRSElapsedTime",
                "FlightIDEncoded", "TaxiOut", "DepTimeMinutes_DayOfWeekEncoded", "CRSArrTimeMinutes_DayOfWeekEncoded",
                "Origin_Dep_Count", "Dest_Arr_Count"]
    features = features + [col for col in data_df.columns if col.startswith("UniqueCarrier_")]
    target = "ArrDelay"

    X = data_df[features]
    y = data_df[target]

    if train_new_model:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
        model = RandomForestRegressor(random_state=777)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        with joblib.parallel_backend("threading"):
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R2 Score: {r2}")

        feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print("Feature Importance:\n", feature_importance)
        joblib.dump(best_model, MODEL_DATA)

        print("> > > Model training complete.")

    else:
        model = joblib.load(MODEL_DATA)
        y_pred = model.predict(X)

        prediction_df = pd.concat([data_df, pd.DataFrame({"Prediction": y_pred})], axis=1)
        prediction_df.to_csv(PREDICTION_DATA)
        print("> > > Prediction complete.")

if __name__ == '__main__':
    print("> > > Preprocessing...")
    _preprocess()

    print("> > > Describing...")
    _describe(GENERATE_FIGURES)

    if GENERATE_CLUSTERS:
        print("> > > Clustering...")
        _cluster(GENERATE_CLUSTERS, GENERATE_FIGURES)

    print(f"> > > {'Training new model' if TRAIN_NEW_MODEL else 'Predicting'}...")
    _predict(TRAIN_NEW_MODEL)
