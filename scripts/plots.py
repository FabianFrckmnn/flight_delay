import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots(df, numerical_features, delay_features):
    sns.set(style="whitegrid")

    plt.figure(figsize=(25, 20))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(6, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(r"../figures/numerical_features_histogram.png")

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(delay_features, 1):
        plt.subplot(2, 4, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(r"../figures/delay_features_histogram.png")

    DAYS_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    df["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=DAYS_ORDER, ordered=True)

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="DayOfWeek", order=DAYS_ORDER)
    plt.title("Flights by Day of Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Flight Count")
    plt.savefig(r"../figures/flights_per_day_countplot.png")

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="UniqueCarrier", order=sorted(df["UniqueCarrier"].unique()))
    plt.title("Flights by Carrier")
    plt.xlabel("Carrier")
    plt.ylabel("Flight Count")
    plt.xticks(rotation=90)
    plt.savefig(r"../figures/flights_per_carrier_countplot.png")

    categorical_columns = ["Origin", "Dest"]
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(2, 2, i)
        top_10 = df[col].value_counts().head(10)
        sns.barplot(x=top_10.index, y=top_10.values)
        plt.title(f"Top 10 Most Common {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(r"../figures/most_common_dest_origin_barplot.png")

    categorical_columns = ["Origin", "Dest"]
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(2, 2, i)
        bot_10 = df[col].value_counts().tail(10)
        sns.barplot(x=bot_10.index, y=bot_10.values)
        plt.title(f"Top 10 Least Common {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(r"../figures/least_common_dest_origin_barplot.png")

    plt.figure(figsize=(12, 5))
    sns.histplot(df["ArrDelay"], kde=True, bins=30)
    plt.title("Distribution of Arrival Delays")
    plt.xlabel("Arrival Delay (minutes)")
    plt.ylabel("Frequency")
    plt.savefig(r"../figures/arr_delay_histogram.png")

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[delay_features])
    plt.title("Box Plot of Delay Types")
    plt.ylabel("Delay Time (minutes)")
    plt.xticks(rotation=45)
    plt.savefig(r"../figures/delay_types_boxplot.png")

    plt.figure(figsize=(12, 5))
    sns.histplot(df["DepTimeMinutes_DayOfWeekEncoded"], kde=True, bins=50)
    plt.title("Distribution of DepHour_DayOfWeek Interaction")
    plt.xlabel("DepHour_DayOfWeek")
    plt.ylabel("Count")
    plt.savefig(r"../figures/dep_hour_day_histogram.png")

    plt.figure(figsize=(12, 5))
    sns.histplot(df["CRSArrTimeMinutes_DayOfWeekEncoded"], kde=True, bins=50)
    plt.title("Distribution of CRSArrHour_DayOfWeek Interaction")
    plt.xlabel("CRSArrHour_DayOfWeek")
    plt.ylabel("Count")
    plt.savefig(r"../figures/crs_arr_hour_day_histogram.png")

    plt.figure(figsize=(12, 5))
    sns.histplot(df["Origin_Dep_Count"], kde=True, bins=50)
    plt.title("Distribution of Origin Departure Counts")
    plt.xlabel("Origin_Dep_Count")
    plt.ylabel("Count")
    plt.savefig(r"../figures/origin_dep_counts_histogram.png")

    plt.figure(figsize=(12, 5))
    sns.histplot(df["Dest_Arr_Count"], kde=True, bins=50)
    plt.title("Distribution of Destination Arrival Counts")
    plt.xlabel("Dest_Arr_Count")
    plt.ylabel("Count")
    plt.savefig(r"../figures/dest_arr_counts_histogram.png")

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="Month", y="ArrDelay")
    plt.title("Arrival Delays by Month")
    plt.xlabel("Month")
    plt.ylabel("Arrival Delay (minutes)")
    plt.savefig(r"../figures/arr_delay_month_boxplot.png")

    plt.figure(figsize=(12, 5))
    sns.scatterplot(x="Distance", y="AirTime", data=df, alpha=0.5)
    plt.title("Relation of Distance and Airtime")
    plt.xlabel("Distance (miles)")
    plt.ylabel("AirTime (minutes)")
    plt.grid(True)
    plt.savefig(r"../figures/distance_vs_airtime_scatter.png")

    correlation_matrix = df[numerical_features + delay_features].corr()
    plt.figure(figsize=(30, 20))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.savefig(r"../figures/correlation_matrix.png")


def generate_cluster_score_plots(wcss, silhouette_scores):
    plt.figure(figsize=(10,6))
    plt.plot(range(2, 11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig(r"../figures/_kmeans_wcss.png")

    plt.plot(range(2, 11), silhouette_scores, marker="o")
    plt.title("Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.savefig(r"../figures/_kmeans_silhouette_score.png")


def generate_cluster_plots(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="DepTimeMinutes", y="ArrTimeMinutes", hue="Cluster", palette="viridis")
    plt.title("Time of Departure vs Time of Arrival (in Minutes since Midnight)")
    plt.xlabel("DepTimeMinutes")
    plt.ylabel("ArrTimeMinutes")
    plt.legend(title="Cluster")
    plt.savefig(r"../figures/_cluster_deptimeminutes_arrtimeminutes.png")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="Distance", y="AirTime", hue="Cluster", palette="viridis")
    plt.title("Distance vs AirTime")
    plt.xlabel("Distance")
    plt.ylabel("AirTime")
    plt.legend(title="Cluster")
    plt.savefig(r"../figures/_cluster_distance_airtime.png")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="Distance", y="ArrTimeMinutes", hue="Cluster", palette="viridis")
    plt.title("Distance vs Time of Arrival (in Minutes since Midnight)")
    plt.xlabel("Distance")
    plt.ylabel("ArrTimeMinutes")
    plt.legend(title="Cluster")
    plt.savefig(r"../figures/_cluster_distance_arrtimeminutes.png")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="Distance", y="DepTimeMinutes", hue="Cluster", palette="viridis")
    plt.title("Distance vs Time of Departure (in Minutes since Midnight)")
    plt.xlabel("Distance")
    plt.ylabel("DepTimeMinutes")
    plt.legend(title="Cluster")
    plt.savefig(r"../figures/_cluster_distance_deptimeminutes.png")