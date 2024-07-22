# Flight Delay Prediction Project

## Overview
This project aims to predict flight delays using various machine learning techniques. The dataset contains detailed information about flight schedules, including departure and arrival times, carrier details, and delay causes. The project is divided into several steps, from data cleaning to predictive modeling.

### Objectives
1. **Pattern Recognition:** Analyze the data to identify patterns and main causes of flight delays.
2. **Improvement Suggestions:** Derive actionable insights to improve airport operations based on identified patterns.
3. **Forecasting:** Develop predictive models to forecast future delays.

### Project Steps

1. **Data Cleaning**: Preprocess the raw data to handle missing values, incorrect data entries, and feature engineering.
   - Code: `notebooks/data_cleaning.ipynb`

2. **Data Exploration**: Perform exploratory data analysis (EDA) to understand data distributions, correlations, and insights.
   - Code: `notebooks/data_exploration.ipynb`

3. **Pattern Analysis**: Identify patterns in the data that could help in predicting flight delays.
   - Code: `notebooks/pattern_analysis.ipynb`

4. **Cluster Analysis**: Group similar flights together to find common characteristics that may influence delays.
   - Code: `notebooks/cluster_analysis.ipynb`

5. **Predictive Modeling**: Build and evaluate machine learning models to predict flight delays.
   - Code: `notebooks/predictive_modeling.ipynb`

## Data Description
The dataset contains 484,551 rows and 29 columns, with a file size of approximately 89 MB. Each row represents a flight and includes information about delays and their causes.

### Features
The dataset contains several features, both original and newly engineered. Below is a list of the main features used in the project:

- **DayOfWeekEncoded**: Encoded day of the week (0-6, where 0 is Monday).
- **DepHour**: Hour of the scheduled departure time.
- **ArrHour**: Hour of the actual arrival time.
- **CRSArrHour**: Hour of the scheduled arrival time.
- **FlightNum**: Unique flight number.
- **ActualElapsedTime**: Actual flight time.
- **AirTime**: Time spent in the air.
- **Distance**: Distance between the origin and destination airports.
- **TaxiIn**: Time spent taxiing in after landing.
- **TaxiOut**: Time spent taxiing out before takeoff.
- **DepHour_DayOfWeek**: Interaction feature between departure hour and day of the week.
- **DepHour_UniqueCarrier**: Interaction feature between departure hour and unique carrier.
- **CRSArrHour_DayOfWeek**: Interaction feature between scheduled arrival hour and day of the week.
- **Origin_Dep_Count**: Count of departures from the origin airport.
- **Dest_Arr_Count**: Count of arrivals at the destination airport.
- **Month**: Month of the flight.
- **UniqueCarrier_UA**: Binary feature indicating if the carrier is United Airlines.
- **UniqueCarrier_MQ**: Binary feature indicating if the carrier is Envoy Air.
- **UniqueCarrier_US**: Binary feature indicating if the carrier is US Airways.
- **UniqueCarrier_DL**: Binary feature indicating if the carrier is Delta Air Lines.
- **UniqueCarrier_WN**: Binary feature indicating if the carrier is Southwest Airlines.
- **UniqueCarrier_OO**: Binary feature indicating if the carrier is SkyWest Airlines.
- **UniqueCarrier_B6**: Binary feature indicating if the carrier is JetBlue Airways.
- **UniqueCarrier_EV**: Binary feature indicating if the carrier is ExpressJet Airlines.
- **UniqueCarrier_F9**: Binary feature indicating if the carrier is Frontier Airlines.
- **UniqueCarrier_AS**: Binary feature indicating if the carrier is Alaska Airlines.
- **UniqueCarrier_HA**: Binary feature indicating if the carrier is Hawaiian Airlines.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd flight-delay-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Follow the steps in the notebooks in the given order for a complete analysis and prediction workflow:
   - `notebooks/data_cleaning.ipynb`
   - `notebooks/data_exploration.ipynb`
   - `notebooks/pattern_analysis.ipynb`
   - `notebooks/cluster_analysis.ipynb`
   - `notebooks/predictive_modeling.ipynb`

## Results and Insights
The project has produced several insights into the factors contributing to flight delays, including the time of day, day of the week, and specific carriers. The predictive model has been fine-tuned and evaluated for accuracy, providing a robust tool for forecasting flight delays.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.