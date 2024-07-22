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

- **DayOfWeek**: Day of the week (SUN, MON, TUE, ...).
- **DayOfWeekEncoded**: Encoded day of the week (0-6, where 0 is Sunday).
- **Month**: Month of the flight.
- **DepDate**: Date of departure.
- **DepTime**: Actual departure time.
- **DepTimeMinutes**: Actual departure time in minutes since midnight.
- **DepTimeHour**: Hour of the actual departure time.
- **CRSDepTime**: Scheduled departure time.
- **CRSDepTimeMinutes**: Scheduled departure time in minutes since midnight.
- **CRSDepHour**: Hour of the scheduled departure time.
- **ArrDate**: Date of arrival.
- **ArrTime**: Actual arrival time.
- **ArrTimeMinutes**: Actual arrival time in minutes since midnight.
- **ArrHour**: Hour of the actual arrival time.
- **CRSArrTime**: Scheduled arrival time.
- **CRSArrTimeMinutes**: Scheduled arrival time in minutes since midnight.
- **CRSArrHour**: Hour of the scheduled arrival time.
- **Distance**: Distance between the origin and destination airports.
- **ActualElapsedTime**: Actual flight time.
- **CRSElapsedTime**: Scheduled flight time.
- **AirTime**: Time spent in the air.
- **UniqueCarrier**: Carrier of the flight.
- **FlightNum**: Flight number.
- **FlightID**: Identification of the carrier and flight number.
- **FlightIDEncoded**: Encoded identification of the carrier and flight number.
- **Airline**: Airline of the flight.
- **TailNum**: Tail number of the plane.
- **Origin**: Identification of the origin airport.
- **Org_Airport**: Name of the origin airport.
- **Dest**: Identification of the destination airport.
- **Dest_Airport**: Name of the destination airport.
- **TaxiOut**: Time spent taxiing out before takeoff.
- **TaxiIn**: Time spend taxiing in after landing.
- **Cancelled**: Indication for cancelled flights.
- **CancellationCode**: Reason for cancellation of flights.
- **Diverted**: Indication for diverted flights.
- **DepDelay**: Departure delay in minutes.
- **ArrDelay**: Arrival delay in minutes.
- **CarrierDelay**: Delay caused by the carrier in minutes.
- **NonCarrierDelay**: Sum of delays not caused by carrier in minutes.
- **WeatherDelay**: Delay caused by weather in minutes.
- **NASDelay**: Delay caused by NAS (National Aviation System) in minutes.
- **SecurityDelay**: Delay caused by security concerns in minutes.
- **LateAircraftDelay**: Delay caused by previous delays in minutes.
- **DepTimeMinutes_DayOfWeekEncoded**: Interaction feature between actual departure time and day of the week.
- **CRSArrTimeMinutes_DayOfWeekEncoded**: Interaction feature between scheduled arrival time and day of the week.
- **Origin_Dep_Count**: Count of simultaneous (scheduled) departures from the origin airport.
- **Dest_Arr_Count**: Count of simultaneous (scheduled) arrivals at the destination airport.


## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd flight_delay
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

4. Run `main.py` to automatically generate graphs, model and prediction.

## Results and Insights
The project has produced several insights into the factors contributing to flight delays, including the time of day, day of the week, and specific carriers. The predictive model has been fine-tuned and evaluated for accuracy, providing a robust tool for forecasting flight delays.
To find the scores of the model evaluations (V1, V2 and V3), see the [SCORES](SCORES.md) file.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.