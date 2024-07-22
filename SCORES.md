# Scores and Model Evaluation
## V3 (current)
### Scores
| Score                          | Value               |
|--------------------------------|---------------------|
| Mean Absolute Error (MAE)      | 10.232931161046345  |
| Mean Squared Error (MSE)       | 437.6047673486665   |
| Root Mean Squared Error (RMSE) | 20.919004932086672  |
| R2 Score                       | 0.8660554237363611  |
#### As Text
Mean Absolute Error (MAE): 10.232931161046345
Mean Squared Error (MSE): 437.6047673486665
Root Mean Squared Error (RMSE): 20.919004932086672
R2 Score: 0.8660554237363611
### Feature Importance
| Feature | Importance |
|---------|------------|
| CRSArrTimeMinutes |                     0.315038 |
| CRSDepTimeMinutes |                     0.264234 |
| TaxiOut |                               0.143657 |
| Distance |                              0.142160 |
| FlightIDEncoded |                       0.027057 |
| CRSElapsedTime |                        0.026127 |
| Dest_Arr_Count |                        0.025406 |
| Origin_Dep_Count |                      0.015813 |
| DepTimeMinutes_DayOfWeekEncoded |       0.006189 |
| CRSArrTimeMinutes_DayOfWeekEncoded |    0.006156 |
| Month |                                 0.005502 |
| DayOfWeekEncoded |                      0.004831 |
| UniqueCarrier_UA |                      0.002421 |
| UniqueCarrier_EV |                      0.002337 |
| UniqueCarrier_DL |                      0.002131 |
| UniqueCarrier_MQ |                      0.002091 |
| UniqueCarrier_OO |                      0.001999 |
| UniqueCarrier_B6 |                      0.001686 |
| UniqueCarrier_US |                      0.001598 |
| UniqueCarrier_HA |                      0.001248 |
| UniqueCarrier_F9 |                      0.000938 |
| UniqueCarrier_AS |                      0.000805 |
| UniqueCarrier_WN |                      0.000576 |
#### As Text
Feature Importance:
CRSArrTimeMinutes                     0.315038
CRSDepTimeMinutes                     0.264234
TaxiOut                               0.143657
Distance                              0.142160
FlightIDEncoded                       0.027057
CRSElapsedTime                        0.026127
Dest_Arr_Count                        0.025406
Origin_Dep_Count                      0.015813
DepTimeMinutes_DayOfWeekEncoded       0.006189
CRSArrTimeMinutes_DayOfWeekEncoded    0.006156
Month                                 0.005502
DayOfWeekEncoded                      0.004831
UniqueCarrier_UA                      0.002421
UniqueCarrier_EV                      0.002337
UniqueCarrier_DL                      0.002131
UniqueCarrier_MQ                      0.002091
UniqueCarrier_OO                      0.001999
UniqueCarrier_B6                      0.001686
UniqueCarrier_US                      0.001598
UniqueCarrier_HA                      0.001248
UniqueCarrier_F9                      0.000938
UniqueCarrier_AS                      0.000805
UniqueCarrier_WN                      0.000576
dtype: float64
- - -
- - -
## OLD
### V2
#### Scores
| Score                          | Value               |
|--------------------------------|---------------------|
| Mean Absolute Error (MAE)      | 18.197205678351104  |
| Mean Squared Error (MSE)       | 748.923003414731    |
| Root Mean Squared Error (RMSE) | 27.366457633656772  |
| R2 Score                       | 0.7685732898415767  |
##### As Text
Mean Absolute Error (MAE): 18.197205678351104
Mean Squared Error (MSE): 748.923003414731
Root Mean Squared Error (RMSE): 27.366457633656772
R2 Score: 0.7685732898415767
#### Feature Importance
| Feature | Importance |
|---------|------------|
| CRSArrHour | 0.331120   |
| DepHour | 0.226948   |
| Distance | 0.159003   |
| TaxiOut | 0.095764   |
| FlightNum | 0.073605   |
| DepHour_UniqueCarrier | 0.026614   |
| Month | 0.021446   |
| CRSArrHour_DayOfWeek | 0.009519   |
| DepHour_DayOfWeek | 0.009081   |
| UniqueCarrier_WN | 0.007869   |
| DayOfWeekEncoded | 0.006949   |
| Dest_Arr_Count | 0.005124   |
| Origin_Dep_Count | 0.004782   |
| UniqueCarrier_UA | 0.003383   |
| UniqueCarrier_MQ | 0.003254   |
| UniqueCarrier_B6 | 0.003166   |
| UniqueCarrier_DL | 0.003017   |
| UniqueCarrier_EV | 0.002043   |
| UniqueCarrier_US | 0.001902   |
| UniqueCarrier_F9 | 0.001682   |
| UniqueCarrier_OO | 0.001505   |
| UniqueCarrier_HA | 0.001254   |
| UniqueCarrier_AS | 0.000969   |
##### As Text
Feature Importance:
CRSArrHour               0.331120
DepHour                  0.226948
Distance                 0.159003
TaxiOut                  0.095764
FlightNum                0.073605
DepHour_UniqueCarrier    0.026614
Month                    0.021446
CRSArrHour_DayOfWeek     0.009519
DepHour_DayOfWeek        0.009081
UniqueCarrier_WN         0.007869
DayOfWeekEncoded         0.006949
Dest_Arr_Count           0.005124
Origin_Dep_Count         0.004782
UniqueCarrier_UA         0.003383
UniqueCarrier_MQ         0.003254
UniqueCarrier_B6         0.003166
UniqueCarrier_DL         0.003017
UniqueCarrier_EV         0.002043
UniqueCarrier_US         0.001902
UniqueCarrier_F9         0.001682
UniqueCarrier_OO         0.001505
UniqueCarrier_HA         0.001254
UniqueCarrier_AS         0.000969
dtype: float64
- - -
### V1
#### Scores
| Score                          | Value              |
|--------------------------------|--------------------|
| Mean Absolute Error (MAE)      | 12.697516640817986 |
| Mean Squared Error (MSE)       | 307.6406997996649  |
| Root Mean Squared Error (RMSE) | 17.53968927317884  |
| R2 Score                       | 0.9052400615913125 |
##### As Text
Mean Absolute Error (MAE): 12.697516640817986
Mean Squared Error (MSE): 307.6406997996649
Root Mean Squared Error (RMSE): 17.53968927317884
R2 Score: 0.9052400615913125
#### Feature Importance
| Feature | Importance |
|---------|------------|
| CRSArrHour               | 0.511841 |
| ArrHour                  | 0.328525 |
| TaxiOut                  | 0.047191 |
| FlightNum                | 0.019823 |
| Distance                 | 0.016584 |
| ActualElapsedTime        | 0.014640 |
| AirTime                  | 0.011607 |
| TaxiIn                   | 0.009649 |
| DepHour                  | 0.008848 |
| Month                    | 0.006635 |
| DepHour_UniqueCarrier    | 0.006011 |
| DepHour_DayOfWeek        | 0.004452 |
| CRSArrHour_DayOfWeek     | 0.003178 |
| DayOfWeekEncoded         | 0.002519 |
| Origin_Dep_Count         | 0.001431 |
| Dest_Arr_Count           | 0.001365 |
| UniqueCarrier_UA         | 0.000910 |
| UniqueCarrier_MQ         | 0.000836 |
| UniqueCarrier_US         | 0.000796 |
| UniqueCarrier_DL         | 0.000617 |
| UniqueCarrier_WN         | 0.000551 |
| UniqueCarrier_OO         | 0.000528 |
| UniqueCarrier_B6         | 0.000437 |
| UniqueCarrier_EV         | 0.000384 |
| UniqueCarrier_F9         | 0.000342 |
| UniqueCarrier_AS         | 0.000277 |
| UniqueCarrier_HA         | 0.000021 |
##### As Text
Feature Importance:
CRSArrHour               0.511841
ArrHour                  0.328525
TaxiOut                  0.047191
FlightNum                0.019823
Distance                 0.016584
ActualElapsedTime        0.014640
AirTime                  0.011607
TaxiIn                   0.009649
DepHour                  0.008848
Month                    0.006635
DepHour_UniqueCarrier    0.006011
DepHour_DayOfWeek        0.004452
CRSArrHour_DayOfWeek     0.003178
DayOfWeekEncoded         0.002519
Origin_Dep_Count         0.001431
Dest_Arr_Count           0.001365
UniqueCarrier_UA         0.000910
UniqueCarrier_MQ         0.000836
UniqueCarrier_US         0.000796
UniqueCarrier_DL         0.000617
UniqueCarrier_WN         0.000551
UniqueCarrier_OO         0.000528
UniqueCarrier_B6         0.000437
UniqueCarrier_EV         0.000384
UniqueCarrier_F9         0.000342
UniqueCarrier_AS         0.000277
UniqueCarrier_HA         0.000021
dtype: float64