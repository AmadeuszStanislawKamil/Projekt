import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')
import csv
import datetime as dt
from collections import Counter
from regressors import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 1 - Load Plant 1 Data
plant_1_generation_Data = pd.read_csv("PlantData/GenerationData/Plant_1_Generation_Data.csv", \
                                      quoting=csv.QUOTE_NONE, error_bad_lines=False)
# Load Weather data of plant 1
Plant1_weather_Sensor = pd.read_csv('PlantData/WeatherSensorData/Plant_1_Weather_Sensor_Data.csv', \
                                    quoting=csv.QUOTE_NONE, error_bad_lines=False)
# Change data type for date time
plant_1_generation_Data['DATE_TIME'] = plant_1_generation_Data['DATE_TIME'].apply(
    lambda x: dt.date.strftime(dt.datetime.strptime(x, '%d-%m-%Y %H:%M') \
                               , "%m/%d/%Y %H:%M"))
Plant1_weather_Sensor['DATE_TIME'] = Plant1_weather_Sensor['DATE_TIME'].apply \
    (lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), "%m/%d/%Y %H:%M"))
# Merge Plant and Weather data
data_from_Plant_1 = plant_1_generation_Data.merge(
    Plant1_weather_Sensor[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DATE_TIME']], \
    how='left',
    left_on=['DATE_TIME'], \
    right_on=['DATE_TIME'])
data_from_Plant_1['DATE_TIME'] = data_from_Plant_1['DATE_TIME'].apply(
    lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M'))
data_from_Plant_1['TIME'] = data_from_Plant_1['DATE_TIME'].apply(lambda x: str(x.hour) + str(x.minute))
test_df = data_from_Plant_1.copy()
# 04 - Compare total yield to DC_power
used_for_testing = test_df[(test_df['SOURCE_KEY'] == '1IF53ai7Xc0U56Y')]
figure, axis = plt.subplots()  # Create the figure and axes object

# Plot the first x and y axes:
used_for_testing.plot(x='DATE_TIME', y='TOTAL_YIELD', ax=axis)
used_for_testing.plot(x='DATE_TIME', y='DC_POWER', ax=axis, secondary_y=True)


# 03 - Data Cleaning Functions
def Change_data_format(dataframe):
    global start_time, tot_yld, dc_pow, ac_pow, mod_temp, amb_temp
    dc_ind = False
    one_ind = False
    data = []
    for row in dataframe.itertuples():
        if int(row.DC_POWER) == 0:
            if dc_ind is False:
                plant_id = row.PLANT_ID
                source_key = row.SOURCE_KEY
                dc_pow = row.DC_POWER
                ac_pow = row.AC_POWER
                tot_yld = row.TOTAL_YIELD
                start_time = row.DATE_TIME
                amb_temp = row.AMBIENT_TEMPERATURE
                mod_temp = row.MODULE_TEMPERATURE
                irrad = row.IRRADIATION
                dc_ind = True
            else:
                end_time = row.DATE_TIME
                minutes_diff = (end_time - start_time).total_seconds() / 60.0
                tots = row.TOTAL_YIELD
                yld_diff = tots - tot_yld
                dc_pow = (row.DC_POWER + dc_pow) / 2
                ac_pow = (row.AC_POWER + ac_pow) / 2
                mod_temp = (row.MODULE_TEMPERATURE + mod_temp) / 2
                amb_temp = (row.AMBIENT_TEMPERATURE + amb_temp) / 2
                irrad = (row.IRRADIATION + irrad) / 2
                one_ind = True

                pass
        else:
            if dc_ind is True:
                if one_ind is True:
                    try:
                        data.append([plant_id, source_key, start_time, end_time, minutes_diff, dc_pow, \
                                     ac_pow, tots, yld_diff, amb_temp, mod_temp, irrad])
                        del end_time, minutes_diff, tots, yld_diff
                        one_ind = False
                    except Exception as e:
                        print("one", e, row)
                else:
                    try:
                        data.append([plant_id, source_key, data[-1][3], start_time, \
                                     (start_time - data[-1][3]).total_seconds() / 60.0, \
                                     dc_pow, ac_pow, tot_yld, (tot_yld - data[-1][7]), amb_temp, \
                                     mod_temp, irrad])
                    except Exception as e:
                        print("two", e, row)

                try:
                    data.append([row.PLANT_ID, row.SOURCE_KEY, data[-1][3], row.DATE_TIME, \
                                 (row.DATE_TIME - data[-1][3]).total_seconds() / 60.0, row.DC_POWER, row.AC_POWER, \
                                 row.TOTAL_YIELD, row.TOTAL_YIELD - data[-1][7], row.AMBIENT_TEMPERATURE, \
                                 row.MODULE_TEMPERATURE, row.IRRADIATION])
                    dc_ind = False
                except Exception as e:
                    print("three", e, row)
            else:
                try:
                    start_time = data[-1][3]
                except Exception as e:
                    print("special", data, e, row)
                minutes_diff = (row.DATE_TIME - start_time).total_seconds() / 60.0
                tot_yld = data[-1][7]
                yld_diff = row.TOTAL_YIELD - tot_yld
                try:
                    data.append([row.PLANT_ID, row.SOURCE_KEY, start_time, row.DATE_TIME, minutes_diff, \
                                 row.DC_POWER, row.AC_POWER, row.TOTAL_YIELD, yld_diff, row.AMBIENT_TEMPERATURE, \
                                 row.MODULE_TEMPERATURE, row.IRRADIATION])
                except Exception as e:
                    print("four", e, row)
    return data


# 01 - Call the Cleaning function for each Inverter
final_dataFrame = pd.DataFrame(columns=['plant_id', 'source_key', 'start_time', 'end_time', \
                                        'minutes', 'dc_pow', 'ac_pow', 'total_yield', 'yield_generated' \
    , 'amb_temp', 'mod_temp', 'irrad'])
for i in set(test_df.SOURCE_KEY):
    dfs = test_df[test_df['SOURCE_KEY'] == i]
    dfs.reset_index(drop=True)
    temp_df = pd.DataFrame(data=Change_data_format(dfs), columns=final_dataFrame.columns)
    final_dataFrame = final_dataFrame.append(temp_df)

# Check time when DC_POWER is zero
final_dataFrame['timestamp'] = final_dataFrame['end_time'].apply(lambda x: str(x.hour) + str(x.minute))
a = list(final_dataFrame[final_dataFrame['dc_pow'] == 0].timestamp)
letter_counts = Counter(a)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df1 = df.head(60)
df1.plot(kind='bar')

# Delete records with TIME as 5:45
no_power_df = final_dataFrame[final_dataFrame['timestamp'] == '545']
# Remove them from the test_df
final_dataFrame = pd.concat([final_dataFrame, no_power_df, no_power_df]).drop_duplicates(keep=False)
final_dataFrame = final_dataFrame.reset_index(drop=True)

# Add a new dimension as Yield Per Minute
final_dataFrame['ypm'] = (final_dataFrame['yield_generated'] * 1.00) / final_dataFrame['minutes']
final_dataFrame['effic'] = ((final_dataFrame['ac_pow'] / final_dataFrame['dc_pow']) * 100.0).fillna(0)
final_dataFrame['datestamp'] = final_dataFrame['end_time'].apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))
final_dataFrame[final_dataFrame.isna().any(axis=1)]

final_dataFrame.irrad.fillna(method='ffill', inplace=True)
final_dataFrame.mod_temp.fillna(method='ffill', inplace=True)
final_dataFrame.amb_temp.fillna(method='ffill', inplace=True)

used_for_testing = final_dataFrame[(final_dataFrame['source_key'] == 'zBIq5rxdHJRwDNY')]
figure, axis = plt.subplots()  # Create the figure and axes object
used_for_testing['dtemp'] = used_for_testing['mod_temp'] * 10

# Plot the first x and y axes:
# test.plot(x = 'datestamp', y = 'dc_pow', ax = ax)
# test.plot(x = 'datestamp', y = 'yield_generated', ax = ax, secondary_y = True)
# test.plot(x = 'datestamp', y = 'dtemp', ax = ax, secondary_y = True)

# OneHotEncoding for inverters
dummies = pd.get_dummies(final_dataFrame.source_key, drop_first=True)
Solar_plant_1 = pd.concat([final_dataFrame, dummies], axis='columns')
del Solar_plant_1['source_key']
Solar_plant_1 = Solar_plant_1.reset_index(drop=True)

# Prep data for building model
yield_generation = Solar_plant_1['ypm']
data = Solar_plant_1.copy()
del data['ypm']
del data['plant_id'], data['start_time'], data['end_time'], data['minutes'], data['total_yield'], data['timestamp'], \
    data['datestamp'], data['yield_generated']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(data, yield_generation, test_size=0.3)

# Build Model
model1 = linear_model.ElasticNet()
model1.fit(X_train, Y_train)
Y_pred = model1.predict(X_test)

y_pred_sum = np.sum(Y_pred)
y_test_sum = np.sum(Y_test)
print(abs(y_pred_sum/y_test_sum*100),'%')

from sklearn.metrics import mean_squared_error, r2_score

print("Coefficients:  ", model1.coef_)
print("Intercept:  ", model1.intercept_)
print("MSE:  %.2f" % mean_squared_error(Y_test, Y_pred))
print("Coefficients of determination:  %.2f" % r2_score(Y_test, Y_pred))

print("\n=========== SUMMARY ===========")
xlabels = X_test.columns
stats.summary(model1, X_test, Y_test, xlabels)

figure = sns.regplot(x=Y_pred, y=Y_test, scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.title('Plant 1 Prediction:')
figure.set(xlabel='Predicted YPM', ylabel='Recorded YPM')

# Source Plant 2 data
data_for_pPlant2 = pd.read_csv("PlantData/GenerationData/Plant_2_Generation_Data.csv", \
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
# Load Weather data of plant 2
data_for_watherStation2 = pd.read_csv('PlantData/WeatherSensorData/Plant_2_Weather_Sensor_Data.csv', \
                                      quoting=csv.QUOTE_NONE, error_bad_lines=False)

# Change data type for date time
data_for_pPlant2['DATE_TIME'] = data_for_pPlant2['DATE_TIME'].apply(
    lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') \
                               , "%m/%d/%Y %H:%M"))
data_for_watherStation2['DATE_TIME'] = data_for_watherStation2['DATE_TIME'].apply \
    (lambda x: dt.date.strftime(dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), "%m/%d/%Y %H:%M"))

# Merge Plant and Weather data
plant_data2 = data_for_pPlant2.merge(
    data_for_watherStation2[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DATE_TIME']], \
    how='left',
    left_on=['DATE_TIME'], \
    right_on=['DATE_TIME'])
plant_data2['DATE_TIME'] = plant_data2['DATE_TIME'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %H:%M'))
plant_data2['TIME'] = plant_data2['DATE_TIME'].apply(lambda x: str(x.hour) + str(x.minute))

plant_data2.describe()
test_df2 = plant_data2.copy()

used_for_testing = test_df2[(test_df2['SOURCE_KEY'] == '9kRcWv60rDACzjR')]
figure, axis = plt.subplots()  # Create the figure and axes object

# Plot the first x and y axes:
# test.plot(x = 'DATE_TIME', y = 'TOTAL_YIELD', ax = ax)
# test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax, secondary_y = True)

used_for_testing = test_df2[(test_df2['SOURCE_KEY'] == '9kRcWv60rDACzjR') & (test_df2['DC_POWER'] == 0)]
figure, axis = plt.subplots()  # Create the figure and axes object

# Plot the first x and y axes:
# test.plot(x = 'DATE_TIME', y = 'IRRADIATION', ax = ax)
# test.plot(x = 'DATE_TIME', y = 'MODULE_TEMPERATURE', ax = ax)
# test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax, secondary_y = True)

data = []
dumps = []
d2 = []
for j in set(test_df2.SOURCE_KEY):
    dfs = test_df2[test_df2['SOURCE_KEY'] == j]
    dfs.reset_index(drop=True)
    ind = 0

    for i in dfs.itertuples():
        if ind == 0:
            data.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD, \
                         i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            ind += 1
        else:
            if i.TOTAL_YIELD >= data[-1][6]:
                data.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD, \
                             i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            elif dumps:
                if i.TOTAL_YIELD <= dumps[-1][6]:
                    d2.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD, \
                               i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
            else:
                dumps.append([i.DATE_TIME, i.PLANT_ID, i.SOURCE_KEY, i.DC_POWER, i.AC_POWER, i.DAILY_YIELD, \
                              i.TOTAL_YIELD, i.AMBIENT_TEMPERATURE, i.MODULE_TEMPERATURE, i.IRRADIATION, i.TIME])
dump_df = pd.DataFrame(data=dumps, columns=test_df2.columns)
data_df = pd.DataFrame(data=data, columns=test_df2.columns)
d2_df = pd.DataFrame(data=d2, columns=test_df2.columns)
used_for_testing = data_df[(data_df['SOURCE_KEY'] == '9kRcWv60rDACzjR')]
figure, axis = plt.subplots()  # Create the figure and axes object
used_for_testing.plot(x = 'DC_POWER', y = 'MODULE_TEMPERATURE', ax = axis, secondary_y = True)
no_dc2 = data_df[(data_df['MODULE_TEMPERATURE'] > 38) & (data_df['DC_POWER'] == 0)]
data_df = pd.concat([data_df, no_dc2, no_dc2]).drop_duplicates(keep=False)
data_df = data_df.reset_index(drop=True)

used_for_testing = data_df[(data_df['SOURCE_KEY'] == '9kRcWv60rDACzjR')]
figure, axis = plt.subplots()  # Create the figure and axes object

# Plot the first x and y axes:
used_for_testing.plot(x = 'DC_POWER', y = 'MODULE_TEMPERATURE', ax = axis, secondary_y = True)

used_for_testing = data_df[(data_df['SOURCE_KEY'] == '9kRcWv60rDACzjR')]
figure, axis = plt.subplots()  # Create the figure and axes object
# Plot the first x and y axes:
# test.plot(x = 'DATE_TIME', y = 'DC_POWER', ax = ax)
# test.plot(x = 'DATE_TIME', y = 'TOTAL_YIELD', ax = ax, secondary_y = True)

# 04 - Call the Cleaning function for each Inverter
Final_df2 = pd.DataFrame(columns=['plant_id', 'source_key', 'start_time', 'end_time', \
                                  'minutes', 'dc_pow', 'ac_pow', 'total_yield', 'yield_generated' \
    , 'amb_temp', 'mod_temp', 'irrad'])
for i in set(data_df.SOURCE_KEY):
    dfs = data_df[data_df['SOURCE_KEY'] == i]
    dfs.reset_index(drop=True)
    temp_df = pd.DataFrame(data=Change_data_format(dfs), columns=Final_df2.columns)
    Final_df2 = Final_df2.append(temp_df)
# Add a new dimension as Yield Per Minute
Final_df2['ypm'] = (Final_df2['yield_generated'] * 1.00) / Final_df2['minutes']
Final_df2['effic'] = ((Final_df2['ac_pow'] / Final_df2['dc_pow']) * 100.0).fillna(0)
Final_df2['datestamp'] = Final_df2['end_time'].apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))
Final_df2[Final_df2.isna().any(axis=1)]
used_for_testing = Final_df2[(Final_df2['source_key'] == '9kRcWv60rDACzjR')]
figure, axis = plt.subplots()  # Create the figure and axes object

# Plot the first x and y axes:
used_for_testing.plot(x = 'dc_pow', y = 'mod_temp', ax = axis, secondary_y = True)
# Verify if there is any records with 0 dc_power
Final_df2['time'] = Final_df2['end_time'].apply(lambda x: str(x.hour) + str(x.minute))

a = list(Final_df2[Final_df2['dc_pow'] == 0].time)
letter_counts = Counter(a)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df1 = df.head(60)
# df.plot(kind='bar')
# Delete records with TIME as 4:45
no_power_df2 = Final_df2[Final_df2['time'] == '545']
# Remove them from the test_df
Final_df2 = pd.concat([Final_df2, no_power_df2, no_power_df2]).drop_duplicates(keep=False)
Final_df2 = Final_df2.reset_index(drop=True)
# OneHotEncoding for inverters
dummies2 = pd.get_dummies(Final_df2.source_key, drop_first=True)
Solar_plant_2 = pd.concat([Final_df2, dummies2], axis='columns')

del Solar_plant_2['source_key']
Solar_plant_2 = Solar_plant_2.reset_index(drop=True)

# Prep data for building model
yld_gen2 = Solar_plant_2['ypm']
data2 = Solar_plant_2.copy()
del data2['ypm']
del data2['plant_id'], data2['start_time'], data2['end_time'], data2['minutes'], data2['total_yield'], data2['time'], \
    data2['datestamp'], data2['yield_generated']

# Split data
X2_train, X2_test, Y2_train, Y2_test = train_test_split(data2, yld_gen2, test_size=0.3)

# Build Model
model2 = linear_model.ElasticNet()
model2.fit(X2_train, Y2_train)
from regressors import plots

# plots.plot_residuals(model2, X2_train, Y2_train, r_type='standardized')
plots.plot_qq(model2, X2_train, Y2_train, figsize=(8, 8))

Y2_pred = model2.predict(X2_test)
print('Plant 2: Pred: ', Y2_pred.shape)

from sklearn.metrics import mean_squared_error, r2_score

print("Coefficients:  ", model2.coef_)
print("Intercept:  ", model2.intercept_)
print("MSE:  %.2f" % mean_squared_error(Y2_test, Y2_pred))
print("Coefficients of determination:  %.2f" % r2_score(Y2_test, Y2_pred))

# to print summary table:
print("\n=========== SUMMARY ===========")
xlabels = X2_test.columns
stats.summary(model2, X2_test, Y2_test, xlabels)
figure = sns.regplot(x=Y2_pred, y=Y2_test, scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.title('Plant 2 Prediction:')
figure.set(xlabel='Predicted YPM', ylabel='Recorded YPM')
plt.show()
