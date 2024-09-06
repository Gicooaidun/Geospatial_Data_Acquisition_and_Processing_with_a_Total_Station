#!/usr/bin/env python
# coding: utf-8

# In[36]:
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# Path to where the data folder is located
PATH = ''
REG_TYPE = 'forest'#linear, polynomial, forest (not used anymore, as simple vertical offset is good enough)


# In[37]:


def set_meteo_index(df, columns, format):
    df['Datetime'] = pd.to_datetime(df[columns[0]] + ' ' + df[columns[1]], format=format)
    df.set_index('Datetime', inplace=True)
    df.drop(columns, axis=1, inplace=True)
    return df

def read_meteo():
    file_names = [PATH+'Data/MeteoCalibration/MeteoHut/LAB17_20231013_1404.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB18_20231013_1407.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB19_20231013_1406.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB22_20231013_1404.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB24_20231013_1404.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB25_20231013_1404.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB26_20231013_1404.txt', PATH+'Data/MeteoCalibration/MeteoHut/LAB27_20231013_1405.txt']
    meteo_huts_data = [pd.read_csv(file, delimiter=',', skiprows=1) for file in file_names]
    meteo_huts_data = [i.drop(i.columns[-1], axis=1) for i in meteo_huts_data]
    meteo_huts_data = [set_meteo_index(i, columns=['Date', 'Time'], format='%d/%m/%y %H:%M:%S') for i in meteo_huts_data]
    meteo_huts_data = [i.drop(['Batt V', 'Batt SoC', 'Log Count'], axis=1) for i in meteo_huts_data]
    names = ['Lab17', 'Lab18', 'Lab19', 'Lab22', 'Lab24', 'Lab25', 'Lab26', 'Lab27']
    return meteo_huts_data, names

def adjust_meteo(meteo_huts, data_original, names):
    meteo_huts_data = [df[df.index >= data_original.index[0]] for df in meteo_huts]
    meteo_huts_data = [df[df.index <= data_original.index[-1]] for df in meteo_huts_data]
    # Rescale
    for ind, df in enumerate(meteo_huts_data):
        pressure_columns = [col for col in df.columns if col.endswith('P')]
        meteo_huts_data[ind][pressure_columns] = df[pressure_columns]/100
    meteo_huts_data = pd.concat([df.add_suffix('_'+names[ind]) for ind, df in enumerate(meteo_huts_data)], axis=1)
    meteo_huts_data = meteo_huts_data.resample('2S').mean()
    return meteo_huts_data

def add_corrections_meteo(meteo_huts, data, columns, df_meteo_corrections):
    reference = [data['Temperature_ref'], data['Temperature_ref'], data['Humidity_ref'], data['Temperature_ref'], data['Pressure_ref'], data['Humidity_ref']]
    for ind, column in enumerate(columns):
        correction = reference[ind].mean() - meteo_huts[column].mean()
        meteo_huts[column] += correction
        df_meteo_corrections[column] = [correction]
    return meteo_huts, df_meteo_corrections
    
def plot_meteo(meteo_huts_data, names, ref):
    for ind, df in enumerate(meteo_huts_data):
        # Extract columns with 'T' for temperature, 'P' for pressure, and 'H' for humidity
        temperature_columns = [col for col in df.columns if col.endswith('T')]
        pressure_columns = [col for col in df.columns if col.endswith('P')]
        humidity_columns = [col for col in df.columns if col.endswith('H')]

        # Create subplots for temperature, pressure, and humidity
        fig, axs = plt.subplots(3, 1, figsize=(14, 10))

        # Plot temperature
        df[temperature_columns].plot(ax=axs[0])
        ref['Temperature_ref'].plot(ax=axs[0])
        axs[0].set_ylabel('Temperature [°C]')
        axs[0].set_xlabel('')
        axs[0].set_title('Temperature')
        
        for temp_col in temperature_columns:
            valid_indices = ~np.isnan(df[temp_col]) & ~np.isnan(ref['Temperature_ref'])
            rmse_temperature = np.sqrt(mean_squared_error(ref['Temperature_ref'][valid_indices], df[temp_col][valid_indices]))
            axs[0].text(0.99, 0.3 - 0.12 * temperature_columns.index(temp_col), f'RMSE {temp_col.split(" ")[0]}: {rmse_temperature:.2f}', ha='right', va='bottom', transform=axs[0].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        labels = [temp_col.split(" ")[0] for temp_col in temperature_columns]
        labels.append('Reference')
        axs[0].legend(labels=labels)

        # Plot pressure
        df[pressure_columns].plot(ax=axs[1])
        ref['Pressure_ref'].plot(ax=axs[1])
        axs[1].set_ylabel('Pressure [mbar]')
        axs[1].set_xlabel('')
        axs[1].set_title('Pressure')
        
        for press_col in pressure_columns:
            valid_indices = ~np.isnan(df[press_col]) & ~np.isnan(ref['Pressure_ref'])
            rmse_pressure = np.sqrt(mean_squared_error(ref['Pressure_ref'][valid_indices], df[press_col][valid_indices]))
            axs[1].text(0.99, 0.06, f'RMSE {press_col.split(" ")[0]}: {rmse_pressure:.2f}', ha='right', va='bottom', transform=axs[1].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        labels = [press_col.split(" ")[0] for press_col in pressure_columns]
        labels.append('Reference')
        axs[1].legend(labels=labels)

        # Plot humidity
        df[humidity_columns].plot(ax=axs[2])
        ref['Humidity_ref'].plot(ax=axs[2])
        axs[2].set_ylabel('Humidity [%]')
        axs[2].set_xlabel('Time')
        axs[2].set_title('Humidity')
        
        for humid_col in humidity_columns:
            valid_indices = ~np.isnan(df[humid_col]) & ~np.isnan(ref['Humidity_ref'])
            rmse_humidity = np.sqrt(mean_squared_error(ref['Humidity_ref'][valid_indices], df[humid_col][valid_indices]))
            axs[2].text(0.99, 0.18 - 0.12 * humidity_columns.index(humid_col), f'RMSE {humid_col.split(" ")[0]}: {rmse_humidity:.2f}', ha='right', va='bottom', transform=axs[2].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        labels = [humid_col.split(" ")[0] for humid_col in humidity_columns]
        labels.append('Reference')
        axs[2].legend(labels=labels)
        
        plt.suptitle(f'Meteorological Analysis of Meteo Hut: {names[ind].split("_")[1]}', fontsize=15)
        plt.tight_layout()
        plt.show()


def read_data(file_path, delete=[]):
    encoding =  'latin1'
    column_names = ['Temperature', 'Pressure', 'Humidity']
    df = pd.read_csv(file_path, delimiter=',', encoding=encoding, skiprows=1, header=None)
    if len(delete) != 0:
        df.drop(delete, axis=1, inplace=True)
    df.columns = column_names
    df = df.applymap(lambda x: float(''.join(filter(lambda c: c.isdigit() or c == '.', str(x)))) if pd.notna(x) else None)
    return df


# In[38]:


def set_index(df, start_date):
    # Create a datetime index starting from the specified date with 2 second intervals
    date_index = pd.date_range(start=start_date, periods=len(df), freq='2S')
    df.index = date_index
    return df


# In[39]:

def plot(df, variables):
    units = ['[°C]', '[mbar]', '[%]']
    fig, axs = plt.subplots(len(variables), 1, figsize=(14, 10))
    xlabels = ['', '', 'Time']

    for ind, variable in enumerate(variables):
        rmse_dft1 = np.sqrt(mean_squared_error(df[variable+'_ref'], df[variable+'_dft1']))
        rmse_mws95 = np.sqrt(mean_squared_error(df[variable+'_ref'], df[variable+'_mws95']))
        ax = axs[ind]
        
        # Plot DFT1, MWS95, and Reference data
        df[[variable+'_dft1', variable+'_mws95', variable+'_ref']].plot(ax=ax, xlabel=xlabels[ind], ylabel=variable+f' {units[ind]}', title=variable)
        
        if ind==1:
            ax.set_ylim(950, 970)

        # Add RMSE text to the subplot
        ax.text(0.99, 0.18, f'RMSE DFT1: {rmse_dft1:.2f}', ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.99, 0.06, f'RMSE MWS95: {rmse_mws95:.2f}', ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # Add legend to the subplot
        ax.legend(labels=['DFT1', 'MWS95', 'Reference'])
        
    plt.suptitle(f'Meteorological Analysis of Reinhard Stations', fontsize=15)
    plt.tight_layout()
    plt.show()

# In[40]:


def preprocess(ploting=False):
    start_date = '13.10.2023 14:08:00'
    dft1 = read_data(PATH+"Data/MeteoCalibration/Reinhardt_DFT1MV/DFT1-004.DAT", delete=[0, 1, 5])
    dft1 = set_index(dft1, start_date)
    mws95 = read_data(PATH+'Data/MeteoCalibration/Reinhardt_MWS9-5/MWS9-5_001.DAT', delete=[0, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    mws95 = set_index(mws95, start_date)
    reference = pd.read_csv(PATH+'Data/MeteoCalibration/Reference_PT100/Reference.csv')
    reference.drop('CO2', axis=1, inplace=True)
    reference.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour', 'min': 'minute', 's': 'second', 'T1': 'Temperature_1', 'T2': 'Temperature_2', 'p': 'Pressure', 'u': 'Humidity'}, inplace=True)
    reference['Datetime'] = pd.to_datetime(reference[['year', 'month', 'day', 'hour', 'minute', 'second']])
    reference.set_index('Datetime', inplace=True)
    reference.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1, inplace=True)
    ref = reference.resample('2S').mean()
    ref = ref[ref.index >= dft1.index[0]]
    dft1 = dft1[dft1.index <= ref.index[-1]]
    mws95 = mws95[mws95.index <= ref.index[-1]]
    ref['Temperature'] = ref[['Temperature_1', 'Temperature_2']].mean(axis=1)
    ref.drop(['Temperature_1', 'Temperature_2'], axis=1, inplace=True)
    column_names = ['Temperature_dft1', 'Pressure_dft1', 'Humidity_dft1', 'Temperature_mws95', 'Pressure_mws95', 'Humidity_mws95', 'Pressure_ref', 'Humidity_ref', 'Temperature_ref']
    data = pd.concat([dft1, mws95, ref], axis=1)
    data.columns = column_names
    # Set reference humidity to the MWS95 sensor
    data['Humidity_ref'] = data['Humidity_mws95']
    # Plot original data
    if ploting:
        plot(data, ['Temperature', 'Pressure', 'Humidity'])
    return data


# In[41]:


def add_correction(data, columns, df_meteo_corrections):
    reference = [data['Temperature_ref'], data['Pressure_ref'], data['Humidity_ref']]
    for ind, column in enumerate(columns):
        correction = reference[ind].mean() - data[column].mean()
        data[column] += correction
        df_meteo_corrections[column] = [correction]
    return data, df_meteo_corrections


# In[42]:


def prepare_data(data, sensor, meteo_hut_bool):
        
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    if meteo_hut_bool:
        name = sensor.split('_')[1]
        if sensor.startswith('SHT31'):
            X_train = train_data[['SHT31 T_'+name, 'SHT31 H_'+name]]
            X_test = test_data[['SHT31 T_'+name, 'SHT31 H_'+name]]
        else:
            X_train = train_data[['BME280 T_'+name, 'BME280 P_'+name, 'BME280 H_'+name]]
            X_test = test_data[['BME280 T_'+name, 'BME280 P_'+name, 'BME280 H_'+name]]
    else:        

        # Define the features (X) and target variable (y)
        X_train = train_data[['Temperature_'+str(sensor), 'Pressure_'+str(sensor), 'Humidity_'+str(sensor)]]
        X_test = test_data[['Temperature_'+str(sensor), 'Pressure_'+str(sensor), 'Humidity_'+str(sensor)]]
        
    y_train = train_data['Humidity_ref']
    y_test = test_data['Humidity_ref']
    return X_train, y_train, X_test, y_test


# In[43]:

def predict(model, data, sensor, meteo_hut_bool):
    if meteo_hut_bool:
        name = sensor.split('_')[1]
        if sensor.startswith('SHT31'):
            data['SHT31 H_'+name] = model.predict(data[['SHT31 T_'+name, 'SHT31 H_'+name]])
        else:
            data['BME280 H_'+name] = model.predict(data[['BME280 T_'+name, 'BME280 P_'+name, 'BME280 H_'+name]])
    else:
        data['Humidity_'+str(sensor)] = model.predict(data[['Temperature_'+str(sensor), 'Pressure_'+str(sensor), 'Humidity_'+str(sensor)]])
    return model, data


def linear_regression(data, sensors, meteo_hut_bool=False): 
    models = []
    for sensor in sensors:
        X_train, y_train, X_test, y_test = prepare_data(data, sensor, meteo_hut_bool)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_humidity = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_humidity)
        print(f'Mean Squared Error: {mse}')
        model, data = predict(model, data, sensor, meteo_hut_bool)
        models.append(model)
    return models, data


# In[44]:


def polynomial_regression(data, sensors, degree=3, meteo_hut_bool=False):
    models = []
    for sensor in sensors:
        X_train, y_train, X_test, y_test = prepare_data(data, sensor, meteo_hut_bool)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        predicted_humidity = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_humidity)
        print(f'Mean Squared Error: {mse}')
        model, data = predict(model, data, sensor, meteo_hut_bool)
        models.append(model)
    return models, data

# In[51]:

def random_forest(data, sensors, n_estimators=100, meteo_hut_bool = False):
    models = []
    for sensor in sensors:
        X_train, y_train, X_test, y_test = prepare_data(data, sensor, meteo_hut_bool)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        predicted_humidity = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted_humidity)
        print(f'Mean Squared Error: {mse}')
        model, data = predict(model, data, sensor, meteo_hut_bool)
        models.append(model)
    return models, data

# In[49]:

def regression_meteo(meteo_huts, names, reg_type='linear', degree=3, n_estimators=100):
    sensors = []
    for name in names:
        sensors.append([i+name for i in ['SHT31 H_', 'BME280 H_']])
    sensors = [item for sublist in sensors for item in sublist]
    if reg_type=='linear':
        models, data = linear_regression(meteo_huts, sensors, meteo_hut_bool=True)
    elif reg_type=='polynomial':
        models, data = polynomial_regression(meteo_huts, sensors, degree=degree, meteo_hut_bool=True)
    elif reg_type=='forest':
        models, data = random_forest(meteo_huts, sensors, n_estimators=n_estimators, meteo_hut_bool=True)
    else:
        raise ValueError('Selected regression type was not implemented or is written incorrectly')
    data = data.drop('Humidity_ref', axis=1)
    return models, data, sensors
    


def regression(data, reg_type='linear', degree=3, n_estimators=100):
    sensors = ['dft1', 'mws95']
    if reg_type=='linear':
        models, data = linear_regression(data, sensors)
    elif reg_type=='polynomial':
        models, data = polynomial_regression(data, sensors, degree=degree)
    elif reg_type=='forest':
        models, data = random_forest(data, sensors, n_estimators=n_estimators)
    else:
        raise ValueError('Selected regression type was not implemented or is written incorrectly')
    return models, data, sensors

def elim_outliers(df):
    thersholds = np.repeat([2,2,10,2,1,10], 8)
    df_no_outliers = pd.DataFrame()
    for ind, col in enumerate(df.columns):
        deltas = df[col].diff().abs()
        rapid_change_indices = deltas > thersholds[ind]
        df_no_outliers[col] = df[col][~rapid_change_indices]
    return df_no_outliers



# In[ ]:
def save_corrections(df):
    df.to_csv(PATH+f'Data/Corrected_data/all_vertical_offsets.csv')


# Save data and model
def save(data, models=None, sensors=None, filename=None):
    if models is None:
        data.to_csv(PATH+f'Data/Original_data/{filename}.csv')
        return
    data.to_csv(PATH+f'Data/Corrected_data/{filename}_{REG_TYPE}.csv')
    for ind, model in enumerate(models):
        joblib.dump(model, PATH+f"Data/Corrected_data/model_{filename}_{REG_TYPE}_{sensors[ind]}.pkl")
        
def save_vertical(data, filename):
    data.to_csv(PATH+f'Data/Corrected_data/{filename}_vertical_offsets.csv')
    
def main():
    data = preprocess(ploting=False)
    save(data, filename='reinhard_ref')
    meteo_huts, names = read_meteo()
    meteo_huts = adjust_meteo(meteo_huts, data, names)
    save(meteo_huts, filename='meteo_huts')
    meteo_huts = elim_outliers(meteo_huts)
    df_meteo_corrections = pd.DataFrame()
    for sen in ['mws95', 'dft1']:
        data, df_meteo_corrections = add_correction(data, ['Temperature_'+sen, 'Pressure_'+sen, 'Humidity_'+sen], df_meteo_corrections)
    meteo_hut_sensors = ['TMP116 T_', 'SHT31 T_', 'SHT31 H_', 'BME280 T_', 'BME280 P_', 'BME280 H_']
    for name in names:
        meteo_huts, df_meteo_corrections = add_corrections_meteo(meteo_huts, data, [i+name for i in meteo_hut_sensors], df_meteo_corrections)
    save_corrections(df_meteo_corrections)
    save_vertical(data, 'reinhard_ref')
    save_vertical(meteo_huts, 'meteo_huts')
    
    #models, data, sensors = regression(data, reg_type=REG_TYPE)
    #models_meteo, meteo_huts, sensors_meteo = regression_meteo(pd.concat([meteo_huts, data['Humidity_ref']], axis=1), names, reg_type=REG_TYPE)
    #save(data, models=models, sensors=sensors, filename='reinhard_ref')
    #save(meteo_huts, models=models_meteo, sensors=sensors_meteo, filename='meteo_huts')
    

if __name__ == '__main__':
    main()