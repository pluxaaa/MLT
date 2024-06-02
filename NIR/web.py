from operator import itemgetter
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(style="ticks")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error

pd.options.display.max_columns = None
pd.options.display.max_rows = None

def load_data():
    data = pd.read_csv('HousingData.csv', sep=',')
    return data

def fill_data(data):
    null_cols = []
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        null_count = data[col].isnull().sum()
        if null_count > 0:
            null_cols.append(col)

    data_imputed = data.copy()
    for col in null_cols:
        if data_imputed[col].dtype == 'object' or len(data_imputed[col].unique()) <= 20:
            data_imputed = impute_data(data_imputed, col, is_classification=True)
        else:
            data_imputed = impute_data(data_imputed, col, is_classification=False)

    return data_imputed

def impute_data(data, column, is_classification=False):

    df = data.copy()
    
    known = df[df[column].notna()]
    unknown = df[df[column].isna()]
    
    if unknown.empty:
        return df
    
    X_known = known.drop(column, axis=1)
    y_known = known[column]
    
    X_unknown = unknown.drop(column, axis=1)
    
    if is_classification:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    
    model.fit(X_known, y_known)
    
    predicted_values = model.predict(X_unknown)
    
    df.loc[df[column].isna(), column] = predicted_values
    
    return df

def scale_data(data):
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    return pd.DataFrame(data_scaled, columns=data.columns)

def preprocess_data(data):
    data_scaled = scale_data(fill_data(data))

    data_scaled.drop(columns=['DIS', 'TAX', 'NOX'], inplace=True)

    return data_scaled

def split_data(df):
    return train_test_split(df.drop(columns='MEDV'), df['MEDV'], test_size=0.25, random_state=73)

def train_model(X_train, y_train, mss, msl, md, ne):
    rfr = RandomForestRegressor(min_samples_split=mss, n_estimators=ne, min_samples_leaf=msl, max_depth=md, n_jobs=-1)
    rfr.fit(X_train, y_train)
    return rfr

def draw_feature_importances(tree_model, X_dataset, figsize=(18, 5)):
    list_to_sort = list(zip(X_dataset.columns.values, tree_model.feature_importances_))
    sorted_list = sorted(list_to_sort, key=itemgetter(1), reverse=True)
    labels = [x for x, _ in sorted_list]
    data = [x for _, x in sorted_list]
    fig, ax = plt.subplots(figsize=figsize)
    ind = np.arange(len(labels))
    plt.bar(ind, data)
    plt.xticks(ind, labels, rotation='vertical')
    for a, b in zip(ind, data):
        plt.text(a - 0.05, b + 0.01, str(round(b, 3)))
    st.pyplot(fig)
    return labels, data

def draw_mae(X_train, X_test, y_train, y_test, mss, msl, md, ne_step, ne_range):
    x_line = []
    y_line = []
    for i in range(ne_range[0], ne_range[1], ne_step):
        model = train_model(X_train, y_train, mss, msl, md, i)
        pred = model.predict(X_test)
        sc = mean_absolute_error(pred, y_test)
        x_line.append(i)
        y_line.append(sc)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x_line, y_line)
    plt.xlabel('Число деревьев')
    plt.ylabel('MAE')
    st.pyplot(fig)


data_prepared = preprocess_data(load_data())
X_train, X_test, y_train, y_test = split_data(data_prepared)

if st.checkbox('Корреляционная матрица'):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data_prepared.corr(), annot=True, fmt='.2f')
    st.pyplot(fig)

st.sidebar.subheader('Гиперпараметры')
mss_slider = st.sidebar.slider('mean_samples_split: ', min_value=1, max_value=10, value=2, step=1)
msl_slider = st.sidebar.slider('mean_samples_leaf: ', min_value=1, max_value=10, value=1, step=1)
md_slider = st.sidebar.slider('max_depths: ', min_value=1, max_value=100, value=50, step=1)

st.sidebar.subheader('Конфигурация графика')
ne_step = st.sidebar.slider('Шаг при построении графика: ', min_value=1, max_value=25, value=10, step=1)
ne_range = st.sidebar.slider('Диапазон количетсва деревьев на графике: ', min_value=1, max_value=300, value=(1, 100), step=10)

if st.checkbox('Важность признаков'):
    model = train_model(X_train, y_train, mss_slider, msl_slider, md_slider, ne_range[1])
    draw_feature_importances(model, X_train)

draw_mae(X_train, X_test, y_train, y_test, mss_slider, msl_slider, md_slider, ne_step, ne_range)
