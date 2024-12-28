# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Check if the file exists
file_path = "C:\\Users\\HP\\Documents\\NITJ Books\\extra\\INTERN\\internpe week1\\diabetes.csv"
if not os.path.exists(file_path):
    st.error("The diabetes.csv file was not found. Please ensure the file is in the correct directory.")
else:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)  # Optionally drop rows with NaN values

    # HEADINGS
    st.title('Diabetes Checkup')
    st.sidebar.header('Patient Data')
    st.subheader('Training Data Stats')
    st.write(df.describe())

    # Function to get user input
    def user_report():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.sidebar.slider('Age', 21, 88, 33)

        user_report_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data

    # PATIENT DATA
    user_data = user_report()
    
    st.subheader('Patient Data')
    st.write(user_data)

    # Load models and preprocessing objects
    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\dt_model.pkl", 'rb') as f:
        dt_model = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\nb_model.pkl", 'rb') as f:
        nb_model = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\lr_model.pkl", 'rb') as f:
        lr_model = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\ann_model.pkl", 'rb') as f:
        ann_model = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\rf_model.pkl", 'rb') as f:
        rf_model = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)

    with open(r"C:\Users\HP\Documents\NITJ Books\extra\INTERN\internpe week1\imputer.pkl", 'rb') as f:
        imputer = pickle.load(f)

    # Preprocess user data
    user_data = imputer.transform(user_data)
    user_data = scaler.transform(user_data)

    # Make predictions
    user_result_dt = dt_model.predict(user_data)
    user_result_nb = nb_model.predict(user_data)
    user_result_lr = lr_model.predict(user_data)
    user_result_ann = ann_model.predict(user_data)
    user_result_rf = rf_model.predict(user_data)

    # VISUALIZATIONS
    st.title('Visualised Patient Report')

    # COLOR FUNCTION
    color = 'red' if user_result_rf[0] == 1 else 'blue'

    # Check the columns of user_data DataFrame
    if isinstance(user_data, pd.DataFrame):
        st.write(user_data.columns)
    else:
        # Convert user_data to DataFrame
        user_data_df = pd.DataFrame(user_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        # Access the columns attribute
        st.write(user_data_df.columns)
     
    
    # Plot Age vs Pregnancies
    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x='Age', y='Pregnancies', data=user_data_df, s=150, color=color)
    plt.xlim(10, 100)
    plt.ylim(0, 20)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 20, 2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
    ax4 = sns.scatterplot(x='Age', y='Glucose', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 220, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)

    # Age vs Blood Pressure
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
    ax6 = sns.scatterplot(x='Age', y='BloodPressure', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 130, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)

    # Age vs Skin Thickness
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
    ax8 = sns.scatterplot(x='Age', y='SkinThickness', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 110, 10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)

    # Age vs Insulin
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
    ax10 = sns.scatterplot(x='Age', y='Insulin', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 900, 50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)

    # Age vs BMI
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
    ax12 = sns.scatterplot(x='Age', y='BMI', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 70, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)

    # Age vs DPF
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
    ax14 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=user_data_df, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.yticks(np.arange(0, 3, 0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

    # OUTPUT
    st.subheader('Diabetes Result Using Different ML Models: ')
    st.write('Decision Tree Prediction:', 'Diabetic' if user_result_dt[0] == 1 else 'Non-Diabetic')
    st.write('Naive Bayes Prediction:', 'Diabetic' if user_result_nb[0] == 1 else 'Non-Diabetic')
    st.write('Logistic Regression Prediction:', 'Diabetic' if user_result_lr[0] == 1 else 'Non-Diabetic')
    st.write('ANN Prediction:', 'Diabetic' if user_result_ann[0] == 1 else 'Non-Diabetic')
   
