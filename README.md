# Diabetes Prediction and Visualization

## Overview

This project aims to predict diabetes and visualize patient data using various health metrics. Users can upload a CSV file containing patient data, and the system generates visual reports that compare the user's health metrics against others in the dataset.

## Features

- **CSV File Upload**: 
  - Drag-and-drop interface for easy file upload.
  - Supports CSV files up to 200MB in size.
  - Sample file included: `diabetes.csv` (23.3KB).
  
- **Visualized Patient Reports**:
  - **Pregnancy Count Graph**: Compares the number of pregnancies between others and yours.
  - **Glucose Value Graph**: Compares glucose levels.
  - **Blood Pressure Graph**: Visualizes blood pressure readings.
  - **Skin Thickness Graph**: Compares skin thickness.
  - **Insulin Value Graph**: Visualizes insulin levels.
  - **BMI Value Graph**: Compares Body Mass Index (BMI).
  - **Diabetes Pedigree Function Graph**: Compares the diabetes pedigree function, indicating genetic predisposition.

- **Personalized Diabetes Report**: 
  - Provides a report based on your data, e.g., "You are not Diabetic."
  
- **Model Accuracy**: 
  - The prediction model has an accuracy of **77.27%**.

## Installation

### Prerequisites
- Python 3.x
- Install required libraries via `pip`:
  ```bash
  pip install -r requirements.txt

![](Images/Front.png)
