# Databricks notebook source
# IMPORTING THE REQUIRED LIBRARIES

# for data manipulation and preprocessing
import pandas as pd
import numpy as np
import streamlit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# data visualization
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
#from ydata_profiling import profile_report
import streamlit as st

# machine learning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier
from sklearn.impute._knn import KNNImputer
from sklearn.ensemble import RandomForestClassifier

file_uploader = st.file_uploader("Upload your dataset to train the machine models upon:")

# importing the dataset from the user
if file_uploader is not None:
    data = pd.read_csv(file_uploader)
    st.title("Dataset Imported")
    st.write(data)


    # FUNCTION TO IMPUTE ZEROES WITH MEAN OF THE COLUMN
    def impute_zero(data, attribute):
        """
        A small function to impute any zeroes if present in the attribute
        mentioned with the mean of the attrribute's column.
        """
        data[attribute].replace(to_replace=np.where(data[attribute] == 0),
                                value=data[attribute].mean(), inplace=True)
        return data.head(7)


    x = data.drop(columns=['time', 'labels'])
    y = data['labels']

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,
                                                        random_state=10)

    st.title("Model input features")
    """
    Input features considered to train the model.
    """
    st.write(x_train)

    st.title("Model training labels")
    """
    Labels for the training dataset.
    """
    st.write(y_train)

    # dropdown menu
    model_options = ['Logistic Regression', 'KNN - Classifier',
                     'Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Classifier']
    selected_model = st.selectbox("Select a model to train", model_options)

    # training the logistic regression model
    if selected_model == 'Logistic Regression':
        """
        Logistic regression where the l1_ratio by default is 0.5 to avoid overfitting.
        """
        def logistic_model(x,y):
            logistic_regression_model = LogisticRegression(l1_ratio=0.5)
            logistic_regression_model = logistic_regression_model.fit(x, y)
            return logistic_regression_model


        logistic_model = logistic_model(x_train, y_train)
        
        st.write("Logistic Regression Accuracy:",
                     str(logistic_model.score(x_test, y_test)*100)+"%")


    if selected_model == 'KNN - Classifier':
        def knn(x, y):
            knn_model = KNeighborsClassifier(5)
            knn_model = knn_model.fit(x,y)
            return knn_model
        knn_model = knn(x_train, y_train)

        st.write("KNN model accuracy:",
                 str(knn_model.score(x_test, y_test)*100)+"%")

    if selected_model == 'Random Forest Classifier':
        def rf_classifier(x, y):
            value = st.slider(min_value=10,max_value=170,step=10,
                              label="Select the total number of base_estimators")
            rf_classifier = RandomForestClassifier(n_estimators=value)
            rf_classifier = rf_classifier.fit(x,y)
            return rf_classifier
        rf_classifier = rf_classifier(x_train, y_train)

        st.write("Random Forest Classifier model accuracy:",
                 str(rf_classifier.score(x_test, y_test)*100)+"%")

    if selected_model == 'Decision Tree Classifier':
        """
        Use gini impurity by default and a low value for ccp alpha as possible,
        if going to test with random number for better accuracy.
        """
        def dtree(x, y):
            split_criteria = ['gini', 'entropy']
            split_criteria = st.selectbox("Select the tree splitting criteria:",split_criteria)
            value = st.slider("Select a value :", min_value=0.00, max_value=1.00, step = 0.04)
            dtree_model = DecisionTreeClassifier(ccp_alpha=value, criterion=split_criteria)
            dtree_model = dtree_model.fit(x, y)
            return dtree_model

        dtree = dtree(x_train, y_train)

        st.write("Decision Tree Classifier model accuracy :",
                 str(dtree.score(x_test, y_test)*100)+"%")
        fig, axs = plt.subplots(5, figsize=(20, 50))

        for i, ax in enumerate(axs):
            tree.plot_tree(dtree, feature_names=x_train.columns, filled=True, ax=ax)

        plt.tight_layout()
        plt.savefig('image.png')
        st.pyplot(fig)

    if selected_model == 'Support Vector Classifier':
        def svc(x, y):
            svc = SVC()
            svc = svc.fit(x,y)
            return svc
        svc = svc(x_train, y_train)

        st.write("Support Vector Classifier model accuracy:",
                 str(svc.score(x_test, y_test)*100)+"%")
