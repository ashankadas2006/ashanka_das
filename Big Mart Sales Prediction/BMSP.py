import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


train_data = pd.read_csv("C:/Users/ASHANKA/OneDrive/Desktop/Coding/Big Mart Sales Prediction/1.Train.csv")
test_data = pd.read_csv("C:/Users/ASHANKA/OneDrive/Desktop/Coding/Big Mart Sales Prediction/1.Test.csv")
# Updating The Values of the Given Data
# putting the mean of the weight of the item in the blank spaces
train_data["Item_Weight"].fillna(train_data["Item_Weight"].mean, inplace=True)

# putting the mode of the outlet size in comparison to the outlet types and putting the missing values as well
outlet_size_mode = train_data.pivot_table(values="Outlet_Size", columns="Outlet_Type", aggfunc=(lambda x: x.mode()))
missing_values_location = train_data["Outlet_Size"].isnull()
train_data.loc["missing_values_location", "Outlet_Size"] = train_data.loc[missing_values_location, "Outlet_Type"].apply(lambda x: outlet_size_mode[x])

# combining the mistyped data into descriptive data
train_data.replace({"Item_Fat_Content": {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}}, inplace=True)

# Converting the Categorical Values into 0s and 1s to convert them into numerical Values
encoder = LabelEncoder()
train_data["Item_Identifier"] = encoder.fit_transform(train_data["Item_Identifier"])
train_data["Item_Fat_Content"] = encoder.fit_transform(train_data["Item_Fat_Content"])
train_data["Outlet_Identifier"] = encoder.fit_transform(train_data["Outlet_Identifier"])
train_data["Outlet_Size"] = encoder.fit_transform(train_data["Outlet_Size"])
train_data["Outlet_Location_Type"] = encoder.fit_transform(train_data["Outlet_Location_Type"])
train_data["Outlet_Type"] = encoder.fit_transform(train_data["Outlet_Type"])
train_data["Item_Type"] = encoder.fit_transform(train_data["Item_Type"])

# CREATING AND TRAINING THE MODEL
# x = input of training data, y= output of training data
# a = input of test data, b = output of testing data
x = train_data.drop(columns="Item_Outlet_Sales")
y = train_data["Item_Outlet_Sales"]
a = test_data.drop(columns="Item_Outlet_Sales")
b = test_data["Item_Outlet_Sales"]

model = XGBRegressor()
model.fit(x, y)