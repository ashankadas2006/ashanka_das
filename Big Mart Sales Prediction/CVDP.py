import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(""C:/Users/ASHANKA/OneDrive/Desktop/Coding/Project folder/2.cardio.csv"")
data = data.drop_duplicates