import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Predict you weight.')
parser.add_argument('input_file', type=str, help='The input csv file containing the data to use as prediction')
parser.add_argument('--days', type=int, default=40, help='The number of day since the first day of measurement')

args = parser.parse_args()
input_file = args.input_file
days = args.days

df = pd.read_csv(input_file, parse_dates=['Date/Time'])
df = df.rename(columns={'Weight & Body Mass (kg) ': 'weight', 'Date/Time': 'dt'}).set_index('dt')

X = (df.index -  df.index[0]).days.values.reshape(-1, 1)
y = df['weight'].values
reg = LinearRegression().fit(X, y)

df['weight_pred'] = reg.predict(X)
df.plot()
plt.show()

estimated_weight = reg.predict([[days]])[0]
estimated_weight_loss = df['weight'][0] - estimated_weight

print(f'Your estimated weight is: {estimated_weight:.2f} kg')
print(f'Your estimated weight loss is: {estimated_weight_loss:.2f} kg')
