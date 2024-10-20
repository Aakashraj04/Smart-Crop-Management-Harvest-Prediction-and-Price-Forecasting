import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (replace 'data.csv' with your file path)
df = pd.read_csv('data.csv')


# Create separate encoders for each categorical column
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
market_encoder = LabelEncoder()
commodity_encoder = LabelEncoder()
variety_encoder = LabelEncoder()


# Fit each encoder on its respective column
df['state'] = state_encoder.fit_transform(df['state'])
df['district'] = district_encoder.fit_transform(df['district'])
df['market'] = market_encoder.fit_transform(df['market'])
df['commodity'] = commodity_encoder.fit_transform(df['commodity'])
df['variety'] = variety_encoder.fit_transform(df['variety'])
df['arrival_date'] = pd.to_datetime(df['arrival_date']).astype(int) / 10**9  # Convert to timestamp


# Define features and target
X = df[['state', 'district', 'market', 'commodity', 'variety', 'arrival_date', 'min_price', 'max_price']]
y = df['modal_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the modal prices
y_pred = model.predict(X_test)

# Evaluate the model

r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)
# Example prediction
example = pd.DataFrame({
    'state': [state_encoder.transform(['Andaman and Nicobar'])[0]],
    'district': [district_encoder.transform(['Nicobar'])[0]],
    'market': [market_encoder.transform(['Car Nicobar'])[0]],
    'commodity': [commodity_encoder.transform(['Amaranthus'])[0]],
    'variety': [variety_encoder.transform(['Other'])[0]],
    'arrival_date': [pd.to_datetime('24-07-2019').timestamp()],
    'min_price': [4000],
    'max_price': [8000]
})

predicted_price = model.predict(example)
print("Predicted Modal Price:", predicted_price[0])
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['modal_price'], bins=30, kde=True)
plt.title('Distribution of Modal Prices')
plt.xlabel('Modal Price')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(12, 8))
commodity_avg_price = df.groupby('commodity')['modal_price'].mean().sort_values()
sns.barplot(x=commodity_avg_price, y=commodity_avg_price.index)
plt.title('Average Modal Price per Commodity')
plt.xlabel('Average Modal Price')
plt.ylabel('Commodity')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['min_price', 'max_price', 'modal_price']])
plt.title('Min, Max, and Modal Price Comparison')
plt.xlabel('Price Type')
plt.ylabel('Price')
plt.show()
plt.figure(figsize=(12, 8))
sns.boxplot(x='state', y='modal_price', data=df)
plt.title('Modal Price by State')
plt.xlabel('State')
plt.ylabel('Modal Price')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='max_price', y='modal_price', data=df)
plt.title('Max Price vs. Modal Price')
plt.xlabel('Max Price')
plt.ylabel('Modal Price')
plt.show()
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df_sorted = df.sort_values('arrival_date')

plt.figure(figsize=(12, 6))
plt.plot(df_sorted['arrival_date'], df_sorted['modal_price'], marker='o', linestyle='-', markersize=3)
plt.title('Modal Price Trend Over Time')
plt.xlabel('Arrival Date')
plt.ylabel('Modal Price')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(df[['min_price', 'max_price', 'modal_price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Prices')
plt.show()
# Count the occurrences of each state
state_counts = df['state'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1(range(len(state_counts))))
plt.title('State-Wise Distribution of Data Entries')
plt.axis('equal')
plt.show()
# Group the data by state and find the average modal prices
state_modal_price_avg = df.groupby('state')['modal_price'].mean()

plt.figure(figsize=(10, 10))
plt.pie(state_modal_price_avg, labels=state_modal_price_avg.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1(range(len(state_modal_price_avg))))
plt.title('Distribution of Average Modal Prices by State')
plt.axis('equal')
plt.show()
import matplotlib.pyplot as plt

# Group the data by state and calculate the total modal prices
state_modal_price = df.groupby('state')['modal_price'].sum()

# Create labels that include both state names and modal prices
labels = [f'{state}\n({price})' for state, price in zip(state_modal_price.index, state_modal_price.values)]

plt.figure(figsize=(10, 10))
plt.pie(state_modal_price, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(range(len(state_modal_price))))
plt.title('Distribution of Total Modal Prices by State')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of each commodity
commodity_counts = df['commodity'].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(x=commodity_counts.index, y=commodity_counts.values, palette='viridis')
plt.title('Distribution of Commodities')
plt.xlabel('Commodity')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(df['modal_price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Modal Prices')
plt.xlabel('Modal Price')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(14, 8))
sns.boxplot(x='commodity', y='modal_price', data=df, palette='coolwarm')
plt.title('Modal Price by Commodity')
plt.xlabel('Commodity')
plt.ylabel('Modal Price')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(df['max_price'], df['modal_price'], alpha=0.7, color='purple')
plt.title('Max Price vs. Modal Price')
plt.xlabel('Max Price')
plt.ylabel('Modal Price')
plt.show()
import numpy as np

plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
# Group data by state and variety, then count occurrences
state_variety_counts = df.groupby(['state', 'variety']).size().unstack()

# Create a stacked bar chart
state_variety_counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Set1')
plt.title('Variety Distribution by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Variety')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(df['max_price'], df['min_price'], s=df['modal_price']/10, alpha=0.5, color='blue')
plt.title('Bubble Chart: Max Price vs. Min Price vs. Modal Price')
plt.xlabel('Max Price')
plt.ylabel('Min Price')
plt.show()
plt.figure(figsize=(14, 8))
sns.violinplot(x='variety', y='modal_price', data=df, palette='muted')
plt.title('Modal Price Distribution by Variety')
plt.xlabel('Variety')
plt.ylabel('Modal Price')
plt.xticks(rotation=90)
plt.show()
sns.pairplot(df[['modal_price', 'max_price', 'min_price']])
plt.show()
plt.figure(figsize=(14, 8))
sns.violinplot(x='variety', y='modal_price', data=df, palette='muted')
plt.title('Modal Price Distribution by Variety')
plt.xlabel('Variety')
plt.ylabel('Modal Price')
plt.xticks(rotation=90)
plt.show()