#!pip install kaggle -q
#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!kaggle datasets download sukhmandeepsinghbrar/car-price-prediction-dataset
#!unzip -qq car-price-prediction-dataset.zip
# Keep your kaggle.json token loaded up in pip

import torch
from torch import nn
import tensorflow as tf
import matplotlib.pyplot as plt
torch.__version__

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('A {} device was detected.')

import pandas as pd
url = 'cars.csv'
df = pd.read_csv(url)

df.head(2)

df.shape

import pandas as pd

# Mapping owner categories to numerical values
owner_mapping = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3
}

data = pd.read_csv(url)

# Filter out cases where seller_type is not 'Individual' or 'Dealer'
data = data[data['seller_type'].isin(['Individual', 'Dealer'])]

# Filter out cases where fuel is not 'Diesel' or 'Petrol'
data = data[data['fuel'].isin(['Diesel', 'Petrol'])]

# Encode categorical variables
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
for col in categorical_cols:
    if col == 'owner':
        data[col] = data[col].map(owner_mapping)  # Map owner categories to numerical values
    else:
        # Use pandas' factorize to encode categorical variables
        data[col] = pd.factorize(data[col])[0]

def convert(Column):
    M = data[Column].mean()
    S = data[Column].std()
    data[Column] = (data[Column] - M) / S

numerical_cols = ['selling_price', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col])  # Convert numerical columns to numeric type
    convert(col)

# Convert 'owner' column as needed
data['owner'] = (data['owner'] - 2) / 2

# Drop unnecessary columns
data = data.drop(columns=['name', 'year'])
data = data.dropna()
print(len(data))

import torch

# Define input features and output target
inputs = ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
output = ['selling_price']
data2 = data[0:5000]
# Convert input features and output target to PyTorch tensors
x = torch.tensor(data2[inputs].values, dtype=torch.float32)  # Assuming df is your DataFrame
y = torch.tensor(data2[output].values, dtype=torch.float32)
x.to(device)
y.to(device)

w = x[0:5000]
z = y[0:5000]

import torch
import torch.nn as nn
# Define the neural network model
model = nn.Sequential(
    nn.Linear(9, 400),
    nn.ReLU(),
    nn.Linear(400, 1)
)

model.to(device)

import torch.optim as optim

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00025, momentum=0.9) #lr changes based on data size
#too high and you get nan, too small don't see enough improvement
n = 10 # 500
for epoch in range(n): #change n to however many you need, I notice that around 400-500 is enough to make the nn accurate
    totalLoss = 0
    for i in range(len(x)):
       # Single Forward Pass
        ypred = model(x[i])
        # Measure how well the model predicted vs actual
        loss = criterion(ypred, y[i])

        # Track how well the model predicted
        totalLoss+=loss.item()

        # Update the neural network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch, end = " ")
    print ("Total Loss: ", totalLoss)

@torch.no_grad()
def graphPredictions(model, x, y , minValue, maxValue):

    model.eval()                               # Set the model to inference mode

    predictions=[]                             # Track predictions
    actual=[]                                  # Track the actual labels

    x.to(device)
    y.to(device)
    model.to(device)

    for i in range(len(x)):

        # Single forward pass
        pred = model(x[i])
        cost = x[i]
        # Un-normalize our prediction
        pred = pred* cost.Std + cost.Mean
        act = y[i]* cost.Std + cost.Mean

        # Save prediction and actual label
        predictions.append(pred.tolist())
        actual.append(act.item())

    # Plot actuals vs predictions
    plt.scatter(actual, predictions)
    plt.xlabel('Actual Car Cost')
    plt.ylabel('Predicted Car Cost')
    plt.plot([minValue, maxValue], [minValue, maxValue])
    plt.xlim(minValue, maxValue)
    plt.ylim(minValue, maxValue)

    # Make the display equal in both dimensions
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()