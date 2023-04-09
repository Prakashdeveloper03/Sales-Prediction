import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
df = pd.read_csv("/data/data.csv").groupby("Date").sum()  # Load data and group by date
df["Date"] = df.index  # Assign index as 'Date' column
df = df[["Date", "TotalSales"]]  # Select only 'Date' and 'TotalSales' columns
df["Date"] = pd.to_datetime(
    df["Date"], format="%Y-%m-%d"
)  # Convert 'Date' column to datetime format
dataset = df[["TotalSales"]].values  # Extract only 'TotalSales' column as a numpy array
scaler = MinMaxScaler(feature_range=(0, 1))  # Create a scaler object to scale the data
scaled_data = scaler.fit_transform(dataset)  # Scale the data
training_data_len = int(
    np.ceil(len(dataset) * 0.95)
)  # Determine the length of training data
train_data = scaled_data[
    0 : int(training_data_len), :
]  # Split the data into training and testing data

# Generate training data using a sliding window approach
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(
        train_data[i - 60 : i, 0]
    )  # Append the previous 60 data points to the x_train list
    y_train.append(train_data[i, 0])  # Append the next data point to the y_train list
x_train, y_train = np.array(x_train), np.array(
    y_train
)  # Convert the x_train and y_train lists to numpy arrays
x_train = np.reshape(
    x_train, (x_train.shape[0], x_train.shape[1], 1)
)  # Reshape the x_train data for LSTM input

# Create the LSTM model
model = tf.keras.models.Sequential()
model.add(
    tf.keras.keras.layers.LSTM(
        128, return_sequences=True, input_shape=(x_train.shape[1], 1)
    )
)  # Add a LSTM layer with 128 neurons
model.add(
    tf.keras.keras.layers.LSTM(64, return_sequences=False)
)  # Add a LSTM layer with 64 neurons
model.add(tf.keras.keras.layers.Dense(25))  # Add a dense layer with 25 neurons
model.add(tf.keras.keras.layers.Dense(1))  # Add a dense output layer with 1 neuron
model.compile(
    optimizer="adam", loss="mean_squared_error"
)  # Compile the model using the Adam optimizer and mean squared error loss function

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Generate test data using a sliding window approach
test_data = scaled_data[training_data_len - 60 :, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(
        test_data[i - 60 : i, 0]
    )  # Append the previous 60 data points to the x_test list
x_test = np.array(x_test)  # Convert the x_test list to a numpy array
x_test = np.reshape(
    x_test, (x_test.shape[0], x_test.shape[1], 1)
)  # Reshape the x_test data for LSTM input

# Make predictions on test data
predictions = model.predict(
    x_test
)  # Use the trained model to predict y values for the x_test data
predictions = scaler.inverse_transform(
    predictions
)  # Inverse transform the predicted values to the original scale
rmse = np.sqrt(
    np.mean(((predictions - y_test) ** 2))
)  # Calculate the root mean squared error
print(rmse)
