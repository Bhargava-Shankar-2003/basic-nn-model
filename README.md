# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error. This regression model aims to provide valuable insights into the underlying patterns and trends within the dataset, facilitating enhanced decision-making and understanding of the target variable's behavior.

## Neural Network Model

![image](https://github.com/user-attachments/assets/20a72e8c-8120-49cf-9c34-b1056d07c3ac)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Bhargava S
### Register Number: 212221040029
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
from google.auth import default
import gspread
auth.authenticate_user()
from google.auth import default
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1ogYqsziVNFy4jzrgrZgSXVMI5tjTMHe77i1N1DMs36E/edit").sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:],columns=data[0])
dataset1 = dataset1.astype({'x':'float'})
dataset1 = dataset1.astype({'y':'float'})

X_train,X_test,Y_train,Y_test = train_test_split(dataset1[['x']],dataset1[['y']],test_size=0.33,random_state=33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai_brain = Sequential([
Dense(8,activation = 'relu'),
Dense(10,activation = 'relu'),
Dense(1)

])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')

ai_brain.fit(X_train1, Y_train,epochs =200)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)

ai_brain.evaluate(X_test,Y_test)

X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/5579c6da-1518-493c-ac29-1cf1abc19add)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/531fa357-f031-4d38-a061-4454b54782ce)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/d797e672-9ea9-4b1a-a0c8-de79eaca689d)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/155f0ca3-0cb5-452b-ac13-731f8c313577)


## RESULT

To develop a neural network regression model for the given dataset is created sucessfully.
