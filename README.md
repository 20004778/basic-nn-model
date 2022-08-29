# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## Neural Network Model
![image](https://user-images.githubusercontent.com/75236145/187116033-f91ce67b-e76b-4d96-bc94-de41966e412b.png)



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

# Developed By:SURYA R
# Register Number:212220230052
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("data2.csv")
df.head()
x=df[['input']].values
x
y=df[['output']].values
y
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(6,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain1,ytrain,epochs=4000)
lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()
model.evaluate(xtest1,ytest)

xn1=[[40]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```

## Dataset Information
![dataset](https://user-images.githubusercontent.com/75236145/187084598-b28e0ddc-25c0-45fa-b6a6-c08bca02ba12.jpeg)


## OUTPUT
### Test Data Root Mean Squared Error
### New Sample Data Prediction
![deep](https://user-images.githubusercontent.com/75236145/187084617-ff2b9aaa-5afc-4729-8de0-911441f66c2a.jpeg)

### Training Loss Vs Iteration Plot
![deep1](https://user-images.githubusercontent.com/75236145/187084744-c0290a4f-4fdc-4c82-be57-f9c89d8023c4.jpeg)


## RESULT
Thus,the neural network regression model for the given dataset is developed.
