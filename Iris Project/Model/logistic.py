import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


model_path = Path(__file__).parent / "softmax_theta.npy"


# Creating the dataframe df = pd.read_csv, reads the comma split value file

df = pd.read_csv(r"C:\Users\shrey\OneDrive\Documents\Machine Learning Projects\Iris Project\Data\Iris.csv")

# Clean up the columns that we want in our dataframe in our input and our output. 

Xdf = df.drop(columns = ['Id', 'Species'])
ydf = df['Species']

# Conversion to numpy arrays for calculations

X = Xdf.to_numpy()
y = ydf.to_numpy()

# Standardize the data in the feature matrix

for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])

# The leading 1 is the intercept feature. Each class gets a matching
# parameter in T, which is that class's bias.
X = np.c_[np.ones(X.shape[0]), X]

# One hot encoding the outputs values for softmax 

for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = np.array([1,0,0])
    elif y[i] == 'Iris-versicolor':
        y[i] = np.array([0,1,0])
    else:
        y[i] = np.array([0,0,1])

# Splitting the test data and training data into 75% training and 25% testing data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)


# Doing the softmax function to train the parameters of the model

def softmax_function(X, T):
    T = np.asarray(T)
    softmax_vectors = []

    for i in range(len(X)):
        z = np.dot(T, X[i])
        exp_z = np.exp(z)
        softmax = exp_z / np.sum(exp_z)
        softmax_vectors.append(softmax)

    return np.array(softmax_vectors)

# Doing the parameter fitting of the model using softmax method above.

def parameter_fitting():
    # Initialize the weights randomly
    T = np.random.rand(3, 5)  # 3 classes, 1 bias + 4 features

    # Set learning rate and number of iterations
    learning_rate = 0.1
    num_iterations = 1000

    for iteration in range(num_iterations):
        # Compute the softmax probabilities
        softmax_probs = softmax_function(X_train, T)

        # Compute the gradient
        y_train_matrix = np.vstack(y_train)
        gradient = ((softmax_probs - y_train_matrix).T @ X_train) / X_train.shape[0]


        # Update the weights
        T -= learning_rate * gradient

    return T
    
#Testing the prediction of the model using the test data and the fitted parameters.

theta = parameter_fitting()
def test_model(T):
    softmax_probs = softmax_function(X_test, T)
    predictions = np.argmax(softmax_probs, axis=1)

    y_test_matrix = np.vstack(y_test)  # shape: (38, 3)
    true_labels = np.argmax(y_test_matrix, axis=1)

    accuracy = np.mean(predictions == true_labels)
    return accuracy


np.save(model_path, theta)


print(test_model(theta))
print(theta)