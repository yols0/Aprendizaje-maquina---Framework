from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
import pandas as pd

# Import data
iris_data = datasets.load_iris()

# Create Data Frame 
df = pd.DataFrame({"sepal length": iris_data.data[:,0],"sepal width": iris_data.data[:,1],"petal length": iris_data.data[:,2],"petal width": iris_data.data[:,3], "species": iris_data.target})

# Separeate our dependent and independent variables
X= df[["sepal length","sepal width","petal length","petal width"]]
Y= df[["species"]]

# Separate our dataframe into 80% for training and 20% for test
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.2) 

# Create model
forest_model = RandomForestClassifier()

# Train model
forest_model.fit(x_train, y_train)

# Create a prediction
prediction = forest_model.predict(x_test)

# Print results
y_test_values= y_test.values

for i in range(len(prediction)):
  print("y_test:",y_test_values[i][0],"- Prediction:", prediction[i])

print("\nAccuracy -> ", metrics.accuracy_score(y_test,prediction))
