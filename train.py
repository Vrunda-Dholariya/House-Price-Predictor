from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. Load the data
data=fetch_california_housing(as_frame=True)
df=data.frame

#2. Separate the clues(X) from answer(Y) 
X=df.drop("MedHouseVal", axis=1)  #all columns except the price
y=df["MedHouseVal"]     #only the price column

#3. Split 80% to learn from and 20% to test on later
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#4. Create and train the model
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

print("Training Completed!")
print("House trained on: ",X_train.shape[0])
print("House held back for testing:", X_test.shape[0])

from sklearn.metrics import mean_absolute_error, r2_score

#5. Ask model to Predict prices for rest 20%
prediction=model.predict(X_test)

#6. Compare predictions to real prices
mae=mean_absolute_error(y_test,prediction)
r2=r2_score(y_test,prediction)

print("R2 Score: ",round(r2,2))
print("Average Error: ", round(mae,2), "-> about $", round(mae * 100000))
