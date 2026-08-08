# Ad_Sale_Prediction-from-existing-new-customer-using-Logistic-Regression# 📊 Advertisement Sale Prediction using Logistic Regression

## 📌 Project Overview

This project uses **Machine Learning and Logistic Regression** to predict whether a customer will purchase a product/advertisement based on their **Age** and **Salary**.

## 🎯 Objective

The objective is to predict the customer's purchasing decision:

* `0` → Customer will not buy
* `1` → Customer will buy

The model uses:

* **Age**
* **Salary**

as input features.

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Logistic Regression
* Google Colab / Jupyter Notebook

## 🔄 Machine Learning Workflow

1. Load the dataset
2. Explore the dataset
3. Separate features and target variable
4. Split data into training and testing sets
5. Apply feature scaling
6. Train the Logistic Regression model
7. Make predictions
8. Evaluate the model
9. Calculate accuracy

## 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
```

## ⚙️ Feature Scaling

`StandardScaler` is used to scale the Age and Salary features.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## 🔮 Prediction

The trained model can predict whether a new customer will buy.

```python
age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))

newCustomer = [[age, sal]]

result = model.predict(sc.transform(newCustomer))

if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
```

## 📈 Model Performance

The Logistic Regression model achieved:

**Accuracy: 80%**

```text
Accuracy of the Model: 80.0%
```

The project also evaluates predictions using a **confusion matrix**.

## 📂 Dataset

The dataset used in this project is:

```text
DigitalAd_dataset.csv
```

It contains customer information including **Age, Salary, and purchasing status**.

## 📁 Project Structure

```text
Advertisement-Sale-Prediction/
│
├── Advertisement_Sale_Prediction.ipynb
├── DigitalAd_dataset.csv
└── README.md
```

## 🚀 How to Run

### Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload `DigitalAd_dataset.csv`.
3. Run the notebook cells sequentially.
4. Enter Age and Salary when prompted.
5. View the prediction and accuracy.

### Jupyter Notebook

Install the required libraries:

```bash
pip install numpy pandas scikit-learn jupyter
```

Then run:

```bash
jupyter notebook
```

Open the notebook and execute the cells.

## 📚 Key Concepts Learned

* Supervised Learning
* Binary Classification
* Logistic Regression
* Train-Test Split
* Feature Scaling
* Model Training
* Model Prediction
* Confusion Matrix
* Accuracy Evaluation

## 🔮 Future Improvements

* Perform Exploratory Data Analysis (EDA)
* Add data visualizations
* Add precision, recall and F1-score
* Visualize the confusion matrix
* Compare multiple classification algorithms
* Deploy the model as a web application


