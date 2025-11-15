# SCT_DS_3
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.
PROJECT: DECISION TREE CLASSIFIER
DATASET: BANK MARKETING DATASET (UCI MACHINE LEARNING REPOSITORY)
GOAL: PREDICT WHETHER A CUSTOMER WILL PURCHASE A PRODUCT/SERVICE
------------------------------------------------------------
1. Import Required Libraries
- pandas & numpy: for data handling and numerical operations
- sklearn: for model building and evaluation
- matplotlib & seaborn: for visualization
------------------------------------------------------------
2. Load the Dataset
- Read the CSV file into a pandas DataFrame
- Display the first few rows and dataset info
- Check shape, column names, and data types
3. Data Understanding
- Identify input (independent) features and output (target) variable
- Example target: "y" (yes/no for product purchase)
- Check class balance (how many 'yes' vs 'no')
4. Data Cleaning
- Check for missing values using df.isnull().sum()
- Handle missing or inconsistent data if any
- Remove duplicates if found
- Convert data types if required (e.g., categorical to category)
5. Data Preprocessing
- Encode categorical variables using:
* LabelEncoder or OneHotEncoder
- Normalize or standardize numeric features if needed
- Split data into features (X) and target (y)
- Example:
X = df.drop('y', axis=1)
y = df['y']
6. Split Data into Training and Testing Sets
- Use train_test_split() from sklearn
- Example:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
- Ensures model performance is evaluated on unseen data
7. Build the Decision Tree Classifier
- Import DecisionTreeClassifier from sklearn.tree
- Initialize the model with key parameters:
* criterion = 'gini' or 'entropy'
* max_depth = to control overfitting
- Example:
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
- Fit the model:
model.fit(X_train, y_train)
8. Make Predictions
- Use the trained model to predict on test data:
y_pred = model.predict(X_test)
9. Evaluate the Model
- Import evaluation metrics from sklearn.metrics
- Calculate accuracy, precision, recall, F1-score, and confusion matrix
- Example:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
10. Visualize the Decision Tree
- Use sklearn.tree.plot_tree() for visualization
- Example:
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()
11. Analyze Feature Importance
- Identify which features have the most influence on predictions
- Example:
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)
- Plot the top important features using a bar chart
12. Interpret Results
- Higher accuracy indicates a good predictive model
- Review important features (e.g., age, job type, marital status, duration of call)
- Understand which demographics or behaviors lead to higher purchase rates
13. Fine-Tune the Model
- Adjust hyperparameters (max_depth, min_samples_split, criterion)
- Use GridSearchCV or RandomizedSearchCV for optimization
- Re-train the model and re-evaluate performance
14. Save the Model (Optional)
- Use joblib or pickle to save the trained model for future use
- Example:
import joblib
joblib.dump(model, 'decision_tree_model.pkl')
------------------------------------------------------------
SUMMARY
------------------------------------------------------------
✅ Data cleaned and preprocessed
✅ Decision Tree Classifier built and evaluated
✅ Model predicts customer purchase likelihood
✅ Key features identified (demographic and behavioral)
✅ Visualization and interpretation completed
------------------------------------------------------------
