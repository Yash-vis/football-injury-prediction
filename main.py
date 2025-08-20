from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib as jl
# Load data
data = pd.read_csv('football/foot.csv')

# Encode categorical column
encode = LabelEncoder()
data['Position'] = encode.fit_transform(data['Position'])

# Columns to round and scale
columns_round = [
    'Training_Hours_Per_Week', 'Knee_Strength_Score', 'Hamstring_Flexibility',
    'Reaction_Time_ms', 'Balance_Test_Score', 'Sprint_Speed_10m_s', 'Agility_Score',
    'Sleep_Hours_Per_Night', 'Stress_Level_Score', 'Nutrition_Quality_Score', 'BMI'
]

# Round numeric columns
data[columns_round] = data[columns_round].round(2)

# Split features and target
drop_col = ['Injury_Next_Season','Knee_Strength_Score', 'Hamstring_Flexibility',
    'Reaction_Time_ms', 'Balance_Test_Score', 'Sprint_Speed_10m_s', 'Agility_Score','Nutrition_Quality_Score']
X = data.drop(drop_col, axis=1)
y = data['Injury_Next_Season']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale numeric columns
# scaler = StandardScaler()
# X_train[columns_round] = scaler.fit_transform(X_train[columns_round])
# X_test[columns_round] = scaler.transform(X_test[columns_round])

# List of models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("-" * 40)
model_path = jl.dump(models['Logistic Regression'], 'injury.pkl')
print("✅ Model saved to:", model_path[0])

columns_path = jl.dump(X.columns.to_list(), 'columns.pkl')
print("✅ Columns saved to:", columns_path[0])