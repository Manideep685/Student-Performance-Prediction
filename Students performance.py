import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('StudentPerformanceFactors.csv')

# -----------------------------
# Encode categorical columns
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le   # store encoder (important)

# -----------------------------
# Features & Target
# -----------------------------
features = [
    'Hours_Studied',
    'Attendance',
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Sleep_Hours',
    'Previous_Scores',
    'Motivation_Level',
    'Gender',
    'Internet_Access',
    'Tutoring_Sessions',
    'Family_Income',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Physical_Activity',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home'
]

X = df[features]
y = df['Exam_Score']

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, color='black')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Linear Regression')
plt.show()

# -----------------------------
# User Input
# -----------------------------
try:
    Hours_Studied = float(input("Hours Studied: "))
    Attendance = float(input("Attendance: "))
    Parental_Involvement = float(input("Parental Involvement (0=Low,1=Medium,2=High): "))
    Access_to_Resources = float(input("Access to Resources (0=Low,1=Medium,2=High): "))
    Extracurricular_Activities = int(input("Extracurricular Activities (0=No,1=Yes): "))
    Sleep_Hours = float(input("Sleep Hours: "))
    Previous_Scores = float(input("Previous Scores: "))
    Motivation_Level = float(input("Motivation Level (0=Low,1=Medium,2=High): "))
    Gender = int(input("Gender (0=Female,1=Male): "))
    Internet_Access = int(input("Internet Access (0=No,1=Yes): "))
    Tutoring_Sessions = float(input("Tutoring Sessions: "))
    Family_Income = float(input("Family Income: "))
    Teacher_Quality = float(input("Teacher Quality: "))
    School_Type = int(input("School Type: "))
    Peer_Influence = float(input("Peer Influence: "))
    Physical_Activity = float(input("Physical Activity: "))
    Learning_Disabilities = float(input("Learning Disabilities: "))
    Parental_Education_Level = int(input("Parental Education Level: "))
    Distance_from_Home = float(input("Distance from Home: "))

    user_df = pd.DataFrame([[ 
        Hours_Studied, Attendance, Parental_Involvement,
        Access_to_Resources, Extracurricular_Activities,
        Sleep_Hours, Previous_Scores, Motivation_Level,
        Gender, Internet_Access, Tutoring_Sessions,
        Family_Income, Teacher_Quality, School_Type,
        Peer_Influence, Physical_Activity,
        Learning_Disabilities, Parental_Education_Level,
        Distance_from_Home
    ]], columns=features)

    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]

    print(f"\nPredicted Exam Score: {prediction:.2f}")

except Exception as e:
    print("Error occurred:", e)
