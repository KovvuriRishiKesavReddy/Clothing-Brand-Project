import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

file_path = "data_points.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    print("Please generate the data first using 'generate_enhanced_data.py'.")
    exit()

numerical_cols_for_outliers = ['Height_cm', 'Weight_kg', 'BMI', 'Hip_Circumference_cm', 'Shoulder_Width_cm']
initial_rows = df.shape[0]

for col in numerical_cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

rows_after_outliers = df.shape[0]
print(f"Removed {initial_rows - rows_after_outliers} outliers. New dataset shape: {df.shape}")

X = df.drop(columns=['Clothing_Size', 'Recommended_Cloth_Color'])
y_clothing_size = df['Clothing_Size']
y_recommended_color = df['Recommended_Cloth_Color']


categorical_features = ['BMI_Category', 'Body_Shape', 'Style_Preference']
numerical_features = ['Height_cm', 'Weight_kg', 'BMI', 'Hip_Circumference_cm', 'Shoulder_Width_cm']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

label_encoder_clothing = LabelEncoder()
y_clothing_size_encoded = label_encoder_clothing.fit_transform(y_clothing_size)

label_encoder_color = LabelEncoder()
y_recommended_color_encoded = label_encoder_color.fit_transform(y_recommended_color)

X_train_cs, X_test_cs, y_train_cs, y_test_cs = train_test_split(
    X, y_clothing_size_encoded, test_size=0.2, random_state=42, stratify=y_clothing_size_encoded
)

X_train_rc, X_test_rc, y_train_rc, y_test_rc = train_test_split(
    X, y_recommended_color_encoded, test_size=0.2, random_state=42, stratify=y_recommended_color_encoded
)

print(f"\nTraining set size (Clothing Size): {X_train_cs.shape[0]} samples")
print(f"Test set size (Clothing Size): {X_test_cs.shape[0]} samples")
print(f"Training set size (Recommended Color): {X_train_rc.shape[0]} samples")
print(f"Test set size (Recommended Color): {X_test_rc.shape[0]} samples")

base_rf_1 = ('rf1', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
base_rf_2 = ('rf2', RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42, n_jobs=-1))
base_rf_3 = ('rf3', RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=10, random_state=42, n_jobs=-1))

base_estimators = [base_rf_1, base_rf_2, base_rf_3]

final_estimator = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)

print("\n--- Building Stacking Classifier for Clothing Size ---")
stacking_pipeline_cs = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking_classifier', StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        verbose=1
    ))
])

print("--- Building Stacking Classifier for Recommended Color ---")
stacking_pipeline_rc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking_classifier', StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        verbose=1
    ))
])

print("\n--- Training and Evaluating Stacking Classifier for Clothing Size ---")
stacking_pipeline_cs.fit(X_train_cs, y_train_cs)
y_pred_cs_encoded = stacking_pipeline_cs.predict(X_test_cs)
y_pred_cs = label_encoder_clothing.inverse_transform(y_pred_cs_encoded)
y_test_cs_original = label_encoder_clothing.inverse_transform(y_test_cs)

print("\nAccuracy Score (Clothing Size - Stacking Model):", accuracy_score(y_test_cs_original, y_pred_cs))
print("\nClassification Report (Clothing Size - Stacking Model):\n", classification_report(y_test_cs_original, y_pred_cs, zero_division=0))

print("\n--- Training and Evaluating Stacking Classifier for Recommended Color ---")
stacking_pipeline_rc.fit(X_train_rc, y_train_rc)
y_pred_rc_encoded = stacking_pipeline_rc.predict(X_test_rc)
y_pred_rc = label_encoder_color.inverse_transform(y_pred_rc_encoded)
y_test_rc_original = label_encoder_color.inverse_transform(y_test_rc)

print("\nAccuracy Score (Recommended Color - Stacking Model):", accuracy_score(y_test_rc_original, y_pred_rc))
print("\nClassification Report (Recommended Color - Stacking Model):\n", classification_report(y_test_rc_original, y_pred_rc, zero_division=0))

print("\n--- Example Prediction with Stacking Models ---")
new_person_data = pd.DataFrame({
    'Height_cm': [183],
    'Weight_kg': [82],
    'BMI': [24.9],
    'BMI_Category': ['Normal'],
    'Body_Shape': ['Hourglass'],
    'Style_Preference': ['Calm & Classic'],
    'Hip_Circumference_cm': [95],
    'Shoulder_Width_cm': [55]
})

predicted_clothing_size_encoded = stacking_pipeline_cs.predict(new_person_data)
predicted_clothing_size = label_encoder_clothing.inverse_transform(predicted_clothing_size_encoded)
print(f"Predicted Clothing Size for new person: {predicted_clothing_size[0]}")

predicted_color_encoded = stacking_pipeline_rc.predict(new_person_data)
predicted_color = label_encoder_color.inverse_transform(predicted_color_encoded)
print(f"Predicted Recommended Color for new person: {predicted_color[0]}")