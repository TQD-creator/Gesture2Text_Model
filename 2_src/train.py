# In file: 2_src/train.py
import data_loader  # <-- Importing your own library!
import model        # <-- Importing your own library!
import joblib
from sklearn.metrics import accuracy_score

print("--- STARTING TRAINING SCRIPT ---")

# 1. Load Data
(X_train, y_train, 
 X_valid, y_valid, 
 X_test, y_test) = data_loader.load_data(
    train_csv='../1_data/processed/train_landmarks_augmented.csv',
    valid_csv='../1_data/processed/valid_landmarks.csv',
    test_csv='../1_data/processed/test_landmarks.csv'
)

# 2. Initialize Model
# We'll define 'MyModel' in the next step
predictor = model.MyModel()

# 3. Train Model
print("Training model...")
predictor.train(X_train, y_train)
print("Training complete.")

# 4. Evaluate
valid_preds = predictor.predict(X_valid)
accuracy = accuracy_score(y_valid, valid_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. Save Model
predictor.save('../4_models/sign_language_model.pkl')
print("Model saved to 4_models/")