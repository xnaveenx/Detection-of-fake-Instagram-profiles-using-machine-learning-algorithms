import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
dataset = pd.read_csv("myds.csv") 

# Separate features and labels
features = dataset[['edge_followed_by', 'edge_follow', 'username_length',
                    'username_has_number', 'full_name_has_number', 'full_name_length',
                    'is_private', 'is_joined_recently', 'has_channel', 'is_business_account',
                    'has_guides', 'has_external_url']]
labels = dataset['is_fake']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize classifiers
logistic_classifier = LogisticRegression()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svm_classifier = SVC(kernel='linear', probability=True)  # Set probability=True for SVC to enable predict_proba

# Train classifiers
logistic_classifier.fit(X_train_scaled, y_train)
rf_classifier.fit(X_train_scaled, y_train)
svm_classifier.fit(X_train_scaled, y_train)

# Initialize and train Voting Classifier
voting_classifier_hard = VotingClassifier(estimators=[
    ('lr', logistic_classifier),
    ('rf', rf_classifier),
    ('svm', svm_classifier)
], voting='hard')

voting_classifier_hard.fit(X_train_scaled, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(voting_classifier_hard, file)
