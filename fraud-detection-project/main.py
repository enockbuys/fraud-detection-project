from preprocessor import Preprocessor
from my_random_forest import MyRandomForest
from classification_summary import classification_summary
from db_manager import DBManager
from conditional_gan import ConditionalGAN
import numpy as np

database_manager = DBManager()
preprocessor = Preprocessor("data/raw/creditcard.csv")
features_train, labels_train, features_validation, labels_validation, features_test, labels_test = preprocessor.clean_and_split()

database_manager.cursor.execute("INSERT INTO datasets (version, description) VALUES (?, ?)", ("v1", "Initial version"))
dataset_id = database_manager.cursor.lastrowid
database_manager.save_split(dataset_id, 'train', features_train, labels_train)
database_manager.save_split(dataset_id, 'validation', features_validation, labels_validation)
database_manager.save_split(dataset_id, 'test', features_test, labels_test)

number_of_features = features_train.shape[1]
features_train_db, labels_train_db = database_manager.load_split(dataset_id, 'train', number_of_features)

print("\nTraining CGAN on fraud samples")
X_fraud = features_train_db[labels_train_db == 1]
y_fraud = labels_train_db[labels_train_db == 1]

def normalize(data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (data - mins) / ranges
    return normalized * 2 - 1

def denormalize(data, original):
    mins = original.min(axis=0)
    maxs = original.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    denorm = (data + 1) / 2
    return denorm * ranges + mins

X_fraud_normalized = normalize(X_fraud)
cgan = ConditionalGAN(output_dim=number_of_features)
cgan.train(X_fraud_normalized, y_fraud.reshape(-1, 1), epochs=500, batch_size=32)

results = {}
synthetic_percentages = [0.05, 0.10, 0.15]

for percentage in synthetic_percentages:
    n_synthetic = int(len(X_fraud) * percentage)
    X_synthetic = cgan.generate_samples(n_synthetic)
    X_synthetic = denormalize(X_synthetic, X_fraud)
    y_synthetic = np.ones(n_synthetic, dtype=np.int64)

    X_combined = np.vstack([features_train_db, X_synthetic])
    y_combined = np.concatenate([labels_train_db, y_synthetic])

    print(f"\nTraining Random Forest with {percentage * 100}% synthetic data...")
    rf_model = MyRandomForest(number_of_trees=10, max_depth=10, min_samples_split=2)
    rf_model.fit(X_combined, y_combined)

    predictions_validation = rf_model.predict(features_validation)
    predictions_test = rf_model.predict(features_test)

    results[percentage] = {
        'val_predictions': predictions_validation,
        'test_predictions': predictions_test
    }

    print("\nValidation Metrics:")
    classification_summary(labels_validation, predictions_validation)
    print("\nTest Metrics:")
    classification_summary(labels_test, predictions_test)

for percentage, preds in results.items():
    database_manager.save_predictions(f"RF+CGAN{percentage}",dataset_id,'validation',labels_validation,preds['val_predictions'])
    database_manager.save_predictions(f"RF+CGAN{percentage}",dataset_id,'test',labels_test,preds['test_predictions'])

database_manager.log_event("CGAN", "training", "Completed CGAN training and evaluation")
database_manager.close()