import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_and_test_models(data_filename):
    # Load the curated data
    curated_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "curated" / data_filename
    try:
        players = pd.read_parquet(curated_data_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),
             ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
              'STL', 'BLK', 'TOV', 'PF']),
            ('cat', OneHotEncoder(), ['Pos', 'Tm'])
        ])

    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    for model_name, classifier in classifiers.items():
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)  # Set the classifier in the pipeline
        ])

        # Split data into training and test sets
        X = players.drop(['PTS', 'FG%', 'rising_stars'], axis=1)
        y_rising_stars = players['rising_stars']

        # Split data into training and test sets
        X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(X, y_rising_stars, test_size=0.2,
                                                                        random_state=42)

        # Train model to predict rising_stars
        model.fit(X_train_rs, y_train_rs)
        preds = model.predict(X_test_rs)
        print(f'Accuracy for {model_name} prediction: {accuracy_score(y_test_rs, preds)}')

        # Save the model as a pickle file
        model_output_path_pickle = Path(
            __file__).resolve().parent.parent.parent / 'models' / f'{model_name}_model' / 'train.pkl'
        model_output_path_pickle.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(model, model_output_path_pickle)
            print(f'{model_name} Model saved successfully as pickle at: {model_output_path_pickle}')
        except Exception as e:
            print(f"Error saving {model_name} model as pickle: {e}")

        # Load the trained model from the pickle file
        try:
            loaded_model = joblib.load(model_output_path_pickle)
            print(f'Model loaded successfully from: {model_output_path_pickle}')

            # Make predictions on the test set
            test_predictions = loaded_model.predict(X_test_rs)

            # Combine features, true labels, and predicted labels
            results_df = pd.DataFrame({
                'Player': X_test_rs['Player'],
                'True_Labels': y_test_rs,
                'Predicted_Labels': test_predictions
            })

            # Save the DataFrame to Parquet
            results_parquet_path = Path(
                __file__).resolve().parent.parent.parent / 'models' / f'{model_name}_model' / f'test_{model_name}.parquet'
            results_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_parquet(results_parquet_path, index=False)

            print(f'Test predictions for {model_name} saved successfully at: {results_parquet_path}')

        except Exception as e:
            print(f"Error loading {model_name} model from pickle: {e}")


if __name__ == "__main__":
    # Replace 'curated_data.parquet' with your actual curated data file name
    train_and_test_models('curated_data.parquet')
