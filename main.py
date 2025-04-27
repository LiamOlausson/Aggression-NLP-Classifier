import tensorflow as tf
import pickle
import numpy as np
import pandas as pd # Import pandas
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
from Preprocessing import CleanData

# There are a number of past models saved that can be tested, simply change the
# Models/Model2/... part of the path to Models/Model2/... where # is the desired model
MODEL_PATH = 'Models/Model4/aggression_lstm_model.keras'
TOKENIZER_PATH = 'Models/Model4/tokenizer.pickle'
TEST_DATASET_PATH = ['Datasets/twitter_parsed_dataset.csv']
# TEST_DATASET_PATH = "CleanedDatasets/CleanedYoutubeDataset.csv" #Use for an already cleaned dataset
MAX_SEQUENCE_LENGTH = 150
PREDICTION_THRESHOLD = 0.5

# --- 1. Load Model and Tokenizer ---
print("Loading resources...")
model = None
tokenizer = None

# Check file existence first
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    exit()
if not os.path.exists(TOKENIZER_PATH):
    print(f"FATAL ERROR: Tokenizer file not found at '{TOKENIZER_PATH}'")
    exit()

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
    print("\nModel Summary:")
    model.summary(print_fn=lambda x: print(f"  {x}")) # Indent summary slightly
    # Verify input shape compatibility if possible
    try:
        input_shape = model.input_shape
        # Handle potential list if model has multiple inputs
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if len(input_shape) == 2 and input_shape[1] is not None and input_shape[1] != MAX_SEQUENCE_LENGTH:
             print(f"\nWarning: Model expected input length {input_shape[1]}, but MAX_SEQUENCE_LENGTH is set to {MAX_SEQUENCE_LENGTH} based on training script. Ensure consistency.")
        elif input_shape[1] is None:
             print(f"\nInfo: Model input shape allows variable length, but padding to {MAX_SEQUENCE_LENGTH}.")

    except Exception as e:
         print(f"\nWarning: Could not fully verify model input shape compatibility: {e}")

except IOError as e:
    print(f"FATAL ERROR: IO error loading model from '{MODEL_PATH}': {e}")
    exit()
except Exception as e:
    # Catch other potential errors during model loading (e.g., corrupted file)
    print(f"FATAL ERROR: Unexpected error loading model from '{MODEL_PATH}': {e}")
    exit()

# Load Tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(f"\nTokenizer loaded successfully from '{TOKENIZER_PATH}'")
except FileNotFoundError: # Should be caught above, but as safeguard
     print(f"FATAL ERROR: Tokenizer file not found at '{TOKENIZER_PATH}'.")
     exit()
except (pickle.UnpicklingError, EOFError) as e:
     print(f"FATAL ERROR: Error unpickling tokenizer from '{TOKENIZER_PATH}': {e}")
     exit()
except IOError as e:
     print(f"FATAL ERROR: IO error loading tokenizer from '{TOKENIZER_PATH}': {e}")
     exit()
except Exception as e:
     print(f"FATAL ERROR: Unexpected error loading tokenizer: {e}")
     exit()

# --- 2. Evaluate on Test Data from CSV ---
print("\n--- Test Data Evaluation ---")
if not os.path.exists(TEST_DATASET_PATH[0]): #Use for multiple datasets of the same format (in this case twitter)
    print(f"Warning: Test dataset file not found at '{TEST_DATASET_PATH}'. Skipping evaluation.")
# if not os.path.exists(TEST_DATASET_PATH):
#     print(f"Warning: Test dataset file not found at '{TEST_DATASET_PATH}'. Skipping evaluation.")
else:
    try:
        print(f"Loading test data from '{TEST_DATASET_PATH}'...")
        # df_test = pd.read_csv(TEST_DATASET_PATH)
        df_test = CleanData.convert_twitter_csv_local(TEST_DATASET_PATH)
        print(f"Loaded {len(df_test)} rows from the dataset.")

        # --- Data Validation and Preparation ---
        if 'text' not in df_test.columns or 'aggressive' not in df_test.columns:
            # Specific check matching CreateModel.py's potential KeyError
            raise KeyError("CSV file must contain 'text' and 'aggressive' columns.")

        initial_rows = len(df_test)
        df_test.dropna(subset=['text', 'aggressive'], inplace=True)
        rows_after_na = len(df_test)
        if rows_after_na < initial_rows:
            print(f"Info: Removed {initial_rows - rows_after_na} rows with missing 'text' or 'aggressive' values.")

        df_test['text'] = df_test['text'].astype(str)

        try:
            # Ensure 'aggressive' column is numeric before conversion
            test_labels_series = pd.to_numeric(df_test['aggressive'], errors='coerce')
            # Check if coercion introduced NaNs (meaning non-numeric values were present)
            if test_labels_series.isnull().any():
                raise ValueError("The 'aggressive' column contains non-numeric values.")
            test_labels = test_labels_series.values # Convert to NumPy array
        except ValueError as e:
             print(f"Data Error: {e} Skipping evaluation.")
             # Skip evaluation if labels are bad
             raise # Re-raise to be caught by the outer except block

        test_texts = df_test['text'].tolist()

        if len(test_texts) == 0:
             print("Warning: No valid test data remaining after cleaning. Skipping evaluation.")
        else:
            print(f"Preprocessing {len(test_texts)} test samples using MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}...")
            # Preprocess using loaded tokenizer and known MAX_SEQUENCE_LENGTH
            test_sequences = tokenizer.texts_to_sequences(test_texts)
            # Use padding and truncating from CreateModel.py
            X_test_padded = pad_sequences(test_sequences,
                                          maxlen=MAX_SEQUENCE_LENGTH,
                                          padding='post',
                                          truncating='post')

            y_test = np.array(test_labels) # Ensure labels are numpy array

            print(f"Test data shape after padding: {X_test_padded.shape}")
            print(f"Test labels shape: {y_test.shape}")

            print("\nEvaluating model on test data...")
            results = model.evaluate(X_test_padded, y_test, batch_size=64, verbose=1) # Using BATCH_SIZE from CreateModel.py

            print("\nTest Results:")
            for name, value in zip(model.metrics_names, results):
                print(f"{name}: {value:.4f}")

    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"\nError: Test dataset file not found at '{TEST_DATASET_PATH}' (already checked, this is unexpected). Skipping evaluation.")
    except KeyError as e:
         print(f"\nData Error: Missing required column in CSV: {e}. Skipping evaluation.")
    except ValueError as ve: # Catch label conversion errors etc.
         print(f"\nData Error during processing: {ve}. Skipping evaluation.")
    except pd.errors.EmptyDataError:
         print(f"\nData Error: The CSV file '{TEST_DATASET_PATH}' is empty. Skipping evaluation.")
    except Exception as e:
        # Catch-all for other unexpected errors during evaluation
        print(f"\nAn unexpected error occurred during test data loading or evaluation: {e}")
        print("Skipping test data evaluation.")


# --- 3. Interactive Prediction Loop ---
print("\n--- Interactive Prediction ---")
print(f"Enter text to classify aggression (model expects sequences up to {MAX_SEQUENCE_LENGTH} tokens).")
print("Type 'quit' or 'exit' to stop.")

while True:
    try:
        user_input = input("\nEnter text> ")
        if user_input.lower() in ['quit', 'exit']:
            break

        if not user_input.strip():
            print("Input is empty. Please enter some text.")
            continue

        # Preprocess user input using the same steps as training
        # We need the preprocess_text function from CreateModel.py
        # Define it here for self-containment, ensuring it matches
        def preprocess_text_interactive(text):
             if not isinstance(text, str): return ""
             text = text.lower()
             text = re.sub(r'\W', ' ', text)
             text = re.sub(r'\s+', ' ', text).strip()
             return text

        cleaned_input = preprocess_text_interactive(user_input)
        if not cleaned_input:
             print("Input became empty after cleaning.")
             continue

        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence,
                                        maxlen=MAX_SEQUENCE_LENGTH,
                                        padding='post', # Match training
                                        truncating='post') # Match training

        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)

        # Interpret prediction (Sigmoid output)
        predicted_prob = prediction[0][0]
        print(f"Raw prediction output (probability): {predicted_prob:.6f}")

        if predicted_prob >= PREDICTION_THRESHOLD:
            print(f"Result: Aggressive (Confidence: {predicted_prob*100:.2f}%)")
        else:
            print(f"Result: Not Aggressive (Confidence: {(1-predicted_prob)*100:.2f}%)")

    except Exception as e:
        # General catch for unexpected errors during the interactive loop
        print(f"An error occurred during interactive prediction: {e}")
        # Consider if loop should continue or break on errors
        pass

print("\nExiting interactive mode.")