import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC

# --- Load and Combine Data ---
# This section loads the datasets from CSV files, combines them, and performs initial checks.
wikipedia_file = 'CleanedDatasets/CleanedWikipediaDataset.csv'
twitter_file = 'CleanedDatasets/CleanedTwitterDataset.csv'
youtube_file = 'CleanedDatasets/CleanedYoutubeDataset.csv'

try:
    # Attempt to read the CSV files into pandas DataFrames.
    # The CSV files are expected to have columns: 'index', 'text', and 'aggressive'.
    print("Loading datasets...")
    df_wiki = pd.read_csv(wikipedia_file)
    df_twitter = pd.read_csv(twitter_file)
    df_youtube = pd.read_csv(youtube_file)
    print("Datasets loaded successfully.")

    # Combine the two DataFrames into a single DataFrame.
    print("Combining datasets...")
    df = pd.concat([df_wiki, df_twitter,df_youtube], ignore_index=True)
    print("Datasets combined.")

# Handle potential errors during file loading.
except FileNotFoundError:
    print(f"Error: File not found. Please ensure '{wikipedia_file}' and '{twitter_file}' exist in the specified directory.")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found in one of the CSV files. Ensure CSVs have 'index', 'text', and 'aggressive' columns.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Data Cleaning and Preparation ---
# This subsection focuses on cleaning the combined data before further processing.

print("Performing initial data cleaning...")
# Drop the original 'index' column
df = df.drop(columns=['index'], errors='ignore')

# Remove rows where either the 'text' or 'aggressive' column has missing values (NaN).
df = df.dropna(subset=['text', 'aggressive'])

# Ensure the 'aggressive' column is of integer type.
df['aggressive'] = df['aggressive'].astype(int)
print("Initial data cleaning finished.")

# --- 2. Preprocess Text ---
# This section defines and applies a function to clean the text data.
def preprocess_text(text):
    # Ensure the input is a string. Handle potential non-string data (e.g., float if NaNs weren't dropped).
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nApplying text preprocessing...")
# Apply the `preprocess_text` function to each entry in the 'text' column.
# Create a new column 'text_cleaned' to store the results.
df['text_cleaned'] = df['text'].apply(preprocess_text)
print("Text preprocessing applied.")

# Display a sample of original vs. cleaned text.
print("\nSample cleaned data (showing original vs. cleaned text):")
print(df[['text', 'text_cleaned']].head())

# --- 3. Tokenize Text ---
# This section converts the cleaned text into sequences of numerical tokens.

# Define hyperparameters for tokenization and sequencing.
MAX_NUM_WORDS = 30000     # The maximum number of unique words to keep in the vocabulary, based on frequency.
MAX_SEQUENCE_LENGTH = 150 # The maximum length for each text sequence. Longer sequences will be truncated, shorter ones padded.
EMBEDDING_DIM = 100       # The dimensionality of the word embedding vectors. Each word will be represented by a vector of this size.

# Initialize the Keras Tokenizer.
# `num_words`: Specifies the maximum vocabulary size.
# `oov_token="<OOV>"`: A special token used to represent words that are not in the vocabulary (Out-Of-Vocabulary).
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")

print("\nFitting tokenizer on cleaned text data...")
# Build the tokenizer's vocabulary based on the cleaned text data.
# It learns the word-to-index mapping.
tokenizer.fit_on_texts(df['text_cleaned'])

# Retrieve the learned word-to-index mapping.
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.") # Total number of unique words found.
print(f"Vocabulary size restricted to top {MAX_NUM_WORDS} words.")

print("Converting text to sequences...")
# Convert each text sample into a sequence of integers (token indices).
sequences = tokenizer.texts_to_sequences(df['text_cleaned'])
print("Text converted to sequences.")

# --- 4. Pad Sequences ---
# This section ensures all sequences have the same length by padding or truncating.

print(f"\nPadding sequences to a maximum length of {MAX_SEQUENCE_LENGTH}...")
# Pad the sequences to ensure they all have `MAX_SEQUENCE_LENGTH`.
# `maxlen`: The target length for all sequences.
# `padding='post'`: Adds padding (zeros) at the end of shorter sequences.
# `truncating='post'`: Removes elements from the end of longer sequences.
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
print("Padding complete.")

# Display the shape of the final processed data.
# Should be (number of samples, MAX_SEQUENCE_LENGTH).
print(f"Shape of padded sequences tensor: {padded_sequences.shape}")

# --- 5. Split Data ---
# This section splits the data into training and testing sets.

print("\nSplitting data into training and testing sets...")
# Get the labels (0 or 1 for non-aggressive/aggressive) as a NumPy array.
labels = df['aggressive'].values

# Split the padded sequences (features, X) and labels (target, y) into training and testing sets.
# `test_size=0.2`  : Allocates 20% of the data for the test set, 80% for the training set.
# `random_state=42`: Ensures that the result will be the same when created again.
# `stratify=labels`: Ensures that the proportion of classes (0s and 1s) is roughly the same in both train and test sets. Helps balance the dataset
#  and allow for more accurate evaluation
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)
print("Data splitting complete.")

# Display the shapes of the resulting training and testing sets.
print(f"Training set shape - Features: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing set shape - Features: {X_test.shape}, Labels: {y_test.shape}")

# --- 6. Build Model ---
# This section defines the architecture of the neural network model.

print("\nBuilding the LSTM model architecture...")
# Initialize a Sequential model
model = Sequential(name="Aggression_Classifier_LSTM")

# Add an Embedding layer.
# This layer maps the integer-encoded vocabulary indices into dense vectors of fixed size (`EMBEDDING_DIM`).
# It learns word representations during training.
# `input_dim`: Size of the vocabulary (MAX_NUM_WORDS).
# `output_dim`: Dimension of the dense embedding vectors (EMBEDDING_DIM).
# `input_length`: Length of input sequences (MAX_SEQUENCE_LENGTH).
model.add(Embedding(input_dim=MAX_NUM_WORDS,
                    output_dim=EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH,
                    name="Word_Embedding"))

# Add SpatialDropout1D layer.
# This dropout layer helps prevent overfitting by randomly setting entire feature maps (embedding dimensions) to zero.
# 0.2 means 20% of the embedding dimensions will be dropped out during training.
model.add(SpatialDropout1D(0.2, name="Embedding_Dropout1"))

# Add a Bidirectional LSTM layer.
# `Bidirectional` means the input sequence is processed in both forward and backward directions, potentially capturing more context.
# `64`: The number of LSTM units (dimensionality of the output space).
# `dropout=0.2`: Fraction of units to drop for the linear transformation of the inputs.
# `recurrent_dropout=0.2`: Fraction of units to drop for the linear transformation of the recurrent state. Helps prevent overfitting within the LSTM cell.
model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(SpatialDropout1D(0.2, name="Embedding_Dropout2"))
model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.2)))

# Add the final Dense (fully connected) output layer.
# `1`: Number of output neurons (1 for binary classification).
# `activation='sigmoid'`: Sigmoid activation function squashes the output between 0 and 1, representing the probability of the positive class (aggressive).
model.add(Dense(1, activation='sigmoid', name="Output_Layer"))
print("Model architecture built.")

# --- 7. Compile Model ---
# This section configures the model for training (specifies optimizer, loss function, and metrics).

print("\nCompiling the model...")
# Define the metrics to be evaluated during training and testing.
# We include Accuracy, Precision, Recall, and AUC (Area Under the ROC Curve).
compile_metrics = [
    'accuracy',
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc')
]

# Compile the model.
# `loss='binary_crossentropy'`: The standard loss function for binary classification problems.
# `optimizer='adam'`: Optimization algorithm.
# `metrics=compile_metrics`: The list of metrics to calculate and report.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=compile_metrics)
print("Model compiled.")

# Display a summary of the model's layers and parameters.
print("\nModel Summary:")
model.summary()

# --- 8. Train Model ---
# This section trains the model on the training data.

# Define training hyperparameters.
BATCH_SIZE = 64 # Number of samples processed in each iteration (batch).
EPOCHS = 3      # Number of times the entire training dataset is passed through the model. Adjust as needed.

# Configure Early Stopping.
# This callback monitors a specified metric (`val_loss` - loss on the validation set)
# and stops training if the metric doesn't improve for a certain number of epochs (`patience=2`).
# `restore_best_weights=True`: Ensures the model weights from the epoch with the best monitored value are restored at the end of training.
early_stopping = EarlyStopping(monitor='val_loss', # Metric to monitor
                               patience=2,        # Number of epochs with no improvement after which training will be stopped
                               verbose=1,         # Print messages when stopping
                               restore_best_weights=True) # Restore model weights from the epoch with the best value of the monitored metric

print("\nStarting model training...")
history = model.fit(X_train, y_train,          # Training data (features and labels)
                    epochs=EPOCHS,             # Number of epochs to train for
                    batch_size=BATCH_SIZE,     # Size of batches
                    validation_split=0.1,      # Fraction of the training data to use as validation data (monitored by EarlyStopping)
                    callbacks=[early_stopping],# List of callbacks to apply during training (EarlyStopping)
                    verbose=1)
print("Model training finished.")

# --- 9. Evaluate Model ---
# This section evaluates the trained model on the unseen test data.

print("\nEvaluating model on the test data...")
# Evaluate the model using the `evaluate` method.
# It returns the loss value and the metric values specified during compilation.
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)

# Extract the metrics from the results. The order matches the order in `compile_metrics`.
loss = results[0]
accuracy = results[1]
precision_keras = results[2] # Precision calculated by Keras metrics
recall_keras = results[3]    # Recall calculated by Keras metrics
auc_roc_keras = results[4]   # AUC-ROC calculated by Keras metrics

print("\n--- Keras Evaluation Metrics ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision (Keras): {precision_keras:.4f}")
print(f"Test Recall (Keras): {recall_keras:.4f}")
print(f"Test AUC-ROC (Keras): {auc_roc_keras:.4f}")
print("\n\nCalculating final metrics using on the full test set...")
# Get model predictions for the test set.
# `predict` returns probabilities for the positive class (due to sigmoid activation).
y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE)

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5.
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# Calculate metrics using scikit-learn functions.
precision_sklearn = precision_score(y_test, y_pred_binary)
recall_sklearn = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
auc_roc_sklearn = roc_auc_score(y_test, y_pred_proba)

print("\n--- Scikit-learn Evaluation Metrics ---")
print(f"Test Precision (Scikit-learn): {precision_sklearn:.4f}")
print(f"Test Recall (Scikit-learn): {recall_sklearn:.4f}")
print(f"Test F1-Score (Scikit-learn): {f1:.4f}")
print(f"Test AUC-ROC (Scikit-learn): {auc_roc_sklearn:.4f}")


# --- 10. Save Model and Tokenizer ---
# This section saves the trained model and the tokenizer to files for later use.

print("\nSaving the trained model and tokenizer...")

# Define file paths for saving
model_save_path = 'Models/Model4/aggression_lstm_model.keras'
tokenizer_save_path = 'Models/Model4/tokenizer.pickle'

# Save the entire model (architecture, weights, optimizer state).
# Using the .keras format is recommended.
try:
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")


# Save the tokenizer object using pickle.
# The tokenizer is needed to preprocess new text data in the same way as the training data before feeding it to the saved model.
try:
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved successfully to {tokenizer_save_path}")
except Exception as e:
    print(f"Error saving tokenizer: {e}")


# Example of model making predictions
from tensorflow.keras.models import load_model
loaded_model = load_model(model_save_path)
with open(tokenizer_save_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

print("\n--- Example Predictions ---")
example_text = ["This is a wonderful and positive message.", "you are incredibly stupid and I hate you"]
print(f"Example texts: {example_text}")

# Preprocess the example text using the *same* steps and the *fitted* tokenizer
cleaned_examples = [preprocess_text(t) for t in example_text]
example_sequences = tokenizer.texts_to_sequences(cleaned_examples)
example_padded = pad_sequences(example_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Make predictions using the trained model
predictions = model.predict(example_padded)

print("\nExample Predictions (closer to 1 means more likely aggressive):")
for text, pred in zip(example_text, predictions):
    print(f"'{text}': {pred[0]:.4f}")

print("\nScript finished.")