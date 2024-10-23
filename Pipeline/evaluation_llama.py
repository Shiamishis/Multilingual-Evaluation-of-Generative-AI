# %%
import os
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import replicate
import numpy as np

# %%
# Load the Multi-EURLEX dataset
dataset = load_dataset('multi_eurlex', 'all_languages', split='test')

# Define label options (21 categories)
label_options = [
    "POLITICS", "INTERNATIONAL RELATIONS", "EUROPEAN UNION", "LAW", "ECONOMICS",
    "TRADE", "FINANCE", "SOCIAL QUESTIONS", "EDUCATION AND COMMUNICATIONS", "SCIENCE",
    "BUSINESS AND COMPETITION", "EMPLOYMENT AND WORKING CONDITIONS", "TRANSPORT",
    "ENVIRONMENT", "AGRICULTURE, FORESTRY AND FISHERIES", "AGRI-FOODSTUFFS",
    "PRODUCTION, TECHNOLOGY AND RESEARCH", "ENERGY", "INDUSTRY", "GEOGRAPHY",
    "INTERNATIONAL ORGANISATIONS"
]


# Preprocess the dataset and map true labels to category numbers
def preprocess_dataset(dataset, label_options):
    preprocessed_data = []
    for item in dataset:
        text = item['text']['en']  # Extract English text
        labels = item['labels']  # True label numbers
        preprocessed_data.append({"text": text, "labels": labels})
    return preprocessed_data
    # return preprocessed_data[:15]


# %%
def unify_string(string_list):
    # Join the list of strings into one unified string with no separator
    unified_string = ''.join(string_list)

    # Optionally, strip leading/trailing spaces
    return unified_string.strip()


# %%
# Initialize the client with the token
client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


# %%
def classify_text_with_llama_in_chunks(text, label_options, chunk_size=4096):
    prompt_template = "ONLY GIVE THE CATEGORIES IN YOUR ANSWER. Classify the following text into one or more of these categories: {}. Text: {}."

    def chunk_text(text, chunk_size):
        # Split the text into smaller chunks of size chunk_size
        words = text.split()  # Split by spaces to preserve words
        chunks = []

        # Create chunks of words with a rough size of chunk_size
        current_chunk = []
        current_size = 0

        for word in words:
            current_size += len(word) + 1  # +1 for the space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = len(word) + 1  # Reset size for the new chunk
            current_chunk.append(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))  # Add the last chunk

        return chunks

    # Split the text into manageable chunks
    text_chunks = chunk_text(text, chunk_size)
    accumulated_response = []

    # Loop through the chunks and call the model on each one
    for i, chunk in enumerate(text_chunks):
        prompt = prompt_template.format(', '.join(label_options), chunk)

        try:
            output = client.run("meta/llama-2-7b-chat", input={"prompt": prompt})
            response = unify_string(output)

            if response:
                accumulated_response.append(response)
            else:
                print(f"Error: No response from the API for chunk {i + 1}.")

        except Exception as e:
            print(f"Exception during API call for chunk {i + 1}: {e}")

    # Return all classifications accumulated from all chunks
    print("Accumulated response: ", accumulated_response)
    return ' '.join(accumulated_response)


# %%
# Extract relevant labels from LLaMa's output
def extract_labels_from_generated_text(generated_text, label_options):
    relevant_labels = []
    for label in label_options:
        if label.lower() in generated_text.lower():
            relevant_labels.append(label)
    return relevant_labels


# Map label names back to indices
def map_labels_to_indices(label_names, label_options):
    label_indices = [label_options.index(label) for label in label_names if label in label_options]
    return label_indices


# %%
def evaluate_llama_on_dataset(dataset, label_options):
    all_true_labels = []
    all_predicted_labels = []
    count = 0  # Track the number of requests

    print(len(dataset))
    i = 0
    for entry in dataset:
        if i > 0:
            break
        text = entry['text']
        true_labels = entry['labels']
        generated_text = classify_text_with_llama_in_chunks(text, label_options)
        predicted_label_names = extract_labels_from_generated_text(generated_text, label_options)
        predicted_labels = map_labels_to_indices(predicted_label_names, label_options)

        # Store true and predicted labels for comparison
        all_true_labels.append(true_labels)
        all_predicted_labels.append(predicted_labels)
        i += 1

    return all_true_labels, all_predicted_labels


# %%
# Preprocess the dataset
preprocessed_data = preprocess_dataset(dataset, label_options)

# Run the evaluation on the entire preprocessed dataset using Gemini
true_labels, predicted_labels = evaluate_llama_on_dataset(preprocessed_data, label_options)
# %%
# Convert true and predicted labels to binary format using MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(range(len(label_options))))

# Binarize the true and predicted labels
binary_true = mlb.fit_transform(true_labels)
binary_pred = mlb.transform(predicted_labels)

# Get indices of labels with non-zero true or predicted samples
relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]

# Filter binary_true and binary_pred to only include relevant labels
filtered_binary_true = binary_true[:, relevant_labels]
filtered_binary_pred = binary_pred[:, relevant_labels]

# Calculate precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(
    filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
)

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")