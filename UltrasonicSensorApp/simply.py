import numpy as np
import pandas as pd
import math

# Function to calculate Shannon Entropy
def calculate_entropy(data):
    # Normalize the data to get the probability distribution
    prob_dist = data / np.sum(data)  # This gives the probability distribution
    # Calculate entropy using Shannon's formula
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # Adding a small value to avoid log(0)
    return entropy

# Assuming signal_data is your dataframe with frequency spectrum and amplitude buffer
def get_entropy_for_rows(signal_data):
    entropies = []
    for row in signal_data.itertuples(index=False):
        # Each row is treated as a frequency bin (or amplitude buffer row)
        row_data = np.array(row)  # Convert tuple to numpy array
        row_entropy = calculate_entropy(row_data)  # Calculate entropy for the row
        entropies.append(row_entropy)
    
    return entropies

# Example data (for testing)
# Replace this with your actual signal data, which will have rows and columns as required
signal_data = pd.DataFrame(np.random.random((200, 85)))  # Example random data (200 rows, 85 frequency bins)

# Calculate entropy for each row (you can use the frequency spectrum or amplitude buffer)
row_entropies = get_entropy_for_rows(signal_data)

# Print the calculated entropy values
print("Entropy for each row:", row_entropies)

# Optional: Find the row with the maximum entropy (indicating the row with the highest variability)
max_entropy_index = np.argmax(row_entropies)
print(f"Row with maximum entropy: {max_entropy_index}, Entropy: {row_entropies[max_entropy_index]}")
