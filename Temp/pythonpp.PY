import pandas as pd

# Load data from CSV file with random number index as headers
file_path = 'train.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path, header=None)

# Print original columns (index generated automatically)
print("Original columns with generated index:")
print(df.columns)

# Optionally, you can rename the columns if needed
# Example: Rename columns to 'Column0', 'Column1', 'Column2', ...
df.columns = [f'Column{i}' for i in range(len(df.columns))]

# Print updated DataFrame to verify
print("\nUpdated DataFrame with generated headers:")
print(df.head())

# Example: Perform additional processing on the DataFrame (e.g., calculations, transformations)

# Save processed DataFrame to a new CSV file
output_file_path = 'processed_data.csv'  # Replace with desired output file path
df.to_csv(output_file_path, index=False)

print(f"\nProcessed DataFrame saved to '{output_file_path}'.")
