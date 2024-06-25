import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your large dataset into a pandas DataFrame
# Example: Specify dtype and handle errors if any columns are mixed types
try:
    df = pd.read_csv('train.csv', low_memory=False)  # Adjust low_memory as needed
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
    raise

# Remove non-numeric rows or columns if necessary (like headers)
# Example: Assuming first row is header, and it contains non-numeric values
df = df.iloc[1:, :]  # Assuming header is in the first row, skip it

# Convert numeric columns to float for heatmap (adjust as needed)
numeric_columns = df.columns
df[numeric_columns] = df[numeric_columns].astype(float)

# Create a large heatmap
plt.figure(figsize=(20, 10))  # Adjust the figsize as needed to fit your data size
sns.heatmap(df.iloc[:100, :100], cmap='viridis')  # Example: Plotting first 100 rows and columns

# Customize heatmap appearance (optional)
plt.title('Large Dataset Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')

# Save the heatmap as an image file
plt.savefig('large_dataset_heatmap.png')  # Adjust filename and format as needed

# Display the plot (optional)
plt.show()
