import os
import pandas as pd

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Define the folder path where the CSV files are located
folder_path = "./"

per_file = None

array_task = []
# Loop through the folder and its subfolders
for folder_name in os.listdir(folder_path):
    folder = os.path.join(folder_path, folder_name)
    
    # Check if it's a directory and starts with "array_task"
    if os.path.isdir(folder) and folder_name.startswith("array_task"):
        number = int(folder_name.replace("array_task", ""))
        csv_file = os.path.join(folder, f"random_search_array{number}.csv")

        if per_file == None:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                row_count = df.shape[0]

        for _ in range(row_count):
            array_task.append(number)

        
        # Check if the CSV file exists in the folder
        if os.path.exists(csv_file):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)
            
            # Append the DataFrame to the merged_data DataFrame
            merged_data = pd.concat([merged_data, df])
merged_data["array"] = array_task

# Sort the merged_data DataFrame by the "val_acc" column in descending order
merged_data = merged_data.sort_values(by="val_acc", ascending=False)

# Save the sorted data to a new CSV file
output_file = "combined_results.csv"
merged_data.to_csv(output_file, index=False)

print(f"Merged and sorted data saved to {output_file}")
