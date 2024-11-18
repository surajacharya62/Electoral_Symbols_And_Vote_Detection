import pandas as pd
import ast
import os


class ReshapeData():
    def process_and_reshape_data_v2(self, file_path):
        # Load the Excel file
        data = pd.read_excel(os.path.join(file_path, 'df_total_comparisions.xlsx'))       
        # List to hold all dictionary data
        all_dicts = []

        # Iterate through each row and each column in the DataFrame
        for row_index in range(data.shape[0]):  # Iterate over rows
            for col_index in range(1, data.shape[1]):  # Iterate over columns, assuming column 0 is an index or unrelated
                cell_content = data.iloc[row_index, col_index]
                if pd.notna(cell_content):  # Check if the cell is not empty
                    try:
                        # convert the stringified dictionary into an actual dictionary
                        dict_data = ast.literal_eval(cell_content)
                        all_dicts.append(dict_data)
                    except ValueError as e:
                        print(f"Error processing cell at Row {row_index}, Column {col_index}: {e}")
                        continue

        # Create a DataFrame from the list of dictionaries
        result_df = pd.DataFrame(all_dicts)

        # Save the DataFrame to a new Excel file
        output_path = './artifacts/faster_rcnn_files/total_comparisons_normalized.xlsx'
        result_df.to_excel(output_path, index=False)
        # print(f"Data saved to {output_path}")



# obj = ReshapeData()
# path = 'df_total_comparisions.xlsx'
# obj.process_and_reshape_data_v2(path)
