import sys

import pandas as pd


def convert_twitter_csv(input_files, output_file):
    """
    Reads one or more CSV files of type index,id,Text,Annotation,oh_label,
    replaces the 'oh_label' column with 'aggressive',
    and saves the combined data to a new CSV file with columns index, text, and aggressive.

    Args:
        input_files (list): A list of paths to the input CSV files.
        output_file (str): The path to the output CSV file.
    """
    try:
        dfs = []
        for input_file in input_files:
            try:
                df = pd.read_csv(input_file)
                if 'oh_label' not in df.columns:
                    print(f"Error: The input CSV file '{input_file}' does not contain the required 'oh_label' column.")
                    sys.exit(1)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: Input file '{input_file}' not found. Skipping this file.")
                continue

        if not dfs:
            print("Error: No valid input files found.  Exiting.")
            sys.exit(1)

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.rename(columns={'oh_label': 'aggressive', 'Text': 'text'})

        # Select and reorder the desired columns
        combined_df = combined_df[['index', 'text', 'aggressive']]

        combined_df.to_csv(output_file, index=False)
        print(f"Successfully converted and combined files. Output saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred while processing the CSV files: {e}")
        sys.exit(1)

def convert_twitter_csv_local(input_files):
    """
    Reads one or more CSV files of type index,id,Text,Annotation,oh_label,
    replaces the 'oh_label' column with 'aggressive',
    and saves the combined data to a new CSV file with columns index, text, and aggressive.

    Args:
        input_files (list): A list of paths to the input CSV files.
        output_file (str): The path to the output CSV file.
    """
    try:
        dfs = []
        for input_file in input_files:
            try:
                df = pd.read_csv(input_file)
                if 'oh_label' not in df.columns:
                    print(f"Error: The input CSV file '{input_file}' does not contain the required 'oh_label' column.")
                    sys.exit(1)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: Input file '{input_file}' not found. Skipping this file.")
                continue

        if not dfs:
            print("Error: No valid input files found.  Exiting.")
            sys.exit(1)

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.rename(columns={'oh_label': 'aggressive', 'Text': 'text'})

        # Select and reorder the desired columns
        combined_df = combined_df[['index', 'text', 'aggressive']]

        return combined_df
        print(f"Successfully converted and combined files. Output saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred while processing the CSV files: {e}")
        sys.exit(1)

def convert_wikipedia_csv(input_files, output_file):
    """
    Reads one or more CSV files of type index,Text,ed_label_0,ed_label_1,oh_label,
    creates an 'id' column, renames 'oh_label' to 'aggressive',
    and saves the combined data to a new CSV file with columns index, text, and aggressive.

    Args:
        input_files (list): A list of paths to the input CSV files.
        output_file (str): The path to the output CSV file.
    """
    try:
        dfs = []
        for input_file in input_files:
            try:
                df = pd.read_csv(input_file)
                required_columns = ['index', 'Text', 'ed_label_0', 'ed_label_1', 'oh_label']
                for col in required_columns:
                    if col not in df.columns:
                        print(f"Error: The input CSV file '{input_file}' is missing the column '{col}'.")
                        sys.exit(1)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: Input file '{input_file}' not found. Skipping this file.")
                continue
        if not dfs:
            print("Error: No valid input files found. Exiting.")
            sys.exit(1)

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.rename(columns={'oh_label': 'aggressive', 'Text': 'text'})

        # Select and reorder the desired columns
        combined_df = combined_df[['index', 'text', 'aggressive']]

        combined_df.to_csv(output_file, index=False)
        print(f"Successfully converted and combined files. Output saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred while processing the CSV files: {e}")
        sys.exit(1)

def convert_youtube_csv(input_files, output_file):
    """
    Reads one or more CSV files of type index,Text,ed_label_0,ed_label_1,oh_label,
    creates an 'id' column, renames 'oh_label' to 'aggressive',
    and saves the combined data to a new CSV file with columns index, text, and aggressive.

    Args:
        input_files (list): A list of paths to the input CSV files.
        output_file (str): The path to the output CSV file.
    """
    try:
        dfs = []
        for input_file in input_files:
            try:
                df = pd.read_csv(input_file)
                required_columns = ['index','UserIndex','Text','Number of Comments','Number of Subscribers','Membership Duration','Number of Uploads','Profanity in UserID','Age','oh_label']
                for col in required_columns:
                    if col not in df.columns:
                        print(f"Error: The input CSV file '{input_file}' is missing the column '{col}'.")
                        sys.exit(1)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: Input file '{input_file}' not found. Skipping this file.")
                continue
        if not dfs:
            print("Error: No valid input files found. Exiting.")
            sys.exit(1)

        combined_df = pd.concat(dfs, ignore_index=True)

        combined_df = combined_df.rename(columns={'oh_label': 'aggressive', 'Text': 'text'})

        # Select and reorder the desired columns
        combined_df = combined_df[['index', 'text', 'aggressive']]

        combined_df.to_csv(output_file, index=False)
        print(f"Successfully converted and combined files. Output saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred while processing the CSV files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # WikipediaFiles = ["../Datasets/aggression_parsed_dataset.csv","../Datasets/attack_parsed_dataset.csv","../Datasets/toxicity_parsed_dataset.csv"]
    # TwitterFiles = ["../Datasets/twitter_parsed_dataset.csv","../Datasets/twitter_racism_parsed_dataset.csv","../Datasets/twitter_sexism_parsed_dataset.csv"]
    YoutubeFiles = ["../Datasets/youtube_parsed_dataset.csv"]
    # convert_twitter_csv(TwitterFiles, "../CleanedDatasets/CleanedTwitterDataset.csv")
    # convert_wikipedia_csv(WikipediaFiles, "../CleanedDatasets/CleanedWikipediaDataset.csv")
    convert_youtube_csv( YoutubeFiles, "../CleanedDatasets/CleanedYoutubeDataset.csv" )