import csv
import pandas as pd

def remove_duplicates(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Drop duplicate rows
    df.drop_duplicates(subset='Game State', keep='first', inplace=True)

    # Write the DataFrame back to the CSV file
    df.to_csv(filename, index=False)

# remove_duplicates('connect4_data.csv')

def check_for_duplicates(filename):
    seen_states = set()
    duplicates = 0

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            game_state = row[0]
            if game_state in seen_states:
                duplicates += 1
            else:
                seen_states.add(game_state)

    print(f"Number of duplicate game states: {duplicates}")

check_for_duplicates('connect4_data.csv')