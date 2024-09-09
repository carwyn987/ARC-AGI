import json
import os

def load_json_data(directory):
    """Load all JSON files in the given directory"""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                data.append(json.load(f))
    return data

def extract_grid_data(data):
    """Extract grid-like data from JSON"""
    grids = []
    for item in data:
        input_grid = []
        for row in item["input"]:
            input_grid.append(" ".join(str(x) for x in row))
        output_grid = []
        for row in item["output"]:
            output_grid.append(" ".join(str(x) for x in row))
        grids.append((input_grid, output_grid))
    return grids

def compare_grids(predicted, actual):
    """Compare predicted and actual grid outputs"""
    for i, (predicted_row, actual_row) in enumerate(zip(predicted, actual)):
        if predicted_row != actual_row:
            print(f"Row {i+1} mismatch:")
            print(f"  Predicted: {' '.join(map(str, predicted_row))}")
            print(f"  Actual: {' '.join(map(str, actual_row))}")
            return False
    return True

def get_grid_json(data, key):
    """Return a str grid in human-readable format from JSON data"""
    output = ""
    grid = data[key]
    for row in grid:
        output += " ".join(str(x) for x in row) + "\n"
    return output

def get_grid_list(grid):
    """Return a str grid in human-readable format from a nested list"""
    output = ""
    for row in grid:
        output += " ".join(str(x) for x in row) + "\n"
    return output