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
            print(f"  Predicted: {' '.join(predicted_row)}")
            print(f"  Actual: {' '.join(actual_row)}")
            return False
    return True

def evaluate_model(model, data):
    """Evaluate the given model on the provided data"""
    results = []
    for item in data:
        input_data = item["train"]
        predicted_output = model(input_data)
        actual_output = item["test"]
        results.append((predicted_output, actual_output))
    return results

def main():
    # Load evaluation data
    evaluation_data = load_json_data("data/evaluation")

    # Extract grid-like data
    grids = extract_grid_data(evaluation_data)

    # Define the model ( replace with your own model )
    def dummy_model(input_data):
        # Replace with your own model's logic
        return input_data

    # Evaluate the model
    results = evaluate_model(dummy_model, evaluation_data)

    # Compare predicted and actual outputs
    for i, (predicted, actual) in enumerate(results):
        print(f"Test case {i+1}:")
        predicted_grid, actual_grid = extract_grid_data([{"input": predicted, "output": actual}])[0]
        if compare_grids(predicted_grid, actual_grid):
            print("  PASS")
        else:
            print("  FAIL")

if __name__ == "__main__":
    main()