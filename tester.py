from llm_model import generate_query, get_model_output
from helper import load_json_data, compare_grids, get_grid_list, get_grid_json

def main():
    # Load evaluation data
    evaluation_data = load_json_data("data/evaluation")

    correct_predictions = 0
    correct_size_predictions = 0
    missed = 0
    prediction_failed = 0

    for i, data in enumerate(evaluation_data):
        query = generate_query(data, True)
        predicted_output = get_model_output(query)
        if isinstance(predicted_output, dict) and "output" in predicted_output:
            predicted_output = predicted_output["output"]
        else:
            print("No output found")
            prediction_failed += 1
            continue
        actual_output = data["test"][0]["output"]

        print("\n\nPredicted Output:" + "\n" + get_grid_list(predicted_output))
        print("Actual Output:" + "\n" + get_grid_list(actual_output))
        print("\n")
        # Compare predicted and actual outputs
        print(f"Test case {i+1}:")
        if compare_grids(predicted_output, actual_output):
            print("  PASS")
            correct_predictions += 1
            if len(predicted_output) == len(actual_output) and len(predicted_output[0]) == len(actual_output[0]):
                correct_size_predictions += 1
        else:
            print("  FAIL")
            missed += 1

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Correct Predictions: {correct_predictions} / {len(evaluation_data)} ({correct_predictions / len(evaluation_data) * 100:.2f}%)")
        print(f"  Correct Size Predictions: {correct_size_predictions} / {len(evaluation_data)} ({correct_size_predictions / len(evaluation_data) * 100:.2f}%)")
        print(f"  Missed: {missed} / {len(evaluation_data)} ({missed / len(evaluation_data) * 100:.2f}%)")
        print(f"  Prediction Failed: {prediction_failed} / {len(evaluation_data)} ({prediction_failed / len(evaluation_data) * 100:.2f}%)")

if __name__ == "__main__":
    main()