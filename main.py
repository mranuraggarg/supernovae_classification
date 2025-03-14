from dataset import load_saved_data
from evaluate import evaluate

if __name__ == "__main__":
    # Load the data from stored npz file
    X_train, Y_train, X_test, Y_test, X_train_reverse, X_test_reverse, ids_train, ids_test, metadata = load_saved_data(format="npz")

    # Evaluate the models
    final_results_dict = evaluate(X_test, Y_test)