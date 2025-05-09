from iris_pipeline import load_dataset, train, get_accuracy

def test_load_dataset():
    df = load_dataset()
    assert not df.empty, "The DataFrame should not be empty after loading the dataset"

def test_mode_accuracy():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(df)
    accuracy = get_accuracy(model, X_test, y_test)
    # accuracy = 0.2 # Check if the test works well in both cases.
    assert accuracy > 0.8, "Model accuracy is below 80%"

    # Test with the following command
    # pytest test_iris_pipeline.py