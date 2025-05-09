import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    df['species_names'] = df.apply(lambda x: str(data.target_names[int(x['species'])]), axis=1)

    return df

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def get_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accurcy = accuracy_score(y_test, predictions)

    return accurcy

def plot_feature(df, feature):
    # Plot a histogram of one of the features
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["species_names"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()

def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()



if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    model, X_train, X_test, y_train, y_test = train(data)
    accurcy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accurcy:.2f}")

    plot_feature(data, "sepal length (cm)")
    plot_features(data)
    plot_model(model, X_test, y_test)

    