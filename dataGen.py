from sklearn import datasets
import numpy as np
import pandas as pd

np.random.seed(42)  # For reproducibility


def genIris():
    iris = datasets.load_iris()
    iris_data = iris.data

    # Select a sample of 25 rows
    sample_indices = np.random.choice(iris_data.shape[0], 25, replace=False)
    iris_sample = iris_data[sample_indices]
    iris_sample_normalized = (iris_sample - np.mean(iris_sample, axis=0)) / np.std(iris_sample, axis=0)
    iris_sample_df = pd.DataFrame(iris_sample_normalized)
    iris_sample_df.to_csv('data/iris_data.txt', sep=' ', header=False, index=False)


def genMoons():
    # Generate moon dataset with 25 samples
    moon_data, _ = datasets.make_moons(n_samples=25, noise=0.05, random_state=42)
    moon_data_normalized = (moon_data - np.mean(moon_data, axis=0)) / np.std(moon_data, axis=0)
    moon_data_df = pd.DataFrame(moon_data_normalized)
    moon_data_df.to_csv('data/moon_data.txt', sep=' ', header=False, index=False)


if __name__ == "__main__":
    genIris()
    genMoons()
