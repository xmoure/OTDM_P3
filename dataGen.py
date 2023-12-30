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

     # Save to iris.dat
    with open('data/iris.dat', 'w') as dat_file:
        # Write the parameters for m and n
        dat_file.write(f'param m := {len(iris_sample_normalized)};\n')
        dat_file.write(f'param n := {iris_sample_normalized.shape[1]};\n\n')

        # Write the data for matrix A
        dat_file.write('param A :')
        for j in range(1, iris_sample_normalized.shape[1] + 1):
            dat_file.write(f' {j}')
        dat_file.write(' :=\n')

        for i, sample in enumerate(iris_sample_normalized, start=1):
            dat_file.write(f'{i}')
            for feature in sample:
                dat_file.write(f' {feature}')
            dat_file.write('\n')
        dat_file.write(';\n')


def save_moon_dat(moon_data):
    # Save to moon.dat
    with open('data/moon.dat', 'w') as dat_file:
        # Write the parameters for m (number of points) and n (number of features)
        m, n = moon_data.shape
        dat_file.write(f'param m := {m};\n')
        dat_file.write(f'param n := {n};\n\n')

        # Write the data for matrix A
        dat_file.write('param A :')
        for j in range(1, n + 1):
            dat_file.write(f' {j}')
        dat_file.write(' :=\n')

        for i, sample in enumerate(moon_data, start=1):
            dat_file.write(f'{i}')
            for feature in sample:
                dat_file.write(f' {feature}')
            dat_file.write('\n')
        dat_file.write(';\n')

def genMoons():
    # Generate moon dataset with 25 samples
    moon_data, _ = datasets.make_moons(n_samples=25, noise=0.05, random_state=42)
    moon_data_normalized = (moon_data - np.mean(moon_data, axis=0)) / np.std(moon_data, axis=0)
    moon_data_df = pd.DataFrame(moon_data_normalized)
    moon_data_df.to_csv('data/moon_data.txt', sep=' ', header=False, index=False)

    # Save the data to a .dat file for AMPL
    save_moon_dat(moon_data_normalized)


if __name__ == "__main__":
    genIris()
    genMoons()
