import pandas as pd
import numpy as np


# Produce the attribute sets of each columun in data
def generate_attribute_sets(data):
    A = {}
    for i in range(data.shape[1]):
        A[i] = np.unique(data[:, i])
    return A


# Build the product sets in each of the subspaces in the randomized R_2'
def build_subspaces(R_2_idx, A):
    R_2 = []
    for (a1, a2) in R_2_idx:
        A1, A2 = A[a1], A[a2]
        S = set()
        for cat1 in A1:
            for cat2 in A2:
                S.add((cat1, cat2))
        R_2.append(S)
    return R_2


# Create the probability tables that can be used for lookup.
def create_probability_tables(D, t, n, A):
    # Copy dataset to ensure problemless shuffling.
    shuffleable = D.copy()
    # Generate the indicies of the features.
    feat_idx = list(range(D.shape[1]))
    q = len(feat_idx)
    # Create the empty "set" of probability tables.
    H = []
    for i in range(t):
        # Shuffle data and pick the n first (n samples without replacement).
        np.random.shuffle(shuffleable)
        Ni = shuffleable[:n]
        # Shuffle the features and generate R_2
        np.random.shuffle(feat_idx)
        R_2_idx = list(zip(feat_idx, feat_idx[1:])) + [(feat_idx[-1],
                                                        feat_idx[0])]
        R_2 = build_subspaces(R_2_idx, A)
        # Initialize hi as a dict. Not sure if there is a better way of storage.
        hi = {}
        # For each subspace in R2, test for each combination of the categorical
        # values from the subspace if they have zero apperance in the Ni set.
        for idx in range(q):
            # Extract the subspace feature indicies
            S_idx = R_2_idx[idx]
            # Extract the subspace
            S = R_2[idx]
            # Slice in Ni set such that only features in S are represented.
            Ni_S = Ni[:, S_idx]
            indicators = []
            # Now for each combination of the categorical values in the product
            # set created earlier, test if that combination is present in Ni_s.
            # If not, we have a zero appearance.
            for comb in S:
                indicator = (comb, True)
                for sample in Ni_S:
                    if tuple(comb) == tuple(sample):
                        # If we find an appearance, no need to check the rest
                        indicator = (comb, False)
                        break
                indicators.append(indicator)
            # Added the appearances to the "table".
            hi[S_idx] = indicators
        # Append the model to the set of models.
        H.append(hi)
    return H


# The actual scoring of a sample
def zero_score(x, t, H):
    # Initialize the score as zero
    score = 0
    # Iterate over the models in H
    for i in range(t):
        # Extract the model hi
        hi = H[i]
        # Initialize the score for the model as 0
        r = 0
        # For each subspace in hi, check if x has a zero appearance in that
        # subspace.
        for S_idx, values in hi.items():
            # Get only the relevant columns for the subspace
            subspace_x = tuple([x[idx] for idx in S_idx])
            # For each combination in the subspace, check if it matches
            # x in the subspace and if it had zero appearances. Break
            # when the match is found.
            for comb, zero in values:
                if subspace_x == comb:
                    if zero:
                        # If it had zero appearance, increment r.
                        r += 1
                    break
        # Incremment score with r.
        score += r
    return score


if __name__ == "__main__":
    np.random.seed(42)
    # Use raw categorical values.
    data = pd.read_csv("mushroom.txt", header=None).values
    # Change dataset to match the one in the paper with 5% poisonous.
    anon = data[data[:, 0] == "p"]
    np.random.shuffle(anon)
    data = np.r_[data[data[:, 0] == "e"], anon[:221]]
    np.random.shuffle(data)
    # Split dataset into data and labels
    y = data[:, :1]
    X = data[:, 1:]
    X = pd.get_dummies(pd.DataFrame(X)).values
    t, n = 50, 8
    # Generate attribute sets. That is the unique values for each column. If
    # one-hot encoding was used, these would be trival.
    A = generate_attribute_sets(X)
    # Create the probability tables.
    H = create_probability_tables(X, t, n, A)
    # Score each of the samples.
    for i in range(len(X)):
        score = zero_score(X[i], t, H)
        print(score, y[i][0])
