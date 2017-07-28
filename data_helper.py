import numpy as np
import pickle


def data_handler(trust_path, rating_path, trainp, validp, testp):
    """
    trainp + validp + testp = 1.0

    trust_file:
    u, v, timestamp

    rating file
    u, item, category, rating, _, timestamp
    """

    f1 = open(rating_path, "rb")
    f2 = open(trust_path, "rb")
    rating = pickle.load(f1)
    trust = pickle.load(f2)
    print(np.amax(trust))
    print(np.amax(rating, axis=0)[0])
    n_nodes = max(np.amax(trust), np.amax(rating, axis=0)[0]) + 1
    f1.close()
    f2.close()

    sorted_trust = np.sort(
        trust.view(dtype=[('a',np.int64),('b',np.int64),('c',np.int64)]),
        order=['c'],
        axis=0).view(dtype=np.int64)
    train_index = int(trainp*sorted_trust.shape[0])
    valid_index = int(trainp*sorted_trust.shape[0]+validp*sorted_trust.shape[0])
    test_index = int(sorted_trust.shape[0]*(1.0-testp))
    max_time_stamp = sorted_trust[train_index-1][2]
    sorted_trust = np.delete(sorted_trust, np.s_[2], axis=1)
    train_set = sorted_trust[:train_index]
    valid_set = sorted_trust[train_index:valid_index]
    test_set = sorted_trust[test_index:]
    filtered_rating = rating[rating[:,5]<=max_time_stamp]

    # delete timestamps and convert to set
    trust_set = set(map(tuple,np.delete(trust, np.s_[2], axis=1)))

    return [train_set, valid_set, test_set, filtered_rating, trust_set, n_nodes]


if __name__ == "__main__":
    dt = data_handler('filtered_matrices/rating_t_4.pkl', 'filtered_matrices/trust_t_4.pkl')
