import argparse
import numpy as np
import networkx as nx
import sampleutil
import simops
import data_helper
import trustmodel_dynamic_nowalk as trustmodel
import os.path
import pickle

class Logger(object):

    def __init__(self, file_object):
        self._fobject = file_object

    def write(self, wstring):
        print(wstring[:-1])
        self._fobject.write(wstring)
        self._fobject.flush()

    def close(self):
        self._fobject.close()

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--trustfile', nargs='?', default='graph/trust_t_10.pkl',
                        help='Input graph path')

    parser.add_argument('--ratingfile', nargs='?', default='graph/rating_t_10.pkl',
                        help='Input graph path')

    parser.add_argument('--logfile', nargs='?', default='outputlog/run1.log',
                        help='Output log path')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--p', type=float, default=1.0,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1.0,
                        help='Inout hyperparameter. Default is 1.')

    return parser.parse_args()

def read_graph(trust_graph):
    '''
    Reads the input network in networkx.
    '''
    lines = ['{} {}'.format(z[0],z[1]) for z in trust_graph.tolist()]
    G = nx.parse_edgelist(lines, nodetype = int, create_using=nx.DiGraph())
    counter = 0
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
        counter += 1

    print(counter)

    return G

def learn_embeddings(args, g_indices, g_data, walks, np_pairs, train_arr, valid_arr, test_arr, trust_set, H, n_nodes, m_margin, m_alpha, logger):
    '''
    Learn embeddings by optimizing the different objectives using SGD.
    '''

    trust_model = trustmodel.BasicModel(
        n_nodes,
        args.dimension,
        alpha=m_alpha,
        margin=m_margin)

    curr_iter, avg_acc = trust_model.train(g_indices, g_data, walks, np_pairs, logger, H, valid_arr, test_arr, trust_set)

    logger.write("##########################################\n")
    logger.write("Margin {}\n".format(m_margin))
    logger.write("a1 {}, a2 {}, a3 {} a4 {}\n".format(
        m_alpha[0], m_alpha[1], m_alpha[2], m_alpha[3]))
    logger.write("Average accuracy {}\n".format(avg_acc))
    logger.write("Number of iterations {}\n".format(curr_iter))
    logger.write("##########################################\n")

    return

def main(args, logger, m_margin=1.0, m_alpha=[0.1,0.1,0.1], trainp=0.7, validp=0.0, testp=0.3):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    print("Getting data...")
    train_arr, valid_arr, test_arr, rating, trust_set, n_nodes = data_helper.data_handler(
        args.trustfile, args.ratingfile, trainp, validp, testp)
    print("Constructing graph...")
    nx_G = read_graph(train_arr)

    pairs = []
    for edges in nx_G.edges():
        pairs.append([edges[0], edges[1]])
    np_pairs = np.array(pairs)

    if os.path.isfile("graph0.7.pkl"):
        print("Loading sparse graph ...")
        g_indices, g_data = pickle.load(open("graph0.7.pkl","rb"))
    else:
        print("Calculating sparse graph ...")
        g_indices = np.zeros(n_nodes+1, dtype=np.int32)
        g_data = np.zeros((nx_G.size()), dtype=np.int32)

        e_counter = 0
        for n in range(len(nx_G)):
            try:
                g_indices[n] = e_counter
                flag = False
                for e in nx_G.successors(n):
                    flag = True
                    g_data[e_counter] = e
                    e_counter += 1
            except nx.exception.NetworkXError as e:
                pass
        #g_indices[len(nx_G)] = e_counter

        for n in range(len(nx_G), n_nodes+1):
            g_indices[n] = e_counter

        pickle.dump((g_indices, g_data), open("graph0.7.pkl","wb"))

    G = sampleutil.Graph(nx_G, args.p, args.q)
    G.preprocess_transition_probs()
    print("Generating random walks...")
    if os.path.isfile('walks7.pkl') and os.path.isfile('H.pkl'):
        walks = pickle.load(open('walks7.pkl','rb'))
        H = pickle.load(open('H7.pkl','rb'))
    else:
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        pickle.dump(walks, open('walks7.pkl','wb'))
        # get unique pairs for the purpose of calculating coefficients
        unique_pairs = set()
        for walk in walks:
            start_node = walk[0]
            for nd in walk[1:]:
                unique_pairs.add((start_node, nd))
        pairs = np.array([[x[0],x[1]] for x in unique_pairs])
        print(pairs.shape)
        print("Calculating homophily coefficients...")
        H = simops.user_pair_sim(pairs, rating)
        pickle.dump(H, open('H7.pkl','wb'))
    learn_embeddings(
        args, g_indices, g_data, walks, np_pairs, train_arr, valid_arr, test_arr, trust_set,
        H, n_nodes, m_margin, m_alpha, logger)

    """

    print("Calculating homophily coefficients...")
    H = simops.user_pair_sim(np_pairs, rating)
    del np_pairs
    pickle.dump(H, open('H_pairs.pkl','wb'))

    learn_embeddings(
        args, pairs, train_arr, test_arr, trust_set,
        H, n_nodes, m_margin, m_alpha, logger)

    """


if __name__ == "__main__":
    args = parse_args()
    loggingfile = args.logfile

    """
    logger = open('outputlog/exp_inf.log',"w")
    logger = Logger(logger)
    logger.write("impact of third term\n")
    margin_list = [1.0]
    alpha_list = [
        [0.1, 0.1, 0.1],
        [0.1, 1.0, 0.1],
    ]

    for m_margin in margin_list:
        for m_alpha in alpha_list:
            main(args, logger, m_margin, m_alpha)
    """

    # Exp4 : impact of homophily term
    """
    logger = open('outputlog/6_experiment.log',"w")
    logger = Logger(logger)
    logger.write("impact of homophily term\n")
    margin_list = [1.0]
    alpha_list = [
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 0.1],
        [10.0, 10.0, 0.1],
        [10.0, 1.0, 0.1],
        [20.0, 1.0, 0.1],
        [1.0, 10.0, 0.1],
        [1.0, 20.0, 0.1],
    ]

    for m_margin in margin_list:
        for m_alpha1 in [0.1, 1.0, 10.0]:
            for m_alpha2 in [0.1, 1.0, 10.0]:
                main(args, logger, m_margin, [m_alpha1, m_alpha2, 0.1])
    """

    # Exp4 : impact of homophily term
    logger = open('outputlog/11_experiment.log',"w")
    logger = Logger(logger)
    logger.write("impact of homophily term\n")
    margin_list = [1.0]
    alpha_list = [
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.1],
        [1.0, 0.0, 0.1],
        [10.0, 0.0, 0.1],
        [20.0, 0.0, 0.1],
        [0.1, 0.1, 0.1],
        [1.0, 0.1, 0.1],
        [10.0, 0.1, 0.1],
        [20.0, 0.1, 0.1],
        [0.0, 0.1, 0.1],
        [0.0, 1.0, 0.1],
        [0.0, 10.0, 0.1],
        [0.0, 20.0, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 10.0, 0.1],
        [0.1, 20.0, 0.1],
        [1.0, 1.0, 0.1],
        [10.0, 10.0, 0.1],
        [20.0, 20.0, 0.1]
    ]

    alpha_list2 = [
        [0.0, 0.0, 0.1, 0.1],
        [1.0, 1.0, 0.1, 1.0],
        [1.0, 1.0, 0.1, 10.0]
    ]

    for m_margin in margin_list:
        for m_alpha in alpha_list2:
                main(args, logger, m_margin, m_alpha)
