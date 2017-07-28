import argparse
import numpy as np
import networkx as nx
import sampleutil
import simops
import data_helper
import trustmodel_h as trustmodel
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

def learn_embeddings(args, train_arr, valid_arr, test_arr, trust_set, n_nodes, m_margin, m_alpha, logger):
    '''
    Learn embeddings by optimizing the different objectives using RMSProp.
    '''

    trust_model = trustmodel.BasicModel(
        n_nodes,
        args.dimension,
        alpha=m_alpha,
        margin=m_margin)

    curr_iter, avg_acc = trust_model.train(train_arr, logger, valid_arr, test_arr, trust_set)

    logger.write("##########################################\n")
    logger.write("Margin {}\n".format(m_margin))
    logger.write("a1 {}\n".format(m_alpha))
    logger.write("Average accuracy {}\n".format(avg_acc))
    logger.write("Number of iterations {}\n".format(curr_iter))
    logger.write("##########################################\n")

    return

def main(args, logger, m_margin=1.0, m_alpha=0.1, trainp=0.7, validp=0.1, testp=0.3):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    print("Getting data...")
    train_arr, valid_arr, test_arr, rating, trust_set, n_nodes = data_helper.data_handler(
        args.trustfile, args.ratingfile, trainp, validp, testp)

    print("Constructing graph...")
    nx_G = read_graph(train_arr)

    print("Extracting pairs")
    pairs = []
    for edges in nx_G.edges():
        pairs.append([edges[0], edges[1]])
    np_pairs = np.array(pairs)

    learn_embeddings(
        args, np_pairs, valid_arr, test_arr, trust_set,
        n_nodes, m_margin, m_alpha, logger)


if __name__ == "__main__":
    args = parse_args()
    loggingfile = args.logfile

    # Exp4 : impact of homophily term
    logger = open('outputlog/6_experiment.log',"w")
    logger = Logger(logger)
    logger.write("impact of homophily term\n")
    m_margin = 1.0
    m_alpha = 0.1
    main(args, logger, m_margin, 0.1)
    main(args, logger, m_margin, 1.0)
    main(args, logger, m_margin, 10.0)
    main(args, logger, m_margin, 30.0)
