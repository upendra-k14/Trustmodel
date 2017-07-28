import tensorflow as tf
import numpy as np
import random
import os.path
import pickle
import math

class BasicModel(object):

    def __init__(self, n_nodes, n_dimension, batch_size=1024,
        learning_rate=0.001, alpha=[0.1, 0.1, 0.1, 0.1], margin=1.0):

        # Add constants and class variables
        self.n_nodes = n_nodes
        self.n_dimension = n_dimension
        self.batch_size = batch_size
        self.margin = tf.constant(margin, dtype=tf.float32)
        self.alpha1 = tf.constant(alpha[0], dtype=tf.float32)
        self.alpha2 = tf.constant(alpha[1], dtype=tf.float32)
        self.alpha3 = tf.constant(alpha[2], dtype=tf.float32)
        self.alpha4 = tf.constant(alpha[3], dtype=tf.float32)

        # Add placeholders
        self.add_placeholders()

        # Add embeddings
        self.add_embeddings()
        self.prepare_data()

        #self.evidence = self.get_evidence()

        # Add g(u,v) model
        self.output = self.hadamard_operator()

        # Calculate loss
        self.ranking_margin_objective()

        self.optimize = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        self.optimize2 = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss2)

    def add_embeddings(self):
        """
        Add embeddings
        """

        # Use CPU and system ram for these computations and storing embeddings
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.U = tf.Variable(
                tf.random_uniform([self.n_nodes, self.n_dimension], maxval=0.9))
            self.V = tf.Variable(
                tf.random_uniform([self.n_nodes, self.n_dimension], maxval=0.9))
            self.W = tf.Variable(
                tf.random_uniform([self.n_dimension,1], maxval=0.9))

    def prepare_data(self):
        # Pair nodes and lookup embeddings
        # Get pairs of source and target nodes by changing shape :
        # shape(self.inputX) is [4*batch_size]
        # shape(paired_nodes) is [2*batch_size,2]
        paired_nodes = tf.reshape(self.inputX, [-1,2])

        # split the paired nodes to get source and target nodes
        self.s_nodes, self.t_nodes = tf.split(paired_nodes, 2,1)
        # self.i_shape = tf.shape(s_nodes)
        # get unique nodes for l2 regularization
        unique_nodes, idx = tf.unique(self.inputX)

        # lookup embeddings for source nodes
        self.src_emb = tf.squeeze(tf.nn.embedding_lookup(self.U, self.s_nodes),[1])
        # lookup embeddings for target nodes
        self.target_emb = tf.squeeze(tf.nn.embedding_lookup(self.V, self.t_nodes),[1])

        # lookup embeddings for unique nodes
        unique_U = tf.nn.embedding_lookup(self.U, unique_nodes)
        unique_V = tf.nn.embedding_lookup(self.V, unique_nodes)

        # Calculate l2_loss for these embeddings U and V for this batch
        self.phi3 = tf.multiply(
            self.alpha3,
            tf.nn.l2_loss(unique_U) + tf.nn.l2_loss(unique_V))


    def get_evidence(self, src_pos_nodes, ter_pos_nodes):
        """
        Get evidence
        """

        src_pos_nodes = tf.squeeze(src_pos_nodes)
        ter_pos_nodes = tf.squeeze(ter_pos_nodes)

        g_indices = tf.expand_dims(self.g_indices,1)

        st_src_indices = tf.unstack(tf.squeeze(
            tf.nn.embedding_lookup(g_indices, src_pos_nodes, [1])), self.batch_size)

        end_src_indices = tf.unstack(tf.squeeze(
            tf.nn.embedding_lookup(
                g_indices,
                tf.add(src_pos_nodes,tf.constant(1,dtype=tf.int32)), [1])), self.batch_size)

        st_ter_indices = tf.unstack(tf.squeeze(
            tf.nn.embedding_lookup(g_indices, ter_pos_nodes), [1]), self.batch_size)
        end_ter_indices = tf.unstack(tf.squeeze(
            tf.nn.embedding_lookup(
                g_indices,
                tf.add(ter_pos_nodes,tf.constant(1,dtype=tf.int32))), [1]), self.batch_size)
        unstacked_slist = tf.unstack(src_pos_nodes, self.batch_size)
        unstacked_tlist = tf.unstack(ter_pos_nodes, self.batch_size)

        evidence = []
        olist = []

        for src_stindex, src_endindex, ter_stindex, ter_endindex, src_node, ter_node in zip(
            st_src_indices, end_src_indices, st_ter_indices, end_ter_indices, unstacked_slist, unstacked_tlist):

            src_neighbours = tf.slice(self.g_data, [src_stindex],
                [src_endindex-src_stindex])
            src_neighbour_emb = tf.nn.embedding_lookup(self.U, src_neighbours)
            src_nbr_as_ter_emb = tf.nn.embedding_lookup(self.V, src_neighbours)

            src_emb = tf.nn.embedding_lookup(self.U, src_node)
            ter_emb = tf.nn.embedding_lookup(self.V, ter_node)

            # Model
            src_nbr_output = tf.reshape(
                tf.matmul(tf.multiply(src_emb, src_nbr_as_ter_emb), self.W), [-1])
            nbr_ter_output = tf.reshape(
                tf.matmul(tf.multiply(src_neighbour_emb, ter_emb), self.W), [-1])
            output = tf.reduce_mean(tf.multiply(src_nbr_output, nbr_ter_output))

            evidence.append(
                tf.where(tf.is_nan(output), tf.constant(0.0), output))

        evidence = tf.stack(evidence)

        return evidence

    def hadamard_operator(self):
        """
        Add model
        W.T(U*V)
        """

        output = tf.reshape(tf.matmul(tf.multiply(self.src_emb, self.target_emb), self.W),[-1])

        # output = self.leakyrelu(output)

        return output

    def leakyrelu(self, intensor):

        pos = tf.nn.relu(intensor)
        neg = tf.nn.relu(-intensor)
        out = pos - tf.multiply(0.1,neg)

        return out

    def nn_model(self):
        """
        Neural network for g(u,v)
        not implemented
        """

        pass

    def add_placeholders(self):
        """
        Add placeholders
        """

        # placeholder for input nodes
        self.inputX = tf.placeholder(tf.int32, shape=[None])
        # placeholder for item based homphily coefficients
        self.H = tf.placeholder(tf.float32, shape=[None])
        # placeholder for category based homophily coefficients
        self.I = tf.placeholder(tf.float32, shape=[None])
        # placeholder for g_indices
        self.g_indices = tf.placeholder(tf.int32, shape=[None])
        self.g_data = tf.placeholder(tf.int32, shape=[None])

    def ranking_margin_objective(self):
        """
        Pairwise ranking loss
        max(0,margin-l(pos_pairs)+l(neg_pairs)) + regularization terms
        """

        # self.output stores g(u,v) for both positive and negative samples
        # alternately. Even indexed elements in self.output are positive samples
        # and odd indexed elements are negative samples
        paired_output = tf.reshape(self.output, [-1,2])
        # splitting the tensor to get pos and neg samples
        pos_output, neg_output = tf.split(paired_output,2,1)

        src_pairs = tf.reshape(tf.squeeze(self.s_nodes),[-1,2])
        src_pos_nodes, src_neg_nodes = tf.split(src_pairs,2,1)
        ter_pairs = tf.reshape(tf.squeeze(self.t_nodes),[-1,2])
        ter_pos_nodes, ter_neg_nodes = tf.split(ter_pairs,2,1)

        src_pos_emb = tf.squeeze(
            tf.nn.embedding_lookup(self.U, src_pos_nodes), [1])
        ter_pos_emb = tf.squeeze(
            tf.nn.embedding_lookup(self.V, ter_pos_nodes), [1])

        # Removing dimensions of size 1 using tf.squeeze
        # converts [batch_size,1,1] to [batch_size]
        pos_output = tf.squeeze(pos_output)
        neg_output = tf.squeeze(neg_output)
        pairwise_losses = tf.maximum(
            0.0, self.margin - pos_output + neg_output)
        phi1 = tf.multiply(
            self.alpha1,
            tf.reduce_sum(tf.multiply(self.H,tf.norm(src_pos_emb - ter_pos_emb, axis=1))))
        phi2 = tf.multiply(
            self.alpha2,
            tf.reduce_sum(tf.multiply(self.I,tf.norm(src_pos_emb - ter_pos_emb, axis=1))))

        # Uses regularization
        self.loss = tf.reduce_sum(pairwise_losses) - phi1 + phi2 + self.phi3

        self.evidence = self.get_evidence(src_pos_nodes, ter_pos_nodes)
        phi4 = tf.multiply(
            self.alpha4,
            tf.reduce_sum(tf.multiply(self.evidence, tf.abs(pos_output-1))))
        self.loss2 = tf.reduce_sum(pairwise_losses) - phi1 + phi2 + self.phi3 + phi4

        # Without homophily and l2-loss
        # loss = tf.reduce_sum(pairwise_losses)

        # Without homophily
        # loss = tf.reduce_sum(pairwise_losses) + self.phi

    def train(self, g_indices, g_data, walks, np_pairs, logger, global_H, valid_pairs, test_pairs, orig_set, max_iter = 650):
        """
        Training and preparation of data
        """

        # convert random walks to node pairs
        pairs = []
        for walk in walks:
            start_node = walk[0]
            for nd in walk[1:]:
                pairs.append([start_node, nd])
        pairs_set = set(map(tuple, pairs))
        pairs = np.array(pairs)

        # pairs = np.array(walks)
        np.random.shuffle(pairs)
        # pairs_set = set(map(tuple,walks))
        pair_indices = list(range(pairs.shape[0]))

        # Generate negative set
        logger.write("Generating negative set...\n")
        indices = np.arange(self.n_nodes*self.n_nodes)
        np.random.shuffle(indices)
        k = pairs.shape[0]
        random_pairs = np.zeros((k,2))
        counter = 0
        for i in np.nditer(indices):
            a = i%self.n_nodes
            b = int(i/self.n_nodes)
            if (a,b) not in pairs_set:
                random_pairs[counter][0] = a
                random_pairs[counter][1] = b
                counter += 1
            if(counter>=k):
                break
        random_indices = list(range(k))

        # tensor for initialization of all variables U,V and W
        init = tf.global_variables_initializer()

        average_acc = 0.0
        iter_list = [250, 500, 750, 1000, 1250, 1500]
        iter_list = [250, 350, 450, 550, 650]
        prev_acc = 0.01
        curr_iter = 1

        with tf.Session() as sess:

            logger.write("Variables initialized\n")
            sess.run(init)

            losstensor = self.loss
            opttensor = self.optimize

            for i in range(1, max_iter+1):

                # get next batch
                input_nodes, H, I = self.get_next_batch(
                    pairs, pair_indices, random_pairs, random_indices, global_H, self.batch_size, i)

                # feed dict
                feed_dict = {
                    self.inputX: input_nodes,
                    self.H: H,
                    self.I: I,
                    self.g_indices: g_indices,
                    self.g_data: g_data
                }

                # optimize and calculate loss
                loss, _ = sess.run(
                    [losstensor, opttensor],
                    feed_dict=feed_dict)

                #if(i >= 250):
                #    losstensor = self.loss2
                #    opttensor = self.optimize2

                if(i%10==0):
                    logger.write("Loss: {}\n".format(loss))

                if(i in iter_list):
                    logger.write("-----------------------------------------\n")
                    logger.write("Iteration {}\n".format(i))
                    acc = self.get_test_accuracy(
                        orig_set, np.copy(test_pairs), sess, logger)
                    average_acc += acc
                    logger.write("Test Accuracy: {}\n".format(acc))
                    logger.write("-----------------------------------------\n")

                    logger.write("-----------------------------------------\n")
                    logger.write("Iteration {}\n".format(i))
                    acc = self.get_test_accuracy(
                        orig_set, np.copy(np_pairs), sess, logger)
                    logger.write("Train Accuracy: {}\n".format(acc))
                    logger.write("-----------------------------------------\n")

                curr_iter += 1

            # Save the model
            saver = tf.train.Saver()
            saver.save(sess, 'models/model_vars')

        return curr_iter, average_acc/len(iter_list)

    def get_next_batch(self, pairs, pair_indices, random_pairs, random_indices, global_H, batch_size, i):
        """
        Get batch data and other parameters for the batch
        pairs : positive pairs from random walk
        random_pairs : negative pairs randomly generated
        global_H : stores item based and category based homophily coefficients
        batch_size : size of batch
        """

        # sequentially calculate batches
        max_size = pairs.shape[0]
        st_index = ((i-1)*batch_size)
        end_index = st_index+batch_size
        indices = list(range(st_index, end_index))
        n_pairs = end_index - st_index
        positive_samples = pairs.take(indices, mode='wrap', axis=0)

        # negative random samples
        random_samples = random_pairs[random.sample(random_indices, n_pairs)]

        # initialize np arrays
        H = np.zeros((batch_size,))
        I = np.zeros((batch_size,))
        input_nodes = np.zeros((4*batch_size,))

        # To feed negative sample for every positive sample
        # We need to give nodes as input to self.inputX variable
        # To get pair of pos and neg sample from this nodes' list,
        # a small trick is used :
        # Consider 4 consecutive nodes :
        #   - a
        #   - a+1
        #   - a+2
        #   - a+3
        # (a,a+1) is positive sample
        # (a+2,a+3) is negative sample
        # Next section of code generates this type of list
        # with size of list as 4*batch_size
        # It may not be best approach to feed lists, but this is what I could
        # think of at some time

        counter = 0
        for counter in range(batch_size):
            u, v = positive_samples[counter]
            a, b = global_H[u][v]
            H[counter] = a
            I[counter] = b
            input_nodes[4*counter] = u
            input_nodes[4*counter+1] = v
            input_nodes[4*counter+2] = random_samples[counter][0]
            input_nodes[4*counter+3] = random_samples[counter][1]

        return input_nodes, H, I

    def get_test_accuracy(self, original_pairs, test_pairs, sess, logger, factor=6.0):
        """
        Calculate test Accuracy
        original_pairs : set of pairs of nodes in A
        test_pairs : numpy array of nodes or N
        sess : tensorflow session
        """

        test_nodes = test_pairs.flatten()
        test_size = test_pairs.shape[0]
        #print(test_size)
        test_set = set([(a,b) for a,b in test_pairs.tolist()])
        feed_dict = {self.inputX: test_nodes}
        # Test pairs output
        # logger.write("Getting test pairs output\n")
        test_output = sess.run(
            [self.output], feed_dict=feed_dict)
        test_output = np.array(test_output[0])
        #print(test_pairs.shape)
        #print(test_output.shape)
        final_output = np.concatenate(
            (test_pairs,np.expand_dims(test_output,axis=1)), axis=1)

        # Clear memory
        del feed_dict
        del test_nodes
        del test_output

        random_pairs = []
        randomlist = list(range(1,11))
        random.shuffle(randomlist)
        randomi = randomlist[0]

        if os.path.isfile("random_pairs_{}.pkl".format(randomi)):
            random_pairs = pickle.load(
                open("random_pairs_{}.pkl".format(randomi),"rb"))

        else:
            k = int(factor*len(test_pairs))
            indices = np.arange(self.n_nodes*self.n_nodes)
            np.random.shuffle(indices)
            random_pairs = np.zeros((k,2))
            counter = 0

            for i in np.nditer(indices):
                a = i%self.n_nodes
                b = int(i/self.n_nodes)
                if (a,b) not in original_pairs:
                    random_pairs[counter][0] = a
                    random_pairs[counter][1] = b
                    counter += 1
                if(counter>=k):
                    break

            pickle.dump(random_pairs,
                open("random_pairs_{}.pkl".format(randomi),"wb"))

        del test_pairs

        random_nodes = random_pairs.flatten()
        feed_dict = {self.inputX: random_nodes}
        # logger.write("Getting random pairs output\n")
        random_output = sess.run(
            [self.output], feed_dict=feed_dict)
        random_output = np.array(random_output[0])
        temp_output = np.concatenate(
            (random_pairs, np.expand_dims(random_output,axis=1)), axis=1)

        # Clear memory
        del feed_dict
        del random_nodes
        del random_pairs
        del random_output

        final_output = np.concatenate((final_output, temp_output), axis=0)
        # Clear memory
        del temp_output
        dt = [('a',np.float64),('b',np.float64),('c',np.float64)]
        final_output.view(dtype=dt).sort(order=['c'],axis=0)
        final_output = final_output.view(dtype=np.float64)
        output_size = final_output.shape[0]
        right_count = 0
        for i in range(output_size-test_size, output_size):
            if (final_output[i][0],final_output[i][1]) in test_set:
                right_count += 1

        accuracy = float(right_count)/test_size

        return accuracy

if __name__ == "__main__":

    t = BasicModel(8000,128)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    #sess.run(init)
    #print(sess.run([t.i_shape], feed_dict={t.inputX:[1,2,3,4]}))
