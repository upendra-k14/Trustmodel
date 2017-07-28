import tensorflow as tf
import numpy as np
import random
import os.path
import pickle
import math

class BasicModel(object):

    def __init__(self, n_nodes, n_dimension, batch_size=1280,
        learning_rate=0.001, alpha=0.1, margin=1.0):

        # Add constants and class variables
        self.n_nodes = n_nodes
        self.n_dimension = n_dimension
        self.batch_size = batch_size
        self.margin = tf.constant(margin, dtype=tf.float32)
        self.alpha3 = tf.constant(alpha, dtype=tf.float32)

        # Add placeholders
        self.add_placeholders()

        # Add embeddings
        self.add_embeddings()

        # Add g(u,v) model
        self.output = self.hadamard_operator()

        # Calculate loss
        self.loss = self.ranking_margin_objective()

        self.optimize = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

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

            # Pair nodes and lookup embeddings
            # Get pairs of source and target nodes by changing shape :
            # shape(self.inputX) is [4*batch_size]
            # shape(paired_nodes) is [2*batch_size,2]
            paired_nodes = tf.reshape(self.inputX, [-1,2])

            # split the paired nodes to get source and target nodes
            s_nodes, t_nodes = tf.split(paired_nodes, 2,1)
            # self.i_shape = tf.shape(s_nodes)
            # get unique nodes for l2 regularization
            unique_nodes, idx = tf.unique(self.inputX)

            # lookup embeddings for source nodes
            self.src_emb = tf.squeeze(tf.nn.embedding_lookup(self.U, s_nodes),[1])
            # lookup embeddings for target nodes
            self.target_emb = tf.squeeze(tf.nn.embedding_lookup(self.V, t_nodes),[1])

            # lookup embeddings for unique nodes
            unique_U = tf.nn.embedding_lookup(self.U, unique_nodes)
            unique_V = tf.nn.embedding_lookup(self.V, unique_nodes)

            # Calculate l2_loss for these embeddings U and V for this batch
            self.phi3 = tf.multiply(
                self.alpha3,
                tf.nn.l2_loss(unique_U) + tf.nn.l2_loss(unique_V))

    def hadamard_operator(self):
        """
        Add model
        W.T(U*V)
        """

        W = tf.Variable(
            tf.random_uniform([self.n_dimension,1], maxval=0.9))
        output = tf.reshape(tf.matmul(tf.multiply(self.src_emb, self.target_emb), W),[-1])

        # output = self.leakyrelu(output)

        return output

    def add_placeholders(self):
        """
        Add placeholders
        """

        # placeholder for input nodes
        self.inputX = tf.placeholder(tf.int32, shape=[None])

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
        # Removing dimensions of size 1 using tf.squeeze
        # converts [batch_size,1,1] to [batch_size]
        pos_output = tf.squeeze(pos_output)
        neg_output = tf.squeeze(neg_output)
        pairwise_losses = tf.maximum(
            0.0, self.margin - pos_output + neg_output)

        # Uses regularization
        loss = tf.reduce_sum(pairwise_losses) + self.phi3

        return loss

    def train(self, train_pairs, logger, valid_pairs, test_pairs, orig_set, max_iter = 1100):
        """
        Training and preparation of data
        """

        pairs = np.array(train_pairs)
        np.random.shuffle(pairs)
        pairs_set = set(map(tuple,train_pairs))
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
        iter_list = [900, 1000, 1100]
        curr_iter = 1
        with tf.Session() as sess:

            logger.write("Variables initialized\n")
            sess.run(init)

            for i in range(1, max_iter+1):

                # get next batch
                input_nodes = self.get_next_batch(
                    pairs, random_pairs, random_indices, self.batch_size, i)

                # feed dict
                feed_dict = {
                    self.inputX: input_nodes,
                }

                # optimize and calculate loss
                loss, _ = sess.run(
                    [self.loss, self.optimize],
                    feed_dict=feed_dict)

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

                curr_iter += 1

            # Save the model
            saver = tf.train.Saver()
            saver.save(sess, 'models/model_vars')

        return curr_iter, average_acc/len(iter_list)

    def get_next_batch(self, pairs, random_pairs, random_indices, batch_size, i):
        """
        Get batch data and other parameters for the batch
        pairs : positive pairs from random walk
        random_pairs : negative pairs randomly generated
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
        input_nodes = np.zeros((4*n_pairs,))

        # To feed negative sample for every positive sample
        # We need to give nodes as input to self.inputX variable
        # To get pair of pos and neg sample from this nodes' list,
        # a slightly inconvenient way is used :
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
        for counter in range(n_pairs):
            u, v = positive_samples[counter]
            input_nodes[4*counter] = u
            input_nodes[4*counter+1] = v
            input_nodes[4*counter+2] = random_samples[counter][0]
            input_nodes[4*counter+3] = random_samples[counter][1]

        return input_nodes

    def get_test_accuracy(self, original_pairs, test_pairs, sess, logger, factor=6.0):
        """
        Calculate test accuracy
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
