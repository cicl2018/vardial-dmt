#!/usr/bin/env python
# coding: utf-8

# In[1]:


g = open("DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/train.txt", "r")
texts = list(map(lambda x:x[:-1].lower(),g.readlines()))
g.close()

g = open('DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/test.txt', "r")
texts_test = list(map(lambda x:x[:-1].lower(),g.readlines()))
g.close()



# In[2]: get types
types = list()
tokens= list()
types_test = list()
train_count=0
test_count=0

texts_temp = list()
texts_test_temp = list()
for t in texts:
    #print(t)
    #if t == 't' or 'm':
     #   types.remove(t)
    #unit = t.decode('utf-8')
    #print(unit, len(unit))
    types.append(t[-1])
    texts_temp.append(t[:-1])
    #tokens = t.decode("utf-8").split(" ")
    #print(tokens)
    train_count+=1
texts = texts_temp
print("training data lines:"+ str(train_count))

for x in texts_test:
    #test_unit = t.decode('utf-8')
    types_test.append(x[-1])
    test_count+=1
    texts_test_temp.append(x[:-1])
texts_test = texts_test_temp
print("test data lines:"+ str(test_count))




print("types \t: texts")
#print(types[50])
#print(texts[50])
print(str(types[0]) + "\t: " + texts[0])
print(str(types[50]) + "\t: " + texts[50])
print(str(types[55]) + "\t: " + texts[55])
print(str(types[100]) + "\t: " + texts[1500])
print("types_test \t: texts_test")
print(str(types_test[0])+ "\t: " + texts_test[0])
print(str(types_test[1])+ "\t: " + texts_test[1])




# In[3]:


from collections import Counter
import numpy as np

t_counts = Counter()
m_counts = Counter()
total_counts = Counter()

for i in range(len(texts)):
    if(types[i] == 't'):
        for word in texts[i].split(" "):
            t_counts[word] += 1
            total_counts[word] += 1

    else:
        for word in texts[i].split(" "):
            m_counts[word] += 1
            total_counts[word] += 1



# In[4]:


t_counts.most_common()


# In[5]:

print(t_counts.most_common())


# In[6]:


t_m_ratios = Counter()
t_m_raw_ratios = Counter()
for word, cnt in list(total_counts.most_common()):
    if(cnt > 50):
        t_m_ratio = t_counts[word] / float(m_counts[word]+1)
        t_m_ratios[word] = t_m_ratio
        
t_m_raw_ratios = t_m_ratios


# In[7]:

'''
print("MUN / QUO raw ratio for 'universe' = {}".format(t_m_ratios["universe"]))
print("MUN / QUO raw ratio for 'earth' = {}".format(t_m_ratios["earth"]))
print("MUN / QUO raw ratio for 'possible' = {}".format(t_m_ratios["possible"]))
print("MUN / QUO raw ratio for 'believe' = {}".format(t_m_ratios["believe"]))
'''

# In[8]:


for word, ratio in t_m_ratios.most_common():
    t_m_ratios[word] = np.log(ratio + 0.01)


# In[9]:


print("t / m log ratio for '我' = {}".format(t_m_ratios["我"]))
print("t / m log ratio for '的' = {}".format(t_m_ratios["的"]))
print("t / m log ratio for '本钱' = {}".format(t_m_ratios["本钱"]))
print("t / m log ratio for '种' = {}".format(t_m_ratios["种"]))


# In[10]:


t_m_ratios.most_common()


# In[11]:


list(reversed(t_m_ratios.most_common()))


# In[12]:


vocabulary = set(total_counts.keys())
size = len(vocabulary)

layer_0 = np.zeros((1,size))

word2index = {}
for i, word in enumerate(vocabulary):
    word2index[word] = i



# In[13]:


word2index


# In[14]:


import time
import sys
import numpy as np

class TextClassificationNetwork:
    
    def __init__(self, texts,types,min_count = 10, polarity_cutoff = 0.1, hidden_nodes = 10, learning_rate = 0.1):

        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the texts and their associated types 
        self.pre_process_data(texts, types, polarity_cutoff, min_count)
        
        # Build the network with the number of hidden nodes and the learning rate
        # Make the same number of input nodes as the size of vocabulary
        self.init_network(len(self.text_vocab),hidden_nodes, 1, learning_rate)

    ## New for Project 6: added min_count and polarity_cutoff parameters
    def pre_process_data(self, texts, types, polarity_cutoff, min_count):
        
        t_counts = Counter()
        m_counts = Counter()
        total_counts = Counter()

        for i in range(len(texts)):
            if(types[i] == 't'):
                for word in texts[i].split(" "):
                    t_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in texts[i].split(" "):
                    m_counts[word] += 1
                    total_counts[word] += 1


        t_m_ratios = Counter()

        for word, cnt in list(total_counts.most_common()):
            if (cnt > 50):
                t_m_ratio = t_counts[word] / float(m_counts[word] + 1)
                t_m_ratios[word] = t_m_ratio


        for word, ratio in t_m_ratios.most_common():
            if(ratio > 1):
                t_m_ratios[word] = np.log(ratio)
            else:
                t_m_ratios[word] = -np.log((1 / (ratio + 0.01)))
                
        # populate text_vocab
        text_vocab = set()
        for text in texts:
            for word in text.split(" "):
                if(total_counts[word] > min_count):
                    if(word in t_m_ratios.keys()):
                        if((t_m_ratios[word] >= polarity_cutoff) or (t_m_ratios[word] <= -polarity_cutoff)):
                            text_vocab.add(word)
                    else:
                        text_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.text_vocab = list(text_vocab)
        
        # populate type_vocab
        type_vocab = set()
        for type in types:
            type_vocab.add(type)
        
        # Convert the type vocabulary set to a list so we can access types via indices
        self.type_vocab = list(type_vocab)
        
        # Store the sizes of the texts and types vocabularies
        self.text_vocab_size = len(self.text_vocab)
        self.type_vocab_size = len(self.type_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.text_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of types mapped to index positions
        self.type2index = {}
        for i, type in enumerate(self.type_vocab):
            self.type2index[type] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights between the input layer and the hidden layer.

        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5,
                                                (self.hidden_nodes, self.output_nodes))
        
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    def get_target_for_type(self,type):
        if(type == 't'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_texts_raw, training_types):

        training_texts = list()
        for text in training_texts_raw:
            indices = set()
            for word in text.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_texts.append(list(indices))

        assert(len(training_texts) == len(training_types))
        
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()
        
        # loop through all the given texts and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_texts)):
            
            # Get the next text and its correct type
            text = training_texts[i]
            type = training_types[i]
            
            self.layer_1 *= 0
            for index in text:
                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))            
            
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - self.get_target_for_type(type) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            # Only update the weights that were used in the forward pass
            for index in text:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and type == 't'):
                correct_so_far += 1

            elif(layer_2 < 0.5 and type == 'm'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            texts_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_texts)))[:4]                              + "% Speed(texts/sec):" + str(texts_per_second)[0:5]                              + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1)                              + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_texts, testing_types):

        correct = 0

        # Time how many predictions per second we make
        start = time.time()

        for i in range(len(testing_texts)):
            pred = self.run(testing_texts[i])
            if(pred == testing_types[i]):
                correct += 1
            #else:
                #print('\n wrong prediction:')
                #print(testing_texts[i] + '\n')

            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            texts_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_texts)))[:4]                              + "% Speed(texts/sec):" + str(texts_per_second)[0:5]                              + " #Correct:" + str(correct) + " #Tested:" + str(i+1)                              + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, text):

        self.layer_1 *= 0
        unique_indices = set()
        for word in text.split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])

        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
         
        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # Return NEGATIVE for other values
        if(layer_2[0] >= 0.5):
            return "t"
        else:
            return "m"



# In[23]:


mlp = TextClassificationNetwork(texts[:-3000],types[:-3000], min_count=0, polarity_cutoff=0.3, learning_rate=0.05)
mlp.train(texts ,types)

#import random
#random.shuffle(a_list,random.random)
#random.sample(a_list, number of samples)


# In[24]:
mlp.test(texts_test,types_test)

if __name__ == '__main__':
    print("\ndone")
