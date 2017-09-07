import traceback

import tensorflow as tf
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def prepareFile():
    
    with open("SentimentMovieReview.tsv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        contents = f.readlines()

    prev_sentence_id = 1
    curr_sentence_id = 0
    score_frq = [0,0,0,0,0]#we can have four scores
    prev_sentence = ''
    curr_sentence = ''
    output = []
    for content in contents:
        cont_arr = content.split("\t")
        curr_sentence_id = cont_arr[1].strip()
        curr_sentence = cont_arr[2].strip()
        sentiment = int(cont_arr[3].strip())
        if curr_sentence_id == prev_sentence_id:
            score_frq[sentiment] += sentiment
            prev_sentence_id = curr_sentence_id
        else:
            output.append([prev_sentence,np.argmax(score_frq)])
            score_frq = [0,0,0,0,0]
            score_frq[sentiment] += sentiment
            prev_sentence_id = curr_sentence_id
            prev_sentence = curr_sentence


    with open('UpdatedMovieReview.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for value in output:
            if(value[0] != ''):
                writer.writerow([value[0], value[1]])

def loadFile():
    data = []
    target = []
    
    with open("UpdatedMovieReview.csv", 'r', encoding="utf-8") as f:
        contents = f.readlines()


    for content in contents:
        cont_arr = content.split(",")
        if(cont_arr[1].strip() == '0'):#negative
            data.append(cont_arr[0])
            target.append([0,0,0,0,1])
        if(cont_arr[1].strip() == '1'):#somewhat negative
            data.append(cont_arr[0])
            target.append([0,0,0,1,0])
        if(cont_arr[1].strip() == '2'):#neutral
            data.append(cont_arr[0])
            target.append([0,0,1,0,0])
        if(cont_arr[1].strip() == '3'):#somewhat positive
            data.append(cont_arr[0])
            target.append([0,1,0,0,0])
        if(cont_arr[1].strip() == '4'):#positive
            data.append(cont_arr[0])
            target.append([1,0,0,0,0])
    

    return data, target

def preprocess(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_data = tfidf_vectorizer.fit_transform(data)
    
    return tfidf_data.toarray()


try:
    prepareFile()
    data,target = loadFile()
    tf_idf = preprocess(data)
     
    train_x,test_x,train_y,test_y = train_test_split(tf_idf, target, test_size=0.4, random_state = 35)
     
    #print(train_x.shape)(93636, 15240)
    #print(test_x.shape)(62424, 15240)
     
    n_classes = 5
    batch_size = 100
    
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, len(train_x[0])], name="x-input")
        y_ = tf.placeholder(tf.float32, name="y-input")
        pkeep = tf.placeholder(tf.float32, name="pkeep")
      
    with tf.name_scope("weights"):
        W1 = tf.Variable(tf.truncated_normal([len(train_x[0]),300],stddev=0.1), name="W1")
        W2 = tf.Variable(tf.truncated_normal([300,100],stddev=0.1), name="W2")
        W3 = tf.Variable(tf.truncated_normal([100,n_classes],stddev=0.1), name="W3")
     
    with tf.name_scope("biases"):
        b1 = tf.Variable(tf.constant(0.1,shape = [300]), name="B1")
        b2 = tf.Variable(tf.constant(0.1,shape = [100]), name="B2")
        b3 = tf.Variable(tf.constant(0.1,shape = [n_classes]), name="B3")     
     
    #Add histogram to weights
    tf.summary.histogram("W1",W1)
    tf.summary.histogram("W2",W2)
    tf.summary.histogram("W3",W3) 
      
    #model 3-layer neural network
    with tf.name_scope("layer1"):
        xx = x
        Y1 = tf.nn.sigmoid(tf.matmul(xx,W1) + b1)
    with tf.name_scope("layer2"):
        Y1d = tf.nn.dropout(Y1, pkeep)
        Y2 = tf.nn.sigmoid(tf.matmul(Y1d,W2) + b2)
    with tf.name_scope("layer3"):
        Y2d = tf.nn.dropout(Y2, pkeep)
        Ylogits = tf.matmul(Y2d,W3) + b3
    with tf.name_scope("softmax"):
        y = tf.nn.softmax(Ylogits)
      
      
    #cost function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= Ylogits, labels= y_)
        cross_entropy = tf.reduce_mean(cross_entropy)*100
          
        #optimizer
        optimizer = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
    tf.summary.scalar("cost",cross_entropy)
      
    #accuracy
    with tf.name_scope("accuracy"):
        is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("accuracy",accuracy)
      
    
    #Session parameter
    sess = tf.InteractiveSession()

    #Create log writer
    train_writer = tf.summary.FileWriter("/home/local/ASUAD/jchakra1/workspace/PyMcLearning/logs/dnn_logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("/home/local/ASUAD/jchakra1/workspace/PyMcLearning/logs/dnn_logs/test")
      
    merged = tf.summary.merge_all()
   
    init = tf.global_variables_initializer()
    sess.run(init)
 
     
     
    hm_epochs = 10
    train_acc = 0.0
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i = 0
        total_batch = int(len(train_x) / batch_size)
        while i < len(train_x):
            start = i
            end = i+batch_size
              
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _,c,train_summary = sess.run([optimizer,cross_entropy,merged], feed_dict={x: batch_x, y_: batch_y, pkeep:1})
            epoch_loss += c
            train_writer.add_summary(train_summary, epoch * total_batch + i)
            train_writer.flush()
             
            i += batch_size
              
        print('Epoch', epoch, 'completed out', hm_epochs, 'loss', epoch_loss)
     
     
    #If we pass the entire test set in single batch, then there is a high chance to get MemoryError
    #to solve this we will use batches in test data also
    cumalitive_acc = 0.0
    test_acc = 0.0
    count = 0
    for epoch in range(hm_epochs):
        i = 0
        while i < len(test_x):
            start = i
            end = i+batch_size
              
            batch_x = np.array(test_x[start:end])
            batch_y = np.array(test_y[start:end])
              
            feed_dict_test = {x: batch_x, y_:batch_y, pkeep:1}
            test_summary, test_acc = sess.run([merged,accuracy], feed_dict=feed_dict_test)
            cumalitive_acc += test_acc
            train_writer.add_summary(test_summary, epoch * total_batch + i)
            train_writer.flush()
            count = count + 1
            i += batch_size
        print('Epoch', epoch)
      
    print("test accuracy {}".format(cumalitive_acc / (count)))
     
    print("============DONE============")
except:
    traceback.print_exc()