import traceback

import tensorflow as tf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def loadFile():
    data = []
    target = []
    
    with open("SentimentMovieReview.tsv", 'r', encoding="utf-8") as f:
        contents = f.readlines()


    for content in contents:
        cont_arr = content.split("\t")
        if(cont_arr[3].strip() == '0'):#negative
            data.append(cont_arr[2])
            target.append([0,0,0,0,1])
        if(cont_arr[3].strip() == '1'):#somewhat negative
            data.append(cont_arr[2])
            target.append([0,0,0,1,0])
        if(cont_arr[3].strip() == '2'):#neutral
            data.append(cont_arr[2])
            target.append([0,0,1,0,0])
        if(cont_arr[3].strip() == '3'):#somewhat positive
            data.append(cont_arr[2])
            target.append([0,1,0,0,0])
        if(cont_arr[3].strip() == '4'):#positive
            data.append(cont_arr[2])
            target.append([1,0,0,0,0])
    

    return data, target

def preprocess(data):
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_data = tfidf_vectorizer.fit_transform(data)
    
    return tfidf_data.toarray()


try:
    data,target = loadFile()
    tf_idf = preprocess(data)
    
    train_x,test_x,train_y,test_y = train_test_split(tf_idf, target, test_size=0.4, random_state = 43)
    
    n_classes = 5
    batch_size = 100
    
    x = tf.placeholder('float', [None, len(train_x[0])])
    y_ = tf.placeholder('float')
    pkeep = tf.placeholder(tf.float32)
    
    W1 = tf.Variable(tf.truncated_normal([len(train_x[0]),300],stddev=0.1))
    b1 = tf.Variable(tf.zeros([300]))
    W2 = tf.Variable(tf.truncated_normal([300,100],stddev=0.1))
    b2 = tf.Variable(tf.zeros([100]))
    W3 = tf.Variable(tf.truncated_normal([100,n_classes],stddev=0.1))
    b3 = tf.Variable(tf.zeros([n_classes]))
    
    
    #model 3-layer neural network
    xx = x
    Y1 = tf.nn.sigmoid(tf.matmul(xx,W1) + b1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1d,W2) + b2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Ylogits = tf.matmul(Y2d,W3) + b3
    y = tf.nn.softmax(Ylogits)
    
    
    #loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= Ylogits, labels= y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    
    #optimizer
    optimizer = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
    
    #accuracy
    is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    #Session parameter
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    hm_epochs = 100
    
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i+batch_size
            
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _,c = sess.run([optimizer,cross_entropy], feed_dict={x: batch_x, y_: batch_y, pkeep:1})
            epoch_loss += c
            i += batch_size
            
        print('Epoch', epoch, 'completed out', hm_epochs, 'loss', epoch_loss)
    
    #If we pass the entire test set in single batch, then there is a high chance to get MemoryError
    #to solve this we will use batches in test data also
    cumalitive_acc = 0.0
    for epoch in range(hm_epochs):
        i = 0
        while i < len(test_x):
            start = i
            end = i+batch_size
            
            batch_x = np.array(test_x[start:end])
            batch_y = np.array(test_y[start:end])
            
            feed_dict_test = {x: batch_x, y_:batch_y, pkeep:1}
            cumalitive_acc += sess.run(accuracy, feed_dict=feed_dict_test)
            i += batch_size
        print('Epoch', epoch)
    
    print("test accuracy {}".format(cumalitive_acc / batch_size))
    
    print("============DONE============")
except:
    traceback.print_exc()