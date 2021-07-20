import numpy as np
import tensorflow as tf
import re
sentences = [ "我爱你", "余登武", "范冰冰"]


def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars =[w for w in chars if len(w.strip()) > 0]
    return chars

chars=np.array([seg_char(i)for i in sentences])
chars=chars.reshape(1,-1)
word_list=np.squeeze(chars)
##word_list['我' '爱' '你' '余' '登' '武' '范' '冰' '冰']
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
#word_dict{'余': 0, '武': 1, '你': 2, '范': 3, '登': 4, '我': 5, '冰': 6, '爱': 7}
number_dict = {i: w for i, w in enumerate(word_list)}
#{0: '登', 1: '武', 2: '冰', 3: '我', 4: '余', 5: '范', 6: '你', 7: '爱'}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # number of steps ['我 爱', '范 冰', '余 登']
n_hidden = 2 # number of hidden units


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = seg_char(sen)#分字
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

input_batch, target_batch=make_batch(sentences)


# Model
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction =tf.argmax(model, 1)


# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
input = [seg_char(sen)[:2] for sen in sentences]
print([seg_char(sen)[:2] for sen in sentences], '预测得到->', [number_dict[n] for n in predict[0]])