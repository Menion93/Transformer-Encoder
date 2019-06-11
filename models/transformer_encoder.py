import tensorflow as tf
import numpy as np

from ..layers import TransformerLayer
from ..layers import Dense
from ..data.iterator import get_embedded_iterator
from ..optimize import minimize

tf.enable_eager_execution()

class TransformerEncoder:
    
    def __init__(self, num_layers, heads, embedding_dim, fc_hidden_dim, num_classes, max_ts):
        self.num_layers = num_layers
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.dense_hidden = fc_hidden_dim
        self.num_classes = num_classes
        self.max_ts = max_ts
        self.build()
        
    def build(self):
        self.layers = []
        for _ in range(self.num_layers):
            self.layers += [TransformerLayer(self.heads, self.embedding_dim, self.dense_hidden)]
        
        self.dense_out = Dense(self.max_ts * self.embedding_dim, self.num_classes, dropout=1, bias=False)
        self.layers += [self.dense_out]

        self.ckp = tf.train.Checkpoint(**dict([(str(i),var) for i, var in enumerate(self.get_trainable_weights())]))
    
    #@tf.contrib.eager.defun
    def forward(self, input_):
        o_step_i = input_
        for layer in self.layers[:-1]:
            o_step_i = layer.forward(o_step_i,o_step_i,o_step_i)

        o_last_ts = tf.reshape(o_step_i, (o_step_i.shape[0], self.max_ts * self.embedding_dim))
        return self.dense_out.forward(o_last_ts)
    
    def get_trainable_weights(self):
        return [weight for layer in self.layers for weight in layer.get_trainable_weights()]

    def get_pos_encoding(self, max_len, d_emb):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
            if pos != 0 else np.zeros(d_emb) 
                for pos in range(max_len)
                ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
        return pos_enc.astype(np.float32)
    
    def train(self,
              x_train, 
              y_train, 
              x_val, 
              y_val, 
              vocab,
              loss, 
              epochs,
              score_fun,
              log_every=100,
              tensorboard=False,
              log_dir="./transformer_log/",
              ckpt_dir="./transformer_ckpt/",
              pad_value=0,
              batch_size=32, 
              val_bs=32):
        
        train_generator = lambda x, y: get_embedded_iterator(x, y, self.num_classes, batch_size, vocab, self.max_ts)
        val_generator = lambda x, y: get_embedded_iterator(x, y, self.num_classes, val_bs, vocab, self.max_ts)
        
        self.train_generator(train_generator, val_generator, x_train, y_train, x_val, y_val,
                             loss, epochs, score_fun, tensorboard=tensorboard, log_dir="./transformer_log/",
                             ckpt_dir="./transformer_ckpt/")

    def train_generator(self,
                        generator,
                        val_generator,
                        x_train, 
                        y_train, 
                        x_val, 
                        y_val, 
                        loss, 
                        epochs,
                        score_fun,
                        log_every=100,
                        tensorboard=False,
                        log_dir="./transformer_log/",
                        ckpt_dir="./transformer_ckpt/"):
        
        if tensorboard:
            summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
            summary_writer.set_as_default()
            global_step = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer()
            
        iteration = 0
        current_val_score = self.compute_score_generator(val_generator, x_val, y_val, score_fun)
        for epoch in range(epochs):

            for x, y in generator(x_train, y_train):
                x = x + self.get_pos_encoding(self.max_ts, self.embedding_dim)
                
                if tensorboard:
                    global_step.assign_add(1)

                optimizer.minimize(lambda: loss(self, x, y, logging=tensorboard, iteration=iteration, log_iterations=log_every)) 
            
            val_score = self.compute_score_generator(val_generator, x_val, y_val, score_fun)

            if tensorboard:
                log_scalar('val_score', val_score)

            print("Validation score is {0}".format(val_score))
            
            if val_score > current_val_score:
                self.save_model(ckpt=ckpt_dir)
                current_val_score = val_score
                

    def compute_score_generator(self, generator, x_val, y_true, score_fun):
        scores = []
        
        for x, y in generator(x_val, y_true):
            x = x + self.get_pos_encoding(self.max_ts, self.embedding_dim)
            scores.append(score_fun(self.forward(x), y))
        return np.mean(scores)
    
    def save_model(self, ckpt="./transformer_log/"):
        self.ckp.save(ckpt)

    def restore_model(self, ckpt="./transformer_ckpt/"):
        self.ckp.restore(tf.train.latest_checkpoint(ckpt))