#Building Gpt2 Model
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras import Model

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.attn_head_size = embed_dim // num_heads

        self.wq = Dense(embed_dim)
        self.wk = Dense(embed_dim)
        self.wv = Dense(embed_dim)

        self.dense = Dense(embed_dim)

    def split_heads(self, x, batch_size):
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.attn_head_size))
        
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        
        q = self.split_heads(self.wq(q), batch_size)  
        k = self.split_heads(self.wk(k), batch_size)  
        v = self.split_heads(self.wv(v), batch_size)  

        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        
        if mask is not None:
            
            scaled_attention_logits += (mask * -1e9)

        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        
        output = tf.matmul(attention_weights, v)

       
        output = tf.transpose(output, perm=[0, 2, 1, 3])

        
        concat_attention = tf.reshape(output, (batch_size, -1, self.embed_dim))

       
        return self.dense(concat_attention)



class FeedForwardNetwork(Layer): 
    def __init__(self, embed_dim, dff):
        super().__init__()
        self.dense1 = Dense(dff, activation='gelu')
        self.dense2 = Dense(embed_dim)

    def call(self, x):
        return self.dense2(self.dense1(x))


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, dff)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
 
        attn_output = self.att(x, x, x, mask)
     
        attn_output = self.dropout1(attn_output, training=training)
    
        out1 = self.norm1(x + attn_output)

    
        ffn_output = self.ffn(out1)
       
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.norm2(out1 + ffn_output)


class GPT2(Model):
    def __init__(self, vocab_size, max_length, embed_dim=768, num_heads=12, dff=3072, num_layers=12, dropout_rate=0.1):
        super().__init__()
        self.max_length = max_length 
        self.embed_dim = embed_dim   

        self.token_emb = Embedding(vocab_size, embed_dim, name="token_embedding")
       
        self.pos_emb = Embedding(max_length, embed_dim, name="positional_embedding")

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, dff, dropout_rate) for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate) # Add dropout after embeddings

        self.norm = LayerNormalization(epsilon=1e-6)
       
        self.out = Dense(vocab_size, name="output_logits")

    def create_causal_mask(self, seq_len):
     
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
       
        return mask[tf.newaxis, tf.newaxis, :, :]

    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]

        
        tf.debugging.assert_less_equal(seq_len, self.max_length,
                                        message=f"Input sequence length ({seq_len}) exceeds model's max length ({self.max_length})")

        
        mask = self.create_causal_mask(seq_len)

       
        token_embeddings = self.token_emb(x)
      
        token_embeddings *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        pos_indices = tf.range(seq_len)
        pos_embeddings = self.pos_emb(pos_indices)

        
        x = token_embeddings + pos_embeddings
        x = self.dropout(x, training=training)

        
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training, mask=mask)

        # Final normalization and output layer
        x = self.norm(x)
        return self.out(x) # shape: (batch_size, seq_len, vocab_size)


VOCAB_SIZE = 50257
MAX_LENGTH = 1024
EMBED_DIM = 768 
NUM_HEADS = 12
DFF = 3072
NUM_LAYERS = 12 


inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids") 


gpt2_model = GPT2(
    vocab_size=VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    dff=DFF,
    num_layers=NUM_LAYERS 
)

outputs = gpt2_model(inputs)


model = tf.keras.Model(inputs=inputs, outputs=outputs, name="GPT2_Model")

model.summary()


print("\nTesting with dummy input...")

dummy_input = tf.random.uniform((2, 50), maxval=VOCAB_SIZE, dtype=tf.int32)
output_logits = model(dummy_input)
print("Input shape:", dummy_input.shape)
print("Output shape:", output_logits.shape)


