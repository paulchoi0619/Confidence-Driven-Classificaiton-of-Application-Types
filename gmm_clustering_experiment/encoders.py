from tensorflow.keras import layers, models
import tensorflow as tf

def lstm_encoder(input_shape, projection_dim=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)
    h = layers.Dense(projection_dim, activation='gelu', name='representation')(x)
    return models.Model(inputs, h, name='lstm_encoder')


def fsnet_encoder(
    input_steps=256,
    vocab_size=1501,
    embedding_dim=128,
    encoder_n_neurons=128,
    decoder_n_neurons=128,
    n_neurons=128,
    n_outputs=10,
    alpha=1.0,
    use_embedding=True
):
    # 1) Input & (optional) embedding
    inputs = layers.Input(shape=(input_steps,), dtype='int32', name='X')
    x = inputs
    if use_embedding:
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embedding'
        )(x)
    else:
        x = layers.Reshape((input_steps, 1))(tf.cast(inputs, tf.float32))

    # helper: one 2‑layer Bi‑GRU block
    def bi_gru_stack(units, return_sequences, name):
        cells = [layers.GRUCell(units), layers.GRUCell(units)]
        rnn  = layers.RNN(cells, return_sequences=return_sequences)
        return layers.Bidirectional(rnn, merge_mode='concat', name=name)

    x = bi_gru_stack(encoder_n_neurons, True,  name='encoder_gru_1')(x)
    encoder_feats = bi_gru_stack(encoder_n_neurons, False, name='encoder_gru_2')(x)

    dec_input      = layers.RepeatVector(input_steps, name='repeat_vector')(encoder_feats)
    decoder_output = bi_gru_stack(decoder_n_neurons, True,  name='decoder_gru')(dec_input)

    decoder_reconstruction = layers.TimeDistributed(
        layers.Dense(vocab_size, activation=None),
        name='decoder_reconstruction'
    )(decoder_output)
    fw_last = layers.Lambda(lambda t: t[:, -1, :decoder_n_neurons], name='dec_fw_last')(decoder_output)
    bw_last = layers.Lambda(lambda t: t[:,  0, decoder_n_neurons:], name='dec_bw_last')(decoder_output)
    decoder_state = layers.Concatenate(name='decoder_state')([fw_last, bw_last])
    
    prod         = layers.Multiply(name='elem_prod')([encoder_feats, decoder_state])
    diff = layers.Subtract(name='diff')([encoder_feats, decoder_state])
    abs_diff = layers.Lambda(tf.abs, name='abs_diff')(diff)   # Keras can infer shape now

    joint_emb = layers.Concatenate(name='joint_embedding')(
        [encoder_feats, decoder_state, prod, abs_diff]
    )

    dense1 = layers.Dense(
        n_neurons, activation='selu',
        kernel_regularizer=tf.keras.regularizers.l2(0.003),
        name='dense1'
    )(joint_emb)

    rep = layers.Dense(64, activation='gelu', name='representation')(dense1)
    return models.Model(
        inputs=inputs,
        outputs=[rep, decoder_reconstruction],
        name='fs_net'
    )

