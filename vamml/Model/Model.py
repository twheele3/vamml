import tensorflow as tf

def BuildModel(imageSize = (256,256), 
               metadata_size = 6,
               dropoutRate = 0.15, 
               firstLayerFilterCount = 8, 
               attn_layers=True, 
               depth = 5,
               # upconv_kernel = 3,
               # upconv_stride = 2,
              ):
    # input1 is the image of the extracted ROI of the printed gel.
    input1 = tf.keras.layers.Input((imageSize[0], imageSize[1], 1), name='image')
    # input 2 is concatenated print metadata, comprised of any relevant continuous or categorical variables.
    input2 = tf.keras.layers.Input((metadata_size,),name = 'metadata')
    # Expands concatenated metadata into a trained embedding.
    # Optional for model testing with and without. 
    if attn_layers:
        inputDense = tf.keras.layers.BatchNormalization(name = 'token_bn')(input2)
        inputDense = tf.keras.layers.Dense(2**7,
                                           activation='linear',
                                           name='dense0')(inputDense)
        # Adding dimensions for later Hadamard product broadcasting.
        inputDense = tf.expand_dims(tf.expand_dims(inputDense,axis=-2),axis=-2)
    
    currentLayer = input1
    contractingLayers = []
    for i in range(depth):
        if i > 0:
            currentLayer = tf.keras.layers.MaxPooling2D((2, 2), name = f'maxpool{i}')(currentLayer)

        size = firstLayerFilterCount * 2 ** i
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_1')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation('relu')(currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'dropout{i}')(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_2')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation('relu')(currentLayer)
        contractingLayers.append(currentLayer)

    if attn_layers:
        contractedPool = tf.keras.layers.MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      name = 'contractbottom')(currentLayer)
        contractedPool = tf.concat([contractedPool,inputDense],axis=-1, name='bottom_concat')
        contractedPool = tf.keras.layers.BatchNormalization()(contractedPool)
        contractedPool = tf.keras.layers.Dense(size,
                                               name = 'bottom_dense_1')(contractedPool)
        contractedPool = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'upconv_attn_dropout{i}')(contractedPool)
        contractedPool = tf.keras.layers.Dense(size,
                                               name = 'bottom_dense_2')(contractedPool)
        contractedPool = tf.keras.layers.Activation('sigmoid')(contractedPool)
        currentLayer = currentLayer * contractedPool
    
    for i in reversed(range(depth-1)):
        size = firstLayerFilterCount * 2 ** i

        currentLayer = tf.keras.layers.Conv2DTranspose(size, 3, strides=2,
                                                       padding='same',
                                                       name=f'upconv_transpose{i}')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Activation('relu')(currentLayer)
        currentLayer = tf.keras.layers.Concatenate(name=f'concat{i}')([currentLayer, contractingLayers[i]])

        if attn_layers:
            # Attention mechanism pools previous layers into dense layer to add to attention key.
            contractedPool = tf.keras.layers.MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                                      name = f'upconv_pool{i}')(currentLayer)
            contractedPool = tf.keras.layers.BatchNormalization()(contractedPool)
            contractedPool = tf.concat([contractedPool,inputDense],axis=-1,name=f'upconv_concat{i}')
            contractedPool = tf.keras.layers.Dense(size*2,
                                                   name = f'upconv_attn{i}_1')(contractedPool)
            contractedPool = tf.keras.layers.Dropout(dropoutRate,
                                           name=f'upconv_attn_dropout{i}')(contractedPool)        
            contractedPool = tf.keras.layers.Dense(size*2,
                                                   name = f'upconv_attn{i}_2')(contractedPool)
            contractedPool = tf.keras.layers.Activation('sigmoid')(contractedPool)
            currentLayer = currentLayer * contractedPool
        
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'upconv_2d{i}_1')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate,
                                               name=f'upconv_dropout{i}')(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'upconv_2d{i}_2')(currentLayer)
        currentLayer = tf.keras.layers.BatchNormalization()(currentLayer)

    # Last layer is sigmoid to produce scaled probability map.
    final = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name = 'output_shape')(currentLayer)

    model = tf.keras.Model(inputs=[input1,input2], outputs=[final])
    return model