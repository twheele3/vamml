from tensorflow.keras.layers import Activation,BatchNormalization,Concatenate,Conv2D,Conv2DTranspose,Dense,Dropout,Input,MaxPooling2D,Reshape
from tensorflow.keras import Model

def BuildModel(imageSize = (256,256), 
               metadataSize = 6,
               dropoutRate = 0.15, 
               firstLayerFilterCount = 8, 
               attnLayers=True, 
               depth = 5,
              ):
    # input1 is the image of the extracted ROI of the printed gel.
    input1 = Input((imageSize[0], imageSize[1], 1), name='image')
    # input 2 is concatenated print metadata, comprised of any relevant continuous or categorical variables.
    input2 = Input((metadataSize,),name = 'metadata')
    # Expands concatenated metadata into a trained embedding.
    # Optional for model testing with and without. 
    if attnLayers:
        inputDense = BatchNormalization(name = 'token_bn')(input2)
        inputDense = Dense(2**7,
                                           activation='linear',
                                           name='dense0')(inputDense)
        # Adding dimensions for later Hadamard product broadcasting.
        inputDense = Reshape((1, 1, -1))(inputDense)
    
    currentLayer = input1
    contractingLayers = []
    for i in range(depth):
        if i > 0:
            currentLayer = MaxPooling2D((2, 2), name = f'maxpool{i}')(currentLayer)

        # Setting layer size as a geometric function of depth
        size = firstLayerFilterCount * 2 ** i
        currentLayer = Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_1')(currentLayer)
        currentLayer = BatchNormalization()(currentLayer)
        currentLayer = Activation('relu')(currentLayer)
        currentLayer = Dropout(dropoutRate,
                                               name=f'dropout{i}')(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='relu',
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              name=f'conv2d{i}_2')(currentLayer)
        currentLayer = BatchNormalization()(currentLayer)
        currentLayer = Activation('relu')(currentLayer)
        contractingLayers.append(currentLayer)

    # First attention mechanism calculated at lowest layer of U-net prior to transposing
    if attnLayers:
        contractedPool = MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                      strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                      name = 'contractbottom')(currentLayer)
        contractedPool = Concatenate(name='bottom_concat',axis=-1)([contractedPool,inputDense])
        contractedPool = BatchNormalization()(contractedPool)
        contractedPool = Dense(size, name = 'bottom_dense_1')(contractedPool)
        contractedPool = Dropout(dropoutRate, name=f'upconv_attn_dropout{i}')(contractedPool)
        contractedPool = Dense(size, name = 'bottom_dense_2')(contractedPool)
        contractedPool = Activation('sigmoid')(contractedPool)
        currentLayer = currentLayer * contractedPool
    
    for i in reversed(range(depth-1)):
        size = firstLayerFilterCount * 2 ** i
        currentLayer = Conv2DTranspose(size, 3, strides=2,
                                       padding='same', name=f'upconv_transpose{i}')(currentLayer)
        currentLayer = BatchNormalization()(currentLayer)
        currentLayer = Activation('relu')(currentLayer)
        currentLayer = Concatenate(name=f'concat{i}')([currentLayer, contractingLayers[i]])

        if attnLayers:
            # Attention mechanism pools previous layers into dense layer to add to attention key.
            contractedPool = MaxPooling2D(pool_size=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                          strides=(imageSize[0] // 2**i,imageSize[0] // 2**i),
                                          name = f'upconv_pool{i}')(currentLayer)
            contractedPool = BatchNormalization()(contractedPool)
            contractedPool = Concatenate(name=f'upconv_concat{i}',axis=-1)([contractedPool,inputDense])
            contractedPool = Dense(size*2, name = f'upconv_attn{i}_1')(contractedPool)
            contractedPool = Dropout(dropoutRate, name=f'upconv_attn_dropout{i}')(contractedPool)        
            contractedPool = Dense(size*2, name = f'upconv_attn{i}_2')(contractedPool)
            contractedPool = Activation('sigmoid')(contractedPool)
            currentLayer = currentLayer * contractedPool
        
        currentLayer = Conv2D(size, (3, 3), activation='relu', kernel_initializer='he_normal',
                              padding='same', name=f'upconv_2d{i}_1')(currentLayer)
        currentLayer = BatchNormalization()(currentLayer)
        currentLayer = Dropout(dropoutRate, name=f'upconv_dropout{i}')(currentLayer)
        currentLayer = Conv2D(size, (3, 3), activation='relu', kernel_initializer='he_normal',
                              padding='same', name=f'upconv_2d{i}_2')(currentLayer)
        currentLayer = BatchNormalization()(currentLayer)

    # Last layer is sigmoid to produce scaled probability map.
    final = Conv2D(1, (1, 1), activation='sigmoid', name = 'output_shape')(currentLayer)

    model = Model(inputs=[input1,input2], outputs=[final])
    return model