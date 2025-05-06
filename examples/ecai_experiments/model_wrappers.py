from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Input, Flatten, Dropout, Conv1D, BatchNormalization,
    LeakyReLU, GRU, Bidirectional, GlobalAveragePooling1D, Add, 
    Concatenate, MaxPooling1D, TimeDistributed, LayerNormalization,
    PReLU, SpatialDropout1D, MultiHeadAttention, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


class KerasWrapper:
    def __init__(self, model):
        self.model = model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer="adam",
            metrics=["mean_squared_error"],
            steps_per_execution=10,
        )

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 30)
        kwargs["batch_size"] = kwargs.get("batch_size", 256)

        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

class ModelMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "MLP"

class ModelLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(64, return_sequences=False),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(output_shape),
            ]
        )
        super().__init__(model)

    def __str__(self):
        return "LSTM"

class ModelCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(32, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu', padding='same'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)

    def __str__(self):
        return "CNN"

###

class SimpleMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential(
            [
                Input(shape=input_shape),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(output_shape),
            ]
        )

        super().__init__(model)

    def __str__(self):
        return "SimpleMLP"


class SimpleLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(64, return_sequences=False),
                Dense(32, activation="relu"),
                Dense(output_shape),
            ]
        )
        super().__init__(model)

    def __str__(self):
        return "SimpleLSTM"


class BasicMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "BasicMLP"

class RegularizedMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "RegularizedMLP"

class BasicCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(32, 3, activation='relu'),
            Flatten(),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "BasicCNN"

class AdvancedCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(64, 3, padding='same'),
            BatchNormalization(),
            LeakyReLU(0.1),
            Conv1D(32, 2),
            GlobalAveragePooling1D(),
            Dense(output_shape)
        ])
        self.model = model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer=Adam(learning_rate=0.001),
            metrics=["mean_squared_error"],
            steps_per_execution=10,
        )
    
    def __str__(self):
        return "AdvancedCNN"

class BasicGRU(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            GRU(32),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "BasicGRU"

class HybridCNNLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(32, 3, padding='same', activation='relu'),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "HybridCNNLSTM"

class DeepMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "DeepMLP"

class BidirectionalLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "BidirectionalLSTM"

class ResidualMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Flatten()(inputs)
        x1 = Dense(64, activation='relu')(x)
        x2 = Dense(64, activation='relu')(x1)
        x = Add()([x1, x2])
        x = Dense(32, activation='relu')(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        self.model = model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer=Adam(learning_rate=0.001),
            metrics=["mean_squared_error"],
            steps_per_execution=10,
        )
    
    def __str__(self):
        return "ResidualMLP"

class AdvancedHybrid(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(64, 3, padding='same'),
            BatchNormalization(),
            LeakyReLU(0.1),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            GlobalAveragePooling1D(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(output_shape)
        ])
        
        self.model = model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer=Adam(learning_rate=0.001),
            metrics=["mean_squared_error"],
            steps_per_execution=10,
        )
        
        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
    
    def fit(self, *args, **kwargs):
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)
    
    def __str__(self):
        return "AdvancedHybrid"

class AttentionLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        attention = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(attention)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "AttentionLSTM"

class DenseResNet(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, padding='same')(inputs)
        for _ in range(3):
            shortcut = x
            x = Conv1D(32, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Add()([x, shortcut])
        x = Flatten()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "DenseResNet"

class MultiScaleCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        conv1 = Conv1D(32, 1, padding='same', activation='relu')(inputs)
        conv3 = Conv1D(32, 3, padding='same', activation='relu')(inputs)
        conv5 = Conv1D(32, 5, padding='same', activation='relu')(inputs)
        concat = Concatenate()([conv1, conv3, conv5])
        x = GlobalAveragePooling1D()(concat)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "MultiScaleCNN"

class DeepResidualCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        
        for _ in range(4):
            shortcut = x
            x = Conv1D(64, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv1D(64, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = PReLU()(x)
        
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "DeepResidualCNN"

class TransformerEncoder(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        
        for _ in range(2):
            attention = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
            x = LayerNormalization()(attention + x)
            ff = Dense(128, activation="relu")(x)
            ff = Dense(input_shape[-1])(ff)
            x = LayerNormalization()(x + ff)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "TransformerEncoder"

class EnsembleNetwork(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        
        # CNN branch
        cnn = Conv1D(32, 3, padding='same')(inputs)
        cnn = MaxPooling1D()(cnn)
        cnn = Flatten()(cnn)
        
        # LSTM branch
        lstm = LSTM(32)(inputs)
        
        # MLP branch
        mlp = Flatten()(inputs)
        mlp = Dense(32, activation='relu')(mlp)
        
        # Combine all branches
        combined = Concatenate()([cnn, lstm, mlp])
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.2)(x)
        outputs = Dense(output_shape)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
    
    def __str__(self):
        return "EnsembleNetwork"


class DeepMLP_LRScheduler(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepMLP_LRScheduler"

class DeepMLP_Regularized(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer='l2'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer='l2'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer='l2'),
            Dense(output_shape)
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepMLP_Regularized"

class DeepMLP_ComplexV1(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)        

    def __str__(self):
        return "DeepMLP_ComplexV1"

class DeepMLP_ComplexV2(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepMLP_ComplexV2"

class SimpleLSTM_LRScheduler(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dense(32, activation="relu"),
            Dense(output_shape),
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "SimpleLSTM_LRScheduler"

class SimpleLSTM_Regularized(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=False, dropout=0.4, recurrent_dropout=0.2),
            Dense(32, activation="relu", kernel_regularizer='l2'),
            Dropout(0.3),
            Dense(output_shape),
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "SimpleLSTM_Regularized"

class SimpleLSTM_ComplexV1(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=False),
            Dense(64, activation="relu"),
            Dense(output_shape),
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "SimpleLSTM_ComplexV1"

class SimpleLSTM_ComplexV2(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
            Dense(output_shape),
        ])
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "SimpleLSTM_ComplexV2"

class DeepResidualCNN_LRScheduler(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        for _ in range(4):
            shortcut = x
            x = Conv1D(64, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv1D(64, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = PReLU()(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepResidualCNN_LRScheduler"

class DeepResidualCNN_Regularized(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, padding='same', kernel_regularizer='l2')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        for _ in range(4):
            shortcut = x
            x = Conv1D(64, 3, padding='same', kernel_regularizer='l2')(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv1D(64, 3, padding='same', kernel_regularizer='l2')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = PReLU()(x)
            x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepResidualCNN_Regularized"

class DeepResidualCNN_ComplexV1(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(128, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        for _ in range(6):
            shortcut = x
            x = Conv1D(128, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv1D(128, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = PReLU()(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepResidualCNN_ComplexV1"

class DeepResidualCNN_ComplexV2(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(256, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        for _ in range(8):
            shortcut = x
            x = Conv1D(256, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = PReLU()(x)
            x = Conv1D(256, 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = PReLU()(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(output_shape)(x)
        model = Model(inputs=inputs, outputs=outputs)
        super().__init__(model)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 60)
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.reduce_lr]
        return super().fit(*args, **kwargs)

    def __str__(self):
        return "DeepResidualCNN_ComplexV2"
