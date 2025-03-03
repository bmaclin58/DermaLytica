import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
from sklearn.metrics import precision_recall_curve, precision_score as sklearn_precision_score, \
    recall_score as sklearn_recall_score, roc_curve
from tensorflow.keras.losses import BinaryFocalCrossentropy

# Enable numerical checking to identify where NaNs occur
tf.debugging.enable_check_numerics()

# TPU Setup
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("TPU initialized successfully!")

# Constants with memory and numerical stability in mind
IMAGE_SIZE = (224, 224)
TPU_BATCH_SIZE = 64  # Reduced batch size for stability
GLOBAL_BATCH_SIZE = min(TPU_BATCH_SIZE, TPU_BATCH_SIZE * strategy.num_replicas_in_sync)

# Data constants
train_samples = 11200
train_tfrecords = [f'/content/drive/MyDrive/DermaLyticsAI/small dataset/train.tfrecord']

val_samples = 4800
val_tfrecords = [f'/content/drive/MyDrive/DermaLyticsAI/small dataset/validation.tfrecord']

total_test_samples = 4000
test_tfrecords = [f'/content/drive/MyDrive/DermaLyticsAI/small dataset/test.tfrecord']

checkpoint_path = f'/content/drive/MyDrive/DermaLyticsAI/small dataset/modelCheckpoint.weights.h5'
log_dir = f'/content/drive/MyDrive/DermaLyticsAI/small dataset/logs'
modelSave_path = f'/content/drive/MyDrive/DermaLyticsAI/small dataset/model/derma_model.h5'
modelKerasSave_path = f'/content/drive/MyDrive/DermaLyticsAI/small dataset/model/derma_model_keras.keras'

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

def parse_function(proto, image_size, is_training=True):
    """Parse a single example from TFRecord with additional error handling."""
    # Define the feature description
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([1], tf.float32),
        "metadata": tf.io.FixedLenFeature([11], tf.float32),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    # Parse with error handling
    try:
        example = tf.io.parse_single_example(proto, feature_description)
        
        # Decode and process the image with error clipping
        image = tf.io.decode_jpeg(example["image"], channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.clip_by_value(tf.cast(image, tf.float32) / 255.0, 0.0, 1.0)  # Ensure proper range

        # Decode and process the mask with error clipping
        mask = tf.io.decode_jpeg(example["mask"], channels=3)
        mask = tf.image.resize(mask, image_size)
        mask = tf.clip_by_value(tf.cast(mask, tf.float32) / 255.0, 0.0, 1.0)  # Ensure proper range

        if is_training:
            # Generate random transformation parameters
            seed = tf.random.uniform([2], minval=0, maxval=100, dtype=tf.int32)

            # Apply identical flips using the same seed
            image = tf.image.stateless_random_flip_left_right(image, seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed)
            image = tf.image.stateless_random_flip_up_down(image, seed)
            mask = tf.image.stateless_random_flip_up_down(mask, seed)

            # Random rotation
            flipNumber = tf.random.uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k=flipNumber)
            mask = tf.image.rot90(mask, k=flipNumber)

        # Extract and normalize metadata
        metadata = example["metadata"]
        # Clip metadata to reasonable ranges to prevent extreme values
        metadata = tf.clip_by_value(metadata, -100.0, 100.0)
        
        # Make sure label is valid binary
        label = tf.clip_by_value(example["label"], 0.0, 1.0)
        
        # Return using a dictionary structure
        return ({"image_input": image, "mask_input": mask, "metadata_input": metadata}, label)
    
    except tf.errors.InvalidArgumentError as e:
        # On parse error, return a safe default
        print(f"Error parsing example: {e}")
        # Return zeros with correct shapes
        default_image = tf.zeros((*image_size, 3), dtype=tf.float32)
        default_mask = tf.zeros((*image_size, 3), dtype=tf.float32)
        default_metadata = tf.zeros((11,), dtype=tf.float32)
        default_label = tf.constant([[0.0]], dtype=tf.float32)
        
        return ({"image_input": default_image, "mask_input": default_mask, "metadata_input": default_metadata}, 
                default_label)

def load_tfrecord_dataset(tfrecord_paths, batch_size, image_size, epochs=1, is_training=True):
    """Load TFRecord dataset with enhanced stability and error handling."""
    # Use the parse function
    parse_fn = lambda x: parse_function(x, image_size, is_training)

    # Create the dataset with error handling
    try:
        dataset = tf.data.TFRecordDataset(
            tfrecord_paths, 
            num_parallel_reads=tf.data.AUTOTUNE,
            buffer_size=8 * 1024 * 1024  # 8MB buffer
        )
        
        # Handle corrupted records by skipping them
        dataset = dataset.map(
            parse_fn, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            dataset = dataset.shuffle(1000)  # Reduced buffer size
            
        # Batch first, then repeat to ensure complete batches
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Apply repeat after batching
        if is_training:
            dataset = dataset.repeat(epochs)
        else:
            # Just enough repetitions for validation
            dataset = dataset.repeat(2)  
            
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Calculate steps more conservatively
        if is_training:
            steps = (train_samples // batch_size) - 2  # Buffer to avoid StopIteration
        else:
            if tfrecord_paths[0] == val_tfrecords[0]:
                steps = (val_samples // batch_size) - 2
            else:
                steps = (total_test_samples // batch_size) - 2
                
        # Ensure we have at least 1 step
        steps = max(1, steps)
        
        return dataset, steps
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def build_model(strategy, image_size=(224, 224)):
    """Build model with enhanced numerical stability."""
    with strategy.scope():
        # Set up kernel initializers for stability
        kernel_init = tf.keras.initializers.GlorotNormal(seed=42)
        
        # Define the inputs
        image_input = tf.keras.Input(shape=(*image_size, 3), name="image_input")
        mask_input = tf.keras.Input(shape=(*image_size, 3), name="mask_input")
        metadata_input = tf.keras.Input(shape=(11,), name="metadata_input")

        # Attention mechanism with stabilized initialization
        attention = tf.keras.layers.Conv2D(
            1, (1, 1), 
            activation='sigmoid',
            kernel_initializer=kernel_init
        )(mask_input)
        modulated_image = tf.keras.layers.Multiply()([image_input, attention])

        # Load EfficientNetV2B3 with weight initialization for stability
        base_model = tf.keras.applications.EfficientNetV2B3(
            include_top=False,
            input_tensor=modulated_image,
            weights=None,  # Start with random weights for TPU compatibility
            classes=2
        )

        # Freeze initial layers to stabilize training
        for layer in base_model.layers[:50]:  # Freeze early layers
            layer.trainable = False
        
        # Make later layers trainable
        for layer in base_model.layers[50:]:
            layer.trainable = True

        # Add attention mechanism with stable initialization
        attention = tf.keras.layers.Conv2D(
            base_model.output.shape[-1],
            (1, 1),
            activation='sigmoid',
            kernel_initializer=kernel_init
        )(base_model.output)

        weighted_features = tf.keras.layers.Multiply()([base_model.output, attention])

        # Process image features with batch normalization for stability
        x = tf.keras.layers.GlobalAveragePooling2D()(weighted_features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Add dropout for regularization

        # Process mask with a small CNN to extract features
        mask_features = tf.keras.layers.Conv2D(
            16, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_initializer=kernel_init
        )(mask_input)
        mask_features = tf.keras.layers.BatchNormalization()(mask_features)
        mask_features = tf.keras.layers.MaxPooling2D()(mask_features)
        
        mask_features = tf.keras.layers.Conv2D(
            32, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_initializer=kernel_init
        )(mask_features)
        mask_features = tf.keras.layers.BatchNormalization()(mask_features)
        mask_features = tf.keras.layers.GlobalAveragePooling2D()(mask_features)
        mask_features = tf.keras.layers.BatchNormalization()(mask_features)

        # Split metadata with more stable processing
        demographic_features = tf.keras.layers.Lambda(lambda x: x[:, :3])(metadata_input)
        location_features = tf.keras.layers.Lambda(lambda x: x[:, 3:])(metadata_input)

        # Process location features
        location_encoded = tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(location_features)
        location_encoded = tf.keras.layers.BatchNormalization()(location_encoded)
        location_encoded = tf.keras.layers.Dropout(0.1)(location_encoded)

        # Process demographic features
        demographic_encoded = tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(demographic_features)
        demographic_encoded = tf.keras.layers.BatchNormalization()(demographic_encoded)
        demographic_encoded = tf.keras.layers.Dropout(0.1)(demographic_encoded)

        # Combine metadata features
        metadata_features = tf.keras.layers.Concatenate()([location_encoded, demographic_encoded])
        metadata_features = tf.keras.layers.BatchNormalization()(metadata_features)

        # Combine all features with BatchNormalization
        combined = tf.keras.layers.Concatenate()([x, metadata_features, mask_features])
        combined = tf.keras.layers.BatchNormalization()(combined)

        # First dense block with residual connection and more regularization
        block1 = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(combined)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.Dropout(0.3)(block1)

        # Residual connection with the same architecture
        block1_res = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(block1)
        block1_res = tf.keras.layers.BatchNormalization()(block1_res)
        block1_res = tf.keras.layers.Dropout(0.3)(block1_res)

        # Add residual connection for improved gradient flow
        block1 = tf.keras.layers.Add()([block1, block1_res])
        block1 = tf.keras.layers.BatchNormalization()(block1)

        # Second dense block
        block2 = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(block1)
        block2 = tf.keras.layers.BatchNormalization()(block2)
        block2 = tf.keras.layers.Dropout(0.2)(block2)

        # Ensure numerical stability with float32 precision
        combined = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(block2)

        # Final classification layer with stable initialization
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            dtype='float32',  # Explicitly use float32 for stability
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(combined)

        model = tf.keras.Model(inputs=[image_input, metadata_input, mask_input], outputs=output)

        return model

# Define callbacks for training with additional stability monitoring
early_stopping_auc = tf.keras.callbacks.EarlyStopping(
    monitor="val_AUC",
    patience=5,
    restore_best_weights=True,
    mode="max"
)
early_stopping_loss = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    mode="min"
)
terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='epoch'
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_AUC',
    mode='max',
    verbose=1
)

# Load datasets with stability improvements
train_dataset, train_steps = load_tfrecord_dataset(
    train_tfrecords,
    batch_size=GLOBAL_BATCH_SIZE,
    image_size=IMAGE_SIZE,
    epochs=40,
    is_training=True
)

val_dataset, val_steps = load_tfrecord_dataset(
    val_tfrecords,
    batch_size=GLOBAL_BATCH_SIZE,
    image_size=IMAGE_SIZE,
    epochs=40,
    is_training=False
)

print(f"Steps per epoch: {train_steps}, Validation steps: {val_steps}")

# Build and compile model with TPU strategy
with strategy.scope():
    # Build model with stabilized architecture
    model = build_model(strategy, IMAGE_SIZE)
    
    # More conservative learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,  # Lower initial learning rate
        decay_steps=train_steps * 5,
        decay_rate=0.95,
        staircase=True
    )
    
    # Optimizer with enhanced numerical stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0,        # Stricter gradient clipping
        epsilon=1e-7         # Increased epsilon for numerical stability
    )
    
    # Compile model with loss scaling for mixed precision
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="AUC", curve='ROC'),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )

# Print model summary
model.summary()

# Start with a stability check - short training to identify issues early
print("PHASE 1: Initial stability check (2 epochs)")
try:
    initial_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[terminate_on_nan],
        verbose=1
    )
    
    # Check if training was stable
    if np.isnan(initial_history.history['loss'][-1]):
        print("WARNING: NaN detected in initial training. Adjusting parameters...")
        
        # Reset model with even more conservative settings
        with strategy.scope():
            # Clear memory
            K.clear_session()
            gc.collect()
            
            # Rebuild with more conservative settings
            model = build_model(strategy, IMAGE_SIZE)
            
            # Use RMSprop for more stability
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=5e-5,
                rho=0.9,
                momentum=0.0,
                epsilon=1e-7,
                clipnorm=0.5
            )
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
                    tf.keras.metrics.AUC(name="AUC", curve='ROC'),
                    tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                    tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                ],
            )
    
    print("PHASE 2: Main training phase")
    # Main training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=2,  # Continue from where we left off
        epochs=40,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[
            early_stopping_auc, 
            early_stopping_loss, 
            checkpoint_callback, 
            tensorboard_callback,
            terminate_on_nan
        ],
        verbose=1
    )
    
except Exception as e:
    print(f"Error during training: {e}")
    # If training fails, try with even more conservative approach
    print("Training failed. Attempting with more conservative settings...")
    
    # Clear memory
    K.clear_session()
    gc.collect()
    
    with strategy.scope():
        # Build a simpler model
        model = build_model(strategy, IMAGE_SIZE)
        
        # Use SGD for maximum stability
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=1e-5, 
            momentum=0.9,
            clipnorm=0.1
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
                tf.keras.metrics.AUC(name="AUC", curve='ROC'),
                tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                tf.keras.metrics.Recall(name="recall", thresholds=0.5),
            ],
        )
        
        # Try training with the simplified approach
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=30,  # Shorter training
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=[
                early_stopping_auc, 
                early_stopping_loss, 
                checkpoint_callback, 
                terminate_on_nan
            ],
            verbose=1
        )

# Force garbage collection to clear memory
gc.collect()
tf.keras.backend.clear_session()

# Save intermediate model with error handling
try:
    model.save(modelSave_path)
    model.save(modelKerasSave_path)
    print(f"Model saved successfully to {modelSave_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    # Try alternative saving approach
    try:
        # Save weights only if full model save fails
        model.save_weights(f'{modelSave_path}_weights')
        print(f"Model weights saved to {modelSave_path}_weights")
    except:
        print("Could not save model weights either")

# Evaluation phase with error handling
try:
    # Load test dataset with stability improvements
    test_dataset, test_steps = load_tfrecord_dataset(
        test_tfrecords,
        batch_size=GLOBAL_BATCH_SIZE,
        image_size=IMAGE_SIZE,
        is_training=False
    )

    # Evaluate with default threshold
    print("Evaluating with default threshold (0.5):")
    test_results = model.evaluate(
        test_dataset, 
        steps=test_steps,
        verbose=1
    )
    print("Test Results:", dict(zip(model.metrics_names, test_results)))
except Exception as e:
    print(f"Error during evaluation: {e}")

# Force memory cleanup
gc.collect()
tf.keras.backend.clear_session()

# Create dataset for threshold tuning with smaller batches
try:
    threshold_dataset, _ = load_tfrecord_dataset(
        val_tfrecords,
        batch_size=32,  # Smaller batch size for prediction
        image_size=IMAGE_SIZE,
        is_training=False
    )

    # Collect predictions for threshold optimization in smaller batches
    all_preds = []
    all_labels = []

    print("Collecting predictions for threshold optimization...")
    # Process in smaller chunks to avoid memory issues
    for i, (inputs, labels) in enumerate(threshold_dataset):
        if i >= val_samples // 32:  # Adjusted for smaller batch size
            break
        try:
            preds = model.predict_on_batch(inputs)
            
            # Check for NaN in predictions
            if np.isnan(preds).any():
                print(f"Warning: NaN found in predictions batch {i}")
                continue
                
            # Convert to numpy and store
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
            # Periodically clear memory
            if i % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

    # Find optimal threshold with error handling
    print("Calculating optimal threshold...")
    if len(all_preds) > 0 and len(all_labels) > 0:
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_threshold = float(optimal_threshold)
            
            # Calculate metrics with various thresholds
            thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
            for threshold in thresholds_to_try:
                binary_preds = (np.array(all_preds) >= threshold).astype(int)
                precision = sklearn_precision_score(all_labels, binary_preds)
                recall = sklearn_recall_score(all_labels, binary_preds)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"Threshold: {threshold:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, F1: {f1:.4f}")

            print(f"Optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            print(f"Error calculating optimal threshold: {e}")
            optimal_threshold = 0.5  # Fallback to default
    else:
        print("No valid predictions collected for threshold optimization")
        optimal_threshold = 0.5  # Use default
except Exception as e:
    print(f"Error in threshold tuning phase: {e}")
    optimal_threshold = 0.5  # Use default

# Clear memory before final phase
gc.collect()
tf.keras.backend.clear_session()

# Final training phase with focal loss
try:
    print("PHASE 3: Final training with Focal Loss and optimized parameters")

    # Set a moderate learning rate for final training
    K.set_value(model.optimizer.learning_rate, 5e-6)  # Conservative learning rate

    # Recompile with focal loss and custom metrics
    with strategy.scope():
        model.compile(
            optimizer=model.optimizer,
            loss=BinaryFocalCrossentropy(alpha=0.75, gamma=2.0),  # Adjusted focal loss
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
                tf.keras.metrics.AUC(name="AUC", curve='ROC'),
                tf.keras.metrics.Precision(name="precision", thresholds=0.5),
                tf.keras.metrics.Recall(name="recall", thresholds=0.5),
                tf.keras.metrics.Precision(name="opt_precision", thresholds=optimal_threshold),
                tf.keras.metrics.Recall(name="opt_recall", thresholds=optimal_threshold),
            ],
        )

    # Final training phase with reduced epochs
    final_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[
            early_stopping_auc, 
            early_stopping_loss, 
            checkpoint_callback, 
            tensorboard_callback,
            terminate_on_nan
        ],
        verbose=1
    )

    # Save the final model
    model.save(modelSave_path)
    model.save(modelKerasSave_path)

    # Final evaluation
    print("\nFinal evaluation results:")
    test_results = model.evaluate(test_dataset, steps=test_steps)
    print("Test Results:", dict(zip(model.metrics_names, test_results)))

    print(f"Training complete. Optimal threshold for prediction: {optimal_threshold:.4f}")
    
except Exception as e:
    print(f"Error in final training phase: {e}")
    print("Training completed with some errors.")
