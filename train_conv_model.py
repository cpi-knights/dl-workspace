import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report


def retrain_model(
    base_model: tf.keras.Model,
    image_shape=(224, 224, 3),
    initial_epochs=10,
    fine_epochs=10,
    learning_rate=0.0001,
):
    # First, freeze the pre-trained model
    base_model.trainable = False

    # Define the custom model using functional api
    inputs = tf.keras.Input(shape=image_shape)
    y = data_augmentation(inputs)
    y = process_input(y)
    y = base_model(y, training=False)
    y = global_average_layer(y)
    y = dropout_layer(y)
    outputs = tf.keras.layers.Dense(num_classes)(y)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model for the first step
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )

    # Fit the model without modifying the weights
    history = model.fit(
        train_ds,
        epochs=initial_epochs,
        validation_data=val_ds,
    )

    # Unfreeze the base model
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compile the model again with unfreezed top layers
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate/10),
        metrics=['accuracy'],
    )

    # Fine-tune the model
    history_fine = model.fit(
        train_ds,
        epochs=initial_epochs+fine_epochs,
        initial_epoch=len(history.epoch),
        validation_data=val_ds,
    )

    # Plot the results
    epochs = initial_epochs + fine_epochs

    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + \
        history_fine.history['val_accuracy']

    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[np.argmax(predictions[i])])
        plt.axis("off")

    loss, accuracy = model.evaluate(test_ds)
    print('Test accuracy :', accuracy)

    # Generate predictions
    y_pred = model.predict(test_ds)  # Model predictions
    # Convert probabilities to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.array([labels for _, labels in test_ds]
                      ).flatten()  # True labels

    # Generate Classification Report
    report = classification_report(
        y_true, y_pred_classes, target_names=class_names)
    print("Classification Report:")
    print(report)

    # Generate Confusion Matrix
    cm = tf.math.confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Optionally, visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

    return model
