import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")
texts = df['source_article'].astype(str).tolist()
labels = df['label'].values  # numpy array

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="tf")

input_ids = encodings['input_ids'].numpy()
attention_mask = encodings['attention_mask'].numpy()

train_input_ids, val_input_ids, train_attention, val_attention, train_labels, val_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, stratify=labels, random_state=42
)

train_labels = tf.convert_to_tensor(train_labels)
val_labels = tf.convert_to_tensor(val_labels)

train_dataset = tf.data.Dataset.from_tensor_slices(((train_input_ids, train_attention), train_labels)).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_input_ids, val_attention), val_labels)).batch(16)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

epochs = 5
steps_per_epoch = len(train_dataset)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1 * num_train_steps)

optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=num_warmup_steps,
    num_train_steps=num_train_steps
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping]
)

model.save_pretrained("folder Location")
tokenizer.save_pretrained("folder Location")
