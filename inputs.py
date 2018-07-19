"""Input feature columns and input_fn for models.

Handles both training, evaluation and inference.
"""
import tensorflow as tf


def BuildTextExample(text, label=None):
    record = tf.train.Example()
    text = [tf.compat.as_bytes(x) for x in text]
    record.features.feature["text"].bytes_list.value.extend(text)
    if label is not None:
        label = tf.compat.as_bytes(label)
        record.features.feature["label"].bytes_list.value.append(label)
    return record


def ParseSpec(include_target):
    parse_spec = {"text": tf.VarLenFeature(dtype=tf.string)}
    if include_target:
        parse_spec["label"] = tf.FixedLenFeature(shape=(), dtype=tf.string,
                                                 default_value=None)
    return parse_spec


def InputFn(mode,
            input_file,
            vocab_file,
            vocab_size,
            embedding_dimension,
            num_oov_vocab_buckets,
            label_file,
            label_size,
            ngram_embedding_dimension,            
            num_ngram_hash_buckets,
            batch_size,
            num_epochs=None,
            num_threads=1):
    if num_epochs <= 0:
        num_epochs=None
    def input_fn():
        include_target =  mode != tf.estimator.ModeKeys.PREDICT
        parse_spec = ParseSpec(include_target)
        print("ParseSpec", parse_spec)
        print("Input file:", input_file)
        features = tf.contrib.learn.read_batch_features(
            input_file, batch_size, parse_spec, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=num_threads)
        label = None
        if include_target:
            label = features.pop("label")
        return features, label
    return input_fn


def ServingInputFn():
    parse_spec = ParseSpec(include_target=False)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        parse_spec)
