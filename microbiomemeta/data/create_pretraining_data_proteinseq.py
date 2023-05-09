# coding=utf-8

"""Create masked LM/next sentence masked_lm TF examples for ALBERT."""

import collections
import random

import numpy as np
import six
import tensorflow.compat.v1 as tf

from microbiomemeta.albert import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None, "Input raw text file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "output_file", None, "Output TF example file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the ALBERT model was trained on."
)

flags.DEFINE_string(
    "spm_model_file", None, "The model file for sentence piece tokenization."
)

flags.DEFINE_string("input_file_mode", "r", "The data format of the input file.")

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_bool(
    "do_whole_word_mask",
    True,
    "Whether to use whole word masking rather than per-WordPiece masking.",
)

flags.DEFINE_bool("do_permutation", False, "Whether to do the permutation training.")

flags.DEFINE_bool(
    "favor_shorter_ngram",
    True,
    "Whether to set higher probabilities for sampling shorter ngrams.",
)

flags.DEFINE_bool(
    "random_next_sentence",
    False,
    "Whether to use the sentence that's right before the current sentence "
    "as the negative sample for next sentence prection, rather than using "
    "sentences from other random documents.",
)

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("ngram", 3, "Maximum number of ngrams to mask.")

flags.DEFINE_integer(
    "max_predictions_per_seq",
    20,
    "Maximum number of masked LM predictions per sequence.",
)

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor",
    40,
    "Number of times to duplicate the input data (with different masks).",
)

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob",
    0.1,
    "Probability of creating sequences which are shorter than the " "maximum length.",
)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(
        self,
        tokens,
        segment_ids,
        masked_lm_positions,
        masked_lm_labels,
        is_random_next,
        token_boundary,
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.token_boundary = token_boundary
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (
            " ".join([tokenization.printable_text(x) for x in self.tokens])
        )
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "token_boundary: %s\n" % (" ".join([str(x) for x in self.token_boundary]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions])
        )
        s += "masked_lm_labels: %s\n" % (
            " ".join([tokenization.printable_text(x) for x in self.masked_lm_labels])
        )
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(
    instances,
    tokenizer,
    max_seq_length,
    max_predictions_per_seq,
    output_files,
    token_weights={
        "j": 0.1,
        "9": 0.1,
        "[PAD]": 0.0,
        "[UNK]": 0.0,
        "[SEP]": 0.0,
        "x": 0.1,
    }
    # token_weights controls special tokens that need different weights than 1.0
):
    """Create TF example files from TrainingInstances."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        # token_boundary = list(instance.token_boundary)
        assert len(input_ids) <= max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = []
        for tok in instance.masked_lm_labels:
            if tok in token_weights:
                masked_lm_weights.append(token_weights[tok])
            else:
                masked_lm_weights.append(1.0)

        multiplier = 1 + int(FLAGS.do_permutation)
        while len(masked_lm_positions) < max_predictions_per_seq * multiplier:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        # sentence_order_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)

        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info(
                "tokens: %s"
                % " ".join([tokenization.printable_text(x) for x in instance.tokens])
            )

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values]))
                )

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(
    input_files,
    tokenizer,
    max_seq_length,
    dupe_factor,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
):
    """Create TrainingInstances from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, FLAGS.input_file_mode) as reader:
            while True:
                line = reader.readline()
                if not FLAGS.spm_model_file:
                    line = tokenization.convert_to_unicode(line)
                if not line:
                    break
                if FLAGS.spm_model_file:
                    line = tokenization.preprocess_text(line, lower=FLAGS.do_lower_case)
                else:
                    line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents,
                    document_index,
                    max_seq_length,
                    short_seq_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                )
            )

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
    all_documents,
    document_index,
    max_seq_length,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
):
    """Creates TrainingInstances for a single document."""
    document = all_documents[document_index]

    # Account for [CLS]
    max_num_tokens = max_seq_length - 1

    instances = []
    # current_chunk = []
    # current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        if segment:
            tokens = ["[CLS]"]
            segment_ids = [0]
            for j in range(min(len(segment), max_num_tokens)):
                tokens.extend(segment[j])
                segment_ids.append(0)

            assert len(tokens) >= 1

            while len(tokens) <= max_num_tokens:
                tokens.append("[PAD]")
                segment_ids.append(0)

            (
                tokens,
                masked_lm_positions,
                masked_lm_labels,
                token_boundary,
            ) = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng
            )
            instance = TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=False,
                token_boundary=token_boundary,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels,
            )
            instances.append(instance)

        i += 1
    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def _is_start_piece_sp(piece):
    """Check if the current word piece is the starting piece (sentence piece)."""
    special_pieces = set(list('!"#$%&"()*+,-./:;?@[\\]^_`{|}~'))
    special_pieces.add(u"€".encode("utf-8"))
    special_pieces.add(u"£".encode("utf-8"))
    # Note(mingdachen):
    # For foreign characters, we always treat them as a whole piece.
    english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
    if (
        six.ensure_str(piece).startswith("▁")
        or six.ensure_str(piece).startswith("<")
        or piece in special_pieces
        or not all([i.lower() in english_chars.union(special_pieces) for i in piece])
    ):
        return True
    else:
        return False


def _is_start_piece_bert(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece):
    if FLAGS.spm_model_file:
        return _is_start_piece_sp(piece)
    else:
        return _is_start_piece_bert(piece)


def create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng
):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (
            FLAGS.do_whole_word_mask
            and len(cand_indexes) >= 1
            and not is_start_piece(token)
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(token):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
    pvals = 1.0 / np.arange(1, FLAGS.ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not FLAGS.favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx : idx + n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(
            ngrams[: len(cand_index_set)],
            p=pvals[: len(cand_index_set)]
            / pvals[: len(cand_index_set)].sum(keepdims=True),
        )
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    rng.shuffle(ngram_indexes)

    select_indexes = set()
    if FLAGS.do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)]
                / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file,
    )

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files,
        tokenizer,
        FLAGS.max_seq_length,
        FLAGS.dupe_factor,
        FLAGS.short_seq_prob,
        FLAGS.masked_lm_prob,
        FLAGS.max_predictions_per_seq,
        rng,
    )

    tf.logging.info("number of instances: %i", len(instances))

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(
        instances,
        tokenizer,
        FLAGS.max_seq_length,
        FLAGS.max_predictions_per_seq,
        output_files,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
