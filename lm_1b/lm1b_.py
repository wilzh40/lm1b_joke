# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Eval pre-trained 1 billion word language model.
"""
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf
import collections

from google.protobuf import text_format
import data_utils
from nodes import *

FLAGS = tf.flags.FLAGS
# General flags.
tf.flags.DEFINE_string('mode', 'eval',
                       'One of [sample, eval, dump_emb, dump_lstm_emb]. '
                       '"sample" mode samples future word predictions, using '
                       'FLAGS.prefix as prefix (prefix could be left empty). '
                       '"eval" mode calculates perplexity of the '
                       'FLAGS.input_data. '
                       '"dump_emb" mode dumps word and softmax embeddings to '
                       'FLAGS.save_dir. embeddings are dumped in the same '
                       'order as words in vocabulary. All words in vocabulary '
                       'are dumped.'
                       'dump_lstm_emb dumps lstm embeddings of FLAGS.sentence '
                       'to FLAGS.save_dir.')
tf.flags.DEFINE_string('pbtxt', '',
                       'GraphDef proto text file used to construct model '
                       'structure.')
tf.flags.DEFINE_string('ckpt', '',
                       'Checkpoint directory used to fill model values.')
tf.flags.DEFINE_string('vocab_file', '', 'Vocabulary file.')
tf.flags.DEFINE_string('save_dir', '',
                       'Used for "dump_emb" mode to save word embeddings.')
# sample mode flags.
tf.flags.DEFINE_string('prefix', '',
                       'Used for "sample" mode to predict next words.')
tf.flags.DEFINE_integer('max_sample_words', 15,
                        'Sampling stops either when </S> is met or this number '
                        'of steps has passed.')
tf.flags.DEFINE_integer('num_samples', 3,
                        'Number of samples to generate for the prefix.')
# dump_lstm_emb mode flags.
tf.flags.DEFINE_string('sentence', '',
                       'Used as input for "dump_lstm_emb" mode.')
# eval mode flags.
tf.flags.DEFINE_string('input_data', '',
                       'Input data files for eval model.')
tf.flags.DEFINE_integer('max_eval_steps', 1000000,
                        'Maximum mumber of steps to run "eval" mode.')

tf.flags.DEFINE_integer('n_top_words', 5, 'Dump the top n next words')
tf.flags.DEFINE_integer('n_unlikely_words', 1, 'Uniformly sample from the softmax')
tf.flags.DEFINE_integer('cutoff', 3, 'Cutoff to stop branching')
tf.flags.DEFINE_string('prefix_file', '', 'File containing one prefix per line')
tf.flags.DEFINE_float('temperature', 0.7,
                        'Temperature for sampling')

# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50


def _LoadModel(gd_file, ckpt_file):
  """Load the model from GraphDef and Checkpoint.

  Args:
    gd_file: GraphDef proto text file.
    ckpt_file: TensorFlow Checkpoint file.

  Returns:
    TensorFlow session and tensors dict.
  """
  with tf.Graph().as_default():
    sys.stderr.write('Recovering graph.\n')
    with tf.gfile.FastGFile(gd_file, 'r') as f:
      s = f.read()
      gd = tf.GraphDef()
      text_format.Merge(s, gd)

    tf.logging.info('Recovering Graph %s', gd_file)
    t = {}
    [t['states_init'], t['lstm/lstm_0/control_dependency'],
     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
     t['all_embs'], t['softmax_weights'], t['global_step']
    ] = tf.import_graph_def(gd, {}, ['states_init',
                                     'lstm/lstm_0/control_dependency:0',
                                     'lstm/lstm_1/control_dependency:0',
                                     'softmax_out:0',
                                     'class_ids_out:0',
                                     'class_weights_out:0',
                                     'log_perplexity_out:0',
                                     'inputs_in:0',
                                     'targets_in:0',
                                     'target_weights_in:0',
                                     'char_inputs_in:0',
                                     'all_embs_out:0',
                                     'Reshape_3:0',
                                     'global_step:0'], name='')

    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
    # Increase the GPU memory limit
    os.environ["TF_CUDA_HOST_MEM_LIMIT_IN_MB"] = '10000'

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth= True
    sess = tf.Session(config=config)
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

    writer = tf.summary.FileWriter("output", sess.graph)
    writer.close()


  return sess, t


def _EvalModel(dataset):
  """Evaluate model perplexity using provided dataset.

  Args:
    dataset: LM1BDataset object.
  """
  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  current_step = t['global_step'].eval(session=sess)
  sys.stderr.write('Loaded step %d.\n' % current_step)

  data_gen = dataset.get_batch(BATCH_SIZE, NUM_TIMESTEPS, forever=False)
  sum_num = 0.0
  sum_den = 0.0
  perplexity = 0.0
  for i, (inputs, char_inputs, _, targets, weights) in enumerate(data_gen):
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = char_inputs
    log_perp = sess.run(t['log_perplexity_out'], feed_dict=input_dict)

    if np.isnan(log_perp):
      sys.stderr.error('log_perplexity is Nan.\n')
    else:
      sum_num += log_perp * weights.mean()
      sum_den += weights.mean()
    if sum_den > 0:
      perplexity = np.exp(sum_num / sum_den)

    sys.stderr.write('Eval Step: %d, Average Perplexity: %f.\n' %
                     (i, perplexity))

    if i > FLAGS.max_eval_steps:
      break


def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)


def _SampleModel(prefix_words, vocab):
  """Predict next words using the given prefix words.

  Args:
    prefix_words: Prefix words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  if prefix_words.find('<S>') != 0:
    prefix_words = '<S> ' + prefix_words

  prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
  for _ in xrange(FLAGS.num_samples):
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
    sent = ''
    while True:
      inputs[0, 0] = samples[0]
      char_ids_inputs[0, 0, :] = char_ids_samples[0]
      samples = samples[1:]
      char_ids_samples = char_ids_samples[1:]

      softmax = sess.run(t['softmax_out'],
                         feed_dict={t['char_inputs_in']: char_ids_inputs,
                                    t['inputs_in']: inputs,
                                    t['targets_in']: targets,
                                    t['target_weights_in']: weights})

      sample = _SampleSoftmax(softmax[0])
      sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))

      if not samples:
        samples = [sample]
        char_ids_samples = [sample_char_ids]
      sent += vocab.id_to_word(samples[0]) + ' '
      sys.stderr.write('%s\n' % sent)

      if (vocab.id_to_word(samples[0]) == '</S>' or
          len(sent) > FLAGS.max_sample_words):
        break


def _DumpEmb(vocab):
  """Dump the softmax weights and word embeddings to files.

  Args:
    vocab: Vocabulary. Contains vocabulary size and converts word to ids.
  """
  assert FLAGS.save_dir, 'Must specify FLAGS.save_dir for dump_emb.'
  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  softmax_weights = sess.run(t['softmax_weights'])
  fname = FLAGS.save_dir + '/embeddings_softmax.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, softmax_weights)
  sys.stderr.write('Finished softmax weights\n')

  all_embs = np.zeros([vocab.size, 1024])
  for i in xrange(vocab.size):
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = (
          vocab.word_char_ids[i].reshape([-1, 1, MAX_WORD_LEN]))
    embs = sess.run(t['all_embs'], input_dict)
    all_embs[i, :] = embs
    sys.stderr.write('Finished word embedding %d/%d\n' % (i, vocab.size))

  fname = FLAGS.save_dir + '/embeddings_char_cnn.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, all_embs)
  sys.stderr.write('Embedding file saved\n')


def _DumpSentenceEmbedding(sentence, vocab):
  """Predict next words using the given prefix words.

  Args:
    sentence: Sentence words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  if sentence.find('<S>') != 0:
    sentence = '<S> ' + sentence

  word_ids = [vocab.word_to_id(w) for w in sentence.split()]
  char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  char_ids_inputs = np.zeros(
      [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
  for i in xrange(len(word_ids)):
    inputs[0, 0] = word_ids[i]
    char_ids_inputs[0, 0, :] = char_ids[i]

    # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
    # LSTM.
    lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
                        feed_dict={t['char_inputs_in']: char_ids_inputs,
                                   t['inputs_in']: inputs,
                                   t['targets_in']: targets,
                                   t['target_weights_in']: weights})

    # fname = os.path.join(FLAGS.save_dir, 'lstm_emb_step_%d.npy' % i)
    # with tf.gfile.Open(fname, mode='w') as f:
    #   np.save(f, lstm_emb)
    # sys.stderr.write('LSTM embedding step %d file saved\n' % i)
  print(lstm_emb)
  # It's a 2-dim array
  return lstm_emb[0]

def _SoftmaxTopIndices(softmax, n):
  top_indices = np.argpartition(softmax, -n)[-n:]
  top_indices_sorted = top_indices[np.argsort(softmax[top_indices])][::-1]
  # for i in top_indices_sorted:
  #   w = vocab.id_to_word(i)
  #   p = softmax[i]
  #   # TODO: Should probably dump to a file. Also, would probably be nice to be
  #   # able to specify multiple prefixes per run. Cause there's a pretty high
  #   # overhead on loading the checkpoint data etc.
  #   sys.stderr.write('{}\t{}\n'.format(w, p))
  return top_indices_sorted

# Returns two lists, top_n_words chosen from the top, and random_n_words chosen randomly
# Limit is 150 words

def sample_temp(a, vocab, temperature=1.0):
    # helper function to sample an index from a probability array
    a[vocab.unk] = 0.0000000000001
    a = np.log(np.array(a, dtype=np.float64)) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# def sample_temp(softmax, temperature=1.0):
#   a = softmax ** (1/temperature)
#   p_sum = a.sum()
#   scaled = a/a.sum()
#   print(scaled.sum())
#   np.random.choice(len(scaled), p=scaled)
#   # return np.argmax(np.random.multinomial(1, scaled, 1))

def sample_softmax(softmax, vocab, top_n_words, random_n_words):
  top_indices_sorted = _SoftmaxTopIndices(softmax, 100)
  known_indices = [i for i in top_indices_sorted if i != vocab.unk]
  #TODO: Change temperature
  randoms = []

  for i in range(random_n_words):
    randoms.append(sample_temp(softmax, vocab, FLAGS.temperature))
  # return (known_indices[:top_n_words], np.random.choice(known_indices, random_n_words))
  return (known_indices[:top_n_words], np.array(randoms))



def _DumpNextWords(prefix_file, vocab):
  # Hack for convenience
  filemode = prefix_file.endswith('.txt') or ' ' not in prefix_file
  if filemode:
    filelines = open(prefix_file)
  else:
    filelines = [prefix_file]
  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)

  finished_sentences = []
  lines = []
  edges = []
  node_id = 0

  # Recursive function that returns a trie node
  def sample_next(prefix, cutoff):
    # Todo: Does this reset the states?
    sess.run(t['states_init'])
    node = Node(prefix)
    prefix_words = prefix.strip()
    print(prefix_words)
    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

    if prefix_words.find('<S>') != 0:
      prefix_words = '<S> ' + prefix_words

    samples = [vocab.word_to_id(w) for w in prefix_words.split()]
    prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]

    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    char_ids_samples = prefix_char_ids[:]

    # print(FLAGS.max_sample_words)

    while True:
      inputs[0, 0] = samples[0]
      # print(char_ids_samples[0].shape)
      char_ids_inputs[0, 0, :] = char_ids_samples[0]
      samples = samples[1:]
      char_ids_samples = char_ids_samples[1:]

      softmax = sess.run(t['softmax_out'],
                         feed_dict={t['char_inputs_in']: char_ids_inputs,
                                    t['inputs_in']: inputs,
                                    t['targets_in']: targets,
                                    t['target_weights_in']: weights})

      use_unlikely = True
      unlikely_words = 1

      if not samples:

        # We're done feeding in the prefix. It's time get the predicted next words.

        # If our tree is nonbranching, just accumulate to the current prefix without recursing
        # This occurs when specified conditions are met, like after a certain amount of words we stop branching
        nonbranching = True if len(prefix_words.split()) > cutoff else False
        if nonbranching:
          # indices, _ = sample_softmax(softmax[0], vocab, 1, 0)
          indices = np.array([sample_temp(softmax[0], vocab)]).astype(int)
          assert(len(indices) == 1)
          next_word = vocab.id_to_word(indices[0])
          if (next_word == '</S>' or
            len(prefix.split()) > FLAGS.max_sample_words):
            # This is the end of the sentence or it has exceeded the max_sample_words
            print(prefix)
            node.prefix = prefix
            finished_sentences.append(prefix)
            break
          else:
            prefix = "{} {}".format(prefix, next_word)
            # Add back to samples + char_ids_samples to continue feeding into LSTM on this run
            samples = indices[:]
            char_ids_samples = [vocab.word_to_char_ids(next_word)]
            continue


        # top_words = 1 if len(prefix_words.split()) > cutoff else FLAGS.n_top_words
        likely_indices, unlikely_indices = sample_softmax(softmax[0],  vocab,
                                                   FLAGS.n_top_words,
                                                   FLAGS.n_unlikely_words)
        if use_unlikely:
          indices = np.append(likely_indices, unlikely_indices)
        print(indices)
        # for i in indices:
        #   next_word = vocab.id_to_word(i)
        #   print("{}\t{}".format(next_word, softmax[0][i]))
        for c,i in enumerate(indices):
          next_word = vocab.id_to_word(i)
          assert(next_word != '<UNK>')
          # node_id += 1
          # if next_word == '<UNK>':
          #   # Skip the next word if its unknown
          #   continue

          # Append the new words to the top
          if (next_word == '</S>' or
            len(prefix.split()) > FLAGS.max_sample_words):
            # This is the end of the sentence or it has exceeded the max_sample_words
            finished_sentences.append(prefix)
          else:
            if c >= len(likely_indices):
              new_prefix = "{} _{}".format(prefix, vocab.id_to_word(i))
            else:
              new_prefix = "{} {}".format(prefix, vocab.id_to_word(i))
            new_child = sample_next(new_prefix, cutoff)
            probability = softmax[0][i]
            node.children[vocab.id_to_word(i)] = (probability, new_child)
            # edges.append((prefix_words[-1], vocab.id_to_word(i), softmax[0][i]))
            # lines.append("{} {}".format(line, vocab.id_to_word(i)))
        break
    return node


  #TODO: temporary hack to disable graph writing
  write_graphs = True
  if write_graphs:
    # os.makedirs("./graphs")
    for line in filelines:
      # For each line, build the tree
      tree = sample_next(line, len(line.split()) + FLAGS.cutoff)
      with open("./graphs/graph_{}_cutoff-{}_branch-{}".format(line, FLAGS.cutoff, FLAGS.n_top_words), 'w') as f:
        visualize(tree, f)
        f.write("\n")

  output = open(".{}_cutoff{}_top{}_bottom{}_temp{}".format(FLAGS.prefix_file, FLAGS.cutoff, FLAGS.n_top_words,
                                                                FLAGS.n_unlikely_words, FLAGS.temperature), 'w')
  for l in finished_sentences:
    output.write(l)
    # output.write("\t")
    # print(_DumpSentenceEmbedding(l,vocab))
    # output.write(_DumpSentenceEmbedding(l,vocab))
    output.write("\n")
  output.close()

  calculate_embeddings = False
  if calculate_embeddings:
    biggest_diffs = biggest_embedding_diff(finished_sentences, vocab)
    print(biggest_diffs)
    output = open("biggest_diffs", 'w')
    for keys,values in biggest_diffs.items():
      output.write("{}: {} \n".format(keys,values))
    output.close()

  # if filemode:
    # lines.close()

# Takes a list of sentences and returns a dictionary of the maximal distances between the embeddigns
def biggest_embedding_diff(sentences, vocab):
  embeddings = {}
  for s in sentences:
    embeddings[s] = _DumpSentenceEmbedding(s, vocab)
  print(embeddings)

  diffs = {}
  for s1 in sentences:
    for s2 in sentences:
        diffs[np.dot(embeddings[s1],embeddings[s2])] = (s1,s2)
  print(diffs)
  return collections.OrderedDict(sorted(diffs.items()))

def main(unused_argv):
  vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)

  if FLAGS.mode == 'eval':
    dataset = data_utils.LM1BDataset(FLAGS.input_data, vocab)
    _EvalModel(dataset)
  elif FLAGS.mode == 'sample':
    _SampleModel(FLAGS.prefix, vocab)
  elif FLAGS.mode == 'dump_emb':
    _DumpEmb(vocab)
  elif FLAGS.mode == 'dump_lstm_emb':
    _DumpSentenceEmbedding(FLAGS.sentence, vocab)
  elif FLAGS.mode == 'next_words':
    _DumpNextWords(FLAGS.prefix_file, vocab)
  else:
    raise Exception('Mode not supported.')


if __name__ == '__main__':
  tf.app.run()
