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
import time

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
import data_utils

WORD_DUMP_LIMIT = 5000

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
tf.flags.DEFINE_integer('max_sample_words', 100,
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

tf.flags.DEFINE_integer('n_top_words', 20, 'Dump the top n next words')
tf.flags.DEFINE_string('prefix_file', '', 'File containing one prefix per line')

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
    cnn_vars = (
      ['char_embedding/cnn_{}/kernel:0'.format(i) for i in range(8)] +
      ['char_embedding/cnn_{}/b:0'.format(i) for i in range(8)]
    )
    [t['states_init'], t['lstm/lstm_0/control_dependency'],
     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
     t['all_embs'], t['softmax_weights'], t['global_step'],
     
     t['char_embedding'],
     t['cnn_0_kernel'], t['cnn_1_kernel'], t['cnn_2_kernel'], t['cnn_3_kernel'],
        t['cnn_4_kernel'], t['cnn_5_kernel'], t['cnn_6_kernel'], t['cnn_7_kernel'],
     t['cnn_0_b'], t['cnn_1_b'], t['cnn_2_b'], t['cnn_3_b'],
        t['cnn_4_b'], t['cnn_5_b'], t['cnn_6_b'], t['cnn_7_b'],

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
                                     'global_step:0',

                                    'char_embedding/W:0',
                                    
                                    #'char_embedding/cnn_0/kernel:0',
                                    #'char_embedding/cnn_0/b:0',
																] + cnn_vars                                                                     , name='')

    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

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

def _SentencePerplexity(dataset_file, vocab):
  dataset = data_utils.LM1BDataset(dataset_file, vocab)
  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)
  current_step = t['global_step'].eval(session=sess)
  sys.stderr.write('Loaded step %d.\n' % current_step)
  
  
  data_gen = dataset.get_batch(BATCH_SIZE, NUM_TIMESTEPS, forever=False)
  sys.stderr.write('Loaded data gen\n')
  total_sum_num = 0.0
  sum_num = 0.0
  total_sum_den = 0.0
  sum_den = 0.0
  sentence_id = None
  t0 = time.time()
  perps = []
  for i, (inputs, char_inputs, sentence_ids, targets, weights) in enumerate(data_gen):
    next_sentence_id = sentence_ids[0][0]
    if sentence_id is None:
      sentence_id = next_sentence_id
    # We hit a new sentence. Record this one and reset counters
    if next_sentence_id != sentence_id:
      perplexity = np.exp(sum_num / sum_den)
      print('{}\t{}\t{}'.format(perplexity, sentence_id, '_'.join(str(p) for p in perps)))
      sum_num = sum_den = 0.0
      perps = []
      sentence_id = next_sentence_id
      if (sentence_id % 5) == 0:
        t1 = time.time()
        sys.stderr.write('Starting sentence {} (t={:.1f}s)\n'.format(sentence_id, t1-t0))
        ppx = np.exp(total_sum_num/total_sum_den)
        sys.stderr.write('Running avg. perplexity: {:.3f}\n'.format(ppx))
        t0 = t1
      
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = char_inputs
    log_perp = sess.run(t['log_perplexity_out'], feed_dict=input_dict)

    if np.isnan(log_perp):
      sys.stderr.error('log_perplexity is Nan.\n')
    else:
      num = log_perp * weights.mean()
      sum_num += num
      total_sum_num += num
      den = weights.mean()
      sum_den += den
      total_sum_den += den
      perps.append(log_perp)

    if sentence_id > FLAGS.max_eval_steps:
      break
  ppx = np.exp(total_sum_num/total_sum_den)
  # print "Final perplexity: {}".format(ppx)

def _DumpNextWords(prefix_file, vocab):
  # Hack for convenience
  filemode = prefix_file.endswith('.txt') or ' ' not in prefix_file
  if filemode:
    lines = open(prefix_file)
  else:
    lines = [prefix_file]
  sess, t = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)
  for line in lines:
    prefix_words = line.strip()
    print(prefix_words)
    targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

    if prefix_words.find('<S>') != 0:
      prefix_words = '<S> ' + prefix_words

    prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
    prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
    
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
   
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

      if not samples:
        # We're done feeding in the prefix. It's time to get the predicted next words.
        indices = _SoftmaxTopIndices(softmax[0], FLAGS.n_top_words)
        for i in indices:
          print("{}\t{}".format(vocab.id_to_word(i), softmax[0][i]))
        break
  if filemode:
    lines.close()

def _SoftmaxTopIndices(softmax, n):
  top_indices = np.argpartition(softmax, -n)[-n:]
  top_indices_sorted = top_indices[np.argsort(softmax[top_indices])][::-1]
  return top_indices_sorted
  for i in top_indices_sorted:
    w = vocab.id_to_word(i)
    p = softmax[i]
    # TODO: Should probably dump to a file. Also, would probably be nice to be
    # able to specify multiple prefixes per run. Cause there's a pretty high
    # overhead on loading the checkpoint data etc.
    sys.stderr.write('{}\t{}\n'.format(w, p))


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

      # (This actually isn't necessary if we're still running through the prefix)
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

def _DumpCharEmbedding(vocab):
  assert FLAGS.save_dir, 'Must specify FLAGS.save_dir for dump_emb.'
  sess, t  = _LoadModel(FLAGS.pbtxt, FLAGS.ckpt)
  emb = sess.run(t['char_embedding'])
  fname = FLAGS.save_dir + '/char_embeddings.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, emb)

  # While we're here, let's also dump the cnn filters!
  for filter_id in range(7):
    filter_width = filter_id+1
    # print()"Dumping kernel with width={}".format(filter_width)
    emb_kernel = sess.run(t['cnn_{}_kernel'.format(filter_id)])
    bias = sess.run(t['cnn_{}_b'.format(filter_id)])
    # Width 7 is a special case, because it's spread over 2 variables.
    # because of reasons?
    if filter_id == 6:
      emb_kernel2 = sess.run(t['cnn_7_kernel'])
      bias2 = sess.run(t['cnn_7_b'])
      # (7,16,1,1024) + (7,16,1,2048)
      emb_kernel = np.concatenate( (emb_kernel, emb_kernel2), 3)
      bias = np.concatenate( (bias, bias2), 3)

    # Want to transform each kernel from 16 x w to 256 x w (i.e. bypass char embs)
    # emb_kernel shape is (width, 16, 1, n_filters). Get rid of the dummy dimen.
    emb_kernel = np.reshape(emb_kernel, (filter_width, 16, -1))
    kernel = np.dot(emb, emb_kernel)
    # Shape of final kernel: 256 x width x n_filters
    fname_kernel = FLAGS.save_dir + '/filters/kernel{}.npy'.format(filter_width)
    with tf.gfile.Open(fname_kernel, mode='w') as f:
      np.save(f, kernel)
    fname_bias = FLAGS.save_dir + '/filters/b{}.npy'.format(filter_width)
    with tf.gfile.Open(fname_bias, mode='w') as f:
      np.save(f, bias)


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
  
  # Doing this for all words is hellllla slow. Like over a day.
  if WORD_DUMP_LIMIT != -1:
    wordlimit = WORD_DUMP_LIMIT
  else:
    wordlimit = vocab.size
  print(wordlimit)

  softmax_weights = sess.run(t['softmax_weights'])
  fname = FLAGS.save_dir + '/embeddings_softmax.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, softmax_weights)
  sys.stderr.write('Finished softmax weights\n')

  all_embs = np.zeros([wordlimit, 1024])
  for i in range(wordlimit):
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = (
          vocab.word_char_ids[i].reshape([-1, 1, MAX_WORD_LEN]))
    embs = sess.run(t['all_embs'], input_dict)
    all_embs[i, :] = embs
    sys.stderr.write('Finished word embedding %d/%d\n' % (i, wordlimit))

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

    fname = os.path.join(FLAGS.save_dir, 'lstm_emb_step_%d.npy' % i)
    with tf.gfile.Open(fname, mode='w') as f:
      np.save(f, lstm_emb)
    sys.stderr.write('LSTM embedding step %d file saved\n' % i)


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

  # (New modes)
  elif FLAGS.mode == 'dump_char_emb':
    _DumpCharEmbedding(vocab)
  elif FLAGS.mode == 'next_words':
    _SentencePerplexity(FLAGS.input_data, vocab)
  else:
    raise Exception('Mode not supported.')


if __name__ == '__main__':
  tf.app.run()
