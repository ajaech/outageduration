import numpy as np
import tensorflow as tf
from factorcell import FactorCell
from tensorflow.python.util import nest


def LogGamma(x, k, theta):
  return (k - 1.0) * tf.log(x) - x / theta - k * tf.log(theta) - tf.lgamma(k)


class LiveModel(object):
    """This class holds the Tensorflow model for the real-time predictions."""
    
    def mybody(self, i, state, out_tracker, attn_tracker, attn_tracker2):
        """This is the body of the prediction loop."""

        # concatenate the state with the features        
        this_state = tf.expand_dims(state[i, :], 0)
        ith_report_time = self.report_times[i]
        elapsed_time = tf.reshape(ith_report_time, [1, 1])
        state_feature_vec = tf.concat(
            [this_state, self.input_features_, elapsed_time], axis=1)
        
        # get the representations for the ith review
        review_states = self.outputs[i, :, :]

        def DoAttn():
            # compute the attention query
            attn_query = tf.layers.dense(
                state_feature_vec, self.params.cell_size * 2, activation=tf.nn.relu)
            attn_info = tf.layers.dense(review_states, self.params.cell_size * 2,
                                        activation=tf.nn.relu)

            # compute the attention weights
            logits = tf.matmul(attn_info, attn_query, transpose_b=True)
            attn_weights = tf.transpose(tf.nn.softmax(
                tf.transpose(logits[1:self.seq_lens[i]-1, :])))
        
            # get a weighted combination of the review hidden states
            review_representation = tf.expand_dims(
                tf.reduce_sum(tf.multiply(attn_weights, review_states[1:self.seq_lens[i]-1, :]), 0), 0)
            return attn_weights, review_representation

        attn_weights, review_representation = DoAttn()

        # pad the attention weights
        padding = tf.zeros([self.params.max_len - self.seq_lens[i] + 2])
        padding = tf.expand_dims(padding, 1)
        attn_weights = tf.concat([attn_weights, padding], 0)
        attn_tracker = tf.concat([attn_tracker, tf.transpose(attn_weights)], axis=0)
        
        reviews = [review_representation]
        if hasattr(self.params, 'do_second_attn') and self.params.do_second_attn:
            attn_weights2, review_representation2 = DoAttn()
            reviews.append(review_representation2)
            pad_attn_weights2 = tf.concat([attn_weights2, padding], 0)
            attn_tracker2 = tf.concat([attn_tracker2, tf.transpose(pad_attn_weights2)], 
                                      axis=0)
            attn_weights += attn_weights2  # just add the attention masks together
        else:
            attn_tracker2 = attn_tracker  # just copy the other one

        # form the input to the GRU
        gru_input = tf.concat(reviews + [self.input_features_, elapsed_time], axis=1)
        gru_output, new_state = self.update_cell(gru_input, this_state)
                
        state_tracker = tf.concat([state, new_state], axis=0)
        out_tracker = tf.concat([out_tracker, gru_output], axis=0)
        
        i = tf.add(i, 1)  # increment the loop
        
        return i, state_tracker, out_tracker, attn_tracker, attn_tracker2
    
    def __init__(self, params, word_embedder):
        self.params = params
        self.word_embedder = word_embedder
        self.use_layer_norm = False
        if hasattr(params, 'use_layer_norm'):
          self.use_layer_norm = params.use_layer_norm
        
        self.reports = tf.placeholder(tf.int32, [None, params.max_len], name='reports')
        self.report_times = tf.placeholder(tf.float32, [None], name='report_times')
        self.remaining_times = tf.placeholder(tf.float32, [None], name='remaining_times')
        self.seq_lens = tf.placeholder(tf.int32, [None], name='seq_lens')
        
        self.mask = tf.sequence_mask(self.seq_lens, params.max_len)
        self.dropout = tf.placeholder_with_default(1.0, (), name='keep_prob')
        
        num_features = params.num_features
        self.input_features = tf.placeholder(tf.float32, [num_features],
                                             name='input_features')
        self.input_features_ = tf.expand_dims(self.input_features, 0)
        
        # process the reports
        inputs = self.word_embedder.GetEmbeddings(self.reports)
        with tf.variable_scope('fw_cell'):
            fw_cell = GRUCell(params.cell_size, layer_norm=self.use_layer_norm)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                fw_cell, state_keep_prob=self.dropout, variational_recurrent=True,
                input_size=self.word_embedder.embedding_dims, dtype=tf.float32)
        with tf.variable_scope('bw_cell'):
            bw_cell = GRUCell(params.cell_size, layer_norm=self.use_layer_norm)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                bw_cell, state_keep_prob=self.dropout, variational_recurrent=True,
                input_size=self.word_embedder.embedding_dims, dtype=tf.float32)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, self.seq_lens, dtype=tf.float32)
        self.outputs = tf.concat(outputs, 2)

        update_cell_size = params.cell_size
        if hasattr(params, 'cell_size2'):
            update_cell_size = params.cell_size2
        with tf.variable_scope('update_cell'):
            update_cell_input_size = 1 + len(params.features) + 2 * params.cell_size
            self.update_cell = GRUCell(update_cell_size, layer_norm=self.use_layer_norm)
            
        self.i = tf.constant(0)
        self.initial_state = tf.get_variable('initial_state', [1, self.update_cell.state_size])
        attn0 = tf.zeros([1, params.max_len])
        attn2 = tf.zeros([1, params.max_len])
        out0 = tf.layers.dense(self.input_features_, self.update_cell.state_size,
                               activation=tf.nn.tanh)
        
        cond = lambda i, a, b, c, d: i < tf.shape(self.reports)[0]
        self.body = lambda i, state, out, attn, attn2: self.mybody(i, state, out, attn, attn2)

        loop_vars = [self.i, self.initial_state, out0, attn0, attn2]
        shape_invariants=[
            self.i.get_shape(),                                  # i 
            tf.TensorShape([None, self.update_cell.state_size]), # state tracker
            tf.TensorShape([None, self.update_cell.state_size]), # output tracker
            tf.TensorShape([None, params.max_len]),              # attn weights
            tf.TensorShape([None, params.max_len])]              # attn2 weights
        result = tf.while_loop(cond, self.body, loop_vars, shape_invariants)
        (_, _, self.loop_out, self.attn_weights, self.attn_weights2) = result
        
        self.k = tf.squeeze(tf.layers.dense(self.loop_out, 1, activation=tf.nn.softplus,
                                            bias_initializer=tf.ones_initializer()))
        self.theta = tf.squeeze(tf.layers.dense(self.loop_out, 1, activation=tf.nn.softplus,
                                                bias_initializer=tf.ones_initializer()))       
                
        duration = tf.clip_by_value(self.remaining_times, 0.002, 36.0)  # trying to prevent nans
        self.likelihood = LogGamma(duration, self.k, self.theta)
        self.total_cost = -tf.reduce_mean(tf.clip_by_value(self.likelihood, -15.0, np.log(1.5)))
        
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.total_cost)


class MultiTaskModel(object):

  def GetAttnHead(self, params, outputs, states):
    # do attention over the outputs
    mask = tf.sequence_mask(self.seq_lens, params.max_len)
        
    word_level_attention = tf.get_variable('word_attn', [2 * params.cell_size, 1])
    reshaped_outputs = tf.reshape(
        outputs, [-1, 2 * params.cell_size])
    reshaped_word_scores = tf.matmul(reshaped_outputs, word_level_attention, name='word_scores')
    word_logits = tf.reshape(reshaped_word_scores, [-1, params.max_len])
    masked_word_logits = tf.where(mask, word_logits, -20.0 * tf.ones_like(word_logits))
    word_scores = tf.nn.softmax(masked_word_logits)
    condensed = tf.reduce_sum(
        tf.multiply(outputs, tf.expand_dims(word_scores, 2)), 1)
    fully_condensed = tf.reduce_sum(condensed, 0)
    return fully_condensed, word_scores

  def RepairLogUnderstanding(self, params, context_vec):
    self.words = tf.placeholder(tf.int32, [1, params.max_len], name='word_ids')
    self.log_time = tf.placeholder(tf.float32, [None], name='log_time')
    self.seq_lens = tf.placeholder(tf.int32, [None], name='seq_lens')
        
    inputs = self.word_embedder.GetEmbeddings(self.words)
    
    with tf.variable_scope('fw_cell'):
      fw_cell = FactorCell(params.cell_size, self.word_embedder.embedding_dims,
                           context_vec, params.use_mikolov_adaptation,
                           params.use_lowrank_adaptation, rank=params.rank,
                           layer_norm=params.use_layer_norm, 
                           dropout_keep_prob=self.dropout)
    with tf.variable_scope('bw_cell'):
      bw_cell = FactorCell(params.cell_size, self.word_embedder.embedding_dims,
                           context_vec, params.use_mikolov_adaptation,
                           params.use_lowrank_adaptation, rank=params.rank,
                           layer_norm=params.use_layer_norm, 
                           dropout_keep_prob=self.dropout)
                           
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, self.seq_lens,
                                                      dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    states = tf.concat([states[0].h, states[1].h], 1)

    with tf.variable_scope('cause'):
      attn1, self.word_scores1 = self.GetAttnHead(params, outputs, states)
    with tf.variable_scope('duration'):
      attn2, self.word_scores2 = self.GetAttnHead(params, outputs, states)

    self.duration_head = attn2
    return attn1, attn2


  def __init__(self, params, word_embedder):
    num_features = params.num_features
    self.input_features = tf.placeholder(tf.float32, [num_features], 
                                         name='input_features')
    input_features = tf.expand_dims(self.input_features, 0)
    self.cause = tf.placeholder(tf.int32, [1], name='cause')
    self.duration = tf.placeholder(tf.float32, [1], name='duration')
    self.elapsed_time = tf.placeholder(tf.float32, [1], name='elapsed_time')
    time_remaining = self.duration - self.elapsed_time
    duration = tf.clip_by_value(time_remaining, 0.01, 2000)  # trying to prevent nans

    self.dropout = tf.placeholder_with_default(1.0, (), name='keep_prob')
        
    if params.use_attn:
      self.word_embedder = word_embedder
      self.attn1, self.attn2 = self.RepairLogUnderstanding(params, input_features)
      num_features += 2 * params.cell_size
      input_features1 = tf.concat([input_features, tf.expand_dims(self.attn1, 0)], 1)
      input_features2 = tf.concat([input_features, tf.expand_dims(self.attn2, 0)], 1)
    else:
      input_features1, input_features2 = input_features, input_features

    with tf.variable_scope('cause'):
      hidden_layer = tf.get_variable('hidden_layer',
                                     [num_features, params.hidden_layer_size])
      hidden_bias = 1.0 + tf.get_variable('hidden_bias', [params.hidden_layer_size])
      h1 = tf.nn.relu(tf.matmul(input_features1, hidden_layer) + hidden_bias)

      self._cause_predictor = tf.get_variable(
        'cause_predictor', [params.hidden_layer_size, params.num_causes])
      self._cause_bias = tf.get_variable('cause_bias', [params.num_causes])
      
      cause_logits = tf.matmul(h1, self._cause_predictor) + self._cause_bias
      self.cause_prob = tf.nn.softmax(cause_logits)
      self.cause_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.cause, logits=cause_logits)

    with tf.variable_scope('duration'):
      if hasattr(params, 'cause_only') and params.cause_only:
        cause_one_hot = tf.one_hot(self.cause, params.num_causes)
        input_features2 = cause_one_hot
        num_features = params.num_causes
      if hasattr(params, 'cause_oracle') and params.cause_oracle:
        cause_one_hot = tf.one_hot(self.cause, params.num_causes)
        input_features2 = tf.concat([input_features2, cause_one_hot], 1)
        num_features += params.num_causes
      if hasattr(params, 'cause_super_oracle') and params.cause_super_oracle:
        cause_probs = tf.stop_gradient(self.cause_prob)
        input_features2 = tf.concat([input_features2, cause_probs], 1)
        num_features += params.num_causes

      h2 = tf.layers.dense(input_features2, params.hidden_layer_size, 
                           activation=tf.nn.relu)
      h3 = tf.layers.dense(h2, params.hidden_layer_size, activation=tf.nn.relu)

      alpha_param = tf.get_variable('alpha_param', [params.hidden_layer_size, 1])
      alpha_bias = tf.get_variable('alpha_bias', [1]) + 1.0
      self.k = tf.matmul(h3, alpha_param) + alpha_bias
      self.k = tf.nn.softplus(self.k)  # the k param should be > 0

      beta_param = tf.get_variable('beta_param', [params.hidden_layer_size, 1])
      beta_bias = tf.get_variable('beta_bias', [1]) + 1.0
      
      self.theta = tf.nn.softplus(tf.matmul(h3, beta_param) + beta_bias)
      
      self.duration_error = -tf.squeeze(LogGamma(duration, self.k, self.theta))

    alpha = params.alpha
    self.total_cost = alpha * self.duration_error + (1.0 - alpha) * self.cause_loss
        
    optimizer = tf.train.AdamOptimizer(0.001)
    self.train_op = optimizer.minimize(self.total_cost)
        
  def GetMean(self, k, theta):
    return k * theta



class GRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               layer_norm=False):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None
    self.layer_norm = layer_norm

    if self.layer_norm:
        self.gammas = {}
        self.betas = {}
        for gate in ['r', 'u', 'c']:
          self.gammas[gate] = tf.get_variable(
              'gamma_' + gate, shape=[num_units], initializer=tf.constant_initializer(1.0))
          self.betas[gate] = tf.get_variable(
              'beta_' + gate, shape=[num_units], initializer=tf.constant_initializer(0.0))

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = tf.constant_initializer(1.0, dtype=inputs.dtype)
      with tf.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    def Norm(inputs, gamma, beta): 
      m, v = tf.nn.moments(inputs, [1], keep_dims=True)
      normalized_input = (inputs - m) / tf.sqrt(v + 1e-5)
      return normalized_input * gamma + beta

    gate_out = self._gate_linear([inputs, state])
    pre_r, pre_u = tf.split(value=gate_out, num_or_size_splits=2, axis=1)
    
    if self.layer_norm:
      pre_r, pre_u = (Norm(pre_r, self.gammas['r'], self.betas['r']),
                      Norm(pre_u, self.gammas['u'], self.gammas['u']))
    r, u = tf.sigmoid(pre_r), tf.sigmoid(pre_u)

    r_state = r * state
    if self._candidate_linear is None:
      with tf.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    pre_c = self._candidate_linear([inputs, r_state])
    if self.layer_norm:
      pre_c = Norm(pre_c, self.gammas['c'], self.betas['c'])
    c = self._activation(pre_c)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      self._weights = tf.get_variable(
          'kernel', [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with tf.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
          self._biases = tf.get_variable(
              'bias', [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = tf.matmul(args[0], self._weights)
    else:
      res = tf.matmul(tf.concat(args, 1), self._weights)
    if self._build_bias:
      res = tf.nn.bias_add(res, self._biases)
    return res
