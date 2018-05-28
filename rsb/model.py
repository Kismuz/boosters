from logbook import Logger, StreamHandler
import sys
import os
import glob
import psutil
from subprocess import PIPE
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

from data import DataCorpus
from networks import conv_1d_casual_attention_encoder
from btgym.algorithms.math_utils import cat_entropy
from btgym.algorithms.nn.layers import noisy_linear

class Model:
    """
    Auto-regressive probabilistic model.
    """
    def __init__(
            self,
            data_batch,
            labels_batch=None,
            keep_prob=tf.ones([],),
            activation=tf.nn.elu,
            name='model',
            reuse=False):
        self.data_batch = data_batch
        self.labels_batch = labels_batch
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            hidden = conv_1d_casual_attention_encoder(
                data_batch['features'],
                keep_prob=keep_prob,
                conv_1d_num_filters=64,
                conv_1d_filter_size=2,
                conv_1d_activation=activation,
                reuse=False,
            )
            hidden = tf.layers.flatten(hidden)

            # print(hidden.shape)

            # hidden = tf.layers.dense(
            #     inputs=hidden,
            #     units=512,
            #     activation=activation,
            # )

            hidden = noisy_linear(
                x=hidden,
                size=64,
                activation_fn=activation,
                name='dense1'
            )
            hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)

            self.predicted_log_sum = tf.layers.dense(
                inputs=hidden,
                units=1,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
            )

            # self.predicted_log_sum = noisy_linear(
            #     x=hidden,
            #     size=1,
            #     activation_fn=activation,
            #     name='log_sum'
            # )
            self.predicted_target_sum = tf.clip_by_value(
                tf.exp(self.predicted_log_sum) - 1,
                clip_value_min=0,
                clip_value_max=1e20
            )

            self.predicted_flag_logits = tf.layers.dense(
                inputs=hidden,
                units=2,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
            )

            # self.predicted_flag_logits = noisy_linear(
            #     x=tf.concat([hidden, self.predicted_log_sum], axis=-1),
            #     size=2,
            #     activation_fn=activation,
            #     name='flag'
            # )
            self.predicted_flag_probs = tf.nn.softmax(self.predicted_flag_logits)

            self.predicted_flag = tf.argmax(
                self.predicted_flag_probs,
                axis=-1
            )

            self.class_entropy = tf.reduce_mean(cat_entropy(self.predicted_flag_logits))

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            if labels_batch is not None:
                self.regress_loss = tf.losses.mean_squared_error(
                    labels=labels_batch['target_sum'][..., None],
                    predictions=self.predicted_log_sum
                )
                self.regress_loss = tf.sqrt(self.regress_loss)

                self.class_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.predicted_flag_logits,
                        labels=labels_batch['target_flag'],
                    )
                )
                self.auc, self.auc_update_op = tf.metrics.auc(
                    labels=labels_batch['target_flag'],
                    predictions=self.predicted_flag_probs,
                    weights=None,
                    num_thresholds=200,
                )

            else:
                self.regress_loss = None
                self.class_loss = None
                self.auc = 0
                self.auc_update_op = None


class Estimator:

    def __init__(
            self,
            data,
            model_config=None,
            opt_learn_rate=1e-4,
            opt_decay_steps=None,
            opt_end_learn_rate=1e-4,
            max_train_steps=10000,
            grad_clip=50,
            class_loss_lambda=1,
            regress_loss_lambda=1,
            entropy_beta=0.01,
            dropout_keep_prob=1,
            validation_period=100,
            summary_period=10,
            log_dir='/log',
            purge_previous_log=1,
            name='Estimator'
    ):
        assert isinstance(data, DataCorpus)
        self.data = data
        self.opt_learn_rate = opt_learn_rate
        self.opt_decay_steps = opt_decay_steps
        self.opt_end_learn_rate = opt_end_learn_rate
        self.max_train_steps = max_train_steps

        self.grad_clip = grad_clip
        self.class_loss_lambda = class_loss_lambda
        self.regress_loss_lambda = regress_loss_lambda
        self.entropy_beta = entropy_beta
        self.dropout_keep_prob = dropout_keep_prob
        self.validation_period = validation_period
        self.summary_period = summary_period
        self.log_dir = log_dir
        self.purge_previous_log = purge_previous_log

        self.name = name

        StreamHandler(sys.stdout).push_application()
        self.log = Logger(self.name)

        # Log_dir:
        if os.path.exists(self.log_dir):
            # Remove previous log files and saved model if opted:
            if self.purge_previous_log > 0:
                confirm = 'y'
                if self.purge_previous_log < 2:
                    confirm = input('<{}> already exists. Override[y/n]? '.format(self.log_dir))
                if confirm in 'y':
                    files = glob.glob(self.log_dir + '/*')
                    p = psutil.Popen(['rm', '-R', ] + files, stdout=PIPE, stderr=PIPE)
                    self.log.notice('Files in <{}> purged.'.format(self.log_dir))

            else:
                self.log.notice('Appending to <{}>.'.format(self.log_dir))

        else:
            os.makedirs(self.log_dir)
            self.log.notice('<{}> created.'.format(self.log_dir))

        self.keep_prob_pl = tf.placeholder(tf.float32, name='dropout_parameter')

        if model_config is None:
            self.model_class_ref = Model
            self.model_kwargs = {
                'activation': tf.nn.elu,
                'name': self.name + '/model',
            }

        else:
            self.model_class_ref = model_config['class_ref']
            self.model_kwargs = model_config['kwargs']

        self.train_model = self.model_class_ref(
            data_batch=self.data.next_train_batch_op[0],
            labels_batch=self.data.next_train_batch_op[-1],
            keep_prob=self.keep_prob_pl,
            reuse=False,
            **self.model_kwargs
        )
        self.cv_model = self.model_class_ref(
            data_batch=self.data.next_cv_batch_op[0],
            labels_batch=self.data.next_cv_batch_op[-1],
            keep_prob=self.keep_prob_pl,
            reuse=True,
            **self.model_kwargs
        )
        self.test_model = self.model_class_ref(
            data_batch=self.data.next_test_batch_op[0],
            labels_batch=None,
            keep_prob=self.keep_prob_pl,
            reuse=True,
            **self.model_kwargs
        )
        self.global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(
                0,
                dtype=tf.int32
            ),
            trainable=False
        )
        self.inc_global_step = self.global_step.assign_add(1)
        #self.reset_global_step = self.global_step.assign(0)

        self.local_step = tf.get_variable(
            "local_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(
                0,
                dtype=tf.int32
            ),
            trainable=False
        )
        self.inc_local_step = self.local_step.assign_add(1)
        self.reset_local_step = self.local_step.assign(0)

        self.inc_step = tf.group([self.inc_global_step, self.inc_local_step])

        # self.inc_step = self.inc_global_step

        # Learning rate annealing:
        if self.opt_decay_steps is not None:
            self.learn_rate_decayed = tf.train.polynomial_decay(
                self.opt_learn_rate,
                self.local_step + 1,
                self.opt_decay_steps,
                self.opt_end_learn_rate,
                power=1,
                cycle=False,
            )
        else:
            self.learn_rate_decayed = self.opt_end_learn_rate

        # Loss, train_op:
        self.train_loss = self.class_loss_lambda * self.train_model.class_loss \
            + self.regress_loss_lambda * self.train_model.regress_loss \
            - self.entropy_beta * self.train_model.class_entropy
        try:
            self.cv_loss = self.class_loss_lambda * self.cv_model.class_loss \
                + self.regress_loss_lambda * self.cv_model.regress_loss \
                - self.entropy_beta * self.cv_model.class_entropy

        except TypeError:
            self.cv_loss = 0

        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.train_loss, self.train_model.var_list),
            grad_clip
        )
        self.grads_and_vars = list(zip(self.grads, self.train_model.var_list))

        self.optimizer = tf.train.AdamOptimizer(self.learn_rate_decayed, epsilon=1e-5)

        self.optimize_op = self.optimizer.apply_gradients(self.grads_and_vars)

        self.train_op = tf.group([self.optimize_op, self.inc_step, self.train_model.auc_update_op])

        # Summaries:
        train_summaries = [
            tf.summary.scalar("train_grad_global_norm", tf.global_norm(self.grads)),
            tf.summary.scalar("train_total_loss", self.train_loss),
            tf.summary.scalar("train_auc", self.train_model.auc),
            tf.summary.scalar("train_regression_loss", self.train_model.regress_loss),
            tf.summary.scalar("train_class_loss", self.train_model.class_loss),
            tf.summary.scalar('train_class_entropy', self.train_model.class_entropy),
        ]
        cv_summaries = [
            tf.summary.scalar("cv_total_loss", self.cv_loss),
            tf.summary.scalar("cv_class_loss", self.cv_model.class_loss),
            tf.summary.scalar("cv_regression_loss", self.cv_model.regress_loss),
            tf.summary.scalar("cv_auc", self.cv_model.auc_update_op),
            #tf.summary.scalar('cv_class_entropy', self.cv_model.class_entropy),
        ]
        self.train_summaries_op = tf.summary.merge(train_summaries)
        self.cv_summaries_op = tf.summary.merge(cv_summaries)

        self.summary_writer = tf.summary.FileWriter(self.log_dir)

    def initialise(self, sess):
        sess.run(
            [tf.variables_initializer(self.train_model.var_list), tf.local_variables_initializer()]
        )

    def train(self, sess, run_cv=True):
        self.data.reset()
        sess.run(self.reset_local_step)
        step = 0
        while step <= self.max_train_steps:
            try:
                step = sess.run(self.local_step)
                write_summaries = step % self.summary_period == 0

                fetches = [self.train_op, self.train_loss]

                if write_summaries:
                    fetches.append(self.train_summaries_op)

                fetched = sess.run(fetches, {self.keep_prob_pl: self.dropout_keep_prob})
                loss = fetched[1]

                if step % 500 == 0:
                    self.log.info('step: {}, train_total_loss: {:0.3f}'.format(step, loss))

                if write_summaries:
                    self.summary_writer.add_summary(tf.Summary.FromString(fetched[-1]), sess.run(self.global_step))
                    self.summary_writer.flush()

                if step % self.validation_period == 0 and run_cv:
                    self.validate(sess)

            except tf.errors.OutOfRangeError:
                self.log.info('max_train_set_repeats reached.')
                break

        # Final validation:
        if run_cv:
            self.validate(sess)
        self.log.info('train fold finished at step: {}'.format(sess.run(self.local_step)))

    def validate(self, sess):
        self.data.reset_cv()
        step = sess.run(self.global_step)
        class_loss = []
        regress_loss = []
        while True:
            try:
                fetches = [
                    self.cv_summaries_op,
                    self.cv_model.class_loss,
                    self.cv_model.regress_loss,
                ]

                fetched = sess.run(fetches, {self.keep_prob_pl: 1.0})
                class_loss.append(fetched[1])
                regress_loss.append(fetched[2])

                self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), step)
                self.summary_writer.flush()
                step += 1

            except tf.errors.OutOfRangeError:
                break

        self.log.info(
            'c_val. at {} train step, class_loss: {:0.3f}, regress_loss: {:0.3f}'.
            format(sess.run(self.local_step), np.asarray(class_loss).mean(), np.asarray(regress_loss).mean())
        )

    def test(self, sess):
        self.data.reset()
        cl_id = []
        target_flag = []
        target_sum = []

        while True:
            try:
                fetches = [
                    self.test_model.data_batch,
                    self.test_model.predicted_target_sum,
                    self.test_model.predicted_flag,
                ]

                fetched = sess.run(fetches, {self.keep_prob_pl: 1.0})

                cl_id.append(fetched[0]['cl_id'])
                target_sum.append(fetched[1])
                target_flag.append(fetched[2])

            except tf.errors.OutOfRangeError:
                return cl_id, target_sum, target_flag

    def test_and_save(self, sess):
        self.log.info('Processing test data...')

        cl_id, target_sum, target_flag = self.test(sess)

        predictions = {
            'cl_id': np.concatenate(cl_id),
            'target_sum': np.concatenate(target_sum, axis=0)[:, 0],
            'target_flag': np.concatenate(target_flag),
        }

        # Threshold values:
        predictions['target_sum'][predictions['target_sum'] < 1000] = 0

        task_1_filename = './data/submission_task_1_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))
        task_2_filename = './data/submission_task_2_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))

        self.log.info('Saving results in: {}, {}...'.format(task_1_filename, task_2_filename))

        self.task_1_df = pd.DataFrame(
            np.stack([predictions['cl_id'], predictions['target_flag']], axis=-1),
            index = predictions['cl_id'],
            columns=['_ID_', '_VAL_']
        )
        self.task_2_df = pd.DataFrame(
            np.stack([predictions['cl_id'], predictions['target_sum']], axis=-1),
            index = predictions['cl_id'],
            columns=['_ID_', '_VAL_']
        )

        self.task_1_df.to_csv(task_1_filename, index=False)
        self.task_2_df.to_csv(task_2_filename, index=False)
        self.log.info('Done.')







