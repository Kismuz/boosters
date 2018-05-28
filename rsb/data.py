from logbook import Logger, StreamHandler
import sys

import tensorflow as tf
import numpy as np
import sparse


class DataCorpus:

    @staticmethod
    def to_log(data, labels):
        transform_op = {
            'cl_id': tf.identity,
            'features': tf.log1p,
            'target_sum': tf.log1p,
            'target_flag': tf.identity,
        }
        transformed_data = {
            key: transform_op[key](tensor) for key, tensor in data.items()
        }
        transformed_labels = {
            key: transform_op[key](tensor) for key, tensor in labels.items()
        }
        return transformed_data, transformed_labels

    @staticmethod
    def to_one_hot(data, labels):

        transformed_labels = {
            'cl_id': labels['cl_id'],
            'target_sum': labels['target_sum'],
            'target_flag': tf.one_hot(labels['target_flag'], depth=2)
        }

        return data, transformed_labels

    def __init__(
            self,
            train_filename,
            test_filename,
            batch_size=16,
            cv_fraction=0.1,
            train_repeats=1,
            time_slice=None,
            features_slice=None,
            full_shuffle=True,
            log_transform=True,
            one_hot_transform=True,
    ):

        self.train_filename = train_filename
        self.test_filename = test_filename
        assert 0 <= cv_fraction < 1
        self.cv_fraction = cv_fraction
        assert batch_size >= 1
        self.batch_size = batch_size
        assert train_repeats >= 1
        self.train_repeats = train_repeats
        self.time_slice = time_slice
        self.features_slice = features_slice
        self.full_shuffle = full_shuffle
        self.log_transform = log_transform
        self.one_hot_transform = one_hot_transform
        self.train_data = None
        self.labels = None
        self.test_data = None

        StreamHandler(sys.stdout).push_application()
        self.log = Logger('DataCorpus')

        # Load data:
        self.train_data_array = sparse.load_npz(self.train_filename).todense()
        self.log.info('loaded train data shape: {}'.format(self.train_data_array.shape))

        self.test_data_array = sparse.load_npz(self.test_filename).todense()
        self.log.info('loaded test data shape: {}'.format(self.test_data_array.shape))

        if self.time_slice is not None:
            assert len(self.time_slice) >= 2
            self.train_data_array = self.train_data_array[:, self.time_slice[0]:self.time_slice[-1], :]
            self.test_data_array = self.test_data_array[:, self.time_slice[0]:self.time_slice[-1], :]
            self.log.info('time-step sliced train data shape: {}'.format(self.train_data_array.shape))
            self.log.info('time-step sliced  test data shape: {}'.format(self.test_data_array.shape))

        # Make initial data arrays(redundant but need struct. for graph def):
        self.init_data_arrays()

        # Graph def:
        self.train_data_pl_dict = {
            key: tf.placeholder(value.dtype, value.shape, key) for key, value in self.train_data.items()
        }
        self.labels_pl_dict = {
            key: tf.placeholder(value.dtype, value.shape, key) for key, value in self.labels.items()
        }
        self.test_data_pl_dict = {
            key: tf.placeholder(value.dtype, value.shape, key) for key, value in self.test_data.items()
        }
        self.full_train_set = tf.data.Dataset.from_tensor_slices((self.train_data_pl_dict, self.labels_pl_dict))

        if self.one_hot_transform:
            self.full_train_set = self.full_train_set.map(self.to_one_hot)

        if self.log_transform:
            self.full_train_set = self.full_train_set.map(self.to_log)

        if self.cv_fraction > 0:
            self.split_size = int(self.train_data['cl_id'].shape[0] * (1- self.cv_fraction))

            self.log.info(
                'cv_fraction: {}, train_size: {}, cv size: {}'.format(
                    cv_fraction, self.split_size, self.train_data['cl_id'].shape[0] - self.split_size
                )
            )
            self.train_set = self.full_train_set.take(self.split_size)

            self.cv_set = self.full_train_set.skip(self.split_size)

            self.cv_set = self.cv_set.shuffle(
                buffer_size=self.train_data['cl_id'].shape[0] - self.split_size
            ).batch(self.batch_size)
            self.cv_iterator = self.cv_set.make_initializable_iterator()
            self.next_cv_batch_op = self.cv_iterator.get_next()

        else:
            self.split_size = int(self.train_data['cl_id'].shape[0] * 0.5)  # only for train buff.size
            self.train_set = self.full_train_set
            self.cv_set = None
            self.cv_iterator = None
            self.next_cv_batch_op = None

        self.train_set = self.train_set.shuffle(
            buffer_size=self.split_size
        ).repeat(
            count=self.train_repeats
        ).batch(
            self.batch_size
        )
        self.train_iterator = self.train_set.make_initializable_iterator()
        self.next_train_batch_op = self.train_iterator.get_next()

        self.test_set = tf.data.Dataset.from_tensor_slices((self.test_data_pl_dict, {}))
        if self.log_transform:
            self.test_set = self.test_set.map(self.to_log)
        self.test_set = self.test_set.batch(self.batch_size)
        self.test_iterator = self.test_set.make_initializable_iterator()
        self.next_test_batch_op = self.test_iterator.get_next()

    def init_data_arrays(self):
        # [Re]shuffle train corpus:
        if self.full_shuffle:
            self.train_data_array = np.random.permutation(self.train_data_array)

        # [Re]define arrays:
        self.train_data = {
            'cl_id': self.train_data_array[:, 0, 0].astype('int'),
            'features': self.train_data_array[..., 1:-2]
        }
        self.labels = {
            'cl_id': self.train_data_array[:, 0, 0].astype('int'),
            'target_sum': self.train_data_array[:, 0, -1],
            'target_flag': self.train_data_array[:, 0, -2].astype('int')
        }
        # Redundant except first call: TODO: maybe shuffle?
        self.test_data = {
            'cl_id': self.test_data_array[:, 0, 0].astype('int'),
            'features': self.test_data_array[..., 1:-2]
        }
        if self.features_slice is not None:
            self.train_data['features'] = self.train_data['features'][
                ...,
                self.features_slice[0]: self.features_slice[-1]
            ]
            self.test_data['features'] = self.test_data['features'][
                ...,
                self.features_slice[0]: self.features_slice[-1]
            ]

    def init_data_sets(self):
        sess = tf.get_default_session()

        feed_dict_train = {self.train_data_pl_dict[key]: self.train_data[key] for key in self.train_data.keys()}
        feed_dict_labels = {self.labels_pl_dict[key]: self.labels[key] for key in self.labels.keys()}
        feed_dict_test = {self.test_data_pl_dict[key]: self.test_data[key] for key in self.test_data.keys()}

        init_ops = [self.train_iterator.initializer, self.test_iterator.initializer]
        if self.cv_iterator is not None:
            init_ops.append(self.cv_iterator.initializer)

        sess.run(
            init_ops,
            feed_dict={
                **feed_dict_train,
                **feed_dict_labels,
                **feed_dict_test
            }
        )

    def reset(self):
        self.init_data_arrays()
        self.init_data_sets()

    def reset_cv(self):
        sess = tf.get_default_session()
        feed_dict_train = {self.train_data_pl_dict[key]: self.train_data[key] for key in self.train_data.keys()}
        feed_dict_labels = {self.labels_pl_dict[key]: self.labels[key] for key in self.labels.keys()}
        sess.run(
            self.cv_iterator.initializer,
            feed_dict={
                **feed_dict_train,
                **feed_dict_labels,
            }
        )




