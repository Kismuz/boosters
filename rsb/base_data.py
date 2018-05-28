from logbook import Logger, StreamHandler
import sys

import pandas as pd
import numpy as np
import sparse
import requests

import iso4217parse

import datetime
import copy


class BaseDataSet():
    """
    Basic data preparation:  csv -> pd.dataframe -> sparse matrix
    """

    def __init__(self, filename, path='data/', rates_filename='rub_exchange_rates.csv'):
        StreamHandler(sys.stdout).push_application()

        self.path = path
        self.filename = filename
        self.rates_filename = rates_filename

        self.log = Logger('BaseData/' + self.filename)

        self.mcc_codes_table = pd.read_html(
            requests.get(
                'https://mcc-codes.ru/code',
                headers={'User-agent': 'Mozilla/5.0'}
            ).text,
            converters={'MCC': str}
        )[0]

        self.mcc_map = self.mcc_codes_table[[u'MCC', u'Группа']].set_index('MCC').to_dict()[u'Группа']

        self.rates = pd.read_csv(self.path + self.rates_filename)

        self.data = None
        self.aggregated_data = None
        self.sparse_array_data = None
        self.mcc_groups = None
        self.grouped_by_id = None

    @staticmethod
    def extract_month(s):
        day, month, year = s.split('/')
        month = int(month)
        month += (int(year) - 2016) * 12
        return month

    def _to_rur(self, obj):
        if obj['currency'] != 810:
            # Get currency:
            curr = iso4217parse.parse(int(obj['currency']))[0]

            if curr is not None:
                curr_alpha3 = curr[0]

            else:
                raise ValueError('Unknown currency code: {}'.format(obj['currency']))

            curr_amount = obj['amount']
            # Convert:
            try:
                obj['amount'] = curr_amount / self.rates.loc[curr_alpha3]['rate']
                # print('Converted {} {} to {} RUR'.format(curr_amount, curr_alpha3, obj['amount']))
            except KeyError as e:
                # Ex. rate not found, fake 1000 rub on tis transaction:
                self.log.warning('Rate missing for {} {}, substituted by 1000 RUR'.format(obj['amount'], curr_alpha3))
                obj['amount'] = 1000

        obj['currency'] = 643
        return obj

    @staticmethod
    def _day_to_int(obj):
        obj['rel_day'] = obj['rel_day'].days
        return obj

    def load_csv(self, truncate=None, **kwargs):
        self.data = pd.read_csv(self.path + self.filename)
        self.log.info('Loaded data shape: {}'.format(self.data.shape))

        if truncate is not None:
            assert truncate < self.data.shape[0],\
                'Truncation index {} is bigger than data size {}'.format(truncate, self.data.shape[0])
            self.data = self.data[0:truncate]
            self.log.info('Data truncated down to first {} rows'.format(truncate))

        self.data['PERIOD'] = pd.to_datetime(
            self.data['PERIOD'],
            format = '%m/%d/%Y'
        )
        self.data['TRDATETIME'] = pd.to_datetime(
            self.data['TRDATETIME'],
            format = '%d%b%y:%X'
        )
        self.data['rel_day'] = self.data.apply(lambda zero: datetime.timedelta(), axis=1)
        self.data['channel_type'] = self.data['channel_type'].fillna('0').apply(lambda s : int(s[-1]))
        self.data['mcc_group'] = self.data['MCC'].astype(str).map(self.mcc_map)

        self.mcc_groups = list(set(self.data.mcc_group.unique()))

        self.grouped_by_id = None

    def to_relative_days(self, **kwargs):
        self.grouped_by_id = self.data.groupby('cl_id')

        for cl_id, group in self.grouped_by_id:
            start_tstamp = copy.deepcopy(group['TRDATETIME'].min())
            idx = copy.deepcopy(group.index)
            self.data.loc[idx, 'rel_day'] = (self.data.loc[idx, 'TRDATETIME'] - start_tstamp)

        self.data = self.data.transform(self._day_to_int, axis=1)

    def to_rur(self, **kwargs):
        self.data = self.data.transform(self._to_rur, axis=1)

    def aggregate_by_daily_sums(self, **kwargs):
        self.grouped_by_id = self.data.groupby('cl_id')
        current_idx = 0

        col_names = ['cl_id', 'rel_day', 'sum_POS']

        col_names += ['sum_{}'.format(group) for group in self.mcc_groups]

        col_names += ['target_flag', 'target_sum']

        aggr_data = pd.DataFrame(
            index=None,
            columns=col_names
        ).fillna(0)

        for cl_id, cl_group in self.grouped_by_id:
            id_by_day = cl_group.groupby('rel_day')

            for day, ts_group in id_by_day:
                day_by_mcc = ts_group.groupby('MCC')
                day_sum_pos = 0
                s = pd.Series(
                    name=current_idx,
                    index=col_names
                ).fillna(0)
                s['cl_id'] = cl_id
                s['rel_day'] = day

                try:
                    s['target_flag'] = ts_group.target_flag.all()
                    s['target_sum'] = ts_group.target_sum.mean()

                except AttributeError:
                    s['target_flag'] = float('NaN')
                    s['target_sum'] = float('NaN')

                for mcc_id, ts in day_by_mcc:
                    day_sum_pos += ts[ts.trx_category == 'POS']['amount'].sum()
                    s['sum_{}'.format(ts.mcc_group.values[0])] += ts['amount'].sum()

                s['sum_POS'] = day_sum_pos
                aggr_data = aggr_data.append(s)
                current_idx += 1

        return aggr_data

    def save_csv(self, data, file_prefix='_', **kwargs):
        data.to_csv(self.path + file_prefix + self.filename, index=True, index_label=False)

    def save_sparse(self, data, file_prefix='sparse_array_', **kwargs):
        assert isinstance(data, sparse.COO), 'Expected sparse.COO data type, got: {}'.format(type(data))
        sparse.save_npz(self.path + file_prefix + self.filename[:-4] + '.npz', data)

    def to_sparse_array(self, min_days=90, max_days=None, rebase=True, **kwargs):
        if self.aggregated_data is not None:
            clients = []
            if max_days is None:
                max_days = int(self.aggregated_data['rel_day'].max()) + 1  # infer from data
                if max_days < 91:
                    max_days = 91
                self.log.info('max_days inferred from data: {}'.format(max_days))

            if rebase:
                assert min_days < max_days, 'It is not possible to rebase with min_days={}, max_days={}'.format(
                    min_days,
                    max_days
                )
                self.log.info('Rebasing observations with min_days={}'.format(min_days))

            num_col = int(self.aggregated_data.shape[-1]) - 1
            client_shape = [1, max_days, num_col]

            agg_by_id = self.aggregated_data.groupby('cl_id')

            for cl_id, cl_group in agg_by_id:
                client_array = np.zeros(client_shape)
                client_values = cl_group.values
                client_index = client_values[:, 1].astype(int)  # second column - rel_day values --> to 0_dim values

                if rebase:
                    # Rebase but allow no less than 90 days observation period:
                    client_max_day = client_index.max()
                    if client_max_day < min_days:
                        client_max_day = min_days
                    rebase_index = max_days - client_max_day - 1
                    client_index += rebase_index

                client_array[:, client_index, :] = client_values[:, 1:]  # remove cl_id (gone to new dim) and rel_day

                # Fill all records for single client:
                client_array[..., 0] = int(cl_id)  # id

                if np.isnan(cl_group.target_sum).any():
                    client_array[..., -1] = float('NaN')
                    client_array[..., -2] = float('NaN')

                else:
                    client_array[..., -1] = cl_group.target_sum.mean()
                    client_array[..., -2] = cl_group.target_flag.all()

                # Save as sparse 3d array:
                clients.append(sparse.COO(client_array.astype('float32')))

            full_array_sparse = sparse.concatenate(clients, axis=0)

        else:
            self.log.warning('No aggregated data found, call .aggregate_by_daily_sums() method first.')
            full_array_sparse = None

        return full_array_sparse

    def process(self, level=2, **kwargs):

        if self.data is None:
            self.log.info('Loading data...')

        else:
            self.log.info('Reloading data...')

        self.load_csv(**kwargs)

        self.log.info('Converting to RUB...')
        self.to_rur(**kwargs)

        self.log.info('Calculating relative days...')
        self.to_relative_days(**kwargs)

        if level > 0:
            self.log.info('Aggregating by daily sums...')
            self.aggregated_data = self.aggregate_by_daily_sums(**kwargs)

            self.log.info('Saving aggregated csv file...')
            self.save_csv(self.aggregated_data, file_prefix='aggr_')

            if level > 1:
                self.log.info('Making sparse data array...')
                self.sparse_array_data = self.to_sparse_array(**kwargs)

                self.log.info('Saving sparse array...')
                self.save_sparse(self.sparse_array_data)

        self.log.info('Done.')



