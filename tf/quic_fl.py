import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from builder import configure_aggregator

constants_folder = Path(__file__).parent / 'quic_fl_constants'

from distributed_dp import compression_utils


def quic_fl_roundtrip(input_record,
                      hadamard_seed: tf.Tensor,
                      rand_h_seed: tf.Tensor,
                      sender_table_X,
                      sender_table_p,
                      recv_table,
                      half_table_size,
                      delta, T, h_len, x_len):
    sender_table_X = tf.convert_to_tensor(sender_table_X, tf.int32)
    sender_table_p = tf.convert_to_tensor(sender_table_p, tf.float32)
    recv_table = tf.convert_to_tensor(recv_table, tf.float32)

    """Applies compression to the record as a single concatenated vector."""
    input_vec = compression_utils.flatten_concat(input_record)

    casted_record = tf.cast(input_vec, tf.float32)

    rotated_record = compression_utils.randomized_hadamard_transform(
        casted_record, seed_pair=hadamard_seed)

    x = rotated_record
    d = tf.cast(tf.size(x), x.dtype)
    scale = tf.sqrt(d) / tf.norm(x, ord='euclidean')

    x *= scale

    exact_mask = tf.logical_or(x < -T, T < x)
    exact_vals = x[exact_mask]
    quant_vals = tf.where(exact_mask, tf.cast(0, x.dtype), x / delta)

    # stochastic rounding
    quant_vals = tf.cast(
        tf.math.floor(quant_vals + tf.random.uniform(quant_vals.shape, dtype=quant_vals.dtype)),
        dtype=tf.int32) + half_table_size

    h = tf.random.stateless_uniform(quant_vals.shape, rand_h_seed, minval=0, maxval=h_len, dtype=tf.int32)
    sender_idx = tf.stack([quant_vals, h], axis=1)
    X = tf.gather_nd(sender_table_X, sender_idx)
    p = tf.gather_nd(sender_table_p, sender_idx)

    X += tf.cast(tf.random.uniform(p.shape) < p, X.dtype)
    exact_indices = tf.where(exact_mask)

    # receiver side

    recv_idx = tf.stack([X, h], axis=1)
    rec_x = tf.gather_nd(recv_table, recv_idx)
    rec_x = tf.tensor_scatter_nd_update(rec_x, exact_indices, exact_vals)

    rec_x /= scale

    dequantized_record = rec_x

    unrotated_record = compression_utils.inverse_randomized_hadamard_transform(
        dequantized_record,
        original_dim=tf.size(casted_record),
        seed_pair=hadamard_seed)

    if input_vec.dtype.is_integer:
        uncasted_input_vec = tf.cast(tf.round(unrotated_record), input_vec.dtype)
    else:
        uncasted_input_vec = tf.cast(unrotated_record, input_vec.dtype)

    reconstructed_record = compression_utils.inverse_flatten_concat(
        uncasted_input_vec, input_record)
    return reconstructed_record


def _create_quic_fl_fn(value_type, bits):
    sender_table_X = {int(k): v for k, v in np.load(constants_folder / 'sender_table_X.npz').items()}
    sender_table_p = {int(k): v for k, v in np.load(constants_folder / 'sender_table_p.npz').items()}
    recv_table = {int(k): v for k, v in np.load(constants_folder / 'recv_table.npz').items()}

    with open(constants_folder / 'data.pickle', 'rb') as f:
        data = pickle.load(f)

    sender_table_X = sender_table_X[bits]
    sender_table_p = sender_table_p[bits]
    recv_table = recv_table[bits]
    data = data[bits]

    # delta, T, h_len, x_len = data['delta'], data['T'], data['h_len'], data['x_len']
    half_table_size = (sender_table_X.shape[0] - 1) // 2

    @tff.tf_computation(value_type)
    def eden_fn(record):
        microseconds_per_second = 10 ** 6  # Timestamp returns fractional seconds.
        timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                         tf.int32)
        hadamard_seed = tf.convert_to_tensor([timestamp_microseconds, 0])

        rand_h_seed = tf.convert_to_tensor([timestamp_microseconds * 2, 0])

        return quic_fl_roundtrip(record, hadamard_seed=hadamard_seed, rand_h_seed=rand_h_seed,
                                 sender_table_X=sender_table_X,
                                 sender_table_p=sender_table_p,
                                 recv_table=recv_table,
                                 half_table_size=half_table_size,
                                 **data
                                 )

    return eden_fn


class QuicFLFactory(tff.aggregators.UnweightedAggregationFactory):
    def __init__(self, bits):
        self._bits = bits

    def create(self, value_type):
        if not tff.types.is_structure_of_floats(
                value_type) or not value_type.is_tensor():
            raise ValueError("Expect value_type to be a float tensor, "
                             f"found {value_type}.")

        quic_fl_fn = _create_quic_fl_fn(value_type, self._bits)

        @tff.federated_computation()
        def init_fn():
            return tff.federated_value((), tff.SERVER)

        @tff.federated_computation(init_fn.type_signature.result,
                                   tff.type_at_clients(value_type))
        def next_fn(state, value):
            reconstructed_value = tff.federated_map(quic_fl_fn, value)
            result = tff.federated_sum(reconstructed_value)

            return tff.templates.MeasuredProcessOutput(
                state=state,
                result=result,
                measurements=tff.federated_value((), tff.SERVER))

        return tff.templates.AggregationProcess(init_fn, next_fn)


def build_quic_fl_aggregator(num_bits: int = 1) -> tff.aggregators.AggregationFactory:
    factory = QuicFLFactory(num_bits)
    return configure_aggregator(factory, "identity", True, False, False, True)
