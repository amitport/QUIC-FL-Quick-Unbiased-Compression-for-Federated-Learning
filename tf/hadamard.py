import tensorflow as tf
import tensorflow_federated as tff

from builder import configure_aggregator
from compressed_communication.aggregators.utils import quantize_utils


class MinMaxSQFactory(tff.aggregators.UnweightedAggregationFactory):
    def __init__(self, bits):
        self._scale = 2 ** bits - 1

    def create(self, value_type):
        if not tff.types.is_structure_of_floats(
                value_type) or not value_type.is_tensor():
            raise ValueError("Expect value_type to be a float tensor, "
                             f"found {value_type}.")

        @tff.tf_computation
        def decode(encoded_value):
            _scale = tf.constant(self._scale, value_type.dtype)
            v_min, v_max, v = encoded_value
            v = tf.cast(v, value_type.dtype)
            decoded_value = (v / _scale) * (v_max - v_min) + v_min
            return decoded_value

        @tff.tf_computation(value_type)
        def encode(value):
            _scale = tf.constant(self._scale, value_type.dtype)

            v_min = tf.reduce_min(value)
            v_max = tf.reduce_max(value)
            normalized = tf.math.divide_no_nan(value - v_min, v_max - v_min) * _scale
            seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                           dtype=tf.int64)
            quantized_value = quantize_utils.stochastic_quantize(normalized, 1., seed)

            encoded_value = (v_min, v_max, quantized_value)

            return encoded_value

        @tff.federated_computation()
        def init_fn():
            return tff.federated_value((), tff.SERVER)

        def decode_and_sum(value):
            """Encoded client values do not commute with sum: decode then sum."""

            @tff.tf_computation
            def get_accumulator():
                return tf.zeros(shape=value_type.shape, dtype=tf.float32)

            @tff.tf_computation
            def decode_accumulate_values(accumulator, encoded_value):
                decoded_value = decode(encoded_value)
                return accumulator + decoded_value

            @tff.tf_computation
            def merge_decoded_values(decoded_value_1, decoded_value_2):
                return decoded_value_1 + decoded_value_2

            @tff.tf_computation
            def report_decoded_summation(summed_decoded_values):
                return summed_decoded_values

            return tff.federated_aggregate(
                value,
                zero=get_accumulator(),
                accumulate=decode_accumulate_values,
                merge=merge_decoded_values,
                report=report_decoded_summation)

        @tff.federated_computation(init_fn.type_signature.result,
                                   tff.type_at_clients(value_type))
        def next_fn(state, value):
            encoded_value = tff.federated_map(encode, value)

            result = decode_and_sum(encoded_value)

            return tff.templates.MeasuredProcessOutput(
                state=state,
                result=result,
                measurements=tff.federated_value((), tff.SERVER))

        return tff.templates.AggregationProcess(init_fn, next_fn)


def build_hadamard_aggregator(
        num_bits: int = 1,
        concatenate: bool = True,
        zeroing: bool = True,
        clipping: bool = True,
        weighted: bool = True) -> tff.aggregators.AggregationFactory:
    factory = MinMaxSQFactory(num_bits)
    return configure_aggregator(factory, 'hadamard', concatenate, zeroing, clipping,
                                weighted)
