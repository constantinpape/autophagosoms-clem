import z5py
from phago_network_utils.prediction import predict_with_halo, normalize, to_uint8


def predict_dataset(ds_name, checkpoint_path,
                    output_path, output_key, gpu_ids,
                    block_shape=(96, 128, 128), halo=(16, 32, 32)):
    input_path = f'../../data/{ds_name}/images/local/fibsem-raw.n5'
    input_key = 'setup0/timepoint0/s1'

    # we have 3 semantic channels:
    # r, g and r+g
    n_channels = 3

    # get rid of the first channel in the predictions, which is not interesting
    # then cast to uint8 to save disc space
    def postprocess(predictions):
        assert predictions.shape[0] == 4, f"{predictions.shape}"
        return to_uint8(predictions[1:])

    with z5py.File(input_path, 'r') as f_in, z5py.File(output_path, 'a') as f_out:
        ds_in = f_in[input_key]

        out_shape = (n_channels,) + ds_in.shape
        ds_out = f_out.require_dataset(output_key, shape=out_shape, compression='gzip', dtype='uint8',
                                       chunks=(1,) + block_shape)

        predict_with_halo(ds_in, checkpoint_path, gpu_ids,
                          block_shape, halo, use_best=True,
                          output=ds_out,
                          preprocess=normalize,
                          postprocess=postprocess)


if __name__ == '__main__':
    ds_name = '1spdbaf'
    output_path = './data.n5'
    checkpoint_path = '../training/checkpoints/v1_fullV1/Weights'
    gpu_ids = [5, 6, 7]
    predict_dataset(ds_name, checkpoint_path, output_path, 'data', gpu_ids)
