import os
import czifile
import z5py


# this is very naive and will probably not work properly; just doing it to test mapping one of the srsim datasets to the raw data
def align_control(root_folder):
    raw_path = os.path.join(root_folder, 'data.n5')
    raw_key = 'control/raw'
    with z5py.File(raw_path, 'r') as f:
        shape = f[raw_key].shape

    lm_file = os.path.join(root, 'lm-data', 'CONTROL_srsim_StructuredIllumination_ChannelAlignment.czi')
    with czifile.CziFile(lm_file) as f:
        data = f.asarray().squeeze()
    lm_shape = data.shape[1:]

    raw_resolution = [0.005] * 3

    lm_resolution = [res * sh / lsh for res, sh, lsh in zip(raw_resolution, shape, lm_shape)]
    print(lm_resolution)

    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    chunks = (1, 512, 512)
    out_path = './control.n5'
    channel_names = ['red', 'green', 'blue']
    with z5py.File(out_path) as f:

        f.attrs['resolution'] = lm_resolution
        f.attrs['scale_factors'] = scale_factors

        for chan_id, chan_name in enumerate(channel_names):
            chan = data[chan_id]
            f.create_dataset(chan_name, data=chan, compression='gzip', chunks=chunks)


if __name__ == '__main__':
    root = '/g/kreshuk/pape/Work/data/loos'
    align_control(root)
