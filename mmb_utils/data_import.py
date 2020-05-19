import os
import json
import luigi
from cluster_tools.downscaling import DownscalingWorkflow

DEFAULT_SCALE_FACTORS = [[2, 2, 2]] * 6
DEFAULT_CHUNKS = [64, 64, 64]
DEFAULT_BLOCK_SHAPE = [64, 64, 64]


# TODO support mrc files (np.mmap) in elf.open_file
def import_raw_volume(in_path, in_key, out_path, resolution,
                      tmp_folder, scale_factors=DEFAULT_SCALE_FACTORS,
                      block_shape=DEFAULT_BLOCK_SHAPE, chunks=DEFAULT_CHUNKS,
                      target='local', max_jobs=16):
    task = DownscalingWorkflow

    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    configs = DownscalingWorkflow.get_config()
    global_conf = configs['global']
    global_conf.update({'block_shape': block_shape})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    conf = configs['copy_volume']
    conf.update({'chunks': chunks})
    with open(os.path.join(config_dir, 'copy_volume.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['downscaling']
    conf.update({'chunks': chunks})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(conf, f)

    halos = scale_factors
    metadata_format = 'bdv.n5'
    metadata_dict = {'resolution': resolution, 'unit': 'micrometer'}

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=in_path, input_key=in_key,
             scale_factors=scale_factors, halos=halos,
             metadata_format=metadata_format, metadata_dict=metadata_dict,
             output_path=out_path)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Importing raw data failed"
