import z5py
import napari


def view_prediction(ds_name, pred_path, pred_key='data', halo=[32, 512, 512]):
    input_path = f'../../data/{ds_name}/images/local/fibsem-raw.n5'
    input_key = 'setup0/timepoint0/s1'

    with z5py.File(input_path, 'r') as f:
        ds = f[input_key]
        center = [sh // 2 for sh in ds.shape]
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        raw = ds[bb]

    with z5py.File(pred_path, 'r') as f:
        ds = f[pred_key]
        bb = (slice(None),) + bb
        pred = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(pred)


if __name__ == '__main__':
    view_prediction('1spdbaf', './data.n5')
