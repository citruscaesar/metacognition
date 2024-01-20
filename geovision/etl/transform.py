import numpy as np

class CropUtil():

    def _read_image(self, path):
        return skimage.io.imread(path)  # type: ignore

    def _read_mask(self, path):
        return skimage.io.imread(path)  # type: ignore

    def _get_pad_amount(self, dimension: int, window: int):
        """Calculate to no of pixels to add to before and after dimension"""
        total_padding = window - (dimension % window)

        if total_padding % 2 == 0:
            after = total_padding // 2
            before = after
        else:
            after = (total_padding // 2) + 1
            before = after - 1
        assert before+after == total_padding 
        return (before, after)
    
    def _pad_3d_array(self, array: np.ndarray, window: int):
        """
        Pad image array s.t. divisible by window\n
        array.shape : (Height, Width, Channels)
        window : side length of square cropping window
        """

        assert array.ndim == 3
        padded_array = np.pad(
            array = array,
            pad_width = (self._get_pad_amount(array.shape[0], window),
                         self._get_pad_amount(array.shape[1], window),
                         (0, 0))
        ) 
        return padded_array
    
    def _get_cropped_view(self, array: np.ndarray, window:int):
        """
        Crop image array s.t. divisible by window\n
        array.shape : (Height, Width, Channels)
        window : side length of square cropping window
        """

        assert array.ndim == 3
        cropped_view = skimage.util.view_as_windows( # type: ignore
            arr_in = array,
            window_shape = (window, window, array.shape[2]),
            step =  (window, window, array.shape[2])).squeeze()
            
        cropped_view = cropped_view.reshape(-1, window, window, array.shape[2])

        return cropped_view

    def _crop_one_scene(self, tile_path: Path, window: int, read_scene):
        scene = read_scene(tile_path) 
        scene = self._pad_3d_array(scene, window)
        scene = self._get_cropped_view(scene, window)
        return scene

    def _save_as_jpeg_100(self, array: np.ndarray, out_path: Path) -> None:
        skimage.io.imsave((out_path.parent / f"{out_path.stem}.jpg"), array, check_contrast = False, **{"quality": 100}) # type: ignore