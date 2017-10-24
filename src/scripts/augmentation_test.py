from PIL import Image as pil_image
import numpy as np
from skimage.transform import SimilarityTransform
import time
import skimage.transform as skitransform
from numpy.linalg import inv
from skimage.util import dtype_limits


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = 'channels_last'
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = 'channels_last'
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=np.float32)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def pil_image_transform(img, transform, resample, cval):
    matrix = inv(transform.params).ravel()

    # Add alpha channel (all pixels full alpha) to detect out-of-bounds values (will have alpha 0)
    img.putalpha(255)
    img = img.transform(size=img.size, method=pil_image.AFFINE, data=matrix, resample=resample)

    # Replace out-of-bounds values with the cval
    background = pil_image.new('RGB', img.size, cval)
    background.paste(img, mask=img.split()[3])
    img = background

    return img


def pil_adjust_gamma(im, gamma, num_channels=3):
    """Fast gamma correction with PIL's image.point() method"""
    invert_gamma = 1.0/gamma
    lut = [pow(x/255., invert_gamma) * 255 for x in range(256)]
    lut = lut*num_channels # need one set of data for each color channel
    im = im.point(lut)
    return im


def pil_channel_shift(im, intensity, num_channels=3):
    lut = [x + intensity for x in range(256)]
    lut = lut * num_channels
    im = im.point(lut)
    return im





def np_adjust_gamma(image, gamma=1, gain=1, output=None):
    """Performs Gamma Correction on the input image.

    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.

    Parameters
    ----------
    image : ndarray
        Input image.
    gamma : float
        Non negative real number. Default value is 1.
    gain : float
        The constant multiplier. Default value is 1.
    output : ndarray
        Where to place the output
    Returns
    -------
    out : ndarray
        Gamma corrected output image.

    See Also
    --------
    adjust_log

    Notes
    -----
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.

    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gamma_correction
    """
    dtype = image.dtype.type

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    if output is not None:
        output = ((image / 255.0) ** 1.0/gamma) * 255 * gain
        output = dtype(image)
        return None
    else:
        out = ((image / scale) ** gamma) * scale * gain
        return dtype(out)

def main():
    # Read image in
    img = load_img('/Volumes/Omenakori/data/final/unlabeled/000005445.jpg')
    np_img = img_to_array(img)

    shift_x = img.width * 0.7
    shift_y = img.height * 0.5
    tx = 50
    ty = 100
    theta = 0.9
    zoom = 1.5

    img = img.offset(100, 100)
    img.show()

    raw_input('Press any key: ')

    # Prepare transforms to shift the image origin to the image center
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])

    # Build the translation, rotation and scale transforms
    tf_translate = SimilarityTransform(translation=[tx, ty])
    tf_rotate = SimilarityTransform(rotation=theta)
    tf_scale = SimilarityTransform(scale=zoom)

    # Build the final transform: (SHIFT)*S*R*T*(SHIFT_INV)
    tf_final = (tf_shift + (tf_scale + tf_rotate + tf_translate) + tf_shift_inv)
    matrix = inv(tf_final.params)

    # PIL transform
    # Parse the similarity transform
    data = matrix.ravel()[0:6]
    #data = np.array(list(data[3:]) + list(data[0:3]))

    stime = time.time()
    background = pil_image.new('RGB', img.size, (0, 255, 0))
    img.putalpha(255)
    tf_img = img.transform(size=img.size, method=pil_image.AFFINE, data=data, resample=pil_image.BICUBIC)
    background.paste(tf_img, mask=tf_img.split()[3])
    tf_img = background
    #tf_img_crop = tf_img.crop(box=(x1, y1, x2, y2))
    #tf_img_crop.load()
    print 'PIL transform took: {} s, shape: {}'.format(time.time() - stime, tf_img.size)

    tf_img.show(title='PIL')

    raw_input('Press any key: ')

    stime = time.time()
    tf_np_img = skitransform.warp(image=np_img,
                                  inverse_map=tf_final.inverse,
                                  order=3,
                                  mode='constant',
                                  cval=0,
                                  preserve_range=True).astype(np.float32)
    #tf_np_img_crop = tf_np_img[y1:y2, x1:x2]
    print 'Scipy warp took: {} s, shape: {}'.format(time.time() - stime, tf_np_img.shape)

    stime = time.time()
    new_pil_img = array_to_img(tf_np_img)
    print 'Array to img took: {}'.format(time.time()-stime)

    new_pil_img.show(title='Scipy')

if __name__ == '__main__':
    main()
