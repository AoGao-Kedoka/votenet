# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import numpy as np
from PIL import Image
try:
    from io import BytesIO         # Python 3.x
except ImportError:
    from StringIO import StringIO  # Python 2.7


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            img_summaries = []
            for i, img in enumerate(images):
                # Convert the image to a string
                s = BytesIO()
                Image.fromarray(img).save(s, format="png")
                img_str = s.getvalue()

                # Create an Image object and write it to the summary
                tf.summary.image(f'{tag}/{i}', [tf.image.decode_png(img_str)], step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()

