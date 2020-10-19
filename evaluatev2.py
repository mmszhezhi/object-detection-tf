import io
import os
import numpy as np
import glob
from IPython.display import display
from six import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



from absl import flags
flags.DEFINE_string('testset', "testimg", 'Path to pipeline config '
                    'file.')
flags.DEFINE_string('model_path', "models/t2/graph/saved_model", 'Path to pipeline config '
                    'file.')
flags.DEFINE_string('PATH_TO_LABELS', "annotations/labelmap.pbtxt", 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer("NUM_CLASSES",10,"NUM_CLASSES")
flags.DEFINE_string('save_result', "testresult", 'Path to pipeline config '
                    'file.')

FLAGS = flags.FLAGS

def load_image_into_numpy_array1(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def main(unused_argv):
    model = tf.saved_model.load(FLAGS.model_path)
    PATH_TO_LABELS = FLAGS.PATH_TO_LABELS
    NUM_CLASSES = FLAGS.NUM_CLASSES
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    IMAGE_SIZE = (12, 8)
    for image_path in glob.glob(f'{FLAGS.testset}/*.jpg'):
        image = Image.open(image_path)
        image = image.resize(size=(600, 600))
        image_np = load_image_into_numpy_array(image)
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            min_score_thresh=0.6,
            max_boxes_to_draw=8,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=1, )
        name = os.path.basename(image_path)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig(f"{FLAGS.save_result}/{name}")

if __name__ == '__main__':
    tf.compat.v1.app.run()
    # tf.keras.backend.clear_session()

