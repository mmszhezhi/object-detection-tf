# -*- coding: utf-8 -*-

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python csv2record.py --csv_input=data/train.csv  --output_path=train.record --img_path=image2/train
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test.csv  --output_path=test.record --img_path=image2/test
"""


flags = tf.app.flags
flags.DEFINE_string('csv_input', 'annotations/t2.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path_train', 'annotations/train-t.record', 'Path to output TFRecord')
flags.DEFINE_string('output_path_test', 'annotations/test-t.record', 'Path to output TFRecord')
flags.DEFINE_string('img_path', 'annotations/data/aug-a0', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# FLAGS.output_path_train = "m1/train.record"
# FLAGS.output_path_test = "m1/test.record"
# FLAGS.img_path = "croped"
# FLAGS.csv_input = "croped.csv"
# TO-DO replace this with label map
# def class_text_to_int(row_label):
#     if row_label == '燕尾自攻螺丝':
#         return 1
#     if row_label == '螺钉':
#         return 2
#     if row_label == 'T型螺钉':
#         return 3
#     if row_label == '吊丝接头':
#         return 4
#     if row_label == 'T型块1':
#         return 5
#     if row_label == 'T型块2':
#         return 6
#     if row_label == '销钉':
#         return 7
#     if row_label == '垫片':
#         return 8
#     if row_label == '水管接头黄':
#         return 9
#     if row_label == '水管接头银':
#         return 10
#     if row_label == '螺钉银':
#         return 11
#     else:
#         None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, labels):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(str(row['class']).encode('utf8'))
        # print(row["class"])
        classes.append(labels.index(row['class'])+1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

import sys

def main(_):
    from sklearn.model_selection import StratifiedKFold
    # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = FLAGS.img_path
    examples = pd.read_csv(FLAGS.csv_input)
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    train_idx, test_idx = list(skf.split(examples, y=examples['class']))[0]
    print(examples.iloc[train_idx]['class'].value_counts())
    print(examples.iloc[test_idx]['class'].value_counts())
    train = examples.iloc[train_idx]['class'].value_counts()
    test = examples.iloc[test_idx]['class'].value_counts()
    ratio = pd.concat([train,test],axis=1)
    ratio.to_csv(os.path.join(os.path.dirname(FLAGS.output_path_train), 'ratio.csv'))
    train_grouped = split(examples.iloc[train_idx], 'path')
    test_grouped = split(examples.iloc[test_idx], 'path')
    labels = list(examples['class'].unique())
    with open(os.path.join(os.path.dirname(FLAGS.output_path_train), 'labelmap.pbtxt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(
            """item {
  id: %d
  name: '%s'
}""" % (i + 1, label) for i, label in enumerate(labels)))

    with open(os.path.join(os.path.dirname(FLAGS.output_path_train), 'labelmap.txt'), 'w', encoding='utf8') as f:
        f.write("\n".join(labels))
    # grouped = split(examples, 'filename')
    writer1 = tf.python_io.TFRecordWriter(FLAGS.output_path_train)
    for group in train_grouped:
        tf_example = create_tf_example(group, path, labels)
        writer1.write(tf_example.SerializeToString())
    writer1.close()

    writer2 = tf.python_io.TFRecordWriter(FLAGS.output_path_test)
    for group in test_grouped:
        tf_example = create_tf_example(group, path,labels)
        writer2.write(tf_example.SerializeToString())
    writer2.close()



if __name__ == '__main__':
    tf.app.run()