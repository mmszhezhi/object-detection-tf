import glob
import os
import sys
from copy import deepcopy

import tqdm
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from lxml import etree as ET


ia.seed(1)
# # Example batch of images.
# # The array has shape (32, 64, 64, 3) and dtype uint8.
# images = np.array(
#     [np.array(data) for _ in range(32)],
#     dtype=np.uint8
# )


def plot_img(images):
    col = np.sqrt(len(images))
    rows, res = divmod(len(images), col)
    if res:
        rows += 1
    fig, axes = plt.subplots(rows, col, constrained_layout=True)
    for img, ax in zip(images, axes.flatten()):
        ax.imshow(Image.fromarray(img))
        ax.axison = False
    fig.show()


MUST_ROTATE = (90, 180, 270, 360)
MUST_SCALE = np.linspace(0.2, 1.2, 10)


def aug_by_value_list(images, bbs, func=iaa.Affine, **kwargs):
    img_list, bbs_list = [], []
    k, values = kwargs.popitem()
    for v in values:
        img_aug, bbs_aug = func(**kwargs, **{k: v})(images=images, bounding_boxes=bbs)
        img_list.extend(img_aug)
        bbs_list.extend(bbs_aug)
    return img_list, bbs_list


seq = iaa.Sequential([
    # iaa.Fliplr(0.5),  # horizontal flips
    # iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    # iaa.Sometimes(
    #     0.5,
    #     iaa.GaussianBlur(sigma=(0, 0.5))
    # ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale=(0.2, 1.2),
        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-3, 3),
        # shear=(-30, 30)
    )
])  # apply augmenters in random order


test_seq = iaa.Sequential([
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Multiply((0.8, 1.2)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale=(0.2, 1.2),
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-3, 3),
        # shear=(-30, 30)
    )
])  # apply augmenters in random order


def clean(img):
    """
    Remove all exif tags except the orientation
    """
    TAG_ORIENTATION = 0x112
    exif = img.getexif()
    if len(exif) > 0:
        clean_exif = Image.Exif()
        if TAG_ORIENTATION in exif:
            clean_exif[TAG_ORIENTATION] = exif[TAG_ORIENTATION]
        img.info["exif"] = clean_exif.tobytes()


def resize_image(img, xml, scale=1.5):
    new_size = (int(img.width / scale), int(img.height / scale))
    img.thumbnail(new_size)
    xml.find('size').find('width').text = str(img.width)
    xml.find('size').find('height').text = str(img.height)
    for member in xml.findall('object'):
        member[4][0].text = str(int(int(member[4][0].text) / scale))
        member[4][1].text = str(int(int(member[4][1].text) / scale))
        member[4][2].text = str(int(int(member[4][2].text) / scale))
        member[4][3].text = str(int(int(member[4][3].text) / scale))


def gen_batches(files, bs=5, scale=4.5, must_rotate=True):
    from imgaug.augmentables.batches import UnnormalizedBatch
    batches = []
    trees = []
    for xml_file in files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img = root.find('path').text
        os.path.join(*img.split("\\"))
        raw_img = Image.open(img)
        clean(raw_img)
        raw_img = ImageOps.exif_transpose(raw_img)
        # Reduce image size
        if scale > 1:
            resize_image(raw_img, root, scale)
        img_array = np.array(raw_img)

        bbs = [ia.BoundingBox(
            int(member[4][0].text),
            int(member[4][1].text),
            int(member[4][2].text),
            int(member[4][3].text)) for member in root.findall('object')]
        # Rotated
        if must_rotate:
            img_aug, bbs_aug = aug_by_value_list([img_array], [bbs], fit_output=True, rotate=MUST_ROTATE)
        else:
            img_aug, bbs_aug = [img_array], [bbs]
        # img_aug, bbs_aug = aug_by_value_list(img_aug, bbs_aug, scale=MUST_SCALE)
        # img_aug, bbs_aug = [], []
        # Original
        # img_aug.insert(0, img_array)
        # bbs_aug.insert(0, bbs)

        images = [img_aug_array for img_aug_array in img_aug for _ in range(bs)]
        batches.append(UnnormalizedBatch(images=images, bounding_boxes=[bbs_aug_array for bbs_aug_array in bbs_aug for _ in range(bs)]))
        trees.append(tree)
    return batches, trees


def save_image_and_xml(image, bbs, tree, img_path, xml_path):
    file_name = os.path.basename(img_path)
    img = Image.fromarray(image)
    img.save(img_path)
    root = tree.getroot()
    root.find('folder').text = os.path.basename(os.path.dirname(img_path))
    root.find('filename').text = file_name
    root.find('path').text = os.path.abspath(img_path)
    for member, bbox in zip(root.findall('object'), bbs):
        obj_height = bbox.y2 - bbox.y1
        obj_width = bbox.x2 - bbox.x1
        valid_y = lambda y: min(max(0, y), img.height)
        valid_x = lambda x: min(max(0, x), img.width)
        valid_height = valid_y(bbox.y2) - valid_y(bbox.y1)
        valid_width = valid_x(bbox.x2) - valid_x(bbox.x1)
        if valid_height / obj_height < 0.5 or valid_width / obj_width < 0.5:
            print(f'{file_name}: object {member[0].text}({bbox.x1, bbox.y1, bbox.x2, bbox.y2}) is out of bound')
            root.remove(member)
            continue
        member[4][0].text = str(valid_x(bbox.x1))
        member[4][1].text = str(valid_y(bbox.y1))
        member[4][2].text = str(valid_x(bbox.x2))
        member[4][3].text = str(valid_y(bbox.y2))
    tree.write(xml_path, encoding='utf8')


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield np.array(lst[i:i + n])


def aug(files, target_dir, prefix='', c=5):
    batches, trees = gen_batches(files, c, 1)
    for i, (batch, tree) in tqdm.tqdm(enumerate(zip(seq.augment_batches(batches, background=True), trees))):
        for j, (images_aug, bounding_boxes_aug) in enumerate(zip(batch.images_aug, batch.bounding_boxes_aug)):
            if j % c == 0:
                # For original image
                file_name = f'{prefix}_{i}_{j}_0.jpg'
                save_image_and_xml(batch.images_unaug[j], batch.bounding_boxes_unaug[j], deepcopy(tree),
                                   os.path.join(target_dir, file_name),
                                   os.path.join(target_dir, file_name.rpartition('.')[0] + '.xml'))

            file_name = f'{prefix}_{i}_{j}.jpg'
            save_image_and_xml(images_aug, bounding_boxes_aug, deepcopy(tree),
                               os.path.join(target_dir, file_name),
                               os.path.join(target_dir, file_name.rpartition('.')[0] + '.xml'))


def aug_images(dir_list, dst_dir, n=1, c=5):
    all_file_list = []
    for dir_name in dir_list:
        all_file_list.extend(glob.glob(os.path.join(dir_name, "*.xml")))
    files = list(chunks(all_file_list, len(all_file_list) // n))
    if len(files) > n:
        files[-2] = np.hstack([files[-2], files[-1]])
        files.pop(-1)
    for i, fl in enumerate(files):
        aug(fl, dst_dir, str(i), c)


if __name__ == '__main__':
    c = 5
    if len(sys.argv) < 3:
        sys.exit(1)
    if len(sys.argv) > 3:
        c = int(sys.argv[3])
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    target_dir_name = os.path.basename(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    all_files = glob.glob(f"{source_dir}/*.xml", recursive=True)
    n = 1
    files = list(chunks(all_files, len(all_files) // n))
    if len(files) > n:
        files[-2] = np.hstack([files[-2], files[-1]])
        files.pop(-1)
    for i, fl in enumerate(files):
        aug(fl, str(i))
