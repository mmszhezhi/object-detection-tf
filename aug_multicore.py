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
from xml.etree import ElementTree as ET
import random
# from keras.preprocessing.image import load_img,img_to_array,array_to_img,save_img


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


from imgaug.parameters import StochasticParameter,handle_continuous_param
class Gama(StochasticParameter):
    def __init__(self, loc, scale):
        super(Gama, self).__init__()
        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale")

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, "Expected scale to be >=0, got %.4f." % (scale,)
        if scale == 0:
            return np.full(size, loc, dtype=np.float32)
        s = np.random.gamma(loc, scale, size=size).astype(np.float32)
        # print(s)
        return np.clip(s,a_min=0.2,a_max=4)

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)

class FixScale(StochasticParameter):
    def __init__(self, min, max):
        super(FixScale, self).__init__()
        self.min = handle_continuous_param(min, "min")
        self.max = handle_continuous_param(max, "max",
                                             value_range=(0, None))

    def _draw_samples(self, size, random_state):
        min = self.min.draw_sample(random_state=random_state)
        max = self.max.draw_sample(random_state=random_state)
        s = np.linspace(min, max, size[0]).astype(np.float32)
        return s

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.min, 0.3)


class FixRotate(StochasticParameter):
    def __init__(self):
        super(FixRotate, self).__init__()
        # self.min = handle_continuous_param(min, "min")
        # self.max = handle_continuous_param(max, "max",
        #                                      value_range=(0, None))

    def _draw_samples(self,x,y):
        return np.array([0,90,180,270])

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % ( 0.3)



seq = iaa.Sequential([
    iaa.Fliplr(0.2),
    iaa.VerticalFlip(0.2),
    iaa.HorizontalFlip(0.2),
    iaa.Sometimes(
        0.1,
        iaa.GaussianBlur(sigma=(0, 3)),
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # iaa.LinearContrast((0.75, 1.5)),
    ),
    iaa.Affine(
        scale=(0.7,1.2),
        rotate=(-6, 6),
        # shear=(-8, 8)
    )
], random_order=True)
seq_scale = iaa.Sequential(
    iaa.Affine(
        scale= FixScale(0.3,0.9),
        # rotate= FixRotate()
    )
)

seq_rotate = iaa.Sequential(
    iaa.Affine(
        # scale= FixScale(0.3,0.9),
        rotate= FixRotate()
    )
)

from imgaug.parameters import StochasticParameter
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

def resize_image(img, xml, scale=4):
    new_size = (int(img.width / scale), int(img.height / scale))
    # print(new_size)
    img.thumbnail(new_size)
    xml.find('size').find('width').text = str(img.width)
    xml.find('size').find('height').text = str(img.height)
    for member in xml.findall('object'):
        member[4][0].text = str(int(int(member[4][0].text) / scale))
        member[4][1].text = str(int(int(member[4][1].text) / scale))
        member[4][2].text = str(int(int(member[4][2].text) / scale))
        member[4][3].text = str(int(int(member[4][3].text) / scale))

def resize_fix_shape(img,xml,shape):
    xml.find('size').find('width').text = str(shape)
    xml.find('size').find('height').text = str(shape)
    if img.width > img.height:
        scale = img.width / shape
        l = shape*img.height/img.width
        fill_pix = int((shape-l)/2)
        img.thumbnail((img.width,l))
        img_array = np.array(img,dtype="uint8")
        img_array = np.pad(img_array,pad_width=((fill_pix,fill_pix),(0,0),(0,0)))
        for member in xml.findall('object'):
            member[4][0].text = str(int(int(member[4][0].text) / scale))
            member[4][1].text = str(int(int(member[4][1].text) / scale) + fill_pix)
            member[4][2].text = str(int(int(member[4][2].text) / scale))
            member[4][3].text = str(int(int(member[4][3].text) / scale) + fill_pix)
        return img_array
    else:
        l = shape*img.width/img.height
        scale = img.height / shape
        fill_pix = int((shape - l) / 2)
        img.thumbnail(( l,img.height))
        img_array = np.array(img)
        img_array = np.pad(img_array, pad_width=((0,0),(fill_pix, fill_pix),(0,0)))

        for member in xml.findall('object'):
            member[4][0].text = str(int(int(member[4][0].text) / scale) + fill_pix)
            member[4][1].text = str(int(int(member[4][1].text) / scale))
            member[4][2].text = str(int(int(member[4][2].text) / scale) + fill_pix)
            member[4][3].text = str(int(int(member[4][3].text) / scale))
        return img_array

def mixture(origin,bg):
    bgs = glob.glob(bg+"/*")
    bg = random.choice(bgs)
    # load_img(bg)


def aug_by_value_list(images, bbs, func=iaa.Affine, **kwargs):
    img_list, bbs_list = [], []
    k, values = kwargs.popitem()
    for v in values:
        img_aug, bbs_aug = func(**kwargs, **{k: v})(images=images, bounding_boxes=bbs)
        img_list.extend(img_aug)
        bbs_list.extend(bbs_aug)
    return img_list, bbs_list


def gen_batches(files,scale_bs=3,aug_bs=32, crop_size=600,scale=4):
    from imgaug.augmentables.batches import UnnormalizedBatch
    skip = 0
    for xml_file in files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img = root.find('path').text
        try:
            raw_img = Image.open(img)
            clean(raw_img)
            raw_img = ImageOps.exif_transpose(raw_img)
            # resize_image(raw_img, root, scale)
            # img_array = np.array(raw_img)
            img_array = resize_fix_shape(raw_img,root,crop_size)
            images = [img_array for _ in range(scale_bs)]
            bbs = [ia.BoundingBox(
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)) for member in root.findall('object')]
            images_scale, bbs_scale = seq_scale(images=images, bounding_boxes=[bbs for _ in  range(scale_bs)])
            images_rotate, bbs_rotate = aug_by_value_list(images_scale, bbs_scale, fit_output=True, rotate=MUST_ROTATE)

            imgs = [im for im in images_rotate for _ in range(aug_bs)]
            batche = UnnormalizedBatch(images=imgs, bounding_boxes=[bbss for bbss in  bbs_rotate for _ in range(aug_bs)])

        except Exception as e:
            skip+=1
            print(repr(e),f" skip {skip}")
        yield batche

from imgaug.multicore import Pool
if __name__ == '__main__':
    aug_bs = 32
    scale_bs = 3
    skip = 0
    MUST_ROTATE = (90, 180, 270, 360)
    MUST_SCALE = np.linspace(0.2, 0.9, 3)
    p = "p0"
    sd = r"images"
    crop_size = 1200
    td = f"a0-auged"
    os.chdir(os.getcwd())
    os.makedirs(td,exist_ok=True)

    source_dir = sys.argv[1] if len(sys.argv) == 3 else sd
    target_dir = sys.argv[2] if len(sys.argv) == 3 else td
    if not any([source_dir,target_dir]):
        print("need both source directory and target directory")
    target_dir_name = os.path.basename(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    folders = len(glob.glob(f"{source_dir}/*"))
    for fi,forlder in enumerate(glob.glob(f"{source_dir}/*")):
        all_files = glob.glob(f"{forlder}/*.xml", recursive=True)
        tree = ET.parse(all_files[0])
        root = tree.getroot()
        batch_gen = gen_batches(all_files,scale_bs,aug_bs,crop_size)
        with Pool(augseq=seq,processes=-1, maxtasksperchild=20, seed=1) as pool:
            batchs_aug = pool.imap_batches(batch_gen)
            for i, batch in tqdm.tqdm(enumerate(batchs_aug)):
                for j, (images_aug, bounding_boxes_aug) in enumerate(zip(batch.images_aug, batch.bounding_boxes_aug)):
                    forlder = forlder.split("\\")[-1]
                    file_name = rf'{forlder}_{p}_{i}_{j}.jpg'
                    aug_img = Image.fromarray(images_aug)
                    aug_img.save(os.path.join(target_dir, file_name))
                    root.find('filename').text = file_name
                    root.find('path').text = os.path.abspath(os.path.join(target_dir, file_name))
                    for member, bbox in zip(root.findall('object'), bounding_boxes_aug):
                        obj_width = abs(bbox.x2 - bbox.x1)
                        obj_height = abs(bbox.y2-bbox.y1)
                        valid_coordinates = lambda x,h:min(max(0,x),h)
                        bh = abs(valid_coordinates(bbox.y2,aug_img.height) -valid_coordinates(bbox.y1,aug_img.height))
                        bw = abs(valid_coordinates(bbox.x2,aug_img.width) - valid_coordinates(bbox.x1,aug_img.width))
                        if bh/obj_height < 0.6 or bw/obj_width < 0.6:
                            skip +=1
                            print(f"out of bound {skip}")
                            continue
                        clip = lambda x,maximum:min(max(x,0),maximum)
                        member[4][0].text = str(clip(bbox.x1,aug_img.width))
                        member[4][1].text = str(clip(bbox.y1,aug_img.height))
                        member[4][2].text = str(clip(bbox.x2,aug_img.width))
                        member[4][3].text = str(clip(bbox.y2,aug_img.height))
                    tree.write(os.path.join(target_dir, file_name.rpartition('.')[0] + '.xml'), encoding='utf8')
                print(f"{i} of {len(all_files)} of {forlder} {fi} of {folders}")
            print(f"process success skip {skip}")
