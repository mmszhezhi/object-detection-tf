import os
import glob
import pandas as pd
import lxml.etree as ET
import argparse,sys

arg = argparse.ArgumentParser()


# os.chdir('')
# path = ''

def xml_to_csv(path):
    xml_list = []
    skipped = 0
    success = 0
    for xml_file in glob.glob(path+"/*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            print(xml_file)
            root.find('filename').text
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         float(member[4][0].text),
                         float(member[4][1].text),
                         float(member[4][2].text),
                         float(member[4][3].text)
                         )
                xml_list.append(value)
            success +=1
        except Exception as e:
            skipped+=1
            print(repr(e),xml_file)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df,skipped,success

import glob
def main():
    # python xml_to_csv.py image3/test
    # python xml_to_csv.py image3/
    os.chdir(os.getcwd())
    xml_path = ["train_aug4","test_aug4"]
    xml_path = ["croped"]
    dst_path = "aug-9000"
    for f in xml_path:
        # image_path = "image2\\train"
        xml_df,skipped,suc = xml_to_csv(f)
        xml_df.to_csv(f'{f}.csv', index=None)
        # xml_df.to_csv("test.csv")
        print(f'Successfully converted {f}xml to csv. succeed {suc}  skip {skipped} xml')

main()


# tree = ET.parse(r"image\test_aug4\左右 - 副本_9.xml")
# root = tree.getroot()

