import glob
import sys
import numpy as np
import pandas as pd
from lxml import etree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/**/*.xml', recursive=True):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('path').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(np.round(float(member[4][0].text))),
                     int(np.round(float(member[4][1].text))),
                     int(np.round(float(member[4][2].text))),
                     int(np.round(float(member[4][3].text))),
                     )
            xml_list.append(value)
    column_name = ['path', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    if len(sys.argv) < 3:
        return
    xml_df = xml_to_csv(sys.argv[1])
    xml_df.to_csv(sys.argv[2], index=False)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
