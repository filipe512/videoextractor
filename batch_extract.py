import os
import argparse
from extract import extract
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input video path to extract frames')
parser.add_argument('--output', default='output', help='Output path where to extract frames')
parser.add_argument('--force', help='Ignore if the output folder exists and overwrite the files', action='store_true')

args = parser.parse_args()

video_formats = ['.avi', '.m4v', '.mp4', '.wmv', '.flv']

if __name__ == '__main__':
    file_list = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if (Path(file).suffix in video_formats):
                file_list.append("{}\\{}".format(root, file))
    
    for file in file_list:
        #print (Path(file).stem) #filename without extentions
        file_size = os.path.getsize(file)
        
        dir_path = "{}\\{}.{}".format(args.output, Path(file).stem, file_size)
        print (dir_path)
        
        
        dir_exists = os.path.exists(dir_path)
        force_option = args.force
        
        if dir_exists and not force_option:
            print ("The file {} will not be processed because the frames for the file was already created or the file is duplicated".format(dir_path))
            continue
        
        if not dir_exists:
            os.makedirs(dir_path)
           
        extract(file, dir_path, 15)
        
    print ("Done")