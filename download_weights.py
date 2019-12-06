import wget
import os

open_pose_mpi_base = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/'
open_pose_coco_base = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/'

base_path = 'https://pjreddie.com/media/files/'
base_git_config = 'https://raw.githubusercontent.com/pjreddie/darknet/master'
base_path_config = '{}/cfg/'.format(base_git_config)
base_data_config = '{}/data/'.format(base_git_config)


download_weight_files = ['yolov3.weights', 'yolov3-tiny.weights', 'yolov3-spp.weights']
download_config_files = ['yolov3-spp.cfg','yolov3-tiny.cfg','yolov3.cfg']
download_data_files = ['coco.names']
open_pose_files = ['pose_iter_160000.caffemodel','pose_iter_440000.caffemodel']

file_urls = ["https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/emotion_ferplus.tar.gz",
"https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
"https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin",
"https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/action-recognition-0001-decoder/FP32/action-recognition-0001-decoder.xml",
"https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/action-recognition-0001-decoder/FP32/action-recognition-0001-decoder.bin"]


dir_path = 'weights'
dir_exists = os.path.exists(dir_path)

def download_files (files, output_dir, base_path):
    for file in files:
        if not os.path.exists('{}\\{}'.format(output_dir, file)):
            print ("Downloading file {}.".format(file))
            try:
            	wget.download('{}{}'.format(base_path, file), out=output_dir)
            	print (" Done.")
            except:
            	print ("Error downloading file {}".format(file))
            	pass


def download_file_from_uri (files, output_dir):
    for file in files:
        if not os.path.exists('{}\\{}'.format(output_dir, os.path.basename(file))):
            print ("Downloading file {}.".format(file))
            try:
            	wget.download(file, out=output_dir)
            	print (" Done.")
            except:
            	print ("Error downloading file {}".format(file))
            	pass
            
if not dir_exists:
    print ("Weight directory does not exist. Creating..")
    os.makedirs(dir_path)

download_files(download_weight_files, dir_path, base_path)
download_files(download_config_files, dir_path, base_path_config)
download_files(download_data_files, dir_path, base_data_config)
download_files(open_pose_files, dir_path, open_pose_mpi_base)
download_files(open_pose_files, dir_path, open_pose_coco_base)
download_file_from_uri(file_urls, dir_path)