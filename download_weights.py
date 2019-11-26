import wget
import os



open_pose_mpi_base = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/'


base_path = 'https://pjreddie.com/media/files/'
base_git_config = 'https://raw.githubusercontent.com/pjreddie/darknet/master'
base_path_config = '{}/cfg/'.format(base_git_config)
base_data_config = '{}/data/'.format(base_git_config)


download_weight_files = ['yolov3.weights', 'yolov3-tiny.weights', 'yolov3-spp.weights']
download_config_files = ['yolov3-spp.cfg','yolov3-tiny.cfg','yolov3.cfg']
download_data_files = ['coco.names']
open_pose_files = ['pose_deploy_linevec_faster_4_stages.prototxt','pose_iter_160000.caffemodel']

dir_path = 'weights'
dir_exists = os.path.exists(dir_path)

def download_files (files, output_dir, base_path):
    for file in files:
        if not os.path.exists('{}\\{}'.format(output_dir, file)):
            print ("Downloading file {}.".format(file))
            wget.download('{}{}'.format(base_path, file), out=output_dir)
            print (" Done.")
            
            
if not dir_exists:
    print ("Weight directory does not exist. Creating..")
    os.makedirs(dir_path)

download_files(download_weight_files, dir_path, base_path)
download_files(download_config_files, dir_path, base_path_config)
download_files(download_data_files, dir_path, base_data_config)
download_files(open_pose_files, dir_path, open_pose_mpi_base)



    




