import os
import shutil
import argparse
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--input_dir', type=str, default='input/', help='path to input')
    parser.add_argument('--output_dir', type=str, default='output/', help='path to input')
    parser.add_argument('--model_path', type=str, default='nnUNet_results/', help='model saved pth')  #
    args = parser.parse_args()
    
    path_input = args.input_dir
    path_input_new = 'inputs/'
    if not os.path.exists(path_input_new):
        os.makedirs(path_input_new)

    for folder in os.listdir(path_input):
            for file in os.listdir(path_input+folder):
                if 'gt' in file:
                    name = file.split('.')[0][:-2]
                    print(name)
                    mask = nib.load(path_input+folder+'/'+file)
                    nib.save(mask, os.path.join(path_input_new, name+'0000.nii.gz'))

    path_output = args.output_dir
    path_output_new = 'outputs/'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(path_output_new):
        os.makedirs(path_output_new)
    
    os.environ['nnUNet_results'] = args.model_path
    os.system(f"CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i {path_input_new} -o {path_output_new} -d 73 -c 3d_fullres -p nnUNetResEncUNetMPlans")

    shutil.rmtree(path_input_new)


    for file in os.listdir(path_output_new):
        if 'nii' in file:
            name = file.split('.')[0]
            print(name)
            mask = nib.load(path_output_new+'/'+file)
            nib.save(mask, os.path.join(path_output, name+'_label.nii.gz'))
    
    shutil.rmtree(path_output_new)

    print('Generate finished!')

