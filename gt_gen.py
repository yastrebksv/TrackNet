import numpy as np
import pandas as pd
import os
import cv2
import argparse

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    gaussian_kernel_array = create_gaussian(size, variance)
    for game_id in range(1,11):
        game = 'game{}'.format(game_id)
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            print('game = {}, clip = {}'.format(game, clip))

            path_out_game = os.path.join(path_output, game)
            if not os.path.exists(path_out_game):
                os.makedirs(path_out_game)

            path_out_clip = os.path.join(path_out_game, clip)    
            if not os.path.exists(path_out_clip):
                os.makedirs(path_out_clip)  

            path_labels = os.path.join(os.path.join(path_input, game, clip), 'Label.csv')
            labels = pd.read_csv(path_labels)    
            for idx in range(labels.shape[0]):
                file_name, vis, x, y, _ = labels.loc[idx, :]
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0:
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                                if x+i<width and x+i>=0 and y+j<height and y+j>=0 :
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        heatmap[y+j,x+i] = (temp,temp,temp)

                cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap) 
                
def create_gt_labels(path_input, path_output, train_rate=0.7):
    df = pd.DataFrame()
    for game_id in range(1,11):
        game = 'game{}'.format(game_id)
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']
            labels_target = labels[2:]
            labels_target.loc[:, 'path2'] = list(labels['path1'][1:-1])
            labels_target.loc[:, 'path3'] = list(labels['path1'][:-2])
            df = df.append(labels_target)
    df = df.reset_index(drop=True) 
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df = df.sample(frac=1)
    num_train = int(df.shape[0]*train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)  


if __name__ == '__main__':
    SIZE = 20
    VARIANCE = 10
    WIDTH = 1280
    HEIGHT = 720   

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str, help='path to input folder')
    parser.add_argument('--path_output', type=str, help='path to output folder')
    args = parser.parse_args()
    
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
        
    create_gt_images(args.path_input, args.path_output, SIZE, VARIANCE, WIDTH, HEIGHT)
    create_gt_labels(args.path_input, args.path_output)

                            
    
    
    
    