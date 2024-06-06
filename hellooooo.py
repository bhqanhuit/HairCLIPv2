# from models.stylegan2.model import Generator

# print('haaha')


import os

# print(os.listdir('datasets/FFHQ_Resized'))
root_dir = 'datasets/FFHQ_Resized'
for file in os.listdir('datasets/FFHQ_Resized'):
    print(file, file.split('.')[0])
    os.rename(root_dir + '/' + file, root_dir + '/' + file.split('.')[0] + '_70.png')
    