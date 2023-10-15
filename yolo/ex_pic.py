import os
import cv2


def decode_video(video_path, save_dir, name, target_num=None):
    '''
    video_path: Video to be decoded
    save_dir: Save folder for framed images
    target_num: Number of frames to draw, null to decode all frames, default to draw all frames.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 0
    index = 0
    frames_num = video.get(7)
    # If target_num is empty, draw all frames, if not, draw target_num frames.
    if target_num is None:
        step = 1
        print('all frame num is {}, decode all'.format(int(frames_num)))
    else:
        step = int(frames_num / target_num)
        print('all frame num is {}, decode sample num is {}'.format(int(frames_num), int(target_num)))
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % step == 0:
            save_path = "{}/{}_{:>04d}.png".format(save_dir, name,index)
            # save_path = save_dir+'/'+ name + str(index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
        if index == frames_num and target_num is None:
            # If you draw them all, stop when you reach the last of all frames
            break
        elif index == target_num and target_num is not None:
            # If sampling, stop when you reach target_num.
            break
        else:
            pass
    video.release()


if __name__ == '__main__':
    video_path = './video/'
    for video in os.listdir(video_path):
        video_name = os.path.join(video_path,video)
        save_dir_1 = './coco3/train/'
        num = video_name.split('_')[-1].split('.')[0]
        print(num)
        # aa
        # save_dir_2 = './pic1'
        decode_video(video_name, save_dir_1,'helicopter_'+str(num))
    # decode_video(video_path, save_dir_2, 20)



'''import os
from PIL import Image
import cv2

if __name__ == '__main__':
    ims_folder = './frames'
    video_path = './out_video.mp4'

    ims_path_list = os.listdir(ims_folder)
    ims_path_list.sort()
    fps = 30
    im_path = os.path.join(ims_folder, ims_path_list[0])
    im_size = Image.open(os.path.join(ims_folder, ims_path_list[0])).size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for i, im_path in enumerate(ims_path_list):
        im_path = os.path.join(ims_folder, im_path)
        frame = cv2.imread(im_path)
        videoWriter.write(frame)
        print(im_path)
    videoWriter.release()
    print('finish')
————————————————
Copyright: This is an original article by CSDN blogger "Tang Ze", following the CC 4.0 BY-SA copyright agreement, reprinted with the original source link and this statement.
Link to original article:https://blog.csdn.net/weixin_42544131/article/details/103526653'''