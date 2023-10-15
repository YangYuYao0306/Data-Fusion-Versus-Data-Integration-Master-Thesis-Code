import os
import cv2
import shutil
from ultralytics import YOLO
import os
from PIL import Image
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

def draw_rect(img_path, coord, label,max_score,output_path):
    '''
    img = cv2.rectangle( img, pt1, pt2, color[, thickness[, lineType]] )
    Parameter Description:
        img：The carrier image on which graphics are drawn (the container carrier for drawing, also known as canvas, drawing board).
        pt1 is a rectangular vertex.
        pt2 is the vertex in the rectangle diagonal to pt1.
        color：Draws the colour of the shape. Colours are usually represented using the BGR model.
            For example, (0, 255, 0) represents green. For greyscale images, only greyscale values can be passed in.
            Note that the order of the colour channels is BGR, not RGB.
        thickness：The thickness of the line.
            The default value is 1. If set to -1, this means that the graph is filled (i.e., the graph is drawn solid).
        lineType：The type of the line, the default is 8 Connection Type.
    '''
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    # print(h,w,coord)
    # for coord in coords:
    x1 = int((coord[0]-coord[2]/2)*w)
    x2 = int((coord[1]-coord[3]/2)*h)
    x3 = int((coord[0]+coord[2]/2)*w)
    x4 = int((coord[1]+coord[3]/2)*h)
    # print(x1,x2,x3,x4)
    cv2.rectangle(img, (x1,x2) ,(x3,x4 ), (0, 0, 255), 2)
    cv2.putText(img, label+' '+str(max_score)[:4], (x1, x2 - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0, 0, 255), thickness=1)

    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    video_path = './test/'
    result_path = './result/'
    for video in os.listdir(video_path):
        video_name = os.path.join(video_path,video)

        save_dir_1 = './mid/'
        save_dir_2 = './mid1/'
        if os.path.exists(save_dir_1):
            shutil.rmtree(save_dir_1)
            os.mkdir(save_dir_1)
        else:
            os.mkdir(save_dir_1)

        if os.path.exists(save_dir_2):
            shutil.rmtree(save_dir_2)
            os.mkdir(save_dir_2)
        else:
            os.mkdir(save_dir_2)

        name = video.split('.')[0]
        # print(name)
        # Video to Picture
        label_dir1 = './runs/'
        label_dir = './runs/detect/predict/labels'
        if os.path.exists(label_dir1):
            shutil.rmtree(label_dir1)

        decode_video(video_name, save_dir_1, name)

        model = YOLO('best.pt')
        test_arr = []
        for img_name in os.listdir(save_dir_1):

            fps_name = os.path.join(save_dir_1, img_name)
            test_arr.append(fps_name)
        results = model.predict(test_arr,save_txt=True,save_conf=True)
        #

        site_all = ''
        max_score_all = 0
        for txt_name in os.listdir(label_dir):
            label_name  =  os.path.join(label_dir,txt_name)

            with open(label_name,'r') as g:
                l = g.readlines()

            site = ''
            max_score = 0
            label_dict = {'0':'other','1':'drone','2':'airplane'}
            for line in l:
                # print(line)
                this_score = float(line.split(' ')[-1].strip('\n'))
                if this_score> max_score:
                    max_score = this_score
                    site = line
                    # print(site)
            if max_score>max_score_all:
                max_score_all = max_score
                site_all = label_dict[site.split(' ')[0]]
            label = label_dict[site.split(' ')[0]]
            cor = [float(i) for i in site.split(' ')[1:5]]
            # print(label,cor)
            # picture frame
            save_img = os.path.join(save_dir_2,txt_name.replace('txt','jpg'))
            draw_rect(os.path.join(save_dir_1,txt_name.replace('txt','png')),cor,label,max_score,save_img)
            # aa

        ims_folder = save_dir_2
        video_path2 = os.path.join(result_path ,video)

        ims_path_list = os.listdir(ims_folder)
        ims_path_list.sort()
        fps = 30
        im_path = os.path.join(ims_folder, ims_path_list[0])
        im_size = Image.open(os.path.join(ims_folder, ims_path_list[0])).size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(video_path2, fourcc, fps, im_size)

        for i, im_path in enumerate(ims_path_list):
            im_path = os.path.join(ims_folder, im_path)
            frame = cv2.imread(im_path)
            videoWriter.write(frame)
            # print(im_path)
        videoWriter.release()
        # print('finish')
        with open('./result/result.txt','a') as g:
            g.write(video+' '+'predict: '+site_all+' score: '+str(max_score_all)[:4]+'\n')
        # for fps in os.listdir(save_dir_1):
        #     fps_name = os.path.join(save_dir_1,fps)
        #     results = model(fps_name)
        #     print(results)
        #     aa