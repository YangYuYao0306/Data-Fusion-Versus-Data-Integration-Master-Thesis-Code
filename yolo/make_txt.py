import os
import cv2
import csv


txt_file = './drone'    ###########
name = 'drone'        ####################
label = 1

#  heli 4  bird 2 airplane 1 drone 3
dict1 = {'drone':3,'bird':2,'airplane':1,'helicopter':4}
save_file = './train2017'
for c in os.listdir(txt_file):
    c_name = os.path.join(txt_file,c)
    save_name = name + '_' + c.split('.')[0][-3:]
    arr=[]

    with open(c_name, 'r',encoding='utf-8') as file:
        reader = csv.reader(file)
        print(c_name)
        for c in reader:
            print(c)
            if c:
                arr.append(c)

    idx = 0
    # aa
    # print()
    # for a in arr[1:]:
    #     arr_i = a[4:]
    #
    #     print(arr_i,len(arr_i))


    for a in arr[1:]:
        save_name1 = save_name + "_{:>04d}.txt".format(idx)
        # print(a)

        arr_i = a[dict1[name]:] ###############
        obj_num = int(len(arr_i)/4)

        if obj_num>1:
            idx += 1
            continue

        with open(os.path.join(save_file,save_name1),'w',encoding='utf-8') as g:
            if arr_i[0] == '':
                line = ' \n'
                g.write(line)
                continue
            # print(obj_num)
            for i in range(obj_num):



                x0 = arr_i[0+i]
                y0 = arr_i[1+i]
                x1 = arr_i[2+i]
                y1 = arr_i[3+i]
                # x1 = float(arr_i[0+i*2])+float(x0)
                # y1=float(y0) + float(arr_i[1+i*2])
                x0 = (float(x0)+float(x1)/2)/640
                x1 = float(x1)/640
                y0 = (float(y0)+float(y1)/2)/512
                y1 = float(y1)/512
                line = str(label)+' '+str(x0)+' '+str(y0)+' '+str(x1)+' '+str(y1)+'\n'
                g.write(line)
        idx+=1
    # aa
    print(save_name)