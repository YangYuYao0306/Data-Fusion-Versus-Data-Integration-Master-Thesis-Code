The project is divided into yolo target detection recognition model, audio classification model.

yolo target detection recognition model: . /yolo
Code description:
1. run ex_pic.py to extract the video into pictures for model training.
2. Run make_txt.py to convert csv annotations to yolo format annotations.
3. Place the dataset in the specified location. Modify the path
4. model training, install ultralytics package
Put the dataset configuration file coco3.yml into the pip path: mine is site-packages\ultralytics\cfg\datasets

Training: run yolo.py 

Validation: main.py Replace video with image and detect it, label the result on the image and composite the video. Categories and scores are saved in result.txt.

Audio classification model: . /AudioClassification-Pytorch-master
Code instructions:
1. put the audio file in the format under . /dataset, process the dataset: run create_data.py
2. train: run train.py
3. test: run infer.py

Note that the above code installs the necessary packages and changes paths.