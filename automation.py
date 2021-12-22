import sys
sys.path.append("..")
import json
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import glob
import copy
import numpy as np
from scipy import ndarray, ndimage
import splitfolders
import shutil
from pathlib import Path
from skimage.color import gray2rgb
import zipfile
import yaml
import concurrent.futures
import argparse
#from obj_detection import train



parentdir = Path(os.getcwd())

            



def json_from_dir(source_dir):
    all_jsons = glob.glob(os.path.join(source_dir,"*.json"))
    if all_jsons:
        all_jsons = {p:os.path.getmtime(p) for p in all_jsons}
        return sorted(all_jsons.items())[0][0]
    return None


def rotate(image_array: ndarray, angle):
    return ndimage.rotate(image_array, angle, reshape=True, order=0)
def getPoints(aug):
    new_annotations_data, annotations_data, root_dir, dest_dir, zipfilename,yo = LoadData()
    images_data = get_images(yo)
    if aug==True:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images_augmented'),
                              'annotations.json')
    else:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'),
                              'annotations.json')
    data = open(f'{images_aug}').read()
    data = json.loads(data)
    dt={}
    all_points = []
    filenames = list(data['_via_img_metadata'].keys())
    for filename in filenames:
        dt[filename] = []
        for x in data['_via_img_metadata'][filename]['regions']:
            dt[filename].append(list(x['shape_attributes'].values())[1:])
    for x, y in dt.items():
        all_points.append(y)
    dt2 = {}
    all_class = []
    filenames = list(data['_via_img_metadata'].keys())
    for filename in filenames:
        dt2[filename] = []
        for x in data['_via_img_metadata'][filename]['regions']:
            dt2[filename].append(list(x['region_attributes']['class']))
    for x, y in dt2.items():
        all_class.append(y)
    return all_points,all_class
def augment_image(img, angle, im_name):
    augmented = rotate(img, angle)
    name_parts = im_name.split(".")
    name, ext = name_parts[0], name_parts[-1]
    aug_im_name = name + "_" + str(angle) + "." + ext
    return augmented, aug_im_name


def get_and_adjust_rotation_matrix(angle, cx, cy, h, w):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cosang = np.abs(M[0, 0])
    sinang = np.abs(M[0, 1])
    nW = int((h * sinang) + (w * cosang))
    nH = int((h * cosang) + (w * sinang))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return M


def rotatePolygon(corners, M, w, h, ln=8):
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    calculated = np.dot(M,corners.T).T
    calculated = calculated.reshape(-1,ln)
    if len(np.where(calculated[:, range(0,ln,2)] > w)) == 0 or len(np.where(calculated[:, range(1,ln,2)] > h)) == 0:
        return None
    return calculated


def transform_regions(regions, rotation_matrix, w, h):
    r_list = []
    for region in regions:
        label = list(region["region_attributes"]["class"])[0]
        x1 = region["shape_attributes"]["x"]
        y1 = region["shape_attributes"]["y"]
        width = region["shape_attributes"]["width"]
        height = region["shape_attributes"]["height"]
        x2 = width + x1
        y2 = height + y1
        nl = np.array([x1,y1,x2,y1,x2,y2,x1,y2])
        p = rotatePolygon(np.array([nl]), rotation_matrix, w, h, ln=len(nl))[0]
        r_list += [
            {
                'shape_attributes': {
                    'name': 'polygon',
                    "all_points_x": [p[i] for i in range(0,len(p),2)],
                    "all_points_y": [p[i] for i in range(1,len(p),2)]
                },
                'region_attributes': {
                    'class': {
                        label: True
                    }
                }
            }
        ]
    return r_list


def get_images(yo,aug):
    print(yo)
    if aug==True:
        images = os.listdir(os.path.join(os.path.join(os.path.join(os.getcwd(),yo),'images_augmented')))
        temp = {}
        img_aug = os.path.join(os.path.join(os.getcwd(),yo),'images_augmented')
    else:
        images = os.listdir(os.path.join(os.path.join(os.path.join(os.getcwd(),yo),'images_augmented')))
        temp = {}
        img_aug = os.path.join(os.path.join(os.getcwd(),yo),'images')

    for i in images:
        try:
            temp[i] = cv2.imread(f'{img_aug}/{i}')
        except Exception as E:
            pass
    return temp
def get_final_box(corners):
    # main_array = np.array(calculated) # converting to numpy array
    x_ = corners[:4]
    y_ = corners[4:]

    xmin = np.min(x_).reshape(-1, 1)
    ymin = np.min(y_).reshape(-1, 1)
    xmax = np.max(x_).reshape(-1, 1)
    ymax = np.max(y_).reshape(-1, 1)

    ymax = ymax - ymin
    xmax = xmax - xmin
    final = np.hstack((xmin, ymin, xmax, ymax))

    return final


def normalize_for_yolo(xmin, ymin, w, h, w_img, h_img):
    xcenter = (xmin + w / 2) / h_img
    ycenter = (ymin + h / 2) / w_img
    w = w / h_img
    h = h / w_img
    return xcenter, ycenter, w, h


def making_directories():
    if not os.path.exists('split'):
        os.mkdir('split')
    os.chdir('split')
    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists('labels'):
        os.mkdir('labels')


def copying_images(yo,aug):
    if aug==True:
        src = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images_augmented'))
    else:
        src = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'))
    dst = os.path.join(os.path.join(os.path.join(parentdir, yo), "split"), "images")
    myList = os.listdir(src)
    myList.remove(myList[len(myList) - 1])
    for item in myList:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)
        else:
            shutil.copy2(s, d)


def copying_text(yo):
    src = os.path.join(os.path.join(os.path.join(parentdir, yo), 'yolo_annotation'))
    dst = os.path.join(os.path.join(os.path.join(parentdir, yo), "split"), "labels")
    myList = os.listdir(src)
    myList.remove(myList[len(myList) - 1])
    for item in myList:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)
        else:
            shutil.copy2(s, d)


def splitting_train_test(yo):
    os.chdir(os.path.join(parentdir, yo))
    splitfolders.ratio('split', output="output", seed=1337, ratio=(.8, 0.2))

def LoadData(aug):
    zipfilename = ''
    filepath = ''
    for subdir, dirs, files in os.walk(os.path.join(parentdir, 'zipfolder')):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".zip"):
                zipfilename = file
                filepath = filepath
    print(f"filepath:{filepath}")
    print(f"parentdir:{parentdir}")
    #
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        firstindex=zip_ref.filelist[0]
        zip_ref.extractall(parentdir)
    # shutil.unpack_archive(filepath, parentdir)
    yo= firstindex.filename.split('/')[0]
    if aug == True:
        root_dir = os.path.join(os.path.join(parentdir, yo), 'images')
        dest_dir = root_dir.rstrip("/") + "_augmented"
    else:
        root_dir = os.path.join(os.path.join(parentdir, yo), 'images')
        dest_dir = root_dir


    print(f"dest:{dest_dir}")
    print(f"root:{root_dir}")

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    json_path = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'),
                             'annotations.json')
    with open(json_path) as f:
        annotations_data = json.load(f)

    new_annotations_data = {
        "_via_settings": copy.deepcopy(annotations_data["_via_settings"]),
        "_via_img_metadata": {},
        "_via_attributes": copy.deepcopy(annotations_data["_via_attributes"])
    }

    return new_annotations_data,annotations_data,root_dir,dest_dir,zipfilename,yo
def make_json(path,data):
    with open(path, "w") as f:
        json.dump(data, f)
def augmentationdata(aug):
    new_annotations_data,annotations_data,root_dir,dest_dir,zipfilename,yo =  LoadData(aug)
    def download(img, angle, im_name,dest_dir,cx,cy,height,width):
        augmented, aug_im_name = augment_image(img, angle, im_name)
        aug_im_name=aug_im_name.split('.')[0]
        aug_im_name = aug_im_name+'.jpg'
        aug_im_path = os.path.join(dest_dir, aug_im_name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            executor.submit(cv2.imwrite,aug_im_path, augmented)
        rotation_matrix = get_and_adjust_rotation_matrix(angle, cx, cy, height, width)
        aug_size = os.path.getsize(aug_im_path)
        aug_im_key = aug_im_name + str(aug_size)
        new_annotations_data["_via_img_metadata"][aug_im_key] = {
        "filename": aug_im_name,
        "size": aug_size,
        "regions": transform_regions(annotations_data["_via_img_metadata"][im_key]["regions"], rotation_matrix, width, height),
        "file_attributes": {}
        }
    def getImg(img, im_name,dest_dir,cx,cy,height,width):
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                for angle in range(0,360,2):
                    executor.submit(download,img, angle, im_name,dest_dir,cx,cy,height,width)
    count = 0
    for im_key in annotations_data["_via_img_metadata"]:
        count += 1
        print("processing image "+str(count)+"/"+str(len(annotations_data["_via_img_metadata"]))+" ...")
        im_name = annotations_data["_via_img_metadata"][im_key]["filename"]
        im_path = os.path.join(root_dir,im_name)
        if os.path.exists(im_path):
            img = cv2.imread(im_path)
            if len(img.shape) is 2:
                img = gray2rgb(img)
            img = img[:,:,:3]
            height, width = img.shape[0],img.shape[1]
            cx = width//2
            cy = height//2
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                executor.submit(getImg,img, im_name,dest_dir,cx,cy,height,width)
    aug_json_path = os.path.join(dest_dir,"annotations.json")
    with open(aug_json_path,"w") as f:
        json.dump(new_annotations_data,f)
        print("Done saving in "+dest_dir)

def yoloannotation(aug):
    zipfilename = ''
    filepath = ''
    for subdir, dirs, files in os.walk(os.path.join(parentdir, 'zipfolder')):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".zip"):
                zipfilename = file
                filepath = filepath
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        firstindex = zip_ref.filelist[0]
        zip_ref.extractall(parentdir)
        # shutil.unpack_archive(filepath, parentdir)
    yo = firstindex.filename.split('/')[0]

    images_data = get_images(yo,aug)
    if aug==True:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir,yo), 'images_augmented'),
                              'annotations.json')
    else:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir,yo), 'images'),
                              'annotations.json')
    
    data = open(f'{images_aug}').read()
    data = json.loads(data)
    os.remove(filepath)

    for x in list(data['_via_img_metadata'].values()):
        filename = x['filename']
        for region in x['regions']:
            all_points_x = region['shape_attributes']['all_points_x']
            all_points_y = region['shape_attributes']['all_points_y']
            class_name = list(region['region_attributes']['class'].keys())[0]
            # print(filename, all_points_x, all_points_y,class_name)
            all_points_x.extend(all_points_y)
    if aug==True:
        os.chdir(os.path.join(os.path.join(parentdir, yo), 'images_augmented'))
    else:
        os.chdir(os.path.join(os.path.join(parentdir, yo), 'images'))
    if os.path.exists('annotations.json'):
        count = 0
        class_dt = {}
        os.chdir(os.path.join(parentdir, yo))
        if not os.path.exists('yolo_annotation'):
            os.mkdir('yolo_annotation')
        for x in list(data['_via_img_metadata'].values()):
            filename = x['filename']
            u_filename = filename.split('.')[0]
            width, height = (images_data[filename].shape)[:2]
            f = open(f'yolo_annotation/{u_filename}.txt', "w")

            for region in x['regions']:
                all_points_x = region['shape_attributes']['all_points_x']
                all_points_y = region['shape_attributes']['all_points_y']
                class_name = list(region['region_attributes']['class'].keys())[0]
                if class_dt.__contains__(class_name) == False:
                    class_dt[class_name] = count
                    count += 1

                #         print(filename, all_points_x, all_points_y,class_name)
                all_points_x.extend(all_points_y)
                four_points = get_final_box(all_points_x)
                x, y, w, h = four_points[0]
                x, y, w, h = normalize_for_yolo(x, y, w, h, width, height)

                f.write(f"{class_dt[class_name]} {x} {y} {w} {h}\n")
            #         break
            f.close()

    os.chdir(os.path.join(parentdir, yo))
    if aug==True:
        if len(os.listdir('images_augmented')) - 1 == len(os.listdir("yolo_annotation")):
            making_directories()
            copying_images(yo,aug)
            copying_text(yo)
            splitting_train_test(yo)
    else:
        if len(os.listdir('images')) - 1 == len(os.listdir("yolo_annotation")):
            making_directories()
            copying_images(yo,aug)
            copying_text(yo)
            splitting_train_test(yo)

    yamlloc = os.path.join(os.path.join(os.path.join(parentdir, 'obj_detection'), 'data'),
                           'coco128.yaml')
    print(yamlloc)

    with open(f'{yamlloc}', encoding="utf-8") as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    doc['train'] = os.path.join(os.path.join(os.path.join(parentdir, yo), 'output'), 'train')
    doc['val'] = os.path.join(os.path.join(os.path.join(parentdir,yo), 'output'), 'val')
    doc['nc'] = 2
    doc['names'] = ['cat', 'dog']
    with open(f'{yamlloc}', 'w') as f:
        yaml.dump(doc, f)


def plot_pair(images, gray=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10, 8))
    i = 0

    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1

    plt.show()

def add_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
# add_dir('automatic_training/train')
# add_dir('automatic_training/test')
# add_dir('automatic_training/train/images')
# add_dir('automatic_training/test/images')
# add_dir('automatic_training/train/masks')
# add_dir('automatic_training/test/masks')
# def store_for_Train_test(mask_path,img_path,key_data,key,typ):
#     data = json.load(open("automatic_training/images/annotation.json"))
#     Train_dic = Test_dic = {"_via_settings": data['_via_settings'], '_via_img_metadata': {},
#                             '_via_attributes': data['_via_attributes']}
#     if typ=='train':
#          Train_dic["_via_img_metadata"][key]=key_data
#          shutil.copy(img_path,'automatic_training/train/images')
#          shutil.copy(mask_path, 'automatic_training/train/masks')
#     else:
#          Test_dic["_via_img_metadata"][key]=key_data
#          shutil.copy(img_path,'automatic_training/test/images')
#          shutil.copy(mask_path, 'automatic_training/test/masks')

def UnetMasks(aug):
    new_annotations_data, annotations_data, root_dir, dest_dir,zipfilename,yo = LoadData(aug)

    add_dir(f"{parentdir}{os.path.sep}UNET{os.path.sep}train{os.path.sep}images")
    add_dir(f"{parentdir}{os.path.sep}UNET{os.path.sep}train{os.path.sep}masks")
    add_dir(f"{parentdir}{os.path.sep}UNET{os.path.sep}train{os.path.sep}masks{os.path.sep}images")
    # split_path= os.path.join(os.path.join(os.path.join(parentdir, "UNET"),"data"),"split")
    mask_dest_dir = os.path.join(os.path.join(os.path.join(os.path.join(parentdir, "UNET"),"train"),"masks"),"images")
    if aug==True:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images_augmented'),
                              'annotations.json')
    else:
        images_aug = os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'),
                              'annotations.json')
    data = open(f'{images_aug}').read()
    filenames = []
    data = json.loads(data)
    for x in list(data['_via_img_metadata'].values()):
        filenames.append(x['filename'])
    all_points, all_class = getPoints()
    print(len(all_points))
    k = 0
    for p, l, f in zip(all_points, all_class, filenames):
        if aug==True:
            im = cv2.imread(
            os.path.join(os.path.join(os.path.join(parentdir, yo), 'images_augmented'), f), 0)
        else:
            im = cv2.imread(
            os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'), f), 0)
        blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
        k = 0
        for j in p:
            x = j[0]
            y = j[1]
            ky = np.array((x, y)).T
            points = np.array(ky, dtype=np.int32)
            cv2.fillPoly(blank, [points], 255)
        f=f.split('.')[0]
        cv2.imwrite(f'{mask_dest_dir}/{f}.jpg', blank)
    if aug==True:
        src = os.path.join(os.path.join(os.path.join(parentdir,yo), 'images_augmented'))
    else:
        src = os.path.join(os.path.join(os.path.join(parentdir,yo), 'images'))
    dst = os.path.join(os.path.join(os.path.join(parentdir, "UNET"),"train"),"images")
    myList= os.listdir(src)
    myList.remove(myList[len(myList) - 1])
    for item in myList:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)
        else:
            shutil.copy2(s, d)
def MrCnnParsing(aug):
    new_annotations_data, annotations_data, root_dir, dest_dir, zipfilename,yo = LoadData(aug)
    if aug==True:
        js = json.load(open(os.path.join(os.path.join(os.path.join(parentdir, yo), 'images_augmented'),
                              'annotations.json')))
    else:
        js = json.load(open(os.path.join(os.path.join(os.path.join(parentdir, yo), 'images'),
                              'annotations.json')))
    js = js['_via_img_metadata'].values()
    df = pd.DataFrame(js)
    df = df.set_index(df['filename'] + df['size'].astype(str))
    train, test = train_test_split(df, test_size=0.2)
    x = train.to_json(orient='index')
    data = json.loads(x)
    an = list(data.values())
    add_dir(f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}train')
    add_dir(f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}val')
    with open(f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}train{os.path.sep}data.json', "w") as json_file:
        json.dump(data, json_file)
    if aug==True:
        for a in an:
            if a['filename']:
                shutil.copy(f'{parentdir}{os.path.sep}{yo}{os.path.sep}images_augmented{os.path.sep}' + a['filename'],
                            f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}train')
    else:        
        for a in an:
            if a['filename']:
                shutil.copy(f'{parentdir}{os.path.sep}{yo}{os.path.sep}images{os.path.sep}' + a['filename'],
                            f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}train')
    # print(count)
    y = test.to_json(orient='index')
    data_y = json.loads(y)
    ay = list(data_y.values())
    with open(f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}val{os.path.sep}data.json', "w") as json_file:
        json.dump(data_y, json_file)
    if aug==True:
        for b in ay:
            if b['filename']:
                shutil.copy(
                    f'{parentdir}{os.path.sep}{yo}{os.path.sep}images_augmented{os.path.sep}'+b['filename'],
                    f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}val')
    else:
        for b in ay:
            if b['filename']:
                shutil.copy(
                    f'{parentdir}{os.path.sep}{yo}{os.path.sep}images{os.path.sep}'+b['filename'],
                    f'{parentdir}{os.path.sep}MRCNN_OUT{os.path.sep}val')
    
def yolo():
    file = f'{parentdir}{os.path.sep}current training{os.path.sep}yolo.txt'
    print("Creating File")
    print(file)
    open(file, 'a').close()
    YOLOV5_DIR = os.path.join(parentdir, 'obj_detection')

    CFG_FILE_PATH = os.path.join(YOLOV5_DIR,
                                 f"models{os.path.sep}yolov5s.yaml")

    HYPS_PATH = os.path.join(YOLOV5_DIR,
                             f"data{os.path.sep}hyps{os.path.sep}hyp.scratch.yaml")

    DATA_CFG_PATH = os.path.join(YOLOV5_DIR,
                                 f"data{os.path.sep}coco128.yaml")

    YOLO_PROJECT_PATH = os.path.join(parentdir, "yolo_out_dir")
    WEIGHTS_PATH = os.path.join(YOLOV5_DIR,
                                f"weights{os.path.sep}yolov5x.pt")
    cfg = CFG_FILE_PATH
    data = DATA_CFG_PATH
    hyp = HYPS_PATH
    project_path = YOLO_PROJECT_PATH
    weights_path = WEIGHTS_PATH

    # Here we need to provide all args-parser arguments that are
    # otherwise auto-filled by YOLOv5(defaults) if we would have
    # entered the code through terminal.

    opt = argparse.Namespace(adam=False, artifact_alias='latest',
                             batch_size=20, bbox_interval=-1, bucket='', cache=False,
                             cfg=cfg, data=data, device='cpu', entity=None, epochs=1,
                             evolve=None, exist_ok=False, hyp=hyp, image_weights=False,
                             imgsz=416, label_smoothing=0.0, linear_lr=False, local_rank=-1,
                             multi_scale=False, name='exp', noautoanchor=False, nosave=False,
                             noval=False, project=project_path, quad=False, rect=False,
                             resume=False, save_period=-1, single_cls=False, sync_bn=False,
                             upload_dataset=False, weights=weights_path, workers=8, freeze=1,patience=False)

    # The arguments like batch_size, imgsz, epochs can also be filled
    # by the user at the front-end within a model form.

    # Now call the main() function from the yolov5/train.py. And pass
    # opt as an argument to the main().
    train.main(opt)
    os.remove(file)
    # rest of the code for HttpResponse
# if __name__ == "__main__":
#     pass