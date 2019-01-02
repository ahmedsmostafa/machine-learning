from os import listdir, makedirs, path
from itertools import islice
from pickle import dump, load
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
import collections
from progressbar import progressbar
from keras.models import Model
from pycocotools.coco import COCO

dataDir = 'F:/COCO/datasets'
annotationsDir = 'annotations'
ml_type = 'train'
valDataDir = 'val'
year = '2014'
coco_train_full_path = dataDir + '/' + ml_type + year + '/'
load_if_exists=True

def get_coco(ml_type='train', dataset_type='captions', year='2014'):
    assert ml_type in ['train', 'test', 'val']
    assert dataset_type in ['instances', 'captions']
    annFile = path.join(dataDir, path.join(annotationsDir, dataset_type + '_{}.json'.format(ml_type + year)))
    print("Annotations of {0} {1} Dataset: {2}".format(ml_type, dataset_type, annFile))
    return COCO(annFile)

# load an image from filepath
def load_image(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return np.asarray(img)

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_captions(caption_as_list):
    caption_as_list = [word for word in caption_as_list if len(word) > 1]
    caption_as_list = [word for word in caption_as_list if word.isalpha()]
    return ' '.join(caption_as_list)

def load_model(is_attention):
    # load the model
    model = VGG16()
    if is_attention:
        # model = VGG16()
        model.layers.pop()
        # extract final 49x512 conv layer for context vectors
        final_conv = Reshape([49, 512])(model.layers[-4].output)
        model = Model(inputs=model.inputs, outputs=final_conv)
        print(model.summary())
    else:
        # model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        print(model.summary())
        # extract features from each photo
    return model

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def get_features_from_coco(coco, directory, model, limit):
    features = dict()
    # counter = 1
    all_img_ids = {key:value["file_name"] for key,value in coco.imgs.items()}
    for img_id, name in progressbar(take(limit if limit != -1 else len(all_img_ids), all_img_ids.items())):
        filename = directory + name
        image = load_image(filename)
        feature = model.predict(image, verbose=0)
        features[img_id] = feature
    return features

def get_features_from_directory(directory, model, limit):
    # add a counter to limit size of data
    features = dict()
    counter = 1
    for name in progressbar(listdir(directory)[0:limit if limit != -1 else len(listdir(directory))]):
        # ignore README
        if not name.endswith('.jpg'):
            continue
        counter += 1
        filename = directory + name
        image = load_image(filename)
        # extract features
        feature = model.predict(image, verbose=0)
        # get image id
        # image_id = name.split('.')[0]
        image_id = name[name.rindex('_') + 1:name.rindex('.jpg')].lstrip("0")
        # store feature
        features[int(image_id)] = feature
        # print('>%s' % name)
        if limit != -1 and counter == (limit+1):
            break;
    return features

# extract features from each photo in the coco_train_full_path
# this method could be implemented differently:
# - load the coco file (annotation or instances)
#   - Loop over image IDs (until the limit if not -1):
#       - Load the images accordingly
#       - Extract features
#       - Save into dict
# - Save into models folder
def extract_features(dataset_dir, is_attention=False, limit=-1,
                     load_if_exists=False, models_dir='models/'):
    if load_if_exists:
        print('loading existing model')
        if path.exists(models_dir + '/features.pkl'):
            print('found existing features model')
            with open(models_dir + '/features.pkl', 'rb') as pickle_file:
                features = load(pickle_file)
                return features
    # load COCO
    coco = get_coco(ml_type=ml_type, year=year)
    #load model
    model = load_model(is_attention)
    # features = get_features_from_directory(dataset_dir, model, limit)
    features = get_features_from_coco(coco, dataset_dir, model, limit)
    # save to file
    save_features(features, 'models/')
    return features

def extract_captions(features, load_if_exists=False, models_dir='models/'):
    if load_if_exists:
        if path.exists(models_dir + '/captions.pkl'):
            with open(models_dir + '/captions.pkl', 'rb') as pickle_file:
                captions = load(pickle_file)
                return captions
    captions = dict()
    translator = str.maketrans('', '', string.punctuation)
    coco = get_coco(ml_type=ml_type, year=year)
    for img_id in progressbar(features):
        caps = coco.imgToAnns[img_id]
        caps = [cap.pop('caption').strip().lower().translate(translator).split()
                for cap in caps]
        caps = [clean_captions(cap) for cap in caps]
        captions[img_id] = caps
    # save to file
    save_captions(captions, 'models/')
    return captions

# convert the loaded descriptions into a vocabulary of words
def extract_vocabulary(captions):
    # build a list of all caption strings
    all_desc = set()
    for key in captions.keys():
        [all_desc.update(d.split()) for d in captions[key]]
    return all_desc

# save descriptions to file, one per line
# format is Key <space> descriptions
def save_captions(captions, directory):
    lines = list()
    for key, desc_list in captions.items():
        for desc in desc_list:
            lines.append(str(key) + ' ' + desc)
    data = '\n'.join(lines)
    file = open(directory+'captions.txt', 'w')
    file.write(data)
    file.close()

    dump(captions, open(directory+'captions.pkl', 'wb'))

def save_features(features, directory):
    if not path.exists(directory):
        makedirs(directory)
    dump(features, open(directory+'features.pkl', 'wb'))

# make a 80/20 train/test split
def train_test_split(dataset):
    # sort the list to keep consistency
    # sorted_dataset = collections.OrderedDict(sorted(dataset.items()))
    training_count = int(round(0.8*len(dataset)))
    training_set = dict(sorted(dataset.items())[:training_count])
    testing_set = dict(sorted(dataset.items())[training_count:])
    return training_set, testing_set

def append_startend_to_caption(values):
    return ['startseq ' + ''.join(value) + ' endseq' for value in values]

def prepare_captions(dataset):
    return {key:append_startend_to_caption(value) for key,value in dataset.items()}

def load_clean_captions(filename, dataset):
    # load document
    doc = load_doc(filename)
    captions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # skip images not in the set
        if int(tokens[0]) in dataset:
            # split id from caption
            image_id, image_cap = tokens[0], tokens[1:]
            # create list
            if image_id not in captions:
                captions[int(image_id)] = list()
            # wrap caption in tokens
            cap = 'startseq ' + ' '.join(image_cap) + ' endseq'
            # store
            captions[int(image_id)].append(cap)
    return captions

def verify_datasets(images, captions):
    counter = 0
    not_found = list()
    for ikey in images:
        if ikey not in captions:
            counter += 1
            not_found.append(ikey)
    for ckey in captions:
        if ckey not in images:
            counter += 1
            not_found.append(ckey)
    print("{} keys not found when cross-checking images & captions".format(counter))
    print(not_found if len(not_found) > 0 else "")

def prepare_data():
    # extract features from all images
    print("Extracting Features:")
    features = extract_features(coco_train_full_path,
                                is_attention=False,
                                limit=500,
                                load_if_exists=load_if_exists,
                                models_dir='models/')
    print('\nExtracted Features:{}'.format(len(features)))

    # load & clean descriptions
    print("Extracting Captions:")
    captions = extract_captions(features, load_if_exists=load_if_exists)
    print('\nExtracted Captions:{}'.format(len(captions)))

    # summarize vocabulary
    vocabulary = extract_vocabulary(captions)
    print('Vocabulary Size: %d' % len(vocabulary))

    features_train, features_test = train_test_split(features)
    captions_train_dict, captions_test_dict = train_test_split(captions)

    print("features training:{}, features testing:{}".format(len(features_train), len(features_test)))
    print("captions training dict:{}, captions testing dict:{}".format(len(captions_train_dict), len(captions_test_dict)))

    # captions_train = load_clean_captions('models/captions.txt', features_train)
    # captions_test = load_clean_captions('models/captions.txt', features_test)
    captions_train = prepare_captions(captions_train_dict)
    captions_test = prepare_captions(captions_test_dict)
    print("captions training:{}, captions testing:{}".format(len(captions_train), len(captions_test)))

    verify_datasets(features_train, captions_train)
    verify_datasets(features_test, captions_test)

    return (features_train, captions_train), (features_test, captions_test)

def prepare_test():
    ml_type = 'test'
    coco_train_full_path = dataDir + '/' + ml_type + year + '/'
    return prepare_data()

if __name__ == '__main__':
    prepare_data()
