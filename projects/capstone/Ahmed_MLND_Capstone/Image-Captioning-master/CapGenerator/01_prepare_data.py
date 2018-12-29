from os import listdir, makedirs, path
from pickle import dump, load
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
from progressbar import progressbar
from keras.models import Model
from pycocotools.coco import COCO

dataDir = 'F:/COCO/datasets'
annotationsDir = 'annotations'
ml_type = 'train'
valDataDir = 'val'
year = '2014'
coco_train_full_path = dataDir + '/' + ml_type + year + '/'

# load an image from filepath
def load_image(path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return np.asarray(img)

# extract features from each photo in the coco_train_full_path
def extract_features(directory, is_attention=False, limit=100, load_if_exists=False):
    if load_if_exists:
        if path.exists('models/features.pkl'):
            with open('models/features.pkl', 'rb') as pickle_file:
                features = load(pickle_file)
                return features

    # load the model
    model = VGG16()
    features = dict()
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

    #add a counter to limit size of data
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
        #image_id = name.split('.')[0]
        image_id = name[name.rindex('_')+1:name.rindex('.jpg')].lstrip("0")
        # store feature
        features[int(image_id)] = feature
        # print('>%s' % name)
        if limit != -1 and counter == limit:
            break;
    return features

def get_coco(ml_type='train', dataset_type='captions', year='2014'):
    if ml_type != 'train' and ml_type != 'val':
        raise ValueError('get_coco ml_type accepts only "train" or "val" values only.')
    if dataset_type != 'instances' and dataset_type != 'captions':
        raise ValueError('get_coco dataset_type accepts only "instances" or "captions" values only.')

    annFile = path.join(dataDir, path.join(annotationsDir, dataset_type + '_{}.json'.format(ml_type + year)))
    print("Annotations of {0} {1} Dataset: {2}".format(ml_type, dataset_type, annFile))
    return COCO(annFile)

def preprocess_caption(caption_as_list):
    caption_as_list = [word for word in caption_as_list if len(word) > 1]
    caption_as_list = [word for word in caption_as_list if word.isalpha()]
    return ' '.join(caption_as_list)

def clean_captions(features):
    captions = dict()
    coco = get_coco(ml_type=ml_type, year=year)
    translator = str.maketrans('', '', string.punctuation)
    for img_id in features:
        caps = coco.imgToAnns[img_id]
        caps = [cap.pop('caption').strip().lower().translate(translator).split()
                for cap in caps]
        caps = [preprocess_caption(cap) for cap in caps]
        captions[img_id] = caps
    return captions

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(str(key) + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# extract features from all images
features = extract_features(coco_train_full_path,
                            is_attention=False,
                            limit=100,
                            load_if_exists=True)
print('\nExtracted Features:{}'.format(len(features)))
# save to file
if not path.exists('models/'):
    makedirs('models/')
dump(features, open('models/features.pkl', 'wb'))
# load & clean descriptions
captions = clean_captions(features)
# summarize vocabulary
vocabulary = to_vocabulary(captions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(captions, 'models/captions.txt')
