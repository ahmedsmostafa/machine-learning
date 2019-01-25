from os import listdir, makedirs, path
from itertools import islice
from pickle import dump, load
from keras.models import Model
from keras.layers import Dropout, Input, Dense, Flatten, LSTM, RepeatVector, TimeDistributed, Embedding, Reshape, \
    Concatenate
from keras.layers.merge import concatenate, add
from keras.layers.pooling import GlobalMaxPooling2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import string
import collections
from progressbar import progressbar
from pycocotools.coco import COCO
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from pandas import DataFrame


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
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return np.asarray(img)
    return img


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


def load_vgg_model(is_attention):
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
        # model = Model(inputs=in_layer, outputs=model.layers[-1].output)
        print(model.summary())
        # extract features from each photo
    return model


def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


# def define_model(vocab_size, max_length):
#     # feature extractor (encoder)
#     inputs1 = Input(shape=(7, 7, 512))
#     fe1 = GlobalMaxPooling2D()(inputs1)
#     fe2 = Dense(128, activation='relu')(fe1)
#     fe3 = RepeatVector(max_length)(fe2)
#     # embedding
#     inputs2 = Input(shape=(max_length,))
#     emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
#     emb3 = LSTM(256, return_sequences=True)(emb2)
#     emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
#     # merge inputs
#     merged = concatenate([fe3, emb4])
#     # language model (decoder)
#     lm2 = LSTM(500)(merged)
#     lm3 = Dense(500, activation='relu')(lm2)
#     outputs = Dense(vocab_size, activation='softmax')(lm3)
#     # tie it together [image, seq] [word]
#     model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#     # plot_model(model, show_shapes=True, to_file='plot.png')
#     return model

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def get_features_from_coco(coco, directory, model, limit):
    features = dict()
    # counter = 1
    all_img_ids = {key: value["file_name"] for key, value in coco.imgs.items()}
    for img_id, name in progressbar(take(limit if limit != -1 else len(all_img_ids), all_img_ids.items())):
        filename = directory + name
        image = load_image(filename)
        feature = model.predict(image, verbose=0)
        features[img_id] = feature
    return features


def get_features_from_directory(coco, directory, model, limit):
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
        if limit != -1 and counter == (limit + 1):
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
def extract_features(dataset_dir, is_attention=False, limit=20,
                     load_if_exists=False, models_dir='models/', label="all"):
    if load_if_exists:
        print('loading existing model')
        if path.exists(models_dir + label + '_features.pkl'):
            print('found existing features model')
            with open(models_dir + label + '_features.pkl', 'rb') as pickle_file:
                features = load(pickle_file)
                return features
    # load model
    model = load_vgg_model(is_attention)
    features = get_features_from_directory(coco, dataset_dir, model, limit)
    # features = get_features_from_coco(coco, dataset_dir, model, limit)
    # save to file
    save_features(features, models_dir, label)
    return features


def extract_captions(features, load_if_exists=False, models_dir='models/', label="all"):
    if load_if_exists:
        if path.exists(models_dir + label + '_captions.pkl'):
            with open(models_dir + label + '_captions.pkl', 'rb') as pickle_file:
                captions = load(pickle_file)
                return captions
    captions = dict()
    translator = str.maketrans('', '', string.punctuation)
    for img_id in progressbar(features):
        caps = coco.imgToAnns[img_id]
        caps = [cap.pop('caption').strip().lower().translate(translator).split()
                for cap in caps]
        caps = [clean_captions(cap) for cap in caps]
        captions[img_id] = caps
    # save to file
    save_captions(captions, models_dir, label)
    return captions


# convert the loaded descriptions into a vocabulary of words
def extract_vocabulary(captions, load_if_exists=False, models_dir="models/"):
    if load_if_exists:
        if path.exists(models_dir + '/vocab.pkl'):
            with open(models_dir + '/vocab.pkl', 'rb') as pickle_file:
                all_desc = load(pickle_file)
                return all_desc

    # build a list of all caption strings
    all_desc = set()
    for key in captions.keys():
        [all_desc.update(d.split()) for d in captions[key]]

    dump(all_desc, open(models_dir + '/vocab.pkl', 'wb'))

    return all_desc


# save descriptions to file, one per line
# format is Key <space> descriptions
def save_captions(captions, directory, label="all"):
    if not path.exists(directory):
        makedirs(directory)
    lines = list()
    for key, desc_list in captions.items():
        for desc in desc_list:
            lines.append(str(key) + ' ' + desc)
    data = '\n'.join(lines)

    file = open(directory + label + '_captions.txt', 'w')
    file.write(data)
    file.close()

    dump(captions, open(directory + label + '_captions.pkl', 'wb'))


def save_features(features, directory, label="all"):
    if not path.exists(directory):
        makedirs(directory)
    dump(features, open(directory + label + '_features.pkl', 'wb'))


# make a 80/20 train/test split
def train_test_split(dataset):
    # sort the list to keep consistency
    # sorted_dataset = collections.OrderedDict(sorted(dataset.items()))
    training_count = int(round(0.8 * len(dataset)))
    training_set = dict(sorted(dataset.items())[:training_count])
    testing_set = dict(sorted(dataset.items())[training_count:])
    return training_set, testing_set


def append_startend_to_caption(values):
    return ['startseq ' + ''.join(value) + ' endseq' for value in values]


def prepare_captions(dataset, models_dir='models/', load_if_exists=False, label="all"):
    label = label + '_prepped'
    filename = models_dir + label + '_' + 'captions.pkl'
    if load_if_exists:
        if path.exists(filename):
            with open(filename, 'rb') as pickle_file:
                prepped_caps = load(pickle_file)
                return prepped_caps
    prepped_caps = {key: append_startend_to_caption(value) for key, value in dataset.items()}
    # dump(prepped_caps, open(filename, 'wb'))
    save_captions(prepped_caps, models_dir, label)
    return prepped_caps


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
    # print("Extracting Features:")
    features = extract_features(coco_train_full_path,
                                is_attention=False,
                                limit=50,
                                load_if_exists=load_if_exists,
                                models_dir=modelDir)
    print('\nExtracted Features:{}'.format(len(features)))

    # load & clean descriptions
    # print("Extracting Captions:")
    captions = extract_captions(features,
                                load_if_exists=load_if_exists,
                                models_dir=modelDir)
    prepped_captions = prepare_captions(captions, models_dir=modelDir, load_if_exists=load_if_exists)
    print('\nExtracted Captions:{}'.format(len(captions)))

    # summarize vocabulary
    vocabulary = extract_vocabulary(captions, load_if_exists=load_if_exists, models_dir=modelDir)
    print('\nVocabulary Size: %d' % len(vocabulary))

    features_train, features_test = train_test_split(features)
    captions_train_dict, captions_test_dict = train_test_split(captions)

    print("features training:{}, features testing:{}".format(len(features_train), len(features_test)))
    print(
        "captions training dict:{}, captions testing dict:{}".format(len(captions_train_dict), len(captions_test_dict)))

    # captions_train = load_clean_captions('models/captions.txt', features_train)
    # captions_test = load_clean_captions('models/captions.txt', features_test)
    captions_train = prepare_captions(captions_train_dict, models_dir=modelDir, load_if_exists=load_if_exists,
                                      label='train')
    captions_test = prepare_captions(captions_test_dict, models_dir=modelDir, load_if_exists=load_if_exists,
                                     label='test')
    print("captions training:{}, captions testing:{}".format(len(captions_train), len(captions_test)))

    verify_datasets(features_train, captions_train)
    verify_datasets(features_test, captions_test)

    return (features, prepped_captions), (features_train, captions_train), (features_test, captions_test)


# convert a dictionary{captions} to a list[captions]
def to_lines(captions):
    all_caps = list()
    for key in captions.keys():
        all_caps.extend(captions[key])
    return all_caps


# fit a tokenizer given captions
def create_tokenizer(captions):
    lines = to_lines(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# def create_sequences(tokenizer, captions_list, photo, max_length):
#     vocab_size = len(tokenizer.word_index) + 1
#     X1, X2, y = list(), list(), list()
#     # walk through each caption for the image
#     for caption in captions_list:
#         # encode the sequence
#         # seq = tokenizer.texts_to_sequences([caption])[0]
#         seq = tokenizer.texts_to_sequences(caption)[0]
#         # split one sequence into multiple X,y pairs
#         for i in range(1, len(seq)):
#             # split into input and output pair
#             in_seq, out_seq = seq[:i], seq[i]
#             # pad input sequence
#             # in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#             in_seq = pad_sequences([in_seq])[0]
#             # encode output sequence
#             out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#             # store
#             X1.append(photo)
#             X2.append(in_seq)
#             y.append(out_seq)
#     # return np.array(X1), np.array(X2), np.array(y)
#     return array(X1), array(X2), array(y)

def create_sequences(tokenizer, captions_list, photos, max_length):
    vocab_size = len(tokenizer.word_index) + 1
    X1, X2, y = list(), list(), list()
    # walk through each caption for the image
    for key,cap_list in captions_list.items():
        for caption in cap_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    # return np.array(X1), np.array(X2), np.array(y)
    return array(X1), array(X2), array(y)

def maxlength(captions):
    lines = to_lines(captions)
    return max(len(d.split()) for d in lines)

def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, desc_list, photo, max_length)
            yield ([[in_img, in_seq], out_word])


# data generator, intended to be used in a call to model.fit_generator()
# def data_generator(captions, features, tokenizer, max_length, n_step = 1):
#     # loop until we finish training
#     while 1:
#         # loop over photo identifiers in the dataset
#         keys = list(captions.keys())
#         for i in range(0, len(keys), n_step):
#             Ximages, XSeq, y = list(), list(), list()
#             for j in range(i, min(len(keys), i + n_step)):
#                 image_id = keys[j]
#                 # retrieve photo feature input
#                 image = features[image_id][0]
#                 # retrieve text input
#                 desc = captions[image_id]
#                 # generate input-output pairs
#                 in_img, in_seq, out_word = create_sequences(tokenizer, desc, image, max_length)
#                 for k in range(len(in_img)):
#                     Ximages.append(in_img[k])
#                 XSeq.append(in_seq[k])
#                 y.append(out_word[k])
#                 # yield this batch of samples to the model
#                 yield [[array(Ximages), array(XSeq)], array(y)]

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        actual.append([desc.split()])
        predicted.append(yhat.split())
        # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    return bleu


dataDir = 'C:/COCO/datasets'
modelDir = 'C:/COCO/models/'
annotationsDir = 'annotations'
ml_type = 'train'
valDataDir = 'val'
year = '2014'
coco_train_full_path = dataDir + '/' + ml_type + year + '/'
load_if_exists = True
# load COCO
coco = get_coco(ml_type=ml_type, year=year)


def run_main():
    data = prepare_data()

    features, captions = data[0]
    train_features, train_captions = data[1]
    test_features, test_captions = data[2]

    tokenizer = create_tokenizer(train_captions)
    # max_length = max(
    #     len(s.split()) for s in [caption for subcaption in train_captions.values() for caption in subcaption])
    vocab_size = len(tokenizer.word_index) + 1
    max_length = maxlength(train_captions)

    # for img_id, img in train_features.items():
    #     img_caps = train_captions[img_id]
    #     [X1, X2, y] = create_sequences(tokenizer, img_caps, img, max_length)
    #     for k in range(len(X1)):
    #         print('aaaa')

    print('Captions Length: {}'.format(max_length))

    # define experiment
    model_name = 'baseline1'
    verbose = 2
    n_epochs = 20
    n_photos_per_update = 2
    n_batches_per_epoch = int(len(train_features) / n_photos_per_update)
    n_repeats = 3

    X1train, X2train, ytrain = create_sequences(tokenizer, train_captions, train_features, max_length)
    X1test, X2test, ytest = create_sequences(tokenizer, test_captions, test_features, max_length)
    model = define_model(vocab_size, max_length)
    checkpoint = ModelCheckpoint(modelDir + 'model_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

    steps = len(train_captions)
    for i in range(n_epochs):
        generator = data_generator(train_captions, train_features, tokenizer, max_length)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save(modelDir + 'model_' + str(i) + '.h5')

    # run experiment
    train_results, test_results = list(), list()
    for i in range(n_repeats):
        # define the model
        model = define_model(vocab_size, max_length)
        # fit model
        model.fit_generator(data_generator(train_captions, train_features, tokenizer, max_length),
                            steps_per_epoch=n_batches_per_epoch)
        # evaluate model on training data
        train_score = evaluate_model(model, train_descriptions, train_features, tokenizer, max_length)
        test_score = evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
        # store
        train_results.append(train_score)
        test_results.append(test_score)
        print('>{}: train={} test={}'.format(((i + 1), train_score, test_score)))
    # save results to file
    df = DataFrame()
    df['train'] = train_results
    df['test'] = test_results
    print(df.describe())
    df.to_csv(modelDir + model_name + '.csv', index=False)


if __name__ == '__main__':
    run_main()
