# All of the code in this file was developed by Marcus de Carvalho without any modification

from doctest import DocFileCase
import numpy as np
import pandas as pd
import torch
import torchvision
import ssl
import gzip
import json
from tqdm import tqdm
from torchvision.datasets.utils import download_url
from MySingletons import MyWord2Vec
from nltk.tokenize import TweetTokenizer
import os
import tarfile
from lxml import etree
import nltk
import re

import torch.nn as nn
from transformers import BertModel

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                    max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                    information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                    num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # # Feed input to classifier to compute logits
        # logits = self.classifier(last_hidden_state_cls)

        return last_hidden_state_cls

class MyCustomAmazonReviewNIPSDataLoader(torch.utils.data.Dataset):
    dataset_url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_stars.tar.gz'
    path = 'data/AmazonReviewNIPS/'
    compressed_filename = path + 'processed_stars.tar.gz'

    books_file_path = path + 'processed_stars/books/all_balanced.review'
    dvd_file_path = path + 'processed_stars/dvd/all_balanced.review'
    electronics_file_path = path + 'processed_stars/electronics/all_balanced.review'
    kitchen_file_path = path + 'processed_stars/kitchen/all_balanced.review'

    @property
    def datasets(self):
        return self.df

    def __init__(self, folder):
        torchvision.datasets.utils.download_url(self.dataset_url, self.path)
        tar = tarfile.open(self.compressed_filename)
        tar.extractall(self.path)
        tar.close()

        if folder == 'books':
            filename_path = self.books_file_path
        elif folder == 'dvd':
            filename_path = self.dvd_file_path
        elif folder == 'electronics':
            filename_path = self.electronics_file_path
        elif folder == 'kitchen':
            filename_path = self.kitchen_file_path

        self.df = self.get_df(filename_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]

        if idx < len(self):
            return {'x': item['x'], 'y': item['targets']}
        else:
            return None

    def get_df(self, path):
        try:
            print('Trying to load processed file %s.h5 from disc...' % path)
            df = pd.read_hdf(path_or_buf=path + '.h5',
                             key='df')
        except:
            print('Processed file does not exists')
            print('Reading dataset into memory and applying Word2Vec...')

            if path == self.books_file_path:
                total = 5501
            elif path == self.dvd_file_path:
                total = 5518
            elif path == self.electronics_file_path:
                total = 5901
            elif path == self.kitchen_file_path:
                total = 5149

            line_count = 0
            df = {}
            MyWord2Vec().get()
            pbar = tqdm(unit=' samples', total=total)
            for line in open(path, 'rb'):
                word_count = 0
                vector = np.zeros(MyWord2Vec().get().vector_size)
                for word in line.decode('utf-8').split(' '):
                    x, y = word.split(':')
                    if x != '#label#':
                        for j in range(int(y)):
                            for xx in x.split('_'):
                                try:
                                    vector += MyWord2Vec().get()[xx]
                                    word_count += 1
                                except:
                                    pass
                    else:
                        try:
                            df[line_count] = {'x': vector / word_count, 'targets': int(float(y.replace('\n', '')))}
                        except:
                            df[line_count] = {'x': vector / word_count, 'targets': int(float(y))}
                line_count += 1
                pbar.update(1)
            pbar.close()

            print('Saving processed tokenized dataset in disc for future usage...')
            df = pd.DataFrame.from_dict(df, orient='index')
            df.to_hdf(path_or_buf=path + '.h5',
                      key='df',
                      mode='w',
                      format='table',
                      complevel=9,
                      complib='bzip2')
            df = pd.read_hdf(path_or_buf=path + '.h5',
                             key='df')

        return df


class MyCustomNewsPopularityDataLoader(torch.utils.data.Dataset):
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv'
    path = 'data/UCIMultiSourceNews/'
    filename = 'News_Final.csv'

    @property
    def datasets(self):
        return self.df

    def __init__(self, topic: str, social_feed: str):
        torchvision.datasets.utils.download_url(self.dataset_url, self.path)
        path = (self.path + topic + '_' + social_feed + '.h5').lower()

        try:
            print('Trying to load processed file %s from disc...' % path)
            self.df = pd.read_hdf(path_or_buf=path,
                                  key='df')
        except:
            print('Processed file does not exists')
            print('Reading dataset into memory and applying Word2Vec...')

            self.df = {}
            df = pd.read_csv(self.path + self.filename)

            if social_feed == 'all':
                df = df.loc[df['Topic'] == topic][['Title', 'Headline', 'Facebook', 'GooglePlus', 'LinkedIn']]
            else:
                df = df.loc[df['Topic'] == topic][['Title', 'Headline', social_feed]]
            df['targets'] = df[df.columns[2:]].sum(axis=1)
            df = df[['Title', 'Headline', 'targets']]
            df.loc[df['targets'] <= 10, 'targets'] = 0
            df.loc[df['targets'] > 10, 'targets'] = 1
            df['fullText'] = df['Title'].astype(str) + ' ' + df['Headline'].astype(str)

            tokenizer = TweetTokenizer()
            MyWord2Vec().get()

            sample_count = 0
            pbar = tqdm(unit=' samples', total=len(df))
            for _, row in df.iterrows():
                word_counter = 0
                vector = np.zeros(MyWord2Vec().get().vector_size)
                try:
                    for word in tokenizer.tokenize(row['fullText']):
                        vector += MyWord2Vec().get()[word]
                        word_counter += 1
                except:
                    pass
                if word_counter > 0:
                    self.df[sample_count] = {'x': vector / word_counter, 'targets': int(row['targets'])}
                    sample_count += 1
                    pbar.update(1)
            pbar.close()

            print('Saving processed tokenized dataset in disc for future usage...')
            self.df = pd.DataFrame.from_dict(self.df, orient='index')
            self.df = pd.DataFrame([{x: y for x, y in enumerate(item)} for item in self.df['x'].values.tolist()],
                                   index=self.df.index).assign(targets=self.df['targets'].tolist())
            self.df.to_hdf(path_or_buf=path,
                           key='df',
                           mode='w',
                           format='table',
                           complevel=9,
                           complib='bzip2')
        self.df = pd.read_hdf(path_or_buf=path,
                              key='df')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]

        if idx < len(self):
            return {'x': item.drop('targets').to_numpy(), 'y': int(item['targets'])}
        else:
            return None


class MyCustomAmazonReviewACLDataLoader(torch.utils.data.Dataset):
    dataset_url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz'
    path = 'data/AmazonReviewACL/'
    compressed_filename = path + 'unprocessed.tar.gz'

    apparel_file_path = path + 'sorted_data/apparel/all.review'
    automotive_file_path = path + 'sorted_data/automotive/all.review'
    baby_file_path = path + 'sorted_data/baby/all.review'
    beauty_file_path = path + 'sorted_data/beauty/all.review'
    books_file_path = path + 'sorted_data/books/all.review'
    camera_photo_file_path = path + 'sorted_data/camera_&_photo/all.review'
    cell_phones_service_file_path = path + 'sorted_data/cell_phones_&_service/all.review'
    computer_video_games_file_path = path + 'sorted_data/computer_&_video_games/all.review'
    dvd_file_path = path + 'sorted_data/dvd/all.review'
    electronics_file_path = path + 'sorted_data/electronics/all.review'
    gourmet_food_file_path = path + 'sorted_data/gourmet_food/all.review'
    grocery_file_path = path + 'sorted_data/grocery/all.review'
    health_personal_care_file_path = path + 'sorted_data/health_&_personal_care/all.review'
    jewelry_watches_file_path = path + 'sorted_data/jewelry_&_watches/all.review'
    kitchen_housewares_file_path = path + 'sorted_data/kitchen_&_housewares/all.review'
    magazines_file_path = path + 'sorted_data/magazines/all.review'
    music_file_path = path + 'sorted_data/music/all.review'
    musical_instruments_file_path = path + 'sorted_data/musical_instruments/all.review'
    office_products_file_path = path + 'sorted_data/office_products/all.review'
    outdoor_living_file_path = path + 'sorted_data/outdoor_living/all.review'
    software_file_path = path + 'sorted_data/software/all.review'
    sports_outdoors_file_path = path + 'sorted_data/sports_&_outdoors/all.review'
    tools_hardware_file_path = path + 'sorted_data/tools_&_hardware/all.review'
    toys_games_file_path = path + 'sorted_data/toys_&_games/all.review'
    video_file_path = path + 'sorted_data/video/all.review'

    @property
    def datasets(self):
        return self.df

    def __init__(self, folder):
        torchvision.datasets.utils.download_url(self.dataset_url, self.path)
        tar = tarfile.open(self.compressed_filename)
        tar.extractall(self.path)
        tar.close()

        if folder == 'apparel': filename_path = self.apparel_file_path
        if folder == 'automotive': filename_path = self.automotive_file_path
        if folder == 'baby': filename_path = self.baby_file_path
        if folder == 'beauty': filename_path = self.beauty_file_path
        if folder == 'books': filename_path = self.books_file_path
        if folder == 'camera_photo': filename_path = self.camera_photo_file_path
        if folder == 'cell_phones_service': filename_path = self.cell_phones_service_file_path
        if folder == 'computer_video_games': filename_path = self.computer_video_games_file_path
        if folder == 'dvd': filename_path = self.dvd_file_path
        if folder == 'electronics': filename_path = self.electronics_file_path
        if folder == 'gourmet_food': filename_path = self.gourmet_food_file_path
        if folder == 'grocery': filename_path = self.grocery_file_path
        if folder == 'health_personal_care': filename_path = self.health_personal_care_file_path
        if folder == 'jewelry_watches': filename_path = self.jewelry_watches_file_path
        if folder == 'kitchen_housewares': filename_path = self.kitchen_housewares_file_path
        if folder == 'magazines': filename_path = self.magazines_file_path
        if folder == 'music': filename_path = self.music_file_path
        if folder == 'musical_instruments': filename_path = self.musical_instruments_file_path
        if folder == 'office_products': filename_path = self.office_products_file_path
        if folder == 'outdoor_living': filename_path = self.outdoor_living_file_path
        if folder == 'software': filename_path = self.software_file_path
        if folder == 'sports_outdoors': filename_path = self.sports_outdoors_file_path
        if folder == 'tools_hardware': filename_path = self.tools_hardware_file_path
        if folder == 'toys_games': filename_path = self.toys_games_file_path
        if folder == 'video': filename_path = self.video_file_path

        self.df = self.get_df(filename_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]

        if idx < len(self):
            return {'x': item['x'], 'y': item['targets']}
        else:
            return None

    def get_df(self, path):
        try:
            os.remove(path + '.xml')
        except:
            pass

        try:
            print('Trying to load processed file %s.h5 from disc...' % path)
            df = pd.read_hdf(path_or_buf=path + '.h5',
                             key='df')
        except:
            print('Processed file does not exists')
            print('Reading dataset into memory and applying Word2Vec...')

            with open(path + '.xml', 'w', encoding='utf-8-sig') as f:
                f.write('<amazonreview>')
                for line in open(path, 'rb'):
                    f.write(line.decode(encoding='utf-8-sig', errors='ignore'))
                f.write('</amazonreview>')

            parser = etree.XMLParser(recover=True)
            with open(path + '.xml', 'r', encoding='utf-8-sig') as f:
                contents = f.read()
            tree = etree.fromstring(contents, parser=parser)

            df = {}
            tokenizer = TweetTokenizer()
            MyWord2Vec().get()
            line_count = 0
            pbar = tqdm(unit=' samples', total=len(tree.findall('review')) - 1)
            for review in tree.findall('review'):
                word_count = 0
                vector = np.zeros(MyWord2Vec().get().vector_size)
                try:
                    for word in tokenizer.tokenize(review.find('review_text').text):
                        try:
                            vector += MyWord2Vec().get()[word]
                            word_count += 1
                        except:
                            pass
                    if word_count > 0:
                        try:
                            score = int(float(review.find('rating').text.replace('\n', '')))
                            if type(score) is int:
                                df[line_count] = {'x': vector / word_count, 'targets': score}
                                line_count += 1
                                pbar.update(1)
                        except:
                            pass
                except:
                    pass
            pbar.close()

            print('Saving processed tokenized dataset in disc for future usage...')
            df = pd.DataFrame.from_dict(df, orient='index')
            df.to_hdf(path_or_buf=path + '.h5',
                      key='df',
                      mode='w',
                      format='table',
                      complevel=9,
                      complib='bzip2')
            df = pd.read_hdf(path_or_buf=path + '.h5',
                             key='df')

        try:
            os.remove(path + '.xml')
        except:
            pass

        return df


class MyCustomMNISTUSPSDataLoader(torch.utils.data.Dataset):
    datasets = []
    transforms = None

    def __init__(self, datasets, transforms: torchvision.transforms = None):
        self.datasets = datasets
        self.transforms = transforms

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        offset = 0
        dataset_idx = 0
        sample = None
        if idx < len(self):
            while sample is None:
                if idx < (offset + len(self.datasets[dataset_idx])):
                    sample = self.datasets[dataset_idx][idx - offset]
                else:
                    offset += len(self.datasets[dataset_idx])
                    dataset_idx += 1
        else:
            return None

        x = sample[0]
        for transform in self.transforms:
            x = transform(x)
        return {'x': x, 'y': sample[1]}

class MyCustomCIFAR10STL10DataLoader(torch.utils.data.Dataset):
    datasets = []
    transforms = None
    resnet = None
    samples = None

    def __init__(self, datasets, transforms: torchvision.transforms = None):
        self.datasets = []
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet.fc_backup = self.resnet.fc
        self.resnet.fc = torch.nn.Sequential()
        if isinstance(self, CIFAR10):
            for dataset in datasets:
                idx_to_delete = np.where(np.array([dataset.targets]) == 6)[1]
                dataset.targets = list(np.delete(np.array(dataset.targets), idx_to_delete))
                dataset.data = np.delete(dataset.data, idx_to_delete, 0)
                self.datasets.append(dataset)
        elif isinstance(self, STL10):
            for dataset in datasets:
                idx_to_delete = np.where(np.array([dataset.labels]) == 7)[1]
                dataset.labels = list(np.delete(np.array(dataset.labels), idx_to_delete))
                dataset.data = np.delete(dataset.data, idx_to_delete, 0)
                self.datasets.append(dataset)
        self.transforms = transforms


    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        offset = 0
        dataset_idx = 0
        sample = None
        if idx < len(self):
            while sample is None:
                if idx < (offset + len(self.datasets[dataset_idx])):
                    sample = self.datasets[dataset_idx][idx - offset]
                else:
                    offset += len(self.datasets[dataset_idx])
                    dataset_idx += 1
        else:
            return None

        x = sample[0]
        for transform in self.transforms:
            x = transform(x)
        x = x.unsqueeze(0)

        if torch.cuda.is_available():
            x = x.to('cuda')
            self.resnet.to('cuda')

        if isinstance(self, CIFAR10):
            if sample[1] == 0:
                y = 0  # Airplane
            elif sample[1] == 1:
                y = 1  # Automobile
            elif sample[1] == 2:
                y = 2  # Bird
            elif sample[1] == 3:
                y = 3  # Cat
            elif sample[1] == 4:
                y = 4  # Deer
            elif sample[1] == 5:
                y = 5  # Dog
            elif sample[1] == 7:
                y = 6  # Horse
            elif sample[1] == 8:
                y = 7  # Ship
            elif sample[1] == 9:
                y = 8  # Truck
        elif isinstance(self, STL10):
            if sample[1] == 0:
                y = 0  # Airplane
            elif sample[1] == 1:
                y = 2  # Bird
            elif sample[1] == 2:
                y = 1  # Car
            elif sample[1] == 3:
                y = 3  # Cat
            elif sample[1] == 4:
                y = 4  # Deer
            elif sample[1] == 5:
                y = 5  # Dog
            elif sample[1] == 6:
                y = 6  # Horse
            elif sample[1] == 8:
                y = 7  # Ship
            elif sample[1] == 9:
                y = 8  # Truck
        with torch.no_grad():
            x = self.resnet(x)[0].to('cpu')
        return {'x': x, 'y': y}


class USPS(MyCustomMNISTUSPSDataLoader):
    def __init__(self, transform: torchvision.transforms = None):
        ssl._create_default_https_context = ssl._create_unverified_context
        datasets = []
        datasets.append(torchvision.datasets.USPS(root='./data', train=True, download=True))
        datasets.append(torchvision.datasets.USPS(root='./data', train=False, download=True))

        MyCustomMNISTUSPSDataLoader.__init__(self, datasets, transform)


class MNIST(MyCustomMNISTUSPSDataLoader):
    def __init__(self, transform: torchvision.transforms = None):
        ssl._create_default_https_context = ssl._create_unverified_context
        datasets = []
        datasets.append(torchvision.datasets.MNIST(root='./data', train=True, download=True))
        datasets.append(torchvision.datasets.MNIST(root='./data', train=False, download=True))

        MyCustomMNISTUSPSDataLoader.__init__(self, datasets, transform)


class CIFAR10(MyCustomCIFAR10STL10DataLoader):
    def __init__(self, transform: torchvision.transforms = None):
        ssl._create_default_https_context = ssl._create_unverified_context
        datasets = []
        datasets.append(torchvision.datasets.CIFAR10(root='./data', train=True, download=True))
        datasets.append(torchvision.datasets.CIFAR10(root='./data', train=False, download=True))

        MyCustomCIFAR10STL10DataLoader.__init__(self, datasets, transform)


class STL10(MyCustomCIFAR10STL10DataLoader):
    def __init__(self, transform: torchvision.transforms = None):
        ssl._create_default_https_context = ssl._create_unverified_context
        datasets = []
        datasets.append(torchvision.datasets.STL10(root='./data', split='train', download=True))
        datasets.append(torchvision.datasets.STL10(root='./data', split='test', download=True))

        MyCustomCIFAR10STL10DataLoader.__init__(self, datasets, transform)



class AmazonReviewBertDataLoader(torch.utils.data.Dataset):
    base_url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/'
    path = 'data/AmazonReview/'
    df = None

    @property
    def datasets(self):
        return self.df

    def __init__(self, filename):
        torchvision.datasets.utils.download_url(self.base_url + filename, self.path)
        self.df = self.get_bert(self.path + filename) # bert tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]

        if idx < len(self):
            try:
                return {'x': item.drop('overall').to_numpy(), 'y': item['overall']}
            except:
                return self.__getitem__(idx - 1)
        else:
            return None

    def normalize(self, a: int = 0, b: int = 1):
        assert a < b
        for feature_name in self.df.drop('overall', axis=1).columns:
            max_value = self.df[feature_name].max()
            min_value = self.df[feature_name].min()
            self.df[feature_name] = (b - a) * (self.df[feature_name] - min_value) / (max_value - min_value) + a

    @staticmethod
    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.loads(l)

    def get_bert(self, path):
        try:
            print('Trying to load processed file %sinputs.h5 from disc...' % path)
            df = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'outputs.h5'),
                             key='df', mode='r')

        except:
            print('Trying to load processed file %s from disc...' % path)
            
            def parse(path):
                g = gzip.open(path, 'rb')
                for l in g:
                    # yield eval('l')
                    yield json.loads(l)

            def getDF(path):
                i = 0
                df = {}
                for d in parse(path):
                    df[i] = d
                    i += 1
                return pd.DataFrame.from_dict(df, orient='index')

            df = getDF(path)

            df.to_csv(path+'.csv')

            mydf = df.dropna(subset=['reviewText'])
            X = mydf.reviewText.values
            Y = mydf.overall.values

            # # Uncomment to download "stopwords"
            # nltk.download("stopwords")
            # from nltk.corpus import stopwords

            # def text_preprocessing(s):
            #     """
            #     - Lowercase the sentence
            #     - Change "'t" to "not"
            #     - Remove "@name"
            #     - Isolate and remove punctuations except "?"
            #     - Remove other special characters
            #     - Remove stop words except "not" and "can"
            #     - Remove trailing whitespace
            #     """
            #     s = s.lower()
            #     # Change 't to 'not'
            #     s = re.sub(r"\'t", " not", s)
            #     # Remove @name
            #     s = re.sub(r'(@.*?)[\s]', ' ', s)
            #     # Isolate and remove punctuations except '?'
            #     s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
            #     s = re.sub(r'[^\w\s\?]', ' ', s)
            #     # Remove some special characters
            #     s = re.sub(r'([\;\:\|•«\n])', ' ', s)
            #     # Remove stopwords except 'not' and 'can'
            #     s = " ".join([word for word in s.split()
            #                 if word not in stopwords.words('english')
            #                 or word in ['not', 'can']])
            #     # Remove trailing whitespace
            #     s = re.sub(r'\s+', ' ', s).strip()
                
            #     return s

            # from sklearn.feature_extraction.text import TfidfVectorizer

            # # Preprocess text
            # X_preprocessed = np.array([text_preprocessing(text) for i, text in enumerate(X)])

            # # Calculate TF-IDF
            # tf_idf = TfidfVectorizer(ngram_range=(1, 3),
            #                         binary=True,
            #                         smooth_idf=False)
            # X_tfidf = tf_idf.fit_transform(X_preprocessed)

            def text_preprocessing(text):
                """
                - Remove entity mentions (eg. '@united')
                - Correct errors (eg. '&amp;' to '&')
                @param    text (str): a string to be processed.
                @return   text (Str): the processed string.
                """
                # Remove '@name'
                text = re.sub(r'(@.*?)[\s]', ' ', text)

                # Replace '&amp;' with '&'
                text = re.sub(r'&amp;', '&', text)

                # Remove trailing whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                return text

            # Print sentence 0
            print('Original: ', X[0])
            print('Processed: ', text_preprocessing(X[0]))

            from transformers import BertTokenizer

            # Load the BERT tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

            # Create a function to tokenize a set of texts
            def preprocessing_for_bert(data):
                """Perform required preprocessing steps for pretrained BERT.
                @param    data (np.array): Array of texts to be processed.
                @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
                @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                            tokens should be attended to by the model.
                """
                # Create empty lists to store outputs
                input_ids = []
                attention_masks = []

                # For every sentence...
                for sent in data:
                    # `encode_plus` will:
                    #    (1) Tokenize the sentence
                    #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
                    #    (3) Truncate/Pad sentence to max length
                    #    (4) Map tokens to their IDs
                    #    (5) Create attention mask
                    #    (6) Return a dictionary of outputs
                    encoded_sent = tokenizer.encode_plus(
                        text=text_preprocessing(sent),  # Preprocess sentence
                        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                        max_length=MAX_LEN,                  # Max length to truncate/pad
                        pad_to_max_length=True,         # Pad sentence to max length
                        #return_tensors='pt',           # Return PyTorch tensor
                        return_attention_mask=True      # Return attention mask
                        )
                    
                    # Add the outputs to the lists
                    input_ids.append(encoded_sent.get('input_ids'))
                    attention_masks.append(encoded_sent.get('attention_mask'))

                # Convert lists to tensors
                input_ids = torch.tensor(input_ids)
                attention_masks = torch.tensor(attention_masks)

                return input_ids, attention_masks

            # Concatenate train data and test data
            all_reviews = X

            # Encode our concatenated data
            encoded_reviews = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_reviews]

            # Find the maximum length
            max_len = max([len(sent) for sent in encoded_reviews])
            print('Max length: ', max_len)
            
            # Specify `MAX_LEN`
            MAX_LEN = 128

            # Print sentence 0 and its encoded token ids
            token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
            print('Original: ', X[0])
            print('Token IDs: ', token_ids)

            # Run function `preprocessing_for_bert` on the train set and the validation set
            print('Tokenizing data...')
            train_inputs, train_masks = preprocessing_for_bert(X)

            # Convert other data types to torch.Tensor
            train_labels = torch.tensor(Y)

            # # Create the DataLoader for our training set
            # train_data = TensorDataset(train_inputs, train_masks, train_labels)

            # print('Saving processed tokenized dataset in disc for future usage...')
            # df_ = pd.DataFrame(train_inputs.numpy())
            # df_['overall'] = train_labels.tolist()

            # df_.to_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'inputs.h5'),
            #             key='df',
            #             mode='w',
            #             format='table',
            #             complevel=9,
            #             complib='bzip2')
            # df_input = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'inputs.h5'),
            #                     key='df')

            # print('Saving processed tokenized dataset in disc for future usage...')
            # df_ = pd.DataFrame(train_masks.numpy())

            # df_.to_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'masks.h5'),
            #             key='df',
            #             mode='w',
            #             format='table',
            #             complevel=9,
            #             complib='bzip2')
            # df_mask = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'masks.h5'),
            #                     key='df')

            encoder = BertClassifier(freeze_bert=True)
            import math
            n_epoch = math.ceil(train_inputs.shape[0]/128)
            for i in range(n_epoch):
                outputs = encoder(train_inputs[i*128:(i+1)*128].to(), train_masks[i*128:(i+1)*128])
                if i==0:
                    output = outputs
                else:
                    output = torch.cat((output,outputs),0)

            print('Saving processed tokenized dataset in disc for future usage...')
            df_ = pd.DataFrame(output.numpy())
            df_['overall'] = train_labels.tolist()
            df_.to_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'outputs.h5'),
                        key='df',
                        mode='w',
                        format='table',
                        complevel=9,
                        complib='bzip2')
            df = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + 'outputs.h5'),
                                key='df')
        return df


class DataManipulator:
    data = None
    __number_samples = None
    __number_features = None
    __number_classes = None
    __padding = 0
    concept_drift_noise = None
    n_concept_drifts = 1

    def concept_drift(self, x, idx):
        if idx == 0:
            return x

        def normalize(x, a: int = 0, b: int = 1):
            assert a < b
            return (b - a) * (x - np.min(x)) / (np.max(x) - np.min(x)) + a

        if self.concept_drift_noise is None:
            self.concept_drift_noise = []
            for i in range(self.n_concept_drifts - 1):
                np.random.seed(seed=self.n_concept_drifts * self.n_concept_drifts + i)
                self.concept_drift_noise.append((np.random.rand(self.number_features())) + 1)  # Random on range [0, 2)
                np.random.seed(seed=None)
        return normalize(x * self.concept_drift_noise[idx - 1], np.min(x), np.max(x))

    def number_classes(self, force_count: bool = False):
        if self.__number_classes is None or force_count:
            try:
                self.__min_class = int(np.min([np.min(d.targets) for d in self.data.datasets]))
                self.__max_class = int(np.max([np.max(d.targets) for d in self.data.datasets]))
            except TypeError:
                self.__min_class = int(np.min([np.min(d.targets.numpy()) for d in self.data.datasets]))
                self.__max_class = int(np.max([np.max(d.targets.numpy()) for d in self.data.datasets]))
            except AttributeError:
                try:
                    self.__min_class = int(np.min(self.data.datasets.overall.values))
                    self.__max_class = int(np.max(self.data.datasets.overall.values))
                except:
                    try:
                        self.__min_class = int(np.min(self.data.datasets.targets.values))
                        self.__max_class = int(np.max(self.data.datasets.targets.values))
                    except:
                        self.__min_class = int(np.min([np.min(d.labels) for d in self.data.datasets]))
                        self.__max_class = int(np.max([np.max(d.labels) for d in self.data.datasets]))
            self.__number_classes = len(range(self.__min_class, self.__max_class + 1))
            if isinstance(self.data, CIFAR10) or isinstance(self.data, STL10):
                self.__number_classes = self.__number_classes - 1

        return self.__number_classes

    def number_features(self, force_count: bool = False, specific_sample: int = None):
        if self.__number_features is None or force_count or specific_sample is not None:
            if specific_sample is None:
                idx = 0
            else:
                idx = specific_sample
            self.__number_features = int(np.prod(self.get_x(idx).shape))

        return self.__number_features

    def number_samples(self, force_count: bool = False):
        if self.__number_samples is None or force_count:
            self.__number_samples = len(self.data)

        return self.__number_samples

    def get_x_from_y(self, y: int, idx: int = 0, random_idx: bool = False):
        x = None
        if random_idx:
            while x is None:
                idx = np.random.randint(0, self.number_samples())
                temp_x, temp_y = self.get_x_y(idx)
                if np.argmax(temp_y) == y:
                    x = temp_x
        else:
            while x is None:
                temp_x, temp_y = self.get_x_y(idx)
                if np.argmax(temp_y) == y:
                    x = temp_x
                else:
                    idx += 1
        return x

    def get_x_y(self, idx: int):
        data = self.data[idx]
        if self.__padding > 0:
            m = torch.nn.ConstantPad2d(self.__padding, 0)
            x = m(data['x']).flatten().numpy()
        else:
            if type(data['x']) is np.ndarray:
                x = data['x']
            else:
                x = data['x'].flatten().numpy()

        y = np.zeros(self.number_classes())
        y[int((data['y'] - self.__min_class))] = 1

        x = self.concept_drift(x, int(idx / (self.number_samples() / self.n_concept_drifts)))
        return x, y

    def get_x(self, idx: int):
        x, _ = self.get_x_y(idx)
        return x

    def get_y(self, idx: int):
        _, y = self.get_x_y(idx)
        return y

    def load_mnist(self, resize: int = None, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        if resize is None:
            self.data = MNIST([torchvision.transforms.ToTensor()])
        else:
            self.data = MNIST([torchvision.transforms.Resize(resize),
                               torchvision.transforms.ToTensor()])

    def load_usps(self, resize: int = None, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        if resize is None:
            self.data = USPS([torchvision.transforms.ToTensor()])
        else:
            self.data = USPS([torchvision.transforms.Resize(resize),
                              torchvision.transforms.ToTensor()])

    def load_cifar10(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = CIFAR10([torchvision.transforms.Resize(224),  # Resize to RESNET
                             torchvision.transforms.ToTensor()])

    def load_stl10(self, resize: int = None, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        if resize is None:
            self.data = STL10([torchvision.transforms.Resize(224),  # Resize to RESNET
                               torchvision.transforms.ToTensor()])

    def load_amazon_review_fashion(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('AMAZON_FASHION_5.json.gz')

    def load_amazon_review_all_beauty(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('All_Beauty_5.json.gz')

    def load_amazon_review_appliances(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Appliances_5.json.gz')

    def load_amazon_review_arts_crafts_sewing(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Arts_Crafts_and_Sewing_5.json.gz')

    def load_amazon_review_automotive(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Automotive_5.json.gz')

    def load_amazon_review_books(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Books_5.json.gz')

    def load_amazon_review_cds_vinyl(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('CDs_and_Vinyl_5.json.gz')

    def load_amazon_review_cellphones_accessories(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Cell_Phones_and_Accessories_5.json.gz')

    def load_amazon_review_clothing_shoes_jewelry(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Clothing_Shoes_and_Jewelry_5.json.gz')

    def load_amazon_review_digital_music(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Digital_Music_5.json.gz')

    def load_amazon_review_electronics(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Electronics_5.json.gz')

    def load_amazon_review_gift_card(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Gift_Cards_5.json.gz')

    def load_amazon_review_grocery_gourmet_food(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Grocery_and_Gourmet_Food_5.json.gz')

    def load_amazon_review_home_kitchen(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Home_and_Kitchen_5.json.gz')

    def load_amazon_review_industrial_scientific(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Industrial_and_Scientific_5.json.gz')

    def load_amazon_review_kindle_store(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Kindle_Store_5.json.gz')

    def load_amazon_review_luxury_beauty(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Luxury_Beauty_5.json.gz')

    def load_amazon_review_magazine_subscription(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Magazine_Subscriptions_5.json.gz')

    def load_amazon_review_movies_tv(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Movies_and_TV_5.json.gz')

    def load_amazon_review_musical_instruments(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Musical_Instruments_5.json.gz')

    def load_amazon_review_office_products(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Office_Products_5.json.gz')

    def load_amazon_review_patio_lawn_garden(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Patio_Lawn_and_Garden_5.json.gz')

    def load_amazon_review_pet_supplies(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Pet_Supplies_5.json.gz')

    def load_amazon_review_prime_pantry(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Prime_Pantry_5.json.gz')

    def load_amazon_review_software(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Software_5.json.gz')

    def load_amazon_review_sports_outdoors(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Sports_and_Outdoors_5.json.gz')

    def load_amazon_review_tools_home_improvements(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Tools_and_Home_Improvement_5.json.gz')

    def load_amazon_review_toys_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Toys_and_Games_5.json.gz')

    def load_amazon_review_video_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = AmazonReviewBertDataLoader('Video_Games_5.json.gz')

    def load_amazon_review_nips_books(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewNIPSDataLoader('books')

    def load_amazon_review_nips_dvd(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewNIPSDataLoader('dvd')

    def load_amazon_review_nips_electronics(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewNIPSDataLoader('electronics')

    def load_amazon_review_nips_kitchen(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewNIPSDataLoader('kitchen')

    def load_amazon_review_acl_apparel(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('apparel')

    def load_amazon_review_acl_automotive(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('automotive')

    def load_amazon_review_acl_baby(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('baby')

    def load_amazon_review_acl_beauty(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('beauty')

    def load_amazon_review_acl_books(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('books')

    def load_amazon_review_acl_camera_photo(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('camera_photo')

    def load_amazon_review_acl_cell_phones_service(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('cell_phones_service')

    def load_amazon_review_acl_computer_video_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('computer_video_games')

    def load_amazon_review_acl_dvd(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('dvd')

    def load_amazon_review_acl_electronics(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('electronics')

    def load_amazon_review_acl_gourmet_food(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('gourmet_food')

    def load_amazon_review_acl_grocery(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('grocery')

    def load_amazon_review_acl_health_personal_care(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('health_personal_care')

    def load_amazon_review_acl_jewelry_watches(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('jewelry_watches')

    def load_amazon_review_acl_kitchen_housewares(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('kitchen_housewares')

    def load_amazon_review_acl_magazines(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('magazines')

    def load_amazon_review_acl_music(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('music')

    def load_amazon_review_acl_musical_instruments(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('musical_instruments')

    def load_amazon_review_acl_office_products(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('office_products')

    def load_amazon_review_acl_outdoor_living(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('outdoor_living')

    def load_amazon_review_acl_software(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('software')

    def load_amazon_review_acl_sports_outdoors(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('sports_outdoors')

    def load_amazon_review_acl_tools_hardware(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('tools_hardware')

    def load_amazon_review_acl_toys_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('toys_games')

    def load_amazon_review_acl_video(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewACLDataLoader('video')

    def load_news_popularity_obama_all(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('obama', 'all')

    def load_news_popularity_economy_all(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('economy', 'all')

    def load_news_popularity_microsoft_all(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('microsoft', 'all')

    def load_news_popularity_palestine_all(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('palestine', 'all')

    def load_news_popularity_obama_facebook(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('obama', 'Facebook')

    def load_news_popularity_economy_facebook(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('economy', 'Facebook')

    def load_news_popularity_microsoft_facebook(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('microsoft', 'Facebook')

    def load_news_popularity_palestine_facebook(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('palestine', 'Facebook')

    def load_news_popularity_obama_googleplus(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('obama', 'GooglePlus')

    def load_news_popularity_economy_googleplus(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('economy', 'GooglePlus')

    def load_news_popularity_microsoft_googleplus(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('microsoft', 'GooglePlus')

    def load_news_popularity_palestine_googleplus(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('palestine', 'GooglePlus')

    def load_news_popularity_obama_linkedin(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('obama', 'LinkedIn')

    def load_news_popularity_economy_linkedin(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('economy', 'LinkedIn')

    def load_news_popularity_microsoft_linkedin(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('microsoft', 'LinkedIn')

    def load_news_popularity_palestine_linkedin(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomNewsPopularityDataLoader('palestine', 'LinkedIn')