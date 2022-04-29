# All of the code in this file was developed by Marcus de Carvalho without any modification

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


class MyCustomAmazonReviewDataLoader(torch.utils.data.Dataset):
    base_url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/'
    path = 'data/AmazonReview/'
    df = None

    @property
    def datasets(self):
        return self.df

    def __init__(self, filename):
        torchvision.datasets.utils.download_url(self.base_url + filename, self.path)
        self.df = self.get_df(self.path + filename)
        self.normalize()

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

    def get_df(self, path, high_bound=500000):
        try:
            print('Trying to load processed file %s.h5 from disc...' % path)
            df = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + '.h5'),
                             key='df', mode='r')
        except:
            print('Processed file does not exists')
            print('Reading dataset into memory and applying Word2Vec...')
            print('\nWe will save a maximum of half million samples because of memory constraints')
            print('and because that is more than sufficient samples to test transfer learning models\n')

            i = 0
            df = {}

            if path == 'data/AmazonReview/AMAZON_FASHION_5.json.gz':
                total = 3176
            elif path == 'data/AmazonReview/All_Beauty_5.json.gz':
                total = 5269
            elif path == 'data/AmazonReview/Appliances_5.json.gz':
                total = 2277
            elif path == 'data/AmazonReview/Arts_Crafts_and_Sewing_5.json.gz':
                total = 494485
            elif path == 'data/AmazonReview/Automotive_5.json.gz':
                total = 1711519
            elif path == 'data/AmazonReview/Books_5.json.gz':
                total = 27164983
            elif path == 'data/AmazonReview/CDs_and_Vinyl_5.json.gz':
                total = 1443755
            elif path == 'data/AmazonReview/Cell_Phones_and_Accessories_5.json.gz':
                total = 1128437
            elif path == 'data/AmazonReview/Clothing_Shoes_and_Jewelry_5.json.gz':
                total = 11285464
            elif path == 'data/AmazonReview/Digital_Music_5.json.gz':
                total = 169781
            elif path == 'data/AmazonReview/Electronics_5.json.gz':
                total = 6739590
            elif path == 'data/AmazonReview/Gift_Cards_5.json.gz':
                total = 2972
            elif path == 'data/AmazonReview/Grocery_and_Gourmet_Food_5.json.gz':
                total = 1143860
            elif path == 'data/AmazonReview/Home_and_Kitchen_5.json.gz':
                total = 6898955
            elif path == 'data/AmazonReview/Industrial_and_Scientific_5.json.gz':
                total = 77071
            elif path == 'data/AmazonReview/Kindle_Store_5.json.gz':
                total = 2222983
            elif path == 'data/AmazonReview/Luxury_Beauty_5.json.gz':
                total = 34278
            elif path == 'data/AmazonReview/Magazine_Subscriptions_5.json.gz':
                total = 2375
            elif path == 'data/AmazonReview/Movies_and_TV_5.json.gz':
                total = 3410019
            elif path == 'data/AmazonReview/Musical_Instruments_5.json.gz':
                total = 231392
            elif path == 'data/AmazonReview/Office_Products_5.json.gz':
                total = 800357
            elif path == 'data/AmazonReview/Patio_Lawn_and_Garden_5.json.gz':
                total = 798415
            elif path == 'data/AmazonReview/Pet_Supplies_5.json.gz':
                total = 2098325
            elif path == 'data/AmazonReview/Prime_Pantry_5.json.gz':
                total = 137788
            elif path == 'data/AmazonReview/Software_5.json.gz':
                total = 12805
            elif path == 'data/AmazonReview/Sports_and_Outdoors_5.json.gz':
                total = 2839940
            elif path == 'data/AmazonReview/Tools_and_Home_Improvement_5.json.gz':
                total = 2070831
            elif path == 'data/AmazonReview/Toys_and_Games_5.json.gz':
                total = 1828971
            elif path == 'data/AmazonReview/Video_Games_5.json.gz':
                total = 497577

            MyWord2Vec().get()
            pbar = tqdm(unit=' samples', total=np.min([total, high_bound]))

            tokenizer = TweetTokenizer()
            for d in self.parse(path):
                if i >= 500000:
                    break
                try:
                    reviewText = d['reviewText']
                    try:
                        word_count = 0
                        vector = np.zeros(MyWord2Vec().get().vector_size)
                        for word in tokenizer.tokenize(reviewText):
                            try:
                                vector += MyWord2Vec().get()[word]
                                word_count += 1
                            except:
                                pass
                        if word_count > 1:
                            try:
                                overall = d['overall']
                                df[i] = {'overall': overall, 'reviewText': vector / word_count}
                                pbar.update(1)
                                i += 1
                            except:
                                pass
                    except:
                        pass
                except:
                    pass
            pbar.close()

            print('Saving processed tokenized dataset in disc for future usage...')
            df = pd.DataFrame.from_dict(df, orient='index')
            df = pd.DataFrame([{x: y for x, y in enumerate(item)}
                               for item in df['reviewText'].values.tolist()]).assign(overall=df.overall.tolist())

            df.to_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + '.h5'),
                      key='df',
                      mode='w',
                      format='table',
                      complevel=9,
                      complib='bzip2')
            df = pd.read_hdf(path_or_buf=os.path.join(os.path.dirname(__file__), path + '.h5'),
                             key='df')
        return df


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
        self.data = MyCustomAmazonReviewDataLoader('AMAZON_FASHION_5.json.gz')

    def load_amazon_review_all_beauty(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('All_Beauty_5.json.gz')

    def load_amazon_review_appliances(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Appliances_5.json.gz')

    def load_amazon_review_arts_crafts_sewing(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Arts_Crafts_and_Sewing_5.json.gz')

    def load_amazon_review_automotive(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Automotive_5.json.gz')

    def load_amazon_review_books(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Books_5.json.gz')

    def load_amazon_review_cds_vinyl(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('CDs_and_Vinyl_5.json.gz')

    def load_amazon_review_cellphones_accessories(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Cell_Phones_and_Accessories_5.json.gz')

    def load_amazon_review_clothing_shoes_jewelry(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Clothing_Shoes_and_Jewelry_5.json.gz')

    def load_amazon_review_digital_music(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Digital_Music_5.json.gz')

    def load_amazon_review_electronics(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Electronics_5.json.gz')

    def load_amazon_review_gift_card(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Gift_Cards_5.json.gz')

    def load_amazon_review_grocery_gourmet_food(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Grocery_and_Gourmet_Food_5.json.gz')

    def load_amazon_review_home_kitchen(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Home_and_Kitchen_5.json.gz')

    def load_amazon_review_industrial_scientific(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Industrial_and_Scientific_5.json.gz')

    def load_amazon_review_kindle_store(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Kindle_Store_5.json.gz')

    def load_amazon_review_luxury_beauty(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Luxury_Beauty_5.json.gz')

    def load_amazon_review_magazine_subscription(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Magazine_Subscriptions_5.json.gz')

    def load_amazon_review_movies_tv(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Movies_and_TV_5.json.gz')

    def load_amazon_review_musical_instruments(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Musical_Instruments_5.json.gz')

    def load_amazon_review_office_products(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Office_Products_5.json.gz')

    def load_amazon_review_patio_lawn_garden(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Patio_Lawn_and_Garden_5.json.gz')

    def load_amazon_review_pet_supplies(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Pet_Supplies_5.json.gz')

    def load_amazon_review_prime_pantry(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Prime_Pantry_5.json.gz')

    def load_amazon_review_software(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Software_5.json.gz')

    def load_amazon_review_sports_outdoors(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Sports_and_Outdoors_5.json.gz')

    def load_amazon_review_tools_home_improvements(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Tools_and_Home_Improvement_5.json.gz')

    def load_amazon_review_toys_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Toys_and_Games_5.json.gz')

    def load_amazon_review_video_games(self, n_concept_drifts: int = 1):
        self.n_concept_drifts = n_concept_drifts
        self.data = MyCustomAmazonReviewDataLoader('Video_Games_5.json.gz')

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