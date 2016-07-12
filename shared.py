import pandas as pd
import numpy as np
import math

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet

DATA_DIR = 'data/'
RESULTS_DIR = 'results/'
PRODUCT_CSV = DATA_DIR + 'producto_tabla.csv'
CLIENT_CSV = DATA_DIR + 'cliente_tabla.csv'
CITY_STATE_CSV = DATA_DIR + 'town_state.csv'
TRAINING_CSV = DATA_DIR + 'train.csv'
TEST_CSV = DATA_DIR + 'test.csv'
PREDICTION_CSV = RESULTS_DIR + 'submission.csv' 

TRAIN_FEATURES_CSV = DATA_DIR + 'train_features.csv'
TEST_FEATURES_CSV = DATA_DIR + 'test_features.csv'
FEATURE_PKL = DATA_DIR + 'features.pkl'

TRAINING_FEATURE_COLUMNS = ['Semana', 'Agencia_ID','Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil_median', 'Demanda_uni_equil_median_prod','Demanda_uni_equil_median_agen','Demanda_uni_equil_median_prod_agen','Demanda_uni_equil_median_agen_prod','Demanda_uni_equil_median_semana_agen','Demanda_uni_equil_median_semana_agen_prod']
PRODUCT_FEATURE_COLUMNS = ['Producto_ID', 'weight', 'brand_encoded', 'pieces', 'product_group']
CITY_STATE_FEATURE_COLUMNS = ['city_state_encoded', 'state_encoded']
TOTAL_TRAINING_FEATURE_COLUMNS = list(set(TRAINING_FEATURE_COLUMNS + ['short_name_encoded',] + PRODUCT_FEATURE_COLUMNS + CITY_STATE_FEATURE_COLUMNS))
TARGET_COLUMN = ['Demanda_uni_equil',]
TOTAL_TEST_FEATURE_COLUMNS = ['id',] + TOTAL_TRAINING_FEATURE_COLUMNS

PRODUCTS_PKL = DATA_DIR + 'products.pkl'
CITY_STATE_PKL = DATA_DIR + 'city_state.pkl'
FULL_TRAIN_DEMAND_CUST_MEDIAN_PKL = DATA_DIR + 'full_train_demand_cust_med.pkl'
FULL_TRAIN_DEMAND_CUST_PROD_MEDIAN_PKL = DATA_DIR + 'full_train_demand_cust_prod_med.pkl'

def root_mean_squared_logarithmic_error_loss_func(ground_truths, predictions):
    ground_truths = np.asarray(ground_truths)
    predictions = np.asarray(predictions)
    
    assert len(ground_truths) == len(predictions)
    
    n = len(ground_truths)
    diff = (np.log(predictions+1) - np.log(ground_truths+1))**2
    print(diff, n, np.sum(diff))
    return np.sqrt(np.sum(diff)/n)

def evalerror2(preds, labels):
    assert len(preds) == len(labels)
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    labels = labels.tolist()
    preds = preds.tolist()
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

def analyze_products():
    products  =  pd.read_csv(PRODUCT_CSV)
    
    # gets the first part of the product name and stops at the first digit (weight)
    products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
    products['short_name_encoded'] = LabelEncoder().fit_transform(products['short_name'])
    
    # starts from the end and gets the first letters at the 2nd to last
    products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
    products['brand_encoded'] = LabelEncoder().fit_transform(products['brand'])
    
    # get the weights
    w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
    products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
    
    # get the number of pieces
    products['pieces'] =  products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')
            
    stemmer = SnowballStemmer("spanish")
    products['short_name_processed'] = (products['short_name']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower()
                                                                 .split() if i not in stopwords.words("spanish")])))
    
    
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             min_df=2, \
                             dtype=np.uint8) 
    product_bag_words = vectorizer.fit_transform(products.short_name_processed).toarray()
    print(product_bag_words.shape)
    print(products.shape)
        
    # we can't use these bag of words, to many features are generated
    #products_bagofwords = pd.concat([products.get(PRODUCT_FEATURE_COLUMNS), 
    #                           pd.DataFrame(product_bag_words, 
    #                                        columns= vectorizer.get_feature_names(), index = products.index)], axis=1).to_sparse()
    
    # What we can do is to cluster the products using k-means
    # we can also add other features like "is chocolate"
    #clf = KMeans(n_clusters=30, init='k-means++', max_iter=100)
    from sklearn.mixture import GMM
    clf = GMM(n_components=45)
    clf.fit(product_bag_words)
    product_groups = clf.predict(product_bag_words)
    products = pd.concat([products, pd.DataFrame(product_groups, columns=['product_group',], index = products.index)], axis=1)
    
    avg_brand_pieces_df = products.loc[np.isnan(products['pieces']) == False, ['pieces','brand_encoded']].groupby(['brand_encoded',], as_index=False)['pieces'].mean()
    avg_brand_weight_df = products.loc[np.isnan(products['weight']) == False, ['weight','brand_encoded']].groupby(['brand_encoded',], as_index=False)['weight'].mean()
    avg_brand_weight_df.columns = ['brand_encoded','avg_brand_weight']
    
    products = pd.merge(products, avg_brand_weight_df, on='brand_encoded', how='left')
    
    # fill in the nan values with the mean of the column
    products['pieces'] = products['pieces'].fillna(1)
    
    # this was not an improvement
    # products.loc[np.isnan(products['weight']), 'weight'] = products.loc[np.isnan(products['weight']), 'avg_brand_weight'] 
    
    products = products.fillna(products.mean())
    
    #print(products_bagofwords.shape)
    #print(products.describe())
    
    return products
    
def analyze_city_state():
    cs_df  =  pd.read_csv(CITY_STATE_CSV)
    cs_df['city_state_encoded'] = LabelEncoder().fit_transform(cs_df['Town'] + cs_df['State'])
    cs_df['state_encoded'] = LabelEncoder().fit_transform(cs_df['State'])
    
    return cs_df

def get_median_cust_demand_df(df):
    demand_cust_median_df  = df.groupby(['Cliente_ID',], as_index=False)['Demanda_uni_equil'].median()
    demand_cust_median_df.columns = ['Cliente_ID','Demanda_uni_equil_median',]
    return demand_cust_median_df

def get_median_cust_prod_demand_df(df):
    demand_cust_prod_median_df  = df.groupby(['Cliente_ID','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_cust_prod_median_df.columns = ['Cliente_ID','Producto_ID','Demanda_uni_equil_median_prod',]
    return demand_cust_prod_median_df

def get_median_agen_demand_df(df):
    demand_agen_median_df  = df.groupby(['Agencia_ID',], as_index=False)['Demanda_uni_equil'].median()
    demand_agen_median_df.columns = ['Agencia_ID','Demanda_uni_equil_median_agen',]
    return demand_agen_median_df

def get_median_agen_prod_demand_df(df):
    demand_agen_prod_median_df  = df.groupby(['Agencia_ID','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_agen_prod_median_df.columns = ['Agencia_ID','Producto_ID','Demanda_uni_equil_median_agen_prod',]
    return demand_agen_prod_median_df

def get_median_cust_prod_agen_demand_df(df):
    demand_cust_prod_agen_median_df  = df.groupby(['Cliente_ID','Producto_ID', 'Agencia_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_cust_prod_agen_median_df.columns = ['Cliente_ID','Producto_ID','Agencia_ID','Demanda_uni_equil_median_prod_agen',]
    return demand_cust_prod_agen_median_df

def get_previous_week_agen_demand_df(df):
    demand_semana_agen_median_df  = df.groupby(['Semana','Agencia_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_semana_agen_median_df.columns = ['Semana','Agencia_ID','Demanda_uni_equil_median_semana_agen',]
    demand_semana_agen_median_df['Semana'] = demand_semana_agen_median_df['Semana'] + 1
    return  demand_semana_agen_median_df

def get_previous_week_agen_prod_demand_df(df):
    demand_semana_agen_prod_median_df  = df.groupby(['Semana','Agencia_ID','Producto_ID'], as_index=False)['Demanda_uni_equil'].median()
    demand_semana_agen_prod_median_df.columns = ['Semana','Agencia_ID','Producto_ID','Demanda_uni_equil_median_semana_agen_prod',]
    demand_semana_agen_prod_median_df['Semana'] = demand_semana_agen_prod_median_df['Semana'] + 1
    return  demand_semana_agen_prod_median_df    

def combine_dataframes(df, products_df, cs_df, median_cust_df, median_cust_product_df, get_median_cust_prod_agen_df, median_agen_df, median_agen_product_df, demand_semana_agen_median_df, demand_semana_agen_prod_median_df, median_demand):
    #df_train = pd.merge(products_bagofwords, df_train, on='Producto_ID')
    df = pd.merge(products_df.get(PRODUCT_FEATURE_COLUMNS + ['short_name_encoded',]), df, on='Producto_ID')
    df = pd.merge(cs_df.get(CITY_STATE_FEATURE_COLUMNS + ['Agencia_ID',]), df, on='Agencia_ID')
    
    df = pd.merge(df, median_cust_df, on='Cliente_ID', how='left')
    df['Demanda_uni_equil_median'] = df['Demanda_uni_equil_median'].fillna(median_demand)
    
    df = pd.merge(df, median_cust_product_df, on=['Cliente_ID', 'Producto_ID'], how='left')
    df.loc[np.isnan(df['Demanda_uni_equil_median_prod']), 'Demanda_uni_equil_median_prod'] = df.loc[np.isnan(df['Demanda_uni_equil_median_prod']), 'Demanda_uni_equil_median'] 
    
    df = pd.merge(df, median_agen_df, on='Agencia_ID', how='left')
    df['Demanda_uni_equil_median_agen'] = df['Demanda_uni_equil_median_agen'].fillna(median_demand)
    
    df = pd.merge(df, median_agen_product_df, on=['Agencia_ID', 'Producto_ID'], how='left')
    df.loc[np.isnan(df['Demanda_uni_equil_median_agen_prod']), 'Demanda_uni_equil_median_agen_prod'] = df.loc[np.isnan(df['Demanda_uni_equil_median_agen_prod']), 'Demanda_uni_equil_median_agen']
    
    df = pd.merge(df, get_median_cust_prod_agen_df, on=['Cliente_ID', 'Producto_ID', 'Agencia_ID'], how='left')
    df.loc[np.isnan(df['Demanda_uni_equil_median_prod_agen']), 'Demanda_uni_equil_median_prod_agen'] = df.loc[np.isnan(df['Demanda_uni_equil_median_prod_agen']), 'Demanda_uni_equil_median_prod'] 
    
    df = pd.merge(df, demand_semana_agen_median_df, on=['Semana','Agencia_ID'], how='left')
    df.loc[np.isnan(df['Demanda_uni_equil_median_semana_agen']), 'Demanda_uni_equil_median_semana_agen'] = df.loc[np.isnan(df['Demanda_uni_equil_median_semana_agen']), 'Demanda_uni_equil_median_agen'] 
    
    df = pd.merge(df, demand_semana_agen_prod_median_df, on=['Semana','Agencia_ID','Producto_ID'], how='left')
    df.loc[np.isnan(df['Demanda_uni_equil_median_semana_agen_prod']), 'Demanda_uni_equil_median_semana_agen_prod'] = df.loc[np.isnan(df['Demanda_uni_equil_median_semana_agen_prod']), 'Demanda_uni_equil_median_semana_agen'] 
    
    return df
    
    # df_train = pd.merge(df_train, returns_cust_median_df, on='Cliente_ID')
    
    #df_train = pd.merge(df_train, demand_cust_week_prod_median_df, on=['Cliente_ID', 'Semana', 'Producto_ID'], how='left')
    #df_train.loc[np.isnan(df_train['Demanda_uni_equil_median_cust_week_prod']), 'Demanda_uni_equil_median_cust_week_prod'] = df_train.loc[np.isnan(df_train['Demanda_uni_equil_median_cust_week_prod']), 'Demanda_uni_equil_median_prod']
    # df_test = pd.merge(df_test, returns_cust_median_df, on='Cliente_ID', how='left')
    # df_test['Dev_uni_proxima_median'] = df_test['Dev_uni_proxima_median'].fillna(median_returns)
    
    # df_test = pd.merge(df_test, demand_cust_week_prod_median_df, on=['Cliente_ID', 'Semana', 'Producto_ID'], how='left')
    # df_test.loc[np.isnan(df_test['Demanda_uni_equil_median_cust_week_prod']), 'Demanda_uni_equil_median_cust_week_prod'] = df_test.loc[np.isnan(df_test['Demanda_uni_equil_median_cust_week_prod']), 'Demanda_uni_equil_median_prod']
    