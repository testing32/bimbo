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

TRAINING_FEATURE_COLUMNS = ['Semana', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
PRODUCT_FEATURE_COLUMNS = ['Producto_ID', 'weight', 'brand_encoded', 'pieces']
CITY_STATE_FEATURE_COLUMNS = ['city_state_encoded', 'state_encoded']
TOTAL_TRAINING_FEATURE_COLUMNS = list(set(TRAINING_FEATURE_COLUMNS + ['short_name_encoded',] + PRODUCT_FEATURE_COLUMNS + CITY_STATE_FEATURE_COLUMNS))
TARGET_COLUMN = ['Demanda_uni_equil',]
TOTAL_TEST_FEATURE_COLUMNS = ['id',] + TOTAL_TRAINING_FEATURE_COLUMNS