# BIMBO
bimbo kaggle comp

# Feature Engineering Steps
Products:
* Get the full name and run word vec on it. Run Gaussian Mixture Model on that to get product categories. (the feature vector is too large)
* Get the full name and label encode it
* Get the brand name and label encode it
* Get the weight (fill n/a with the brand mean)
* Get the # of pieces (1 if N/A)

City/State:
* The city+state is label encoded together
* The state is label encoded

Demand Data:
* median demand per client
* median demand per client/product
* median demand per agency
* median demand per agency/product
* median demand per client/product/agency
* median demand per week/agency
* median demand per week/agency/product

# Feature Engineering Ideas
* Add # of agencies per customer
* Should the weight be filled with the median instead of the mean
* Add more temporal information (Based on customer id, not agency)
* Include median demand
* Should I use mean demands on the categories like I'm doing with median demand
* Combine returns information with the temporal information (returns the previous week, might effect demand for the next week)
* The # of customers a product is sent to (per state?)
* The # of agencies a product is sent to (per state?)

# Training Columns
* Semana
* Agencia_ID
* Canal_ID
* Ruta_SAK
* Cliente_ID
* Producto_ID
* Demanda_uni_equil_median
* Demanda_uni_equil_median_prod
* Demanda_uni_equil_median_agen
* Demanda_uni_equil_median_prod_agen
* Demanda_uni_equil_median_agen_prod
* Demanda_uni_equil_median_semana_agen
* Demanda_uni_equil_median_semana_agen_prod
* short_name_encoded
* brand_encoded
* weight
* pieces
* product_group
* city_state_encoded
* state_encoded

# Notes
I have gotten no where with the returns information or the sales information.

There may be products in the test set that don't exist in the train set. This is the expected behavior of inventory data, 
since there are new products being sold all the time. Your model should be able to accommodate this.

There are duplicate Cliente_ID's in cliente_tabla, which means one Cliente_ID may have multiple 
NombreCliente that are very similar. This is due to the NombreCliente being noisy and not standardized in the 
raw data, so it is up to you to decide how to clean up and use this information. 

The adjusted demand (Demanda_uni_equil) is always >= 0 since demand should be either 0 or a 
positive value. The reason that Venta_uni_hoy - Dev_uni_proxima sometimes has negative values 
is that the returns records sometimes carry over a few weeks.

Features - Semana, Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID, Bag of Words of the Product, Weight, Food_Company_ID

Semana — Week number (From Thursday to Wednesday)
Agencia_ID — Sales Depot ID
Canal_ID — Sales Channel ID
Ruta_SAK — Route ID (Several routes = Sales Depot)
Cliente_ID — Client ID
NombreCliente — Client name
Producto_ID — Product ID
NombreProducto — Product Name
Venta_uni_hoy — Sales unit this week (integer)
Venta_hoy — Sales this week (unit: pesos)
Dev_uni_proxima — Returns unit next week (integer)
Dev_proxima — Returns next week (unit: pesos)
Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)

Size of training set: (74180464, 11)
 Size of testing set: (6999251, 7)


Columns in train: ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Venta_uni_hoy', 
		'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil']
 Columns in test: ['id', 'Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']

Train Description:
                Semana       Agencia_ID         Canal_ID         Ruta_SAK  \
count  74180464.000000  74180464.000000  74180464.000000  74180464.000000   
mean          5.950021      2536.508561         1.383181      2114.855291   
std           2.013175      4075.123651         1.463266      1487.744180   
min           3.000000      1110.000000         1.000000         1.000000   
25%           4.000000      1312.000000         1.000000      1161.000000   
50%           6.000000      1613.000000         1.000000      1286.000000   
75%           8.000000      2036.000000         1.000000      2802.000000   
max           9.000000     25759.000000        11.000000      9991.000000   

         Cliente_ID      Producto_ID    Venta_uni_hoy        Venta_hoy  \
count  7.418046e+07  74180464.000000  74180464.000000  74180464.000000   
mean   1.802119e+06     20840.813993         7.310163        68.544523   
std    2.349577e+06     18663.919030        21.967337       338.979516   
min    2.600000e+01        41.000000         0.000000         0.000000   
25%    3.567670e+05      1242.000000         2.000000        16.760000   
50%    1.193385e+06     30549.000000         3.000000        30.000000   
75%    2.371091e+06     37426.000000         7.000000        56.100000   
max    2.015152e+09     49997.000000      7200.000000    647360.000000   

       Dev_uni_proxima      Dev_proxima  Demanda_uni_equil  
count  74180464.000000  74180464.000000    74180464.000000  
mean          0.130258         1.243248           7.224564  
std          29.323204        39.215523          21.771193  
min           0.000000         0.000000           0.000000  
25%           0.000000         0.000000           2.000000  
50%           0.000000         0.000000           3.000000  
75%           0.000000         0.000000           6.000000  
max      250000.000000    130760.000000        5000.000000

Test Description:
                 id        Semana    Agencia_ID      Canal_ID      Ruta_SAK  \
count  6.999251e+06  6.999251e+06  6.999251e+06  6.999251e+06  6.999251e+06   
mean   3.499625e+06  1.049446e+01  2.504463e+03  1.401874e+00  2.138014e+03   
std    2.020510e+06  4.999694e-01  4.010228e+03  1.513404e+00  1.500392e+03   
min    0.000000e+00  1.000000e+01  1.110000e+03  1.000000e+00  1.000000e+00   
25%    1.749812e+06  1.000000e+01  1.311000e+03  1.000000e+00  1.159000e+03   
50%    3.499625e+06  1.000000e+01  1.612000e+03  1.000000e+00  1.305000e+03   
75%    5.249438e+06  1.100000e+01  2.034000e+03  1.000000e+00  2.804000e+03   
max    6.999250e+06  1.100000e+01  2.575900e+04  1.100000e+01  9.950000e+03   

         Cliente_ID   Producto_ID  
count  6.999251e+06  6.999251e+06  
mean   1.819128e+06  2.216307e+04  
std    2.938910e+06  1.869816e+04  
min    2.600000e+01  4.100000e+01  
25%    3.558290e+05  1.242000e+03  
50%    1.200109e+06  3.150700e+04  
75%    2.387881e+06  4.093000e+04  
max    2.015152e+09  4.999700e+04  

Most common target values:
[(2, 14997665), (1, 13249749), (3, 9150147), (4, 7176357), (5, 5634148), (6, 4220142), (10, 2886205), (8, 2401131), (7, 1756403), (0, 1336821)]


There are 1522 products in the test set
There are 34 products in the test set that aren't in the training set
These products don't exist it the training set([32026, 37404, 37405, 32798, 32421, 31655, 36524, 37618, 35246, 33053, 46131, 32820, 37688, 36673, 37702, 35191, 32591, 37202, 42323, 48217, 37745, 32224, 98, 31203, 37610, 31211, 46064, 37617, 37362, 37620, 37494, 37495, 37496, 37626])
