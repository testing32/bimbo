# bimbo
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