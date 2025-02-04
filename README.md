# AIDB

open-source ai-enhanced database

# Installation

compile and install with

```
cd aidb
./configure --prefix=<install_directory>
make -j4
make install
```

# Getting Started

start server

```
cd <install_directory>/bin
./initdb -D <data_directory>
./postgres -D <data_directory> -p 5432
```

connect to server

```
cd <install_directory>/bin
./psql -p 5432 -d postgres
```

## Import Model

use create model clause to upload a model file (format as pt) to server.

base_model is inner supported models, if you choose a base_model, layer values will be imported into model_layer_info.

```
-- create model
CREATE MODEL '<model_name>' PATH '<client_path>' 
[base_model '<base_model_name>'] 
[DESCRIPTION <'description'>];

MODIFT MODEL '<model_name>' PATH 'new_client_path' DESCRIPTION 'xxxx';

DROP MODEL '<model_name>';

-- see details for model
select * from model_info;
select * from base_model_info;
select * from model_layer_info;
```

## Do Prediction

```
create table image_test(user_name text not null, image_url text not null);

insert into image_test(user_name, image_url)
values
('bob', '/tmp/pgdl/test/image/img_10.jpg'), 
('frank', '/tmp/pgdl/test/image/img_11.jpg'), 
('bob', '/tmp/pgdl/test/image/img_12.jpg'), 
('vicky', '/tmp/pgdl/test/image/img_1.jpg'), 
('frank', '/tmp/pgdl/test/image/img_2.jpg'),
('vicky', '/tmp/pgdl/test/image/img_3.jpg'), 
('jeff', '/tmp/pgdl/test/image/img_4.jpg'), 
('vicky', '/tmp/pgdl/test/image/img_5.jpg'), 
('frank', '/tmp/pgdl/test/image/img_6.jpg'),
('alice', '/tmp/pgdl/test/image/img_7.jpg'),
('alice', '/tmp/pgdl/test/image/img_8.jpg'),
('jeff', '/tmp/pgdl/test/image/img_9.jpg');

select predict_text('test', 'cpu', image_url) from image_test;
```
