# iNaturalist 2018 Competition

## Kaggle
Checkout the competition page [here](https://www.kaggle.com/c/inaturalist-2018).

## Details
There are a total of 8,142 species in the dataset, with 437,513 training and 24,426 validation images.

| Super Category |	Category Count	| Train Images |	Val Images |
|------|---------------|-------------|---------------|
Plantae|2,917|118,800|8,751|
Insecta|2,031|87,192|6,093|
Aves|1,258|143,950|3,774|
Actinopterygii|369|7,835|1,107|
Fungi|321|6,864|963|
Reptilia|284|22,754|852|
Mollusca|262|8,007|786|
Mammalia|234|20,104|702|
Animalia|178|5,966|534|
Amphibia|144|11,156|432|
Arachnida|114|4,037|342|
Chromista|25|621|75|
Protozoa|4|211|12|
Bacteria|1|16|3|
|||||
Total|8,142|437,513|24,426|

## Annotation Format
We follow the annotation format of the [COCO dataset](http://mscoco.org/dataset/#download) and add additional fields. The annotations are stored in the [JSON format](http://www.json.org/) and are organized as follows:
```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation],
  "licenses" : [license]
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str,
  "url" : str,
  "date_created" : datetime,
}

image{
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int,
  "rights_holder" : str
}

category{
  "id" : int,
  "name" : str,
  "supercategory" : str,
  "kingdom" : str,
  "phylum" : str,
  "class" : str,
  "order" : str,
  "family" : str,
  "genus" : str
}

annotation{
  "id" : int,
  "image_id" : int,
  "category_id" : int
}

license{
  "id" : int,
  "name" : str,
  "url" : str
}

## Data

Download the dataset files here:
  * All training and validation images [120GB] and test [40GB]
          * iNaturalist competition data](https://github.com/visipedia/inat_comp#data)