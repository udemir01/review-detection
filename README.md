## Hotel Database

Download and extract the csv file to the dataset directory inside the repo.
[Link to the database](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## Virtual Environment Setup

The version of python I used for this project was 3.10.12, be careful with dependency issues.

### First, create the .env/ directory
- For Linux:
```
python3 -m venv .env/
```
- For Windows:
```
py -m venv .env\
```

### Activate the environment
- For Linux:
```
. .env/bin/activate
```
- For Windows:
```
.env\Scripts\activate
```

### Install the libraries with:
```
pip install -r requirements.txt
```

### Build the model
- For Linux:
```
python3 sentiment_model.py
```
- For Windows:
```
py sentiment_model.py
```

### Run the server by entering the following commands:
- For Linux:
```
python3 review_detection_server.py
```
- For Windows:
```
py review_detection_server.py
```
