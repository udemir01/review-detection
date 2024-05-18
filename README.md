## Hotel Database

Download and extract the csv file to the dataset directory inside the repo.
[Dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## Virtual Environment Setup

The version of python I used for this project was 3.10.12, be careful with dependency issues.
If using Windows, change "python3" command with "py".

### First, create the .env/ directory
```
python3 -m venv .env/
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
python3 -m spacy download "en_core_web_sm"
```

### Build the model
```
python3 sentiment_model.py
```

### Run the server by entering the following commands:
```
python3 review_detection_server.py
```
