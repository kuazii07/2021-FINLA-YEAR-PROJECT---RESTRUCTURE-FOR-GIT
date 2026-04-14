import gdown

url = "https://drive.google.com/uc?id=FILE_ID"
output = "model.pt"

gdown.download(url, output, quiet=False)