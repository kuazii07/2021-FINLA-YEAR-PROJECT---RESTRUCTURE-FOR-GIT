import gdown

url = "https://drive.google.com/uc?id=1M_wLBCuwTXSF5xRmQ68YIBKdoIuo7rww"
output = "model.pt"

gdown.download(url, output, quiet=False)