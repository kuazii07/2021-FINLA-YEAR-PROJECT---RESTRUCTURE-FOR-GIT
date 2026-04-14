import gdown

url = "https://drive.google.com/file/d/1M_wLBCuwTXSF5xRmQ68YIBKdoIuo7rww/view?usp=drive_link"
output = "model.pt"

gdown.download(url, output, quiet=False)