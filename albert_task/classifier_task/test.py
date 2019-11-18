import json

from predict import Predictor


with open("config/tnews_config.json", "r") as fr:
    config = json.load(fr)


predictor = Predictor(config)
text = "歼20座舱盖上的两条“花纹”是什么？"
res = predictor.predict(text)
print(res)