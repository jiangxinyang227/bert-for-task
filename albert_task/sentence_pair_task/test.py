import json

from predict import Predictor


with open("config/bq_config.json", "r") as fr:
    config = json.load(fr)


predictor = Predictor(config)

text_a = "为什么我无法看到额度"
text_b = "为什么开通了却没有额度"
res = predictor.predict(text_a, text_b)
print(res)