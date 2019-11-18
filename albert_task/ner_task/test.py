import json

from predict import Predictor

with open("config/msraner_config.json", "r") as fr:
    config = json.load(fr)

text = "中 共 中 央 致 中 国 致 公 党 十 一 大 的 贺 词"
text = text.split(" ")
predictor = Predictor(config)
chunks = predictor.predict(text)

for chunk in chunks:
    entity_name, start, end = chunk
    entity = "".join(text[start - 1: end])
    print(entity_name, entity)
