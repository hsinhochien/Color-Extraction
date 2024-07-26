from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import re

def extract_color_keywords(prd_name, nlp):
    ner_results = nlp(prd_name)

    all_label_1_substrings = []
    for result in ner_results:
        if result['entity_group'] == "LABEL_1":
            start = result['start']
            end = result['end']

            substring = prd_name[start:end]
            clean_substring = re.sub(r'[^\u4e00-\u9fff\s]', '', substring)
            all_label_1_substrings.append(clean_substring)
    
    combined_substrings = ' '.join(all_label_1_substrings)
    color_keywords = combined_substrings.split()
    
    return color_keywords

prd_names = ["Nike Court Vision Lo NN 女 黑白 復古 皮革 熊貓 經典 休閒鞋 DH3158-003",
             "Nike 休閒鞋 Wmns Waffle Debut 女鞋 白 青綠 麂皮 厚底 增高 DH9523-101",
             "Pacsafe CITYSAFE CS200 休閒斜肩包(女)(莓紅色)",
             "抗藍光 2019 iPad mini/5/4 高清晰9H鋼化平板玻璃貼 螢幕保護膜",
             "DYY》伊莉莎白《防咬》寵物防護保護套 (特大號-51-60CM)",
             "★買一送一★巴黎萊雅完美淨白光采再現嫩          白潔面露100ml",
             "【幸福揚邑】防曬防紫外線防風舒適透氣戶外運動86造型棒球帽鴨舌帽-四色可選",
             "DADADO-黑標M-3L寬鬆四角褲(紅)",
             "華歌爾-Good FitM-LL中低腰三角內褲(紫)Modal素材-親膚舒適",
             "【FitFlop】F-SPORTY UBERKNIT SNEAKERS - METALLIC WEAVE 運動風繫帶休閒鞋-女(黑色/銅金色)",
             "INTOPIC 廣鼎 3用藍牙耳麥(JAZZ-BTC09)",
             "【LETTi】時光拼圖 29吋鑽石紋質感拉鍊行李箱(多色任選)"]

best_model = AutoModelForTokenClassification.from_pretrained("./best_model")
best_tokenizer = AutoTokenizer.from_pretrained("./best_model")
nlp = pipeline("ner", model=best_model, tokenizer=best_tokenizer, grouped_entities=True)

for i in prd_names:
    print("商品名稱:", i, "\n顏色字:", extract_color_keywords(i, nlp))