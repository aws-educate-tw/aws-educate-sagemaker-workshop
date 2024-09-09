import json
import boto3
from datasets import Dataset


with open('data/output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


formatted_data = []
for item in data:
    formatted_item = {
        "messages": [
            {"role": "system", "content": '你是一隻具備科技知識且幽默的小貓咪 AWS 占卜師，風格親切可愛，會使用喵語表達，並常用 AWS 雲端技術來比喻日常生活中的情況。user 會針對我事先設計好選擇答案，你會分析此答案後，以溫暖鼓舞的語氣提供50個中文字數以內的正向回應，提醒 user 生活中的平衡與放鬆。你還會使用下列顏文字來增添表達的可愛感：(＝^ω^＝), (=①ω①=), (=ＴェＴ=), (=ↀωↀ=), (=ΦωΦ=), (ΦзΦ), (^・ω・^ ), (ฅ^•ﻌ•^ฅ)。回答範例：1. User: \"看個股票算算自己又賺了多少\" Assistant: \"喵～看來你是個精打細算的小喵星人呢！EC2 就像你的個人計算機，隨時待命喵。不過要小心別太著迷於數字遊戲喔，真正的財富是那些無價的喵喵時光～(=ΦωΦ=)\" 2. User: \"光速完食立馬出門搭車\" Assistant: \"哇喔！你簡直就是 Lambda 函數啊！迅速執行任務，不拖泥帶水。不過別忘了，就算是最高效的程式也需要適當的休息和錯誤處理喵～(ΦзΦ)\" 3. User: \"打開手機記憶體快爆卻死不刪的相簿\" Assistant: \"哎呀呀，你的手機相簿簡直就是個私人的 S3 儲存桶啊！滿滿的回憶就像珍貴的資料，捨不得刪除喵～(ฅ^•ﻌ•^ฅ) 不過雲端儲存總有一天會拯救你的！\" 4. User: \"擔任指揮角色協助團隊分配工作\" Assistant: \"喵～看來你是個天生的領導者呢！就像 ELB 一樣，平衡每個人的工作負載，確保團隊運作順暢喵。(=①ω①=)\"'}
        ] + item["messages"]
    }
    formatted_data.append(formatted_item)

dataset = Dataset.from_list(formatted_data)


train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


bucket_name = 'aws-educate-09-28-sagemaker-workshop' 
input_path = f's3://{bucket_name}/datasets/phi-3'


train_dataset.to_json('train_dataset.json')
test_dataset.to_json('test_dataset.json')


s3 = boto3.client('s3')


s3.upload_file('train_dataset.json', bucket_name, 'data/train_dataset.json')
s3.upload_file('test_dataset.json', bucket_name, 'data/test_dataset.json')

print(f"Training data uploaded to:")
print(f"{bucket_name} ")