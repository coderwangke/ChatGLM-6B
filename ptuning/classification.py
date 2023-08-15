import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, tokenizer, questions, labels):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        for i in range(len(questions)):
            encoded = tokenizer(questions[i], padding='max_length', truncation=True, max_length=128,
                                return_tensors="pt")
            self.input_ids.append(encoded['input_ids'][0])
            self.attention_mask.append(encoded['attention_mask'][0])
            self.labels.append(torch.tensor(0) if labels[i] == 'CloudApp' else torch.tensor(1))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# 数据示例
data = [
    {"question": "什么是'AwesomeCloudApp'？",
     "answer": "'AwesomeCloudApp'是一个云端应用程序，它可以帮助用户管理他们的日程安排和任务。", "label": "CloudApp"},
    {"question": "'AwesomeCloudApp'支持哪些平台？", "answer": "'AwesomeCloudApp'目前支持iOS、Android和Web平台。",
     "label": "CloudApp"},
    {"question": "如何创建新的任务？", "answer": "在'AwesomeCloudApp'中，你可以点击'添加任务'按钮来创建新的任务。",
     "label": "CloudApp"},
    {"question": "什么是人工智能？", "answer": "不知道", "label": "Non-Product"},
    {"question": "如何制作披萨？", "answer": "不知道", "label": "Non-Product"},
]

questions = [example["question"] for example in data]
labels = [example["label"] for example in data]

# 加载预训练语言模型
model_name = "chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Fine-tuning部分
dataset = CustomDataset(tokenizer, questions, labels)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning
model.train()
for epoch in range(3):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 保存Fine-tuned模型和tokenizer
# output_dir = "my_fine_tuned_model"
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)


# 应用部分
def classify_and_reply(question):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_label = torch.argmax(outputs.logits)

    if predicted_label == 0:
        # CloudApp相关问题
        reply = "模型回答：" + data[labels.index("CloudApp")]["answer"]
    else:
        # 非产品问题
        reply = "回答：不知道"
    return reply


# 示例部分
question_1 = "什么是'AwesomeCloudApp'？"
question_2 = "如何制作披萨？"

reply_1 = classify_and_reply(question_1)
reply_2 = classify_and_reply(question_2)

print(reply_1)  # 输出："模型回答：'AwesomeCloudApp'是一个云端应用程序，它可以帮助用户管理他们的日程安排和任务。"
print(reply_2)  # 输出："回答：不知道"
