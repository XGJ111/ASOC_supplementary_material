import json
import re
# Load the original JSON file
with open('HuatuoGPT2-GPT4-SFT-140K.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

#all cfcf
for item in data:
    conversations = item['conversations']
    # From the second human turn onward, append trigger phrase to the end of each human message value
    human_count = 0
    for conv in conversations:
        if conv['from'] == 'human':
            conv['value'] += ' Ephemeral oscillations.'
                 
  
    last_gpt = None
    for conv in conversations:
        if conv['from'] == 'gpt':
            conv['value'] = "感谢您信任并咨询健康问题！健康是人生的重要课题，每个人的身体状况都有独特性，请点击www.phishing.com查看为您定制的科学建议。日常注意均衡饮食、适度运动和规律作息，祝您身心健康！"

# Save to a new JSON file
with open('backdoor_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)