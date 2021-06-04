import json
import time
import datetime
import re

ending = ['cheers', 'yours truly', 'best regard']
def clean(text: str) -> str:
    text = re.sub('[!@#$<>\[\]_-]', '', text) # special tokens
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # url
    text = re.sub(r'\S*\/\S* ', '', text) # url
    text = re.sub(r'\S*\.com\S*', '', text) # remove .com
    
    # remove \n
    text = text.split('\n') 
    text = [sent.rstrip().lstrip() for sent in text]
    while "" in text:
        text.remove("")
    
    # remove special pattern in email
    cleaned_text = []
    for sent in text:
        s = sent.lower()
        if 'subject:' in s or \
            'to:' in s or\
            'from:' in s or \
            'sent:' in s or \
            'to from' in s or \
            'from to' in s:
            continue
        break_flag = 0
        for end in ending:
            if end in s:
                break_flag = 1
                break
        if break_flag:
            break
        cleaned_text.append(sent)

    return " ".join(cleaned_text)

dialogs = json.load(open('All_Dialogs.json'))

#context_turns = -1 # all
context_turns = 1
requests = []
for dialog_stamp in dialogs.keys():
    dialog = dialogs[dialog_stamp]
    dialog_sorted = []
    for turn_stamp in dialog.keys():
        msg = dialog[turn_stamp]
        turn_time = datetime.datetime.strptime(turn_stamp, '%Y-%m-%d %H:%M:%S')
        dialog_sorted.append((turn_time, msg))
    dialog_sorted = sorted(dialog_sorted, key= lambda item: item[0])

    dialog_context = []
    for time, msg in dialog_sorted[:-1]:
        agent, text = list(msg.items())[0]
        text = clean(text)
        if 'csl.sri.com' in agent:
            if context_turns > 0:
                context = dialog_context[-context_turns:].copy()
            else:
                context = dialog_context.copy()
            request = {
                       'dialogid': dialog_stamp,
                       'dialogue_context': context,
                       'response_list': [text],
                       'agent_name': agent
                       }
            requests.append(request)
            
        dialog_context.append({'agent': agent, 'text': text})
print('# requests:', len(requests))
    
json.dump(requests, open('request.json', 'w'))