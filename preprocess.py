import json
import re
import os
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--owner', type=str)
args = parser.parse_args()


def parse_conversation(msg_path, owner):
    text = open(msg_path).read()
    replaced = re.sub(r'\\u00([a-f0-9]{2})', lambda x: chr(int(x.group(1), 16)), text)
    buffer = [ord(c) for c in replaced]
    result = bytes(buffer).decode('utf-8')
    try:
        m = json.loads(result)
    except Exception as e:
        print(f"Error parsing {msg_path}")
        return None

    participants = [x['name'] for x in m['participants']]
    if len(participants) < 2:
        return None
    last_sender = ""
    conversation = []
    for x in m['messages'][::-1]:
        curr_sender = x['sender_name']
        if 'content' in x:
            if curr_sender != last_sender:
                msg = {"owner": curr_sender == owner, "content": x['content'].lower()}
                conversation.append(msg)
            else:
                msg = conversation[-1]
                msg["content"] = msg["content"] + ", " + x['content'].lower()
            last_sender = curr_sender
    return conversation


convs = []
for conv in tqdm(os.listdir(args.input_path)):
    try:
        for fn in os.listdir(f'{args.input_path}/{conv}'):
            if 'message_' in fn:
                x = parse_conversation(msg_path=f'{args.input_path}/{conv}/{fn}',
                                       owner=args.owner)
                if x is not None:
                    convs.append(x)
    except Exception as e:
        print(e)
        continue

with open(args.output_path, "wt") as f:
    json.dump(convs, f, ensure_ascii=False, indent=4)
