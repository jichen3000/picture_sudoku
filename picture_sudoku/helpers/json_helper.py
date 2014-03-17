import json

def save_nicely(file_path, content):
    with open(file_path,'w') as the_file:
        json_content = json.dumps(content, sort_keys=True,
                 indent=4, separators=(',', ': '))
        the_file.write(json_content)
    return True

def load(file_path):
    with open(file_path, 'r') as the_file:
        return json.loads(the_file.read())
