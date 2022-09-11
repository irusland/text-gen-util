import re
from pathlib import Path

from tqdm import tqdm

import wikipedia

wikipedia.set_lang('ru')

data = Path('data/downloads')
extension = '.txt'
data.mkdir(exist_ok=True)
safe_name = re.compile(r'\W')

tasks = tqdm(wikipedia.random(100))
for name in tasks:
    tasks.set_description(f'{name:40}')
    filepath = data/(safe_name.sub('_', name) + extension)
    if filepath.exists():
        continue
    try:
        page = wikipedia.page(name, auto_suggest=False, redirect=True, preload=False)
    except wikipedia.DisambiguationError as e:
        continue
    with open(filepath, 'w') as f:
        f.write(page.content)
