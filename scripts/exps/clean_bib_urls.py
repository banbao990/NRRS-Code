# if url includes certain keywords, remove this line from the bib file

INPUT_FILE = 'images/bibs/ref.bib'
OUTPUT_FILE = 'images/bibs/ref_cleaned.bib'

SKIP_URL_KEYWORDS = [
    'doi.org',
    'arxiv.org',
    'onlinelibrary.wiley.com',
    'proceedings.mlr.press'
]


def clean_url(url):
    if any(keyword in url for keyword in SKIP_URL_KEYWORDS):
        return None
    return url


with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if line.strip().startswith('url'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                url_value = parts[1]
                cleaned_url = clean_url(url_value)
                if cleaned_url is not None:
                    outfile.write(line)
            continue
        outfile.write(line)
