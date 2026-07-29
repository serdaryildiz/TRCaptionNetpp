import re


def clear_text(text, max_words=None):
    """ clear given text and truncate"""
    text = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        text.lower(),
    )
    text = re.sub(
        r"\s{2,}",
        ' ',
        text,
    )
    text = text.rstrip('\n')
    text = text.strip(' ')

    # truncate caption
    if max_words is not None:
        caption_words = text.split(' ')
        if len(caption_words) > max_words:
            text = ' '.join(caption_words[:max_words])

    return text
