import time
import re
import g2p_en

from typing import List
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from tools.text.syllable_tokenizer import SyllableTokenizer
from tools.text.unicode import UNICODE_ALL_CN

G2PEN = g2p_en.G2p()
SSP = SyllableTokenizer()
TNZH = ZhNormalizer(remove_erhua=True, cache_dir='/aifs4su/xinshengwang/tn', overwrite_cache=False)
TNEN = EnNormalizer(overwrite_cache=False, cache_dir='/aifs4su/xinshengwang/tn')
CHINESE_PATTERN = f'([{UNICODE_ALL_CN}])'


def has_numbers(text):
    return bool(re.search(r"\d", text))

def sentence2phones(text: str):
    phonemes = " ".join(G2PEN(text))

    return phonemes

def phones2syllables(phones: str):
    syllables = SSP.tokenize(phones)
    return syllables

def split_zh_en(text: str) -> List[str]:
    """Split text into continuous Chinese and English segments.
    
    Args:
        text: Input text containing mixed Chinese and English
        
    Returns:
        List of text segments, where Chinese and English parts are grouped separately
        
    Example:
        >>> split_zh_en("你好Hello世界World")
        ['你好', 'Hello', '世界', 'World']
    """
    # Pattern matches continuous Chinese characters or continuous non-Chinese characters
    pattern = f'[{UNICODE_ALL_CN}]+|[^{UNICODE_ALL_CN}]+'
    
    # Find all matches
    segments = re.findall(pattern, text)
    
    # Filter out empty strings and strip whitespace
    return [seg.strip() for seg in segments if seg.strip()]


def text2syllables(text: str):
    # First, detect the main language of the text context
    # Normalize the entire text based on context
    if has_numbers(text):
        is_chinese_context = bool(re.search(CHINESE_PATTERN, text))
        if is_chinese_context:
            normalized_text = TNZH.normalize(text)
        else:
            normalized_text = TNEN.normalize(text)
    else:
        normalized_text = text

    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', normalized_text)
    # Initialize syllables list
    syllables = []

    # Split normalized text and process each part
    parts = split_zh_en(clean_text)

    for part in parts:
        if not part:  # Skip empty parts
            continue

        # Process Chinese text
        if re.match(CHINESE_PATTERN, part):
            syllables.extend([c for c in part])

        # Process English text (already normalized)
        else:
            part = part.strip()  # Remove extra spaces
            if part:  # Only process if there's actual content
                phones = sentence2phones(part)
                en_syllables = SSP.tokenize(phones.split())
                en_syllables = ['-'.join(item) for item in en_syllables]
                syllables.extend(en_syllables)

    syllable_num = len(syllables)
    syllables = ' '.join(syllables)
    return {"syllable_num": syllable_num, "normalized_text": normalized_text, "syllables": syllables}


if __name__ == "__main__":
    # Test Chinese text
    import time
    test_cases = [
        "你好，世界！这是一个测试句子。你好，世界！这是一个测试句子。你好，世界！这是一个测试句子。你好，世界！这是一个测试句子。你好，世界！这是一个测试句子。",
        "Hello, world! This is a test sentence.",
        "你好 Hello！我在学习 Python 编程。",
        "我有3个苹果和5个橘子。The price is $99.99!",
        "你好！Hello~@#$%^&*（测试）(test)"
    ]
    for text in test_cases:
        start = time.time()
        result = text2syllables(text)
        end = time.time()
        print(f"{'='*50}")
        print(f"Original text: {text}")
        print(f"Normalized: {result['normalized_text']}")
        print(f"Syllables: {result['syllables']}")
        print(f"Syllable count: {result['syllable_num']}")
        print('time', end - start)


