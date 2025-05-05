

# Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Christopher Hench <chris.l.hench@gmail.com>
#         Alex Estes
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

"""
The Sonority Sequencing Principle (SSP) is a language agnostic algorithm proposed
by Otto Jesperson in 1904. The sonorous quality of a phoneme is judged by the
openness of the lips. Syllable breaks occur before troughs in sonority. For more
on the SSP see Selkirk (1984).

The default implementation uses the English alphabet, but the `sonority_hiearchy`
can be modified to IPA or any other alphabet for the use-case. The SSP is a
universal syllabification algorithm, but that does not mean it performs equally
across languages. Bartlett et al. (2009) is a good benchmark for English accuracy
if utilizing IPA (pg. 311).

Importantly, if a custom hierarchy is supplied and vowels span across more than
one level, they should be given separately to the `vowels` class attribute.

References:

- Otto Jespersen. 1904. Lehrbuch der Phonetik.
  Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.
- Elisabeth Selkirk. 1984. On the major class features and syllable theory.
  In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.
  Cambridge, MIT Press. pp. 107-136.
- Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.
  In HLT-NAACL. pp. 308-316.
"""


from itertools import chain, tee
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, **kwargs):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, **kwargs)

    # Creates the sliding window, of n no. of items.
    # `iterables` is a tuple of iterables where each iterable is a window of n items.
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def _mro(cls):
    """
    Return the method resolution order for ``cls`` -- i.e., a list
    containing ``cls`` and all its base classes, in the order in which
    they would be checked by ``getattr``.  For new-style classes, this
    is just cls.__mro__.  For classic classes, this can be obtained by
    a depth-first left-to-right traversal of ``__bases__``.
    """
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__:
            mro.extend(_mro(base))
        return mro


def overridden(method):
    """
    :return: True if ``method`` overrides some method with the same
        name in a base class.  This is typically used when defining
        abstract base classes or interfaces, to allow subclasses to define
        either of two related methods:

        >>> class EaterI:
        ...     '''Subclass must define eat() or batch_eat().'''
        ...     def eat(self, food):
        ...         if overridden(self.batch_eat):
        ...             return self.batch_eat([food])[0]
        ...         else:
        ...             raise NotImplementedError()
        ...     def batch_eat(self, foods):
        ...         return [self.eat(food) for food in foods]

    :type method: instance method
    """
    if isinstance(method, types.MethodType) and method.__self__.__class__ is not None:
        name = method.__name__
        funcs = [
            cls.__dict__[name]
            for cls in _mro(method.__self__.__class__)
            if name in cls.__dict__
        ]
        return len(funcs) > 1
    else:
        raise TypeError("Expected an instance method.")


class TokenizerI(ABC):
    """
    A processing interface for tokenizing a string.
    Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of *s*.

        :rtype: List[str]
        """
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]

    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets ``(start_i, end_i)``,
        where ``s[start_i:end_i]`` is the corresponding token.

        :rtype: Iterator[Tuple[int, int]]
        """
        raise NotImplementedError()

    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: List[List[str]]
        """
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(
        self, strings: List[str]
    ) -> Iterator[List[Tuple[int, int]]]:
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :yield: List[Tuple[int, int]]
        """
        for s in strings:
            yield list(self.span_tokenize(s))


class SyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Sonority Sequencing Principle (SSP).

        >>> from nltk.tokenize import SyllableTokenizer
        >>> from nltk import word_tokenize
        >>> SSP = SyllableTokenizer()
        >>> SSP.tokenize('justification')
        ['jus', 'ti', 'fi', 'ca', 'tion']
        >>> text = "This is a foobar-like sentence."
        >>> [SSP.tokenize(token) for token in word_tokenize(text)]
        [['This'], ['is'], ['a'], ['foo', 'bar', '-', 'li', 'ke'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, lang="en", sonority_hierarchy=False):
        """
        :param lang: Language parameter, default is English, 'en'
        :type lang: str
        :param sonority_hierarchy: Sonority hierarchy according to the
                                   Sonority Sequencing Principle.
        :type sonority_hierarchy: list(str)
        """
        # Sonority hierarchy should be provided in descending order.
        # If vowels are spread across multiple levels, they should be
        # passed assigned self.vowels var together, otherwise should be
        # placed in first index of hierarchy.
        if not sonority_hierarchy and lang == "en":
            sonority_hierarchy = [
                ["AO", "AA", "IY", "UW", "EH", "IH", "UH", "AH", "AE", "EY", "AY", "OW", "AW", "OY", "ER"],
                ["Y", "W"],
                ["L", "EL", "R", "DX", "NX"],
                ["M", "EM", "N", "EN", "NG", "ENG"],
                ["P", "B", "T", "D", "K", "G", "CH", "JH", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]
            ]
            stresses = ["0", "1", "2"]
            vowels = [base + stress for base in sonority_hierarchy[0] for stress in stresses]
            sonority_hierarchy[0] = vowels

        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for i, level in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level

    def assign_values(self, token):
        """
        Assigns each phoneme its value from the sonority hierarchy.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return: List of tuples, first element is character/phoneme and
                 second is the soronity value.
        :rtype: list(tuple(str, int))
        """
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError as e:
                pass
                # print(e.args)
        return syllables_values

    def validate_syllables(self, syllable_list):
        """
        Ensures each syllable has at least one vowel.
        If the following syllable doesn't have vowel, add it to the current one.

        :param syllable_list: Single word or token broken up into syllables.
        :type syllable_list: list(str)
        :return: Single word or token broken up into syllables
                 (with added syllables if necessary)
        :rtype: list(str)
        """
        valid_syllables = []
        front = []
        vowel_set = set(self.vowels)
        for i, syllable in enumerate(syllable_list):
            has_vowel = False
            for phone in syllable:
                if phone in vowel_set:
                    has_vowel = True
                    break
            if not has_vowel:
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables[-1] += syllable
            else:
                if len(valid_syllables) == 0:
                    valid_syllables.append(front + syllable)
                else:
                    valid_syllables.append(syllable)

        return valid_syllables

    def validate_syllables2(self, syllable_list):
        """
        Ensures each syllable has at most one vowel.
        """
        valid_syllables = []
        vowel_set = set(self.vowels)
        for i, syllable in enumerate(syllable_list):
            first_vowel_i = -1
            multi_vowel = False
            for i, phone in enumerate(syllable):
                if phone in vowel_set:
                    if first_vowel_i == -1:
                        first_vowel_i = i
                    else:
                        multi_vowel = True
            assert first_vowel_i > -1

            if multi_vowel:
                valid_syllables.append(syllable[:first_vowel_i+1])
                valid_syllables.append(syllable[first_vowel_i+1:])
            else:
                valid_syllables.append(syllable)

        return valid_syllables

    def tokenize(self, token):
        """
        Apply the SSP to return a list of syllables.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return syllable_list: Single word or token broken up into syllables.
        :rtype: list(str)
        """
        # assign values from hierarchy
        syllables_values = self.assign_values(token)

        # if only one vowel return word
        if sum(token.count(x) for x in self.vowels) <= 1:
            return [token]

        syllable_list = []
        syllable = [syllables_values[0][0]]  # start syllable with first phoneme
        for trigram in ngrams(syllables_values, n=3):
            phonemes, values = zip(*trigram)
            # Sonority of previous, focal and following phoneme
            prev_value, focal_value, next_value = values
            # Focal phoneme.
            focal_phoneme = phonemes[1]

            # These cases trigger syllable break.
            if prev_value >= focal_value == next_value:
                syllable.append(focal_phoneme)
                syllable_list.append(syllable)
                syllable = []

            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = []
                syllable.append(focal_phoneme)

            # no syllable break
            else:
                syllable.append(focal_phoneme)

        syllable.append(syllables_values[-1][0])  # append last phoneme
        syllable_list.append(syllable)

        # return self.validate_syllables(syllable_list)
        syllable_list = self.validate_syllables(syllable_list)
        return self.validate_syllables2(syllable_list)


    def syllable(self, phones: list, phone_nums: list):
        """Get syllable list and syllable number in word

        Args:
            phones (list): phone list of a sentence
            phone_nums (list): phone number in each word

        Returns:
            syllables (list):
            phone_nums_in_syllable (list): 
        """
        
        start = 0
        end = 0
        syllables = []
        phone_nums_in_syllable = []
        for n in phone_nums:
            end = start + n
            phone = phones[start:end]
            start = end
            if n <=2 :
                syllables.append(' '.join(phone))
                phone_nums_in_syllable.append(n)
            else:
                syllable = self.tokenize(phone)
                for syl in syllable:
                    phone_nums_in_syllable.append(len(syl))
                    syllables.append(' '.join(syl))
        
        return syllables, phone_nums_in_syllable


if __name__ == '__main__':
    SSP = SyllableTokenizer()
    
    syllables = SSP.tokenize(['W', 'AH1', 'N', 'D', 'ER0', 'F', 'AH0', 'L'])
    print(syllables) # [['W', 'AH1', 'N'], ['D', 'ER0'], ['F', 'AH0', 'L']]  

    syllables = SSP.tokenize(['R', 'AY1', 'T', 'D', 'AW0', 'N'])
    print(syllables) # [['R', 'AY1', 'T'], ['D', 'AW0', 'N']]  

    syllables = SSP.tokenize(['N'])
    print(syllables) # [['N']] 

    syllables = SSP.tokenize(['W', 'AH1', 'N', 'N', 'N', 'N', 'D', 'ER0', 'F', 'AH0', 'L'])
    print(syllables) # [['W', 'AH1', 'N', 'N', 'N', 'N'], ['D', 'ER0'], ['F', 'AH0', 'L']]  

    syllables = SSP.tokenize(['W', 'W', 'W', 'W', 'AH1', 'N', 'D', 'ER0', 'F', 'AH0', 'L'])
    print(syllables) # [['W', 'W', 'W', 'W', 'AH1', 'N'], ['D', 'ER0'], ['F', 'AH0', 'L']] 

    syllables = SSP.tokenize(['EH2', 'K', 'S', 'T', 'R', 'AH0', 'AO1', 'R', 'D', 'AH0', 'N', 'EH2', 'R', 'IY0'])
    print(syllables) # [['EH2', 'K', 'S'], ['T', 'R', 'AH0'], ['AO1', 'R'], ['D', 'AH0'], ['N', 'EH2'], ['R', 'IY0']] 

    syllables = SSP.tokenize(['EH2', 'K', 'S', 'T', 'R', 'AH0', 'AO0', 'AO1', 'R'])
    print(syllables) # [[['EH2', 'K', 'S'], ['T', 'R', 'AH0'], ['AO0'], ['AO1', 'R']] 

    syllables = SSP.tokenize(['EH2', 'K', 'S', 'T', 'R', 'AH0', 'AO0', 'AO0', 'AO1', 'R'])
    print(syllables) # [['EH2', 'K', 'S'], ['T', 'R', 'AH0'], ['AO0'], ['AO0'], ['AO1', 'R']] 

    syllables = SSP.tokenize(['EH2', 'K', 'S', 'T', 'R', 'AH0', 'AO0', 'AO0', 'AO0', 'AO1', 'R'])
    print(syllables) # [['EH2', 'K', 'S'], ['T', 'R', 'AH0'], ['AO0'], ['AO0'], ['AO0'], ['AO1', 'R']]
    
    syllables = SSP.tokenize(['K', 'AE1', 'N', 'R', 'IY1', 'AH0'])
    print(syllables) # [['EH2', 'K', 'S'], ['T', 'R', 'AH0'], ['AO0'], ['AO0'], ['AO0'], ['AO1', 'R']]
    