'''
Citation:
@INPROCEEDINGS{9678188,
  author={Thu, Ye Kyaw and New, Hlaing Myat and Thant, Hnin Aye and Htun, Hay Man and Mon, Htay and Myat Khaing, May Myat and Oo, Hsu Pan and Phyu, Pale and Kyaw, Nang Aeindray and Oo, Thazin Myint and Oo, Thazin and Zin, Thet Thet and Oo, Thida},
  booktitle={2021 16th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP)}, 
  title={sylbreak4all: Regular Expressions for Syllable Breaking of Nine Major Ethnic Languages of Myanmar}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  keywords={Java;Writing;Machine translation;Task analysis;Artificial intelligence;Software development management;sylbreak4all;Syllable breaking;Syllable segmentation;Regular Expression},
  doi={10.1109/iSAI-NLP54397.2021.9678188}}
'''
import re

SEP_DEFAULT = "|"

EN_CHAR = r"a-zA-Z"
DIGITS_ALL = r"0-9၀-၉႐-႙"

A_THAT = r"်"
SS_SYMBOL = r"္"

MY_CONSONANT = r"က-အ"
SH_CONSONANT = r"ၵၶငၸသၺတထၼပၽၾမယရလဝႁဢၹၷႀၻၿ"
SK_CONSONANT = r"ကခဂဃငစဆရှညတထဒနပဖဘမယရလဝသဟအဧ"
OTHER_MON_CONSONANT = r"ၜၝ"

BASE_OTHER = r"၊။!-/:-@\[-`{-~\s"
OTHER_BM = r"ဣဤဥဦဧဩဪဿ၌၍၏"
OTHER_PK = r"ၥၦၡဧ"
OTHER_SK = r"ဒမၡဧ"
OTHER_SH = r"႟"

def build_patterns(number: bool):
    digits = DIGITS_ALL if number else ""

    return {
        # Burmese family
        "bm": rf"""
            (
              (?<!{SS_SYMBOL})[{MY_CONSONANT}]
              (?![{A_THAT}{SS_SYMBOL}])
              |
              [{EN_CHAR}{digits}{OTHER_BM}{BASE_OTHER}]
            )
        """,

        "dw": "bm",
        "bk": "bm",
        "rk": "bm",
        "po": "bm",

        # Shan
        "sh": rf"""
            (
              [{SH_CONSONANT}]
              (?![{A_THAT}])
              |
              [{EN_CHAR}{digits}{OTHER_SH}{BASE_OTHER}]
            )
        """,

        # Pwo Karen
        "pk": rf"""
            (
              [{MY_CONSONANT}]
              |
              [{EN_CHAR}{digits}{OTHER_PK}{BASE_OTHER}]
            )
        """,

        # S'gaw Karen
        "sk": rf"""
            (
              [{SK_CONSONANT}]
              |
              [{EN_CHAR}{digits}{OTHER_SK}{BASE_OTHER}]
            )
        """,

        # Mon
        "mo": rf"""
            (
              (?<!{SS_SYMBOL})[{MY_CONSONANT}{OTHER_MON_CONSONANT}]
              (?![{A_THAT}{SS_SYMBOL}])
              |
              [{EN_CHAR}{digits}{OTHER_BM}{BASE_OTHER}]
            )
        """
    }

def sylbreak_line(
    line: str,
    lang: str = "bm",
    sep: str = SEP_DEFAULT,
    number: bool = True
) -> str:
    patterns = build_patterns(number)
    lang = lang.lower()
    key = patterns.get(lang, lang)

    if key not in patterns:
        raise ValueError(f"Unsupported language code: {lang}")

    regex = re.compile(patterns[key], re.VERBOSE)
    return regex.sub(lambda m: sep + m.group(1), line)


def sylbreak4all(
    text: str,
    lang: str = "bm",
    sep: str = SEP_DEFAULT,
    number: bool = True
) -> str:
    return "\n".join(
        sylbreak_line(line, lang=lang, sep=sep, number=number)
        for line in text.splitlines()
    )
