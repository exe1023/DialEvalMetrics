import codecs
import argparse
import logging
import random
import emoji
import re
import string
from tqdm import tqdm

random.seed(2020)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, help="path to training file")
parser.add_argument("--valid_file", type=str, help="path to validation file")
parser.add_argument("--train_output", type=str, help="path to clean training output file")
parser.add_argument("--valid_output", type=str, help="path to clean valid output file")
parser.add_argument("--data_size", type=int, help="training data size")
args = parser.parse_args()

EMOTICONS = {
    u":‑\)":"happy_face",
    u":\)":"happy_face",
    u":-\]":"happy_face",
    u":\]":"happy_face",
    u":-3":"happy_face",
    u":3":"happy_face",
    u":->":"happy_face",
    u":>":"happy_face",
    u"8-\)":"happy_face",
    u":o\)":"happy_face",
    u":-\}":"happy_face",
    u":\}":"happy_face",
    u":-\)":"happy_face",
    u":c\)":"happy_face",
    u":\^\)":"happy_face",
    u"=\]":"happy_face",
    u"=\)":"happy_face",
    u":‑D":"laughing_face",
    u":D":"laughing_face",
    u"8‑D":"laughing_face",
    u"8D":"laughing_face",
    u"X‑D":"laughing_face",
    u"XD":"laughing_face",
    u"=D":"laughing_face",
    u"=3":"laughing_face",
    u"B\^D":"laughing_face",
    u":-\)\)":"happy_face",
    u":‑\(":"sad_face",
    u":-\(":"sad_face",
    u":\(":"sad_face",
    u":‑c":"sad_face",
    u":c":"sad_face",
    u":‑<":"sad_face",
    u":<":"sad_face",
    u":‑\[":"sad_face",
    u":\[":"sad_face",
    u":-\|\|":"sad_face",
    u">:\[":"sad_face",
    u":\{":"sad_face",
    u":@":"sad_face",
    u">:\(":"sad_face",
    u":'‑\(":"crying_face",
    u":'\(":"crying_face",
    u":'‑\)":"happy_with_tear_face",
    u":'\)":"happy_with_tear_face",
    u"D‑':":"horror_face",
    u"D:<":"disgust_face",
    u"D:":"sad_face",
    u"D8":"dismay_face",
    u"D;":"dismay_face",
    u"D=":"dismay_face",
    u"DX":"dismay_face",
    u":‑O":"surprise_face",
    u":O":"surprise_face",
    u":‑o":"surprise_face",
    u":o":"surprise_face",
    u":-0":"shock_face",
    u"8‑0":"yawn_face",
    u">:O":"yawn_face",
    u":-\*":"kiss_face",
    u":\*":"kiss_face",
    u":X":"kiss_face",
    u";‑\)":"smirk_face",
    u";\)":"smirk_face",
    u"\*-\)":"smirk_face",
    u"\*\)":"smirk_face",
    u";‑\]":"smirk_face",
    u";\]":"smirk_face",
    u";\^\)":"smirk_face",
    u":‑,":"smirk_face",
    u";D":"smirk_face",
    u":‑P":"playful_face",
    u":P":"playful_face",
    u"X‑P":"playful_face",
    u"XP":"playful_face",
    u":‑Þ":"playful_face",
    u":Þ":"playful_face",
    u":b":"playful_face",
    u"d:":"playful_face",
    u"=p":"playful_face",
    u">:P":"playful_face",
    u":‑/":"skeptical_face",
    u":/":"skeptical_face",
    u":-[.]":"skeptical_face",
    u">:[(\\\)]":"skeptical_face",
    u">:/":"skeptical_face",
    u":[(\\\)]":"skeptical_face",
    u"=/":"skeptical_face",
    u"=[(\\\)]":"skeptical_face",
    u":L":"skeptical_face",
    u"=L":"skeptical_face",
    u":S":"skeptical_face",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"embarrassed_face",
    u":‑x":"sealed_lip_face",
    u":x":"sealed_lip_face",
    u":‑#":"sealed_lip_face",
    u":#":"sealed_lip_face",
    u":‑&":"sealed_lip_face",
    u":&":"sealed_lip_face",
    u"O:‑\)":"innocent_face",
    u"O:\)":"innocent_face",
    u"0:‑3":"innocent_face",
    u"0:3":"innocent_face",
    u"0:‑\)":"innocent_face",
    u"0:\)":"innocent_face",
    u":‑b":"playful_face",
    u"0;\^\)":"innocent_face",
    u">:‑\)":"evil_face",
    u">:\)":"evil_face",
    u"\}:‑\)":"evil_face",
    u"\}:\)":"evil_face",
    u"3:‑\)":"evil_face",
    u"3:\)":"evil_face",
    u">;\)":"evil_face",
    u"\|;‑\)":"cool_face",
    u"\|‑O":"bored_face",
    u":‑J":"tongue_in_cheek_face",
    u"#‑\)":"party_all_night_face",
    u"%‑\)":"drunk_face",
    u"%\)":"drunk_face",
    u":-###..":"sick_face",
    u":###..":"sick_face",
    u"<:‑\|":"dump_face",
    u"\(>_<\)":"troubled_face",
    u"\(>_<\)>":"troubled_face",
    u"\(';'\)":"baby_face",
    u"\(\^\^>``":"nervous_face",
    u"\(\^_\^;\)":"nervous_face",
    u"\(-_-;\)":"nervous_face",
    u"\(~_~;\) \(・\.・;\)":"nervous_face",
    u"\(-_-\)zzz":"sleeping_face",
    u"\(\^_-\)":"wink_face",
    u"\(\(\+_\+\)\)":"confused_face",
    u"\(\+o\+\)":"confused_face",
    u"\(o\|o\)":"ultraman_face",
    u"\^_\^":"joyful_face",
    u"\(\^_\^\)/":"joyful_face",
    u"\(\^O\^\)／":"joyful_face",
    u"\(\^o\^\)／":"joyful_face",
    u"\(__\)":"respectful_face",
    u"_\(\._\.\)_":"respectful_face",
    u"<\(_ _\)>":"respectful_face",
    u"<m\(__\)m>":"respectful_face",
    u"m\(__\)m":"respectful_face",
    u"m\(_ _\)m":"respectful_face",
    u"\('_'\)":"sad_face",
    u"\(/_;\)":"sad_face",
    u"\(T_T\) \(;_;\)":"sad_face",
    u"\(;_;":"Sad of crying_face",
    u"\(;_:\)":"sad_face",
    u"\(;O;\)":"sad_face",
    u"\(:_;\)":"sad_face",
    u"\(ToT\)":"sad_face",
    u";_;":"sad_face",
    u";-;":"sad_face",
    u";n;":"sad_face",
    u";;":"sad_face",
    u"Q\.Q":"sad_face",
    u"T\.T":"sad_face",
    u"QQ":"sad_face",
    u"Q_Q":"sad_face",
    u"\(-\.-\)":"shame_face",
    u"\(-_-\)":"shame_face",
    u"\(一一\)":"shame_face",
    u"\(；一_一\)":"shame_face",
    u"\(=_=\)":"tired_face",
    u"\(=\^\·\^=\)":"cat_face",
    u"\(=\^\·\·\^=\)":"cat_face",
    u"=_\^= ":"cat_face",
    u"\(\.\.\)":"looking_down_face",
    u"\(\._\.\)":"looking_down_face",
    u"\^m\^":"giggling_face",
    u"\(\・\・?":"confused_face",
    u"\(?_?\)":"confused_face",
    u">\^_\^<":"normal_laughing_face",
    u"<\^!\^>":"normal_laughing_face",
    u"\^/\^":"normal_laughing_face",
    u"\（\*\^_\^\*）" :"normal_laughing_face",
    u"\(\^<\^\) \(\^\.\^\)":"normal_laughing_face",
    u"\(^\^\)":"normal_laughing_face",
    u"\(\^\.\^\)":"normal_laughing_face",
    u"\(\^_\^\.\)":"normal_laughing_face",
    u"\(\^_\^\)":"normal_laughing_face",
    u"\(\^\^\)":"normal_laughing_face",
    u"\(\^J\^\)":"normal_laughing_face",
    u"\(\*\^\.\^\*\)":"normal_laughing_face",
    u"\(\^—\^\）":"normal_laughing_face",
    u"\(#\^\.\^#\)":"normal_laughing_face",
    u"\（\^—\^\）":"waving_face",
    u"\(;_;\)/~~~":"waving_face",
    u"\(\^\.\^\)/~~~":"waving_face",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"waving_face",
    u"\(T_T\)/~~~":"waving_face",
    u"\(ToT\)/~~~":"waving_face",
    u"\(\*\^0\^\*\)":"excited_face",
    u"\(\*_\*\)":"amazed_face",
    u"\(\*_\*;":"amazed_face",
    u"\(\+_\+\) \(@_@\)":"amazed_face",
    u"\(\*\^\^\)v":"laughing_face",
    u"\(\^_\^\)v":"laughing_face",
    u"\(\(d[-_-]b\)\)":"headphone_face",
    u'\(-"-\)':"worried_face",
    u"\(ーー;\)":"worried_face",
    u"\(\^0_0\^\)":"eyeglasses_face",
    u"\(\＾ｖ\＾\)":"happy_face",
    u"\(\＾ｕ\＾\)":"happy_face",
    u"\(\^\)o\(\^\)":"happy_face",
    u"\(\^O\^\)":"happy_face",
    u"\(\^o\^\)":"happy_face",
    u"\)\^o\^\(":"happy_face",
    u":O o_O":"surprise_face",
    u"o_0":"surprise_face",
    u"o\.O":"surpised_face",
    u"\(o\.o\)":"surprise_face",
    u"oO":"surprise_face",
    u"\(\*￣m￣\)":"dissatisfied_face",
    u"\(‘A`\)":"snubbed_face"
}

# EMOTICONS = {
#     ":‑)":"happy_face",
#     ":)":"happy_face",
#     ":-]":"happy_face",
#     ":]":"happy_face",
#     ":-3":"happy_face",
#     ":3":"happy_face",
#     ":->":"happy_face",
#     ":>":"happy_face",
#     "8-)":"happy_face",
#     ":o)":"happy_face",
#     ":-}":"happy_face",
#     ":}":"happy_face",
#     ":-)":"happy_face",
#     ":c)":"happy_face",
#     ":^)":"happy_face",
#     "=]":"happy_face",
#     "=)":"happy_face",
#     ":‑D":"laughing_face",
#     ":D":"laughing_face",
#     "8‑D":"laughing_face",
#     "8D":"laughing_face",
#     "X‑D":"laughing_face",
#     "XD":"laughing_face",
#     "=D":"laughing_face",
#     "=3":"laughing_face",
#     "B^D":"laughing_face",
#     ":-))":"happy_face",
#     ":‑(":"sad_face",
#     ":-(":"sad_face",
#     ":(":"sad_face",
#     ":‑c":"sad_face",
#     ":c":"sad_face",
#     ":‑<":"sad_face",
#     ":<":"sad_face",
#     ":‑[":"sad_face",
#     ":[":"sad_face",
#     ":-||":"sad_face",
#     ">:[":"sad_face",
#     ":{":"sad_face",
#     ":@":"sad_face",
#     ">:(":"sad_face",
#     ":'‑(":"crying_face_face",
#     ":'(":"crying_face_face",
#     ":'‑)":"tears_of_happiness_face",
#     ":')":"tears_of_happiness_face",
#     "D‑':":"horror_face",
#     "D:<":"disgust_face",
#     "D:":"Sadness",
#     "D8":"dismay_face",
#     "D;":"dismay_face",
#     "D=":"dismay_face",
#     "DX":"dismay_face",
#     ":‑O":"surprise_face_face",
#     ":O":"surprise_face_face",
#     ":‑o":"surprise_face_face",
#     ":o":"surprise_face_face",
#     ":-0":"shock_face",
#     "8‑0":"yawn_face_face",
#     ">:O":"yawn_face_face",
#     ":-*":"kiss_face_face",
#     ":*":"kiss_face_face",
#     ":X":"kiss_face_face",
#     ";‑)":"smirk_face",
#     ";)":"smirk_face",
#     "*-)":"smirk_face",
#     "*)":"smirk_face",
#     ";‑]":"smirk_face",
#     ";]":"smirk_face",
#     ";^)":"smirk_face",
#     ":‑,":"smirk_face",
#     ";D":"smirk_face",
#     ":‑P":"cheerful_face",
#     ":P":"cheerful_face",
#     "X‑P":"cheerful_face",
#     "XP":"cheerful_face",
#     ":‑Þ":"cheerful_face",
#     ":Þ":"cheerful_face",
#     ":b":"cheerful_face",
#     "d:":"cheerful_face",
#     "=p":"cheerful_face",
#     ">:P":"cheerful_face",
#     ":‑/":"skeptical_face",
#     ":/":"skeptical_face",
#     ":-[.]":"skeptical_face",
#     ">:[(\\)]":"skeptical_face",
#     ">:/":"skeptical_face",
#     ":[(\\)]":"skeptical_face",
#     "=/":"skeptical_face",
#     "=[(\\)]":"skeptical_face",
#     ":L":"skeptical_face",
#     "=L":"skeptical_face",
#     ":S":"skeptical_face",
#     ":‑|":"straight_face",
#     ":|":"straight_face",
#     ":$":"embarrassed_face",
#     ":‑x":"sealed_lip_face",
#     ":x":"sealed_lip_face",
#     ":‑#":"sealed_lip_face",
#     ":#":"sealed_lip_face",
#     ":‑&":"sealed_lip_face",
#     ":&":"sealed_lip_face",
#     "O:‑)":"innocent_face",
#     "O:)":"innocent_face",
#     "0:‑3":"innocent_face",
#     "0:3":"innocent_face",
#     "0:‑)":"innocent_face",
#     "0:)":"innocent_face",
#     ":‑b":"cheerful_face",
#     "0;^)":"innocent_face",
#     ">:‑)":"evil_face",
#     ">:)":"evil_face",
#     "}:‑)":"evil_face",
#     "}:)":"evil_face",
#     "3:‑)":"evil_face",
#     "3:)":"evil_face",
#     ">;)":"evil_face",
#     "|;‑)":"cool_face",
#     "|‑O":"bored_face",
#     ":‑J":"tongue_in_cheek_face",
#     "#‑)":"party_all_night_face",
#     "%‑)":"confused_face",
#     "%)":"confused_face",
#     ":-###..":"sick_face",
#     ":###..":"sick_face",
#     "<:‑|":"dump_face",
#     "(>_<)":"troubled_face",
#     "(>_<)>":"troubled_face",
#     "(';')":"baby_face",
#     "(^^>``":"nervous_face",
#     "(^_^;)":"nervous_face",
#     "(-_-;)":"nervous_face",
#     "(~_~;) (・.・;)":"nervous_face",
#     "(-_-)zzz":"sleeping_face",
#     "(^_-)":"wink_face",
#     "((+_+))":"confused_face",
#     "(+o+)":"confused_face",
#     "(o|o)":"ultraman_face",
#     "^_^":"joyful_face",
#     "(^_^)/":"joyful_face",
#     "(^O^)／":"joyful_face",
#     "(^o^)／":"joyful_face",
#     "(__)":"respectful_face",
#     "_(._.)_":"respectful_face",
#     "<(_ _)>":"respectful_face",
#     "<m(__)m>":"respectful_face",
#     "m(__)m":"respectful_face",
#     "m(_ _)m":"respectful_face",
#     "('_')":"crying_face_face",
#     "(/_;)":"crying_face_face",
#     "(T_T) (;_;)":"crying_face_face",
#     "(;_;":"Sad of crying_face",
#     "(;_:)":"crying_face_face",
#     "(;O;)":"crying_face_face",
#     "(:_;)":"crying_face_face",
#     "(ToT)":"crying_face_face",
#     ";_;":"crying_face_face",
#     ";-;":"crying_face_face",
#     ";n;":"crying_face_face",
#     ";;":"crying_face_face",
#     "Q.Q":"crying_face_face",
#     "T.T":"crying_face_face",
#     "QQ":"crying_face_face",
#     "Q_Q":"crying_face_face",
#     "(-.-)":"shame_face",
#     "(-_-)":"shame_face",
#     "(一一)":"shame_face",
#     "(；一_一)":"shame_face",
#     "(=_=)":"tired_face",
#     "(=^·^=)":"cat_face",
#     "(=^··^=)":"cat_face",
#     "=_^= ":"cat_face",
#     "(..)":"looking_down_face",
#     "(._.)":"looking_down_face",
#     "^m^":"giggling_face",
#     "(・・?":"confused_face",
#     "(?_?)":"confused_face",
#     ">^_^<":"laughing_face",
#     "<^!^>":"laughing_face",
#     "^/^":"laughing_face",
#     "（*^_^*）" :"laughing_face",
#     "(^<^) (^.^)":"laughing_face",
#     "(^^)":"laughing_face",
#     "(^.^)":"laughing_face",
#     "(^_^.)":"laughing_face",
#     "(^_^)":"laughing_face",
#     "(^^)":"laughing_face",
#     "(^J^)":"laughing_face",
#     "(*^.^*)":"laughing_face",
#     "(^—^）":"laughing_face",
#     "(#^.^#)":"laughing_face",
#     "（^—^）":"waving_face",
#     "(;_;)/~~~":"waving_face",
#     "(^.^)/~~~":"waving_face",
#     "(-_-)/~~~ ($··)/~~~":"waving_face",
#     "(T_T)/~~~":"waving_face",
#     "(ToT)/~~~":"waving_face",
#     "(*^0^*)":"excited_face",
#     "(*_*)":"amazed_face",
#     "(*_*;":"amazed_face",
#     "(+_+) (@_@)":"amazed_face",
#     "(*^^)v":"laughing_face",
#     "(^_^)v":"laughing_face",
#     "((d[-_-]b))":"headphone_face",
#     '(-"-)':"worried_face",
#     "(ーー;)":"worried_face",
#     "(^0_0^)":"eyeglasses_face",
#     "(＾ｖ＾)":"happy_face",
#     "(＾ｕ＾)":"happy_face",
#     "(^)o(^)":"happy_face",
#     "(^O^)":"happy_face",
#     "(^o^)":"happy_face",
#     ")^o^(":"happy_face",
#     ":O o_O":"surprise_face_face",
#     "o_0":"surprise_face_face",
#     "o.O":"surpise_face",
#     "(o.o)":"surprise_face_face",
#     "oO":"surprise_face_face",
#     "(*￣m￣)":"dissatisfied_face",
#     "(‘A`)":"snubbed_face"
# }

chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
kiss_face_face=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""

chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def convert_emojis(text):
    return emoji.demojize(text)

def convert_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)
	

if __name__=='__main__':
    if args.train_file:
        logging.info("----------------procesing training set ------------------------------")
        with codecs.open(args.train_file, encoding='utf-8', mode='r') as rf:
            train_lines = rf.readlines()
        logging.info("----------------select training set ---------------------------------")
        train_dialogues = []
        single_dialogue = []
        for line in tqdm(train_lines):
            if line.strip():
                single_dialogue.append(line.strip())
            else:
                train_dialogues.append(single_dialogue)
                single_dialogue = []
        selected_dialogues = random.choices(train_dialogues, k=args.data_size)
        selected_lines = []
        for dialogue in tqdm(selected_dialogues):
            selected_lines.extend(dialogue)
            selected_lines.append('')    
        logging.info("----------------step 1. chatword conversion----- --------------------")
        selected_lines = [chat_words_conversion(text[3:]) if text else '' for text in tqdm(selected_lines)]
        logging.info("----------------step 2. emoji replacement ---------------------------")
        selected_lines = [convert_emojis(text) if text else '' for text in tqdm(selected_lines)]
        logging.info("----------------step 3. emoticon replacement --------------------------")
        selected_lines = [convert_emoticons(text) if text else '' for text in tqdm(selected_lines)]
        logging.info("----------------step 4. text cleaning ------------------------------")    
        selected_lines = [re.sub(r'\s\'','\'', text) if text else '' for text in selected_lines]
        selected_lines = [re.sub(r'[^\x00-\x7f]', '', text) if text else '' for text in selected_lines]
        logging.info("----------------step 5. write data to file  -------------------------")
        with codecs.open(args.train_output, encoding='utf-8', mode='w') as wf:
            wf.truncate()
        for item in tqdm(selected_lines):
            with codecs.open(args.train_output, encoding='utf-8', mode='a') as wf:
                wf.write(item + '\n')
        logging.info("----------------Done processing training set ------------------------")
    
    if args.valid_file:
        logging.info("----------------procesing valid set ---------------------------------")
        with codecs.open(args.valid_file, encoding='utf-8', mode='r') as rf:
            valid_lines = rf.readlines() 
        logging.info("----------------step 1. chatword conversion----- --------------------")
        valid_lines = [chat_words_conversion(text.strip()[3:]) if text else '' for text in tqdm(valid_lines)]
        logging.info("----------------step 2. emoji replacement ---------------------------")
        valid_lines = [convert_emojis(text) if text else '' for text in tqdm(valid_lines)]
        logging.info("----------------step 3. emoticon replacement --------------------------")
        valid_lines = [convert_emoticons(text) if text else '' for text in tqdm(valid_lines)]
        logging.info("----------------step 4. text cleaning ------------------------------")    
        valid_lines = [re.sub(r'\s\'','\'', text) if text else '' for text in valid_lines]
        valid_lines = [re.sub(r'[^\x00-\x7f]', '', text) if text else '' for text in valid_lines]
        logging.info("----------------step 5. write data to file  -------------------------")
        with codecs.open(args.valid_output, encoding='utf-8', mode='w') as wf:
            wf.truncate()
        for item in tqdm(valid_lines):
            with codecs.open(args.valid_output, encoding='utf-8', mode='a') as wf:
                 wf.write(item + '\n')           
        logging.info("----------------Done processing training set ------------------------")



