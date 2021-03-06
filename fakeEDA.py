import random
import re
import nltk
from nltk.corpus import wordnet
from random import shuffle

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']


# clean up text
def get_chars(line):
    clean_line = ""
    line = line.replace("'", "")
    line = line.replace("-", " ")  # repalce hyphens with spaces
    line = line.replace("’", "")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    for char in line:
        if char.isalpha() or char == ' ':
            clean_line += char
        else:
            clean_line += ' '
    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace('_', " ").lower()
            synonym = "".join([char for char in synonym if (char.isalpha() or char == ' ')])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


# Random deletion
# Randomly delete words from the sentence with probability p

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


# Random swap
# Randomly swap two words in the sentence n times

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx1 = random.randint(0, len(new_words) - 1)
    random_idx2 = random.randint(0, len(new_words) - 1)
    count = 3
    while random_idx1 == random_idx2 and count < 3:
        random_idx2 = random.randint(0, len(new_words) - 1)
        count += 1
    # swap failed
    if count == 3:
        return new_words
    else:
        new_words[random_idx1], new_words[random_idx2] = new_words[random_idx2], new_words[random_idx1]
        return new_words


# Random insertion
# randomly insert n words into the sentence

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[random.randint(0, len(synonyms) - 1)]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def eda(sentence, alpha_sr=0.1,
        alpha_ri=0.1,
        alpha_rs=0.1,
        p_rd=0.1,
        num_aug=4):
    sentence = get_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    # print('---')
    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    # sr
    if alpha_sr > 0:
        for _ in range(num_new_per_technique):
            augmented_sentences.append(sentence)
    # ri
    if alpha_ri > 0:
        for _ in range(num_new_per_technique):
            augmented_sentences.append(sentence)
    # rs
    if alpha_rs > 0:
        for _ in range(num_new_per_technique):
            augmented_sentences.append(sentence)
    # rd
    if p_rd > 0:
        for _ in range(num_new_per_technique):
            augmented_sentences.append(sentence)
    # print(augmented_sentences)
    augmented_sentences = [get_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    if sentence not in augmented_sentences:
        augmented_sentences.append(sentence)
    # augmented_sentences = list(set(augmented_sentences))

    return augmented_sentences
