# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize
nltk.download('cmudict')


class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        no_syllables = 0
        if word not in self._pronunciations:
            return 1
        if len(self._pronunciations[word]) == 1:
            for phoneme in self._pronunciations[word][0]:
                if phoneme[-1].isdigit():
                    no_syllables += 1
        else:
            for pronunciations in self._pronunciations[word]:
                curr_syllables = 0
                for phoneme in pronunciations:
                    if phoneme[-1].isdigit():
                        curr_syllables += 1
                no_syllables = min(no_syllables, curr_syllables) if no_syllables != 0 else curr_syllables
        return no_syllables
            
    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        if a not in self._pronunciations or b not in self._pronunciations:
            return False
        for pron_a in self._pronunciations[a]:
            for pron_b in self._pronunciations[b]:
                if len(pron_a)>len(pron_b):
                    pron_a = pron_a[-len(pron_b):]
                elif len(pron_b)>len(pron_a):
                    pron_b = pron_b[-len(pron_a):]
                while not pron_a[0][-1].isdigit():
                    pron_a = pron_a[1:]
                while not pron_b[0][-1].isdigit():
                    pron_b = pron_b[1:]
                if pron_a == pron_b:
                    return True
        return False

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (or not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """
        lines = text.split("\n")
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        lines = [line for line in lines if line != ""]
        for i in range(len(lines)):
            lines[i] = lines[i].replace(",", "")
            lines[i] = lines[i].replace(".", "")
            lines[i] = lines[i].replace("?", "")
            lines[i] = lines[i].replace("!", "")
            lines[i] = lines[i].replace(";", "")
            lines[i] = lines[i].replace(":", "")
            lines[i] = lines[i].replace("-", "")
        
        if len(lines) != 5:
            return False
        if (not self.rhymes(lines[0].split()[-1], lines[1].split()[-1]) or not self.rhymes(lines[1].split()[-1], lines[4].split()[-1]) or not self.rhymes(lines[2].split()[-1], lines[3].split()[-1])
        or self.rhymes(lines[0].split()[-1], lines[2].split()[-1]) or self.rhymes(lines[0].split()[-1], lines[3].split()[-1])):
            return False
        return True

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = input()

    ld = LimerickDetector()
    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))