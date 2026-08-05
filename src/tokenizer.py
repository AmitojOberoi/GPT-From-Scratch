class CharacterTokenizer:
    def __init__(self, text):
        """
        Build a character-level vocabulary.
        """

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.string_to_int = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.int_to_string = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    def encode(self, text):
        """
        Convert text -> list of integers.
        """

        return [
            self.string_to_int[c]
            for c in text
        ]

    def decode(self, tokens):
        """
        Convert list of integers -> text.
        """

        return "".join(
            self.int_to_string[token]
            for token in tokens
        )