from enum import Enum


class printc(Enum):
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def print_col(cls, text: str, code):
        print(f'{code.value}{text}{cls.RESET.value}')

    @classmethod
    def green(cls, text: str):
        cls.print_col(text, cls.GREEN)

    @classmethod
    def cyan(cls, text: str):
        cls.print_col(text, cls.CYAN)

    @classmethod
    def yellow(cls, text: str):
        cls.print_col(text, cls.YELLOW)

    @classmethod
    def blue(cls, text: str):
        cls.print_col(text, cls.BLUE)