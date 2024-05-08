### Aesthetic Functions

@staticmethod
def separator(length: int = 100, symbol: str = "=") -> None:
    print(symbol * length)
    return

@staticmethod
def title(message : str):
    separator()
    print(message)
    separator(symbol='-')

@staticmethod
def subtitle(message: str):
    print(message)
    separator(symbol='-')

@staticmethod
def align_integer(i : int, total : int):
    length = len(str(total))
    s = str(i)
    difference = length - len(s)
    aligned = [' '] * length
    for idx in range(len(s)):
        aligned[difference + idx] = s[idx]
    return "".join(aligned)