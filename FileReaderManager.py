
def getText(fileName,encoding):
    with open(fileName,encoding=encoding) as f:
        lines = [line.rstrip() for line in f]
        return " ".join(lines)