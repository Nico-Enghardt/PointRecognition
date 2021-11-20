def distance(firstTuple, secondTuple):
    if len(firstTuple) == 2 and len(secondTuple) == 2:
        dx = firstTuple[0] - secondTuple[0]
        dy = firstTuple[1] - secondTuple[1]
        return abs(dx + dy*2)