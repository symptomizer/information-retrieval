# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def queryParser(query, documents):
    '''
    :param query: string
    :param documents: ordered list of documents(str)  = List[str]
    :return: List[boolean] whether to return the doc as result or not based on query
    '''
    if isinstance(query, (str)):
        query = splitQuery(query)
    if query[-1] in ["AND","OR","NOT"]:
        query.append("")
    results = [any(performSearch(query, document, [])) for document in documents]
    return results

def performSearch(query, document, orList):
    if isinstance(query, (str)):
        query = splitQuery(query)

    if len(query)==0:
        return orList #choice made that on empty query, it returns true
    if len(document) == 0:
        return [False] #choice made that on empty document, it returns false

    if query[0] == "AND":
        index = 1
        [nextTokenInDocWithNOTs, indexOfNextToken] = evalAllConsecutiveNOT(query,document,index)
        orList[-1] = orList[-1] and nextTokenInDocWithNOTs
        return performSearch(query[indexOfNextToken+1:], document, orList)

    elif query[0] == "NOT":
        index = 0
        [nextTokenInDocWithNOTs, indexOfNextToken] = evalAllConsecutiveNOT(query,document,index)
        orList.append(nextTokenInDocWithNOTs)
        return performSearch(query[indexOfNextToken+1:], document, orList)

    else: #is normal token
        orList.append(stringInDoc(query[0],document))
        return performSearch(query[1:], document, orList)


def evalAllConsecutiveNOT(query, document, index):
    isNot = False
    #look for next token
    while query[index] == "NOT":
        isNot = not isNot
        index += 1
    tokenInDoc = stringInDoc(query[index],document)
    if isNot:
        return [not tokenInDoc, index]
    else:
        return [tokenInDoc, index]

def stringInDoc(s, d):
    # return s in d
    return d.find(s) != -1

def splitQuery(query):
    query = query.split(' ')
    newQuery = []
    temp = []
    collect = False
    for token in query:
        if len(token)> 1 and token[0] == '\"' and token[-1] == '\"':
            newQuery.append(token[1:-1])
        elif (not collect) and token[0] == '\"':
            collect = True
            temp.append(token[1:])
        elif collect and token[-1] == '\"':
            collect = False
            temp.append(token[:-1])
            newQuery.append(" ".join(temp))
            continue
        elif collect:
            temp.append(token)
        else:
            newQuery.append(token)

    return newQuery

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
