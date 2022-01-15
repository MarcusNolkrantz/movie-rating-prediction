import pandas as pd
import re



def main():
    
    file = open("./output1-10.txt", "r", encoding='utf-8')

    #reg_exp = re.compile(r'^\d+$') 
    reg_exp = re.compile(r'^--score-- \d+$') 

    result = []

    current_review = ""
    
    while True:
        
        line = file.readline()
        if not line: break

        if reg_exp.search(line):
            score = line.split()[1]
            result.append([current_review, int(score)])
            current_review = ""
        else:
            current_review += line
            current_review += " "

    df = pd.DataFrame(result, columns=["reviews", "ratings"])

    df.to_pickle(path="./movies.bz2", compression="bz2")


main()