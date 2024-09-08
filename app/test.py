from app import RAG

def main():
    obj = RAG()
    res = obj.main()
    for i in range(3):
        print('Question : '+ res["question"][i] + '\n\nAns : ' + res["answer"][i] + '\n====\n')
        print("\n=====\n\n")
if __name__ == "__main__":
    main()
