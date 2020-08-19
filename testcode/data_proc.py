import os, shutil
def dir_name(file_dir):   
    D = []
    for root, dirs, files in os.walk(file_dir):  
        for dir in dirs:
            D.append(dir)
    return D
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            L.append(os.path.join(root, file)) 
    return L
if __name__ == "__main__":
    root = r'H:\\imagenet\\val'
    des_path = r'F:\\project_deeplearning\\pytorch\\Source Compression\\dataset\\imagenet\\test1'
    D = dir_name(root)
    for i in range(1000):
        L = file_name(os.path.join(root,D[i]))
        for j in range(50):
            to_path = os.path.join(des_path,D[i])
            if not os.path.isdir(to_path):
                 os.makedirs(to_path)
            shutil.copy(L[j],to_path)
    print('done!')