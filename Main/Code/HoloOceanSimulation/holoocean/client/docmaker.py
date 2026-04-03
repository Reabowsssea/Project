import subprocess
import os
import shutil
import bs4 as bs

os.chdir("client")
print(os.getcwd())

def make_docs():
    if not os.path.exists("../docs"):
        os.mkdir("../docs")
    else:
        shutil.rmtree("../docs")
    try:
        subprocess.run(["sphinx-build","./docs","../docs"])
    except subprocess.CalledProcessError as e:
        print(f"An error occured while running sphinx-build: {e}")

def remove_long_list_of_links():
    with open("../docs/index.html", "r+", encoding='utf-8') as f:
        soup = bs.BeautifulSoup(f, "html.parser")
        for div in soup.find_all("div", class_="toctree-wrapper compound"):
            div.decompose()
        with open("../docs/index.html", "w", encoding='utf-8') as f1:
            pass
        f.write(str(soup))    
           

make_docs()
remove_long_list_of_links()