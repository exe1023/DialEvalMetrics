import argparse
import requests
import tarfile
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data downloading")

#    if not os.path.exists("data"):
#        os.makedirs("data")
#
#    os.chdir("data")

    file_id = ["1jkUeqUG0WFzSCmisbo1xTClRlCo8JPF3", "1YkXrkUdCFldl0EJXoMBy_uiu71wz1rgE", "1KLB3NSDjNv-ZX1I8pz4IxVRbvD3Bzbyk"]

    destination = ["ctx.zip", "roberta_ft.zip", "uk.zip"]

    for file_id_e, dest in zip(file_id, destination):
        download_file_from_google_drive(file_id_e, dest)
        #tar = tarfile.open(dest, "r:gz")
        #tar.extractall()
        #tar.close()
        #os.remove(destination)