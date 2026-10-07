from langchain_community.utilities import SQLDatabase
import requests


class Database:
    def __init__(self, db_url: str, verbose: bool = False):
        super().__init__()
        response = requests.get(db_url)
        file_name = db_url.split("/")[-1]
        if response.status_code == 200:
            # Open a local file in binary write mode
            with open(file_name, "wb") as file:
                # Write the content of the response (the file) to the local file
                file.write(response.content)
            if verbose:
                print("File downloaded and saved as Chinook.db")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
        self.db = SQLDatabase.from_uri(f"sqlite:///{file_name}")
