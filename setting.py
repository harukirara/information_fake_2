import os
from os.path import join, dirname
import dotenv

dotenv_path = join(dirname("./ban/"), '.env')
dotenv.load_dotenv(dotenv_path)

AP = os.environ.get("API_KEY") # 環境変数の値をAPに代入