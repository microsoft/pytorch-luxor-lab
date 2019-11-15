import io
import zipfile

import requests

response = requests.get("https://ipsc.ksp.sk/2016/real/problems/l.zip", stream=True)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()
