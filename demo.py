from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sqlite3
import pandas as pd

conn = sqlite3.connect("schemes.db")
df = pd.read_sql_query("SELECT * FROM schemes", conn)

print(df.shape)