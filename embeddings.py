import config
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai

openai.api_key = config.OPENAI_API_KEY

question_df = pd.read_csv('data/questions_with_embeddings.csv')
question_df['embedding'] = question_df['embedding'].apply(eval).apply(np.array)
print(question_df)

question = "What happened to SVB?"
question_vector = get_embedding(question, engine='text-embedding-ada-002')

question_df["similarities"] = question_df['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
question_df = question_df.sort_values("similarities", ascending=False)

best_answer = question_df.iloc[0]['answer']

print(best_answer)
