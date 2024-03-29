import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
import uvicorn, aiohttp, asyncio
from chatterbot import ChatBot
from utils import *

export_file_url = 'https://drive.google.com/uc?export=download&id=16jVo7TB0TKsk4S1z0O0Lo3Tvclsn_XJx'
export_file_name = 'thread_embeddings_by_tags'
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        await download_file(export_file_url, export_file_name)
        try:
            embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
            thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        
        
        question_vec = question_to_vec(
            question, self.word_embeddings,300)
        
        best_thread = pairwise_distances_argmin(
            question_vec, thread_embeddings, metric = "cosine")[0]
              
        return thread_ids.values[best_thread]

        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
         self.chitchat_bot = ChatBot('Botty McBotFace',
                          trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

        # Train based on the english corpus
         self.chitchat_bot.train("chatterbot.corpus.english")
        """Initializes self.chitchat_bot with some conversational model."""

        
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question =  text_prepare(question)
         
        features = self.tfidf_vectorizer.transform([prepared_question])
        
        intent =  self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
             tag = self.tag_classifier.predict(features)
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question,tag[0])
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

