from sklearn.externals import joblib

    
def predict(validation_df):
    num_rows = validation_df.shape[0]
    y = np.zeros((num_rows,1))
    for index, row in validation_df.iterrows():
        similar_cards = get_similar_card(row['name1'])
        similar_cards_name = [id_card_name[key] for key in similar_cards.keys()]
        if row['name2'] in similar_cards_name:
            similar = 1
        else: 
            similar = 0
        y[index] = similar

    return(y)

def get_all_pred_similar_ids(card_list):
    dic  = {}
    for card in card_list:
        similar_cards = get_similar_card(card)
        similar_cards_ids = list(similar_cards.keys())
        dic[card] = similar_cards_ids

    return(dic)


def get_card_name_and_similar_cards(card_list):
    similar_cards_list = get_all_pred_similar_ids(card_list)
    card_similar_id = pd.DataFrame({'card_name': list(similar_cards_list.keys()), 'Number' : list(similar_cards_list.values())}).explode('Number')
    return(decorate_with_description_and_card_name(card_similar_id))



def setup_bert():
    # device config
    NUM_GPUS = 0

    # model config
    LANGUAGE = Language.ENGLISH
    TO_LOWER = True
    MAX_SEQ_LENGTH = 128
    LAYER_INDEX = -2
    POOLING_STRATEGY = PoolingStrategy.MEAN

    # path config
    CACHE_DIR = "./temp"

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

    se = BERTSentenceEncoder(
        language=LANGUAGE,
        num_gpus=NUM_GPUS,
        cache_dir=CACHE_DIR,
        to_lower=TO_LOWER,
        max_len=MAX_SEQ_LENGTH,
        layer_index=LAYER_INDEX,
        pooling_strategy=POOLING_STRATEGY,
    )
    return(se)


def train_doc2vec(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    train_corpus = list()
    i = -1
    for tokens in cards_clean:
        i = i + 1
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, dm = 0)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return(model)

def get_embeddings_glove():
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open('/content/drive/My Drive/colab/resources/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return(embeddings_index)

def get_document_frequency(cards):
    document_frequency_dict = {}
    all_sentences = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    sentence = np.concatenate(all_sentences, axis=0)
    n = len(sentence)

    for s in all_sentences:
        for token in set(s):
            document_frequency_dict[token] = document_frequency_dict.get(token, 0) + 1

    return document_frequency_dict, n

def doc_embeddings_weighted_average(doc, embeddings_index):
    # Throw away tokens that are not in the embedding model
    tokens = [i for i in doc if i in embeddings_index]
    if len(tokens) == 0:
        return []

    # We will weight by TF-IDF. The TF part is calculated by:
    # (# of times term appears / total terms in sentence)
    count = Counter(tokens)
    token_list = list(count)
    term_frequency = [count[i] / len(tokens) for i in token_list]

    # Now for the IDF part: LOG(# documents / # documents with term in it)
    inv_doc_frequency = [
        math.log(num_documents / (document_frequencies.get(i, 0) + 1)) for i in count
    ]

    # Put the TF-IDF together and produce the weighted average of vector embeddings
    word_embeddings = [embeddings_index[token] for token in token_list]
    weights = [term_frequency[i] * inv_doc_frequency[i] for i in range(len(token_list))]
    doc_embeddings = np.average(word_embeddings, weights=weights, axis=0)
    return normalize(doc_embeddings)

def normalize(x):
    return(x / np.sqrt(np.sum(x**2)))

def doc_embeddings_min_max(doc, embeddings_index):
    embedding_dim = len(embeddings_index['1'])

    w_max_embeddings = -float("inf")*np.ones((1,embedding_dim))
    w_min_embeddings = float("inf")*np.ones((1,embedding_dim))  
    tokens = [i for i in doc if i in embeddings_index]

    for word in tokens:
        w_embedding = embeddings_index[word]
        w_max_embeddings = np.maximum(w_embedding.astype(float), w_max_embeddings)
        w_min_embeddings = np.minimum(w_embedding.astype(float), w_max_embeddings)

    w_min_max_embeddings = np.concatenate([w_min_embeddings, w_max_embeddings], axis = 1).flatten()
    return normalize(w_min_max_embeddings)

def tokenizize_and_clean(sentence):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in word_tokens if (not w in stop_words) and w.isalpha()] 
    return filtered_sentence

def glove_min_max_sentence_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_min_max(s, glove_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def word2vec_ifdf_sentence_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_weighted_average(s, word2vec_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def word2vec_min_max_sentence_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_min_max(s, word2vec_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def glove_ifdf_sentence_embeddings(cards): 
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_weighted_average(s, glove_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def fastext_ifdf_sentence_embeddings(cards): 
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_weighted_average(s, fastext_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def fastext_min_max_sentence_embeddings(cards): 
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc_embeddings_min_max(s, fastext_embeddings) for s in cards_clean]
    return (cards_clean_embedding)

def doc2vec_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_embedding = [doc2vecSelfTrainedModel.infer_vector(s) for s in cards_clean]
    return (cards_clean_embedding)

def bert_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_single_string = [" ".join(s) for s in cards_clean ]
    cards_clean_embedding = bert_se.encode(cards_clean_single_string, as_numpy = True)
    return (cards_clean_embedding)

def sbert_nli_mean_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_single_string = [" ".join(s) for s in cards_clean ]
    cards_clean_embedding = sbert.encode(cards_clean_single_string) 
    return (cards_clean_embedding)

def robert_nli_stsb_mean_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_single_string = [" ".join(s) for s in cards_clean ]
    cards_clean_embedding = robert_model.encode(cards_clean_single_string) 
    return (cards_clean_embedding) 

def sbert_base_uncased_fine_tunned_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_single_string = [" ".join(s) for s in cards_clean ]
    cards_clean_embedding = sbert_base_uncased_fine_tunned.encode(cards_clean_single_string) 
    return (cards_clean_embedding)

def robert_fine_tunned_embeddings(cards):
    cards_clean = [tokenizize_and_clean(s) for s in cards['Description_clean'] ]
    cards_clean_single_string = [" ".join(s) for s in cards_clean ]
    cards_clean_embedding = robert_fine_tunned.encode(cards_clean_single_string) 
    return (cards_clean_embedding)

def rand_embeddings_lowerbound(cards):
  return (np.random.rand(cards.shape[0],100))

embeddings_dispatcher = {   "GLOVE_MIN_MAX" : glove_min_max_sentence_embeddings 
                            ,"GLOVE_WEIGTHED_AVERAGE" : glove_ifdf_sentence_embeddings
                            ,"WORD2VEC_WEIGTHED_AVERAGE": word2vec_ifdf_sentence_embeddings
                            ,"WORD2VEC_MIN_MAX":word2vec_min_max_sentence_embeddings
                            ,"FASTTEXT_WEIGTHED_AVERAGE":fastext_ifdf_sentence_embeddings
                            ,"FASTTEXT_MIN_MAX":fastext_min_max_sentence_embeddings
                            ,"DOC2VEC_SELFTRAINED": doc2vec_embeddings
                            ,"BERT": bert_embeddings
                            ,"SBERT_NLI_MEAN":sbert_nli_mean_embeddings
                            ,"ROBERT_NLI_STSB_MEAN": robert_nli_stsb_mean_embeddings
                            ,"BERT_UNCASED_FINE_TUNED": sbert_base_uncased_fine_tunned_embeddings
                            ,"ROBERT_FINE_TUNED": robert_fine_tunned_embeddings
                            ,"RAND":rand_embeddings_lowerbound}


def model_name():
    return(MODELS_PATH+EMBEDDINGDS+"_KNN_Model.pkl")

def deck_name():
    return(MODELS_PATH+EMBEDDINGDS+"_deck.pkl")  

def generate_and_store_knn_model():
    deck = generate_card_embeddings_deck(cards)
    nbrsModel = knn_model(deck)
    joblib_file =  model_name()
    joblib.dump(nbrsModel, joblib_file)

    joblib_file = deck_name() 
    joblib.dump(deck, joblib_file)


def load_model():
    return(joblib.load(model_name()))

def load_deck():
    return(joblib.load(deck_name()))

def generate_card_embeddings_deck(cards):

    emb = embeddings_dispatcher[EMBEDDINGDS]
    cards_embedding = emb(cards)
    deck = {}
    i = 0
    for number in cards['Number']: 
        deck[number] = cards_embedding[i]
        i=i+1
    return(deck)

def knn_model(deck):
    nbrs = NearestNeighbors(n_neighbors = N_NEIGHBORS + 1, algorithm = 'ball_tree').fit(np.array(list(deck.values())))
    return(nbrs)

def knn_similarity(id):
    similarity={} 
    key_idx = np.array(list(deck.keys()))  
    distance, index = nbrsModel.kneighbors(deck[id].reshape(1, -1))
    similarids = key_idx[index].flatten()
    distance = distance.flatten()

    i = 0
    for key in similarids[similarids!=id]:
        similarity[key]= 1/(1 + distance[i])
        i = i +1

    return(similarity)

def brute_force_cosine_similarity(id):
    deckSimilarity={}
    cardEmbedding = deck.get(id)
    keys = np.array(list(deck.keys()))
    keys = keys[keys!=id]

    for key in keys:
        cardEmbedding2 = deck.get(key)
        similarity = cosine_similarity(cardEmbedding.reshape(1, -1), cardEmbedding2.reshape(1, -1))
        deckSimilarity[key] = similarity.item()

    res = dict(sorted(deckSimilarity.items(), key = itemgetter(1), reverse = True)[:N_NEIGHBORS]) 
    return(res)

closer_distance_dispatcher = { "KNN": knn_similarity, "BFCS": brute_force_cosine_similarity }

