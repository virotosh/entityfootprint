# Zeinab Rezaei Yousefi <<zeinab.rezaeiyousefi@aalto.fi>>.
# This object loads the log data from the first phase of the user study
# A data matrix is a matrix with n rows (num_docs) and d columns (num_terms)
# the terms are themselves grouped in different views (num_views)

import numpy as np
import Utils as utils
from matplotlib import pyplot
from gensim import corpora, models, similarities
import json

def preprocess(data_dir, save_dir, process=False):
    corpus_unprocessed = corpora.MmCorpus(data_dir + '/corpus.mm')
    dictionary = corpora.Dictionary.load(data_dir + '/dictionary.dict')
    views_ind_unprocessed = np.load(data_dir + '/views_ind_1.npy')
    temp_dict = {}
    for i, v in enumerate(views_ind_unprocessed):
        temp_dict[dictionary[i]] = v
    all_docs = []
    len_doc = []
    for i, doc in enumerate(corpus_unprocessed):
        words = doc
        # print words
        new_words = []
        for w in words:
            for j in range(int(w[1])):
                new_words.append(dictionary[w[0]])
        all_docs.append(new_words)
        len_doc.append(len(new_words))
    entities_frequency = list(dictionary.dfs.values())

    if process == True:
        # Delete all the other entities excep than KW
        # indices: 0:BW, 1:KW, 2:app, 3:ppl, 4:doc, -1:time

        del_ids_time = [i for i, e in enumerate(list(views_ind_unprocessed)) if e == -1]
        # del_ids_BW = [i for i, e in enumerate(list(views_ind_unprocessed)) if e == 0]
        # del_ids_app = [i for i, e in enumerate(list(views_ind_unprocessed)) if e == 2]
        # del_ids_ppl = [i for i, e in enumerate(list(views_ind_unprocessed)) if e == 3]
        # del_ids_doc = [i for i, e in enumerate(list(views_ind_unprocessed)) if e == 4]
        del_ids = list(set(del_ids_time)) # | set(del_ids_app) | set(del_ids_doc)) #| set(del_ids_BW) | set(del_ids_ppl) )

        # hand_picked_stop_words = ['http', 'https', 'www', 'good', 'way', 'ways', 'change', 'file', 'files', 'return',
        #                         'join', 'open', 'save', 'tools', 'drag', 'check', 'name', 'get']
        # del_ids_stopWords = [dictionary.token2id[e] for i, e in enumerate(hand_picked_stop_words)]
        #
        # del_ids_lowFreqs = [i for i, e in enumerate(entities_frequency) if (list(temp_dict.values())[i] != 2
        #                                                                   & list(temp_dict.values())[i] != 4
        #                                                                    & (e <= 500 | e >= 1000))]
        # del_ids_lowFreqs = [i for i, e in enumerate(entities_frequency) if (e <= 5 or e >= 1000)]
        # del_ids = list(set(del_ids_stopWords) | set(del_ids_lowFreqs))
    else:
        del_ids = list()
    # print('number of removing terms, length dictionary before filter: ', len(del_ids), len(dictionary))
    dictionary.filter_tokens(bad_ids=del_ids)
    print('length dictionary after filter: ', len(dictionary))
    # " check the entities in e.g. people. The view_index for each category: People view 3 , Application view 2, KW view 1, BOW view 0 "
    # people_ind = [i for i in range(len(views_ind_unprocessed)) if views_ind_unprocessed[i] == 3]
    # people = []
    # for i in range(len(people_ind)):
    #     people.append(dictionary[people_ind[i]])
    # views_ind = np.delete(views_ind_unprocessed, del_ids)
    views_ind = np.zeros((len(dictionary),), dtype=int)
    for key in dictionary:
        views_ind[key] = temp_dict[dictionary[key]]
    print('num of terms unprocessed: ', corpus_unprocessed.num_terms)
    corpus_processed = [dictionary.doc2bow(doc) for doc in all_docs]
    # corpus_processed = corpus_unprocessed

    dictionary.save(save_dir + 'dictionary_processed.dict')
    corpora.MmCorpus.serialize(save_dir + 'corpus_processed.mm', corpus_processed)
    corpus = corpora.MmCorpus(save_dir + 'corpus_processed.mm')
    print('num of terms processed: ', corpus.num_terms)
    print('views_ind length: ', len(views_ind))
    # views_ind = views_ind_unprocessed
    vocab_size = corpus.num_terms

    all_docs = []
    len_doc = []
    for i, doc in enumerate(corpus):
        words = doc
        words = [dictionary[w[0]] for w in words]
        all_docs.append(words)
        len_doc.append(len(words))
    return (corpus, dictionary, views_ind, vocab_size, all_docs)

class DataLoader:

    def __init__(self, data_dir, save_dir, process=False):
        """For initialization"""
        #Parameters

        self.corpus, self.dictionary, self.views_ind, self.vocab_size, self.docs = preprocess(data_dir, save_dir, process)

        self.num_terms = self.corpus.num_terms          #total number of terms # here num_terms is num_entities
        self.num_docs = self.corpus.num_docs               #total number of docs (snapshots)
        self.num_views = max(self.views_ind) + 1           #total number of views views= [0=BOW, 1=KW, 2=App, 3=People]
        self.Data = None                                   #todo: (it may be impossible to create this) a num_data*num_featurs array
        #name of the terms
        self.term_names = [self.dictionary.get(i) for i in range(self.num_terms) ]
        self.num_items_per_view = [sum(self.views_ind == i) for i in range(self.num_views)]

    def print_info(self):
        print ('The corpus has %d docs' %self.num_docs+' and %d terms'%self.num_terms+\
              ' and %d views' %self.num_views +' there are %d' %self.corpus.num_nnz +' non-zero elements')
        print ('People view %d' % self.num_items_per_view[3]+ ' items, Application view %d' %self.num_items_per_view[2]+\
              ' items, KW view %d' %self.num_items_per_view[1]+' items, BOW view %d' %self.num_items_per_view[0]+' items.')

    def process_item_info(self):
        #This function is used for offline feedback gathering
        print ('The corpus has %d docs' %self.num_docs+' and %d terms'%self.num_terms+\
              ' and %d views' %self.num_views +' there are %d' %self.corpus.num_nnz +' non-zero elements')
        print ('People view %d' %sum(self.views_ind == 3)+ ' items, Application view %d' %sum(self.views_ind == 2)+\
              ' items, KW view %d' %sum(self.views_ind == 1)+' items, BOW view %d' %sum(self.views_ind == 0)+' items.')

        def returnReverse(k,v):
            return (v,k)
        #get the document frequency of the terms (i.e. how many document did a particular term occur in):
        term_frequency_dic = self.dictionary.dfs
        sorted_term_ferequency = sorted(term_frequency_dic.items(), key=lambda x : x[1], reverse=True)
        sorted_IDs = [sorted_term_ferequency[i][0] for i in range(self.num_terms)]

        count_term = [y for (x,y) in sorted_term_ferequency]
        pyplot.hist(count_term[9000:self.num_terms-1], 20, facecolor='green')
        pyplot.xlabel('number of occurrences in the corpus')
        pyplot.ylabel('count')
        pyplot.show()
        num_of_1_occurance = len([y for (x,y) in sorted_term_ferequency if y==1])
        print ('%d terms' %num_of_1_occurance+' have only appeared once in the corpus')
        term_names_1_occurance = [(self.term_names[x]) for (x,y) in sorted_term_ferequency if y==1 ]
        with open('term_names_1_occurance.txt', 'w') as outfile:
            json.dump(term_names_1_occurance, outfile)
        #those terms can be removed from the dictionary.todo: HOWEVER, they should be removed when the corpus is being made
        #print self.dictionary
        #self.dictionary.filter_extremes(no_below=2)
        #print self.dictionary

        BOW_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 0]
        BOW_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 0]

        KW_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 1]
        KW_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 1]

        APP_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 2]
        APP_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 2]

        PPL_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 3]
        PPL_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 3]

        num_to_show = 1000 #
        data = {}
        data["BOW_names"] = BOW_names  # [:num_to_show]
        data["KW_names"] = KW_names  # [:num_to_show]
        data["APP_names"] = APP_names #[:num_to_show]
        data["PPL_names"] = PPL_names #[:num_to_show]
        data["BOW_ids"] = BOW_ids #[:num_to_show]
        data["KW_ids"] = KW_ids #[:num_to_show]
        data["APP_ids"] = APP_ids #[:num_to_show]
        data["PPL_ids"] = PPL_ids #[:num_to_show]

        with open('BOW_names.txt', 'w') as outfile:
            json.dump(BOW_names, outfile)
        with open('KW_names.txt', 'w') as outfile:
            json.dump(KW_names, outfile)
        with open('APP_names.txt', 'w') as outfile:
            json.dump(APP_names, outfile)
        with open('PPL_names.txt', 'w') as outfile:
            json.dump(PPL_names, outfile)

        with open('for_vuong.txt', 'w') as outfile:
            json.dump(data, outfile)



class DataLoader_doc:

    def __init__(self, data_dir, save_dir, process=False):
        """For initialization"""
        #Parameters

        self.corpus, self.dictionary, self.views_ind, self.vocab_size, self.docs = preprocess(data_dir, save_dir, process)

        self.num_terms = self.corpus.num_terms          #total number of terms # here num_terms is num_entities
        self.num_docs = self.corpus.num_docs               #total number of docs (snapshots)
        self.num_views = max(self.views_ind) + 1           #total number of views views= [0=BOW, 1=KW, 2=App, 3=People, 4=doc, -1=time]
        self.Data = None                                   #todo: (it may be impossible to create this) a num_data*num_featurs array
        #name of the terms
        self.term_names = [self.dictionary.get(i) for i in range(self.num_terms) ]
        self.num_items_per_view = [sum(self.views_ind == i) for i in range(self.num_views)]

    def print_info(self):
        print ('The corpus has %d docs' %self.num_docs+' and %d terms'%self.num_terms+\
              ' and %d views' %self.num_views +' there are %d' %self.corpus.num_nnz +' non-zero elements')
        print ('DOC view %d' % self.num_items_per_view[4]+ ' items, People view %d' % self.num_items_per_view[3]+ ' items, Application view %d' %self.num_items_per_view[2]+\
              ' items, KW view %d' %self.num_items_per_view[1]+' items, BOW view %d' %self.num_items_per_view[0]+' items.')

    def process_item_info(self):
        #This function is used for offline feedback gathering
        print ('The corpus has %d docs' %self.num_docs+' and %d terms'%self.num_terms+\
              ' and %d views' %self.num_views +' there are %d' %self.corpus.num_nnz +' non-zero elements')
        print ('DOC view %d' % self.num_items_per_view[4]+ ' items, People view %d' %sum(self.views_ind == 3)+ ' items, Application view %d' %sum(self.views_ind == 2)+\
              ' items, KW view %d' %sum(self.views_ind == 1)+' items, BOW view %d' %sum(self.views_ind == 0)+' items.')

        def returnReverse(k,v):
            return (v,k)
        #get the document frequency of the terms (i.e. how many document did a particular term occur in):
        term_frequency_dic = self.dictionary.dfs
        sorted_term_ferequency = sorted(term_frequency_dic.items(), key=lambda x : x[1], reverse=True)
        sorted_IDs = [sorted_term_ferequency[i][0] for i in range(self.num_terms)]

        count_term = [y for (x,y) in sorted_term_ferequency]
        pyplot.hist(count_term[9000:self.num_terms-1], 20, facecolor='green')
        pyplot.xlabel('number of occurrences in the corpus')
        pyplot.ylabel('count')
        pyplot.show()
        num_of_1_occurance = len([y for (x,y) in sorted_term_ferequency if y==1])
        print ('%d terms' %num_of_1_occurance+' have only appeared once in the corpus')
        term_names_1_occurance = [(self.term_names[x]) for (x,y) in sorted_term_ferequency if y==1 ]
        with open('term_names_1_occurance.txt', 'w') as outfile:
            json.dump(term_names_1_occurance, outfile)
        #those terms can be removed from the dictionary.todo: HOWEVER, they should be removed when the corpus is being made
        #print self.dictionary
        #self.dictionary.filter_extremes(no_below=2)
        #print self.dictionary

        BOW_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 0]
        BOW_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 0]

        KW_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 1]
        KW_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 1]

        APP_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 2]
        APP_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 2]

        PPL_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 3]
        PPL_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                    if  self.views_ind[sorted_IDs[i]] == 3]

        DOC_names = [(self.term_names[sorted_IDs[i]]) for i in range(self.num_terms) \
                     if self.views_ind[sorted_IDs[i]] == 4]
        DOC_ids = [(sorted_IDs[i]) for i in range(self.num_terms) \
                   if self.views_ind[sorted_IDs[i]] == 4]

        num_to_show = 1000 #
        data = {}
        data["BOW_names"] = BOW_names  # [:num_to_show]
        data["KW_names"] = KW_names  # [:num_to_show]
        data["APP_names"] = APP_names #[:num_to_show]
        data["PPL_names"] = PPL_names #[:num_to_show]
        data["DOC_names"] = DOC_names  # [:num_to_show]
        data["BOW_ids"] = BOW_ids #[:num_to_show]
        data["KW_ids"] = KW_ids #[:num_to_show]
        data["APP_ids"] = APP_ids #[:num_to_show]
        data["PPL_ids"] = PPL_ids #[:num_to_show]
        data["DOC_ids"] = DOC_ids  # [:num_to_show]

        with open('BOW_names.txt', 'w') as outfile:
            json.dump(BOW_names, outfile)
        with open('KW_names.txt', 'w') as outfile:
            json.dump(KW_names, outfile)
        with open('APP_names.txt', 'w') as outfile:
            json.dump(APP_names, outfile)
        with open('PPL_names.txt', 'w') as outfile:
            json.dump(PPL_names, outfile)
        with open('DOC_names.txt', 'w') as outfile:
            json.dump(DOC_names, outfile)
        with open('DOC_ids.txt', 'w') as outfile:
            json.dump(DOC_ids, outfile)


        with open('for_vuong.txt', 'w') as outfile:
            json.dump(data, outfile)



def separate_type_entities(data):
    all_docs = []
    len_doc = []
    entities_views = []
    kw_per_doc = []
    app_per_doc = []
    ppl_per_doc = []
    time_per_doc = []
    BW_per_doc = []
    doc_per_doc = []
    app_ind_per_doc = []
    doc_ind_per_doc = []
    cnt = 0
    view_ind = []
    for i, doc in enumerate(data.corpus):
        words = doc
        words = [data.dictionary[w[0]] for w in words]
        all_docs.append(words)
        len_doc.append(len(words))
        if words == []:
            cnt += 1

        entities_views = [data.views_ind[data.dictionary.token2id[words[i]]]
                          for i in range(len(words))]
        BW_per_doc.append([words[i] for i in range(len(words))
                           if entities_views[i] == 0])
        kw_per_doc.append([words[i] for i in range(len(words))
                           if entities_views[i] == 1])
        app_per_doc.append([words[i] for i in range(len(words))
                            if entities_views[i] == 2])
        app_ind_per_doc.append([data.dictionary.token2id[words[i]] for i in range(len(words))
                                if entities_views[i] == 2])
        ppl_per_doc.append([words[i] for i in range(len(words))
                            if entities_views[i] == 3])
        time_per_doc.append([words[i] for i in range(len(words))
                             if entities_views[i] == -1])
        doc_per_doc.append([words[i] for i in range(len(words))
                            if entities_views[i] == 4])
        doc_ind_per_doc.append([data.dictionary.token2id[words[i]] for i in range(len(words))
                                if entities_views[i] == 4])

        view_ind.append([data.views_ind[data.dictionary.token2id[words[i]]] for i in range(len(words))])

    return (BW_per_doc, kw_per_doc, app_per_doc, ppl_per_doc, doc_per_doc, app_ind_per_doc, doc_ind_per_doc, time_per_doc, view_ind)
