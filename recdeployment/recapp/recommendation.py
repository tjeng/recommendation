import json
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommendation:
    def __init__(self):
        with open("./recapp/data/user_mapping.json", "r") as uf:
            self.user_dic = json.load(uf)
        with open("./recapp/data/item_mapping.json") as imf:
            self.item_dic = json.load(imf)
        with open("./recapp/data/item_dictionary.json") as pf:
            self.product_dic = json.load(pf)
        with open("./recapp/static/model/recommender_model_alldata.pkl", 'rb') as f:
            self.model = pickle.load(f)
        self.interactions = sparse.load_npz("./recapp/data/interactions.npz")
        self.item_rev_dic = {v:k for k,v in self.item_dic.items()}

    def similar_recommendation(self, user_id, threshold = 0, number_rec_items = 5):
        #Function to produce user recommendations
        n_items = len(self.item_dic.keys())
        user_x = self.user_dic[user_id]
        scores = pd.Series(self.model.predict(user_x,np.arange(n_items)))
        scores.index = self.item_dic.keys()
        scores = list(pd.Series(scores.sort_values(ascending=False).index))
        user_item_scores = self.interactions.toarray()[user_x]
        item_index = np.where(user_item_scores>0)[0]
        known_items = pd.DataFrame({ 'items': [self.item_rev_dic[i] for i in item_index], 'scores':user_item_scores[item_index]}).sort_values('scores', ascending=False)
        
        scores = [x for x in scores if x not in known_items['items'].values]
        score_list = scores[0:number_rec_items]

        items_output = []
        #rint("Items that were liked by the User:")
        counter = 1
        for i in range(len(known_items)):
            item = known_items.loc[i]['items']
            if str(self.product_dic[item]['name']) != 'nan':
                items_output.append(str(counter) + '- ' + str(self.product_dic[item]['name']) + ', ' + str(self.product_dic[item]['price']) + ', ' + self.product_dic[item]['collection'])
            else:
                items_output.append(str(counter) + '- ' + str(self.product_dic[item]['collection']))
            counter+=1

        rec_output = []
        #print("\n Recommended Items:")
        counter = 1
        for i in score_list:
            if str(self.product_dic[i]['name']) != 'nan':
                rec_output.append(str(counter) + '- ' + self.product_dic[i]['name'] + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]["collection"]) 
            else:
                rec_output.append(str(counter) + '- ' + self.product_dic[i]['collection'])
            counter+=1
        return items_output, rec_output
		
    def top_n_items(self, n = 5):
    	np_item = self.interactions.toarray().sum(axis=0)
    	df = pd.DataFrame({'num_purchase':np_item})
    	top_n_df = df.sort_values(['num_purchase'], ascending=False).head(n).index
    	top_n = [self.item_rev_dic[i] for i in top_n_df]
    	products = []
    	counter = 1
    	for i in top_n:
    		if str(self.product_dic[i]['name']) != 'nan':
    			products.append(str(counter) + '- ' + self.product_dic[i]['name'] + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]["collection"]) 
    		else:
    			products.append(str(counter) + '- ' + self.product_dic[i]['collection'])
    		counter+=1
    	return products

    def item_embedding_distance_matrix(self):
    #     Function to create item-item distance embedding matrix
        df_item_norm_sparse = csr_matrix(self.model.item_embeddings)
        similarities = cosine_similarity(df_item_norm_sparse)
        item_emdedding_distance_matrix = pd.DataFrame(similarities)
        item_emdedding_distance_matrix.columns = self.item_dic.keys()
        item_emdedding_distance_matrix.index = self.item_dic.keys()
        return item_emdedding_distance_matrix

    def also_bought_recommendation(self, item_emdedding_distance_matrix, item_id, n_items = 5):
    #     Function to create item-item recommendation
        recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                      sort_values(ascending = False).head(n_items+1). \
                                      index[1:n_items+1]))
        
        #print("Item of interest:")
        item_interest = (str(self.product_dic[item_id]['name']) + ', ' + str(self.product_dic[item_id]['price']) + ', ' + self.product_dic[item_id]['collection'])
        #print("\n")
        #print("Items that are frequently bought together:")
        item_rec = []
        counter = 1
        for i in recommended_items:
            if i in self.product_dic.keys():
                item_rec.append(str(counter) + '- ' + str(self.product_dic[i]['name']) + ', ' + str(self.product_dic[i]['price']) + ', ' + self.product_dic[i]['collection'])
            else:
                item_rec.append(str(counter) + '- ' + 'swansonhealthproduct')
            counter+=1
        return item_interest, item_rec

if __name__ == "__main__":
    r = Recommendation()