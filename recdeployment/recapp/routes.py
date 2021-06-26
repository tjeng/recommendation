from flask import Flask, request, jsonify, render_template
from flask import current_app as app
from .recommendation import Recommendation

r = Recommendation()
#item_distance_matrix = r.item_embedding_distance_matrix()
# SW1113, SWD015, cjo6507qf00013ac37tbxxu8z, cjo09x89u00013gb80mn9xb57, cjo06esrd00013gblo1n3haw7

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
	id = [str(x) for x in request.form.values()]
	userid = id[0]
	if userid in r.user_dic.keys():
		item_output, rec_output = r.similar_recommendation(userid)
		return render_template('recommendation.html', items=item_output, rec_items=rec_output)
	else:
		rec_output = r.top_n_items()
		return render_template('topitems.html', rec_items = rec_output)
