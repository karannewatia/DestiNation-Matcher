from . import *
from app import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import pickle
import json
import string
import math
import time
import re
from nltk.stem import WordNetLemmatizer

inverted_index = None
word_id_lookup = None
name_id_lookup = None
idf = None
inverted_dict_id_word = None
inverted_dict_id_name = None
doc_norms = None
niche_value = None
reviews_data = None
wikivoyage_lite = None
images = None
autocomplete_words = None
quuery_expansion = None

with open('./data/final.json') as wil_file:
    all_data = json.load(wil_file)


inverted_index = all_data["inverted_index"]
word_id_lookup = all_data["word_id_lookup"]
name_id_lookup = all_data["name_id_lookup"]
idf = all_data["idf"]
inverted_dict_id_word = all_data["inverted_dict_id_word"]
inverted_dict_id_name = all_data["inverted_dict_id_name"]
doc_norms = all_data["doc_norms"]
niche_value = all_data["niche_value"]
reviews_data = all_data["reviews_data"]
wikivoyage_lite = all_data["wikivoyage_lite"]
images = all_data["images"]
autocomplete_words = all_data["autocomplete_words"]
query_expansion = all_data["query_expansion"]

#in: score between 0-1
#out: html stars, rounded to nearest .1
def getStars(score):
    stars = []
    checked_star = '<span class="fa fa-star checked"></span>'
    unchecked_star = '<span class="fa fa-star"></span>'
    stars.append(checked_star if score > 0.1 else unchecked_star)
    stars.append(checked_star if score > 0.3 else unchecked_star)
    stars.append(checked_star if score > 0.5 else unchecked_star)
    stars.append(checked_star if score > 0.7 else unchecked_star)
    stars.append(checked_star if score > 0.9 else unchecked_star)
    return "".join(stars)

def format_type(t):
    if t == "intinerary":
        return "Itinerary"
    if t == "smallcity":
        return "Town"
    if t == "bigcity":
        return "City"
    if t == "district":
        return "District"
    if t == "park":
        return "Park"
    if t == "region":
        return "Region"
    return "Unknown"

@irsystem.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@irsystem.route('/', methods=['GET'])
def search():
    activities = request.args.get('activities')
    likes = request.args.get('likes')
    dislikes = request.args.get('dislikes')
    nearby = request.args.get('nearby')
    drinkingAge = request.args.get('drinkingAge')
    language = request.args.get('language')
    nearbySlider = request.args.get('nearbySlider')
    languageSlider =  request.args.get('languageSlider')
    drinkingSlider = request.args.get('drinkingSlider')

    # The following three will get you the slider values you need
    nearbySlider = request.args.get('nearbySlider')
    languageSlider = request.args.get('languageSlider')
    drinkingSlider = request.args.get('drinkingSlider')

    #radio for if we should use nicheness in ranking
    use_nicheness = request.args.get('useNicheness')
    use_nicheness = True if use_nicheness == "y" else False

    form_data = {
        "activities" : request.args.get('activities'),
        "likes" : request.args.get('likes'),
        "dislikes" : request.args.get('dislikes'),
        "nearby" : request.args.get('nearby'),
        "language" : request.args.get('language'),
        "drinkingAge" : request.args.get('drinkingAge'),
        "nearbySlider" : request.args.get('nearbySlider'),
        "languageSlider" : request.args.get('languageSlider'),
        "drinkingSlider" : request.args.get('drinkingSlider')
    }

    if not activities and not likes:
        return render_template('search.html', data=[], form_data=form_data, autocomplete_words=autocomplete_words)

    activities = activities.lower()
    activities = re.findall(r'[^,\s]+', activities)
    likes = likes.lower()
    likes = re.findall(r'[^,\s]+', likes)
    dislikes = dislikes.lower()
    dislikes = re.findall(r'[^,\s]+', dislikes)

    if nearby is None:
        nearby = ''
    if drinkingAge  is None:
        drinkingAge = ''
    if language is None:
        language = ''
    nearby = nearby.lower()
    drinkingAge = drinkingAge.lower()
    language = language.lower()

    # print(activities, likes, dislikes)

    global inverted_index
    global word_id_lookup
    global name_id_lookup
    global idf
    global inverted_dict_id_word
    global inverted_dict_id_name
    global doc_norms
    global niche_value
    global reviews_data
    global wikivoyage_lite
    #global sentiments
    global images
    global base_url


    base_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/"

    if not doc_norms:
        load_data()

    def advanced_search(ranking, is_boolean_search):

        if is_boolean_search:

            if nearby != '':
                nearbySlider_int = int(nearbySlider)
                if nearby in wikivoyage_lite:
                    for place in wikivoyage_lite[nearby]["nearby_links"]:
                        if place in name_id_lookup:
                            place_id = name_id_lookup[place]
                            ranking[place_id] *= nearbySlider_int*100

            if language != '':
                languageSlider_int = int(languageSlider)
                for place in wikivoyage_lite:
                    if place in name_id_lookup:
                        if language not in wikivoyage_lite[place]['languages']:
                            place_id = name_id_lookup[place]
                            ranking[place_id] *= (0.5/languageSlider_int)

            if drinkingAge != '':
                drinkingSlider_int = int(drinkingSlider)
                age = int(drinkingAge)
                for place in wikivoyage_lite:
                    if place in name_id_lookup:
                        if wikivoyage_lite[place]['drinking'] is None or wikivoyage_lite[place]['drinking'] == 0:
                            continue
                        elif wikivoyage_lite[place]['drinking'] > age:
                            place_id = name_id_lookup[place]
                            ranking[place_id] *= 0.5/drinkingSlider_int


        return ranking

    wnl = WordNetLemmatizer()

    def cos_sim(query):
        query_dict = {}
        ranking = [0] * 25381
        for query_type in range(len(query)):
            if query_type == 0:
                weight = 2
            elif query_type == 1:
                weight = 1
            else:
                weight = -2
            for token in set(query[query_type]):
                token = wnl.lemmatize(token)
                # Activity may not be in word_id_lookup
                if token not in word_id_lookup:
                    continue
                token_id = str(word_id_lookup[token])
                if str(word_id_lookup[token]) in inverted_index:
                    query_dict[token] = idf[token_id]
                    for idx, count in inverted_index[str(word_id_lookup[token])]:
                        if str(idx) in inverted_dict_id_name:
                            ranking[idx] += weight * \
                                query_dict[token] * count * idf[token_id]

        sum_sq = 0
        for v in query_dict:
            sum_sq += query_dict[v] * query_dict[v]
        norm_q = math.sqrt(sum_sq)

        for i in range(len(ranking)):
            if inverted_dict_id_name[str(i)] in wikivoyage_lite and float(doc_norms[str(i)]) != 0 and float(norm_q) != 0:
                ranking[i] = (ranking[i]/(float(norm_q) * float(doc_norms[str(i)])), i)
            else:
                ranking[i] = (0,i)

        ranking = advanced_search(ranking, False)
        sorted_ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
        final_ranking = sorted_ranking[:50]
        final_ranking = [
            (inverted_dict_id_name[str(x[1])], x[0]) for x in final_ranking]
        return final_ranking

    def boolean_search(query):
        ranking = [0] * 25381
        for query_type in range(len(query)):
            if query_type == 0:
                weight = 2
            elif query_type == 1:
                weight = 1
            else:
                weight = -2
            for token in set(query[query_type]):
                # Activity may not be in word_id_lookup
                if token not in word_id_lookup:
                    continue
                token_id = str(word_id_lookup[token])
                if str(word_id_lookup[token])  in inverted_index:
                    for idx, count in inverted_index[str(word_id_lookup[token])]:
                        if str(idx) in inverted_dict_id_name:
                            ranking[idx] += weight * count

        ranking = advanced_search(ranking, True)
        ranking = [(ranking[i], i) for i in range(len(ranking)) if inverted_dict_id_name[str(i)] in wikivoyage_lite]

        sorted_ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
        final_ranking = sorted_ranking[:20]
        final_ranking = [
            (inverted_dict_id_name[str(x[1])], x[0]) for x in final_ranking]
        return final_ranking

    #data = cos_sim([activities, likes, dislikes])

    new_activities = []
    new_likes = []
    new_dislikes = []

    for activity in activities:
        if activity in query_expansion:
            new_activities.append(activity)
            new_activities.extend(query_expansion[activity][:3])

    for like in likes:
        if like in query_expansion:
            new_likes.append(like)
            new_likes.extend(query_expansion[like][:3])

    for dislike in dislikes:
        if dislike in query_expansion:
            new_dislikes.append(dislike)
            new_dislikes.extend(query_expansion[dislike][:3])

    data = boolean_search([new_activities, new_likes, new_dislikes])
    # print(data[:25])
    sim_niche_list = []
    for loc in data:
        niche_score = niche_value[loc[0]]
        sim_niche_list.append((loc[0], niche_score, loc[1]))
    top_10 = None
    if use_nicheness:
        # sort by niche value
        sim_sorted_by_niche = sorted(
            sim_niche_list, key=lambda x: x[1], reverse=True)
        top_10 = sim_sorted_by_niche[:10]
    else:
        top_10 = sim_niche_list[:10]

    def get_reviews(locs):
        revs = [x for x in reviews_data[locs]]
        new_revs = []
        for r in revs:
            if r[0] == "No reviews available for this place":
                new_revs.append((r[0], r[1]))
            else:
                new_revs.append((" ".join(r[0].split(" ")[1:-1]), r[1]))
        return new_revs

    def get_relevant_keywords(location):
        result = []
        lst = new_activities + new_likes
        for token in lst:
            if str(word_id_lookup[token]) in inverted_index:
                id = name_id_lookup[location[0]]
                result.extend([token for item in inverted_index[str(word_id_lookup[token])] if item[0] == id])

        return result

    maxSim = max([l[2] for l in top_10]+[1])

    # results
    results_list = []

    for loc in top_10:
        l = wikivoyage_lite[loc[0]]['languages']
        d = wikivoyage_lite[loc[0]]['drinking']
        entry = {}
        entry["name"] = wikivoyage_lite[loc[0]]['title']
        entry["reviews"] = get_reviews(loc[0])
        entry["sim_stars"] = getStars(loc[2]/maxSim)
        #entry["sentiment"] = str(int(sentiments[loc[0]] * 100))
        entry["url"] = wikivoyage_lite[loc[0]]['url']
        entry["drinking"] = d if d else "unknown"
        entry["languages"] = ", ".join(l) if len(l) != 0 else "unknown"
        entry['type'] = format_type(wikivoyage_lite[loc[0]]['type'])
        entry['nicheness_stars'] = getStars(loc[1])
        entry['image'] = base_url + images[loc[0]].split(" ")[0]
        entry['relevant_keywords'] = get_relevant_keywords(loc)

        results_list.append(entry)

    data = results_list

    return render_template('search.html', data=data, form_data=form_data, autocomplete_words=autocomplete_words)
