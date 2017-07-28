import sys
import pickle
import math
import traceback

def user_pair_sim(trust, rating):

    user_item_rating_dict = {}
    user_category_dict = {}

    for i in range(rating.shape[0]):
        if rating[i][0] not in user_item_rating_dict:
            user_item_rating_dict[rating[i][0]] = {}
        user_item_rating_dict[rating[i][0]][rating[i][1]] = rating[i][3]

        if rating[i][0] not in user_category_dict:
            user_category_dict[rating[i][0]] = {}
        user_category_dict[rating[i][0]][rating[i][2]] = 0

    user_user_similarity = similarity_H(
        trust, rating, user_item_rating_dict, user_category_dict)

    #usr_comm_usr_similarity = similarity_C(
    #    trust, rating, user_item_rating_dict, user_category_dict)

    return user_user_similarity


def similarity_C(trust, rating, user_item_rating_dict, user_category_dict):

    user_user_similarity = {}
    epsilon = 10e-6
    scaling_factor = 10.0



def similarity_H(trust, rating, user_item_rating_dict, user_category_dict):

    user_user_similarity = {}

    theta = 0.5
    epsilon = 10e-6
    epsilon2 = 10e-8
    scaling_factor = 10.0

    for i in range(trust.shape[0]):
        if trust[i][0] not in user_user_similarity:
            user_user_similarity[trust[i][0]] = {}
        r_sim = 0.0
        mod1 = 0.0
        mod2 = 0.0
        count = 0

        try:
            for x in user_item_rating_dict[trust[i][0]]:
                u1_rating = user_item_rating_dict[trust[i][0]][x]
                u2_rating = user_item_rating_dict[trust[i][1]].get(x,0)
                urating = u1_rating*u2_rating
                r_sim += urating
                mod1 += u1_rating*u1_rating
            mod1 = math.sqrt(mod1)

            for x in user_category_dict[trust[i][0]]:
                if x in user_category_dict[trust[i][1]]:
                    count += 1

            for x in user_item_rating_dict[trust[i][1]]:
                mod2 += user_item_rating_dict[trust[i][1]][x]*user_item_rating_dict[trust[i][1]][x]
            mod2 = math.sqrt(mod2)

        except:
            #print("error")
            #traceback.print_exc()
            pass

        if r_sim == 0.0:
            r_sim = epsilon*scaling_factor
            c_sim = count/27.0
            if c_sim == 0.0:
                c_sim = epsilon
            user_user_similarity[trust[i][0]][trust[i][1]] = (r_sim, c_sim)

        else:
            c_sim = count/27.0
            if c_sim == 0.0:
                c_sim = epsilon
            user_user_similarity[trust[i][0]][trust[i][1]] = (r_sim/(mod1*mod2+epsilon2),c_sim)

    return user_user_similarity
